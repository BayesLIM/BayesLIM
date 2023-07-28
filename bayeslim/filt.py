"""
Module for visibility and map filtering
"""
import torch
import numpy as np

from . import utils, dataset, linalg


class BaseFilter(utils.Module):
    """
    Base filter class for 1D filtering of tensors,
    VisData, MapData or CalData
    """
    def __init__(self, dim=0, name=None, attrs=[]):
        """
        Parameters
        ----------
        dim : int, optional
            Dimension of input tensors to filter.
        """
        super().__init__(name=name)
        self.dim = dim
        self.attrs = attrs
        self.device = None

    def push(self, device):
        dtype = isinstance(device, torch.dtype)
        if not dtype:
            self.device = device
        for attr in self.attrs:
            if hasattr(self, attr):
                setattr(self, attr, utils.push(getattr(self, attr), device))


class GPFilter(BaseFilter):
    """
    A Gaussian Process filter.
    MAP prediction of a noisy signal (e.g. Wiener filter)

    y_filt = G y

    where

    G = C_signal [C_signal + C_noise]^-1
    
    where C_signal is the covariance of the signal
    to remove from the data, and C_noise is the
    covariance of the remaining parts of the data
    (can include thermal noise and whatever other
    terms in the data).
    """
    def __init__(self, Cs, Cn, dim=0, hermitian=True, no_filter=False,
                 rcond=1e-15, dtype=None, device=None, residual=False,
                 name=None):
        """
        Parameters
        ----------
        Cs : tensor
            Square covariance of signal you want to estimate.
            of shape (N_pred_samples, N_data_samples)
        Cn : tensor
            Square covariance of the noise (and other things)
            of shape (N_data_samples, N_data_samples)
        dim : int
            Dimension of input data to apply filter
        hermitian : bool, optional
            If input covariances are real-symmetric or complex-hermitian.
            Generally this is true, unless one applies a 
            complex phasor in front of Cs.
        no_filter : bool, optional
            If True, don't filter the input data and
            return as-is
        rcond : float, optional
            rcond parameter when taking pinv of C_data
        dtype : torch dtype, optional
            This is the data type of the input data to-be filtered.
        residual : bool, optional
            If True, subtract MAP estimate of signal from data to form
            the residual, otherwise simply return its MAP estimate (default)
        name : str, optional
            Name of the filter
        """
        attrs = ['Cs', 'Cn', 'C', 'C_inv', 'G', 'V']
        super().__init__(dim=dim, name=name, attrs=attrs)
        self.Cs = torch.as_tensor(Cs, device=device)
        self.Cn = torch.as_tensor(Cn, device=device)
        self.dtype = dtype
        self.rcond = rcond
        self.ein = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.hermitian = hermitian
        self.no_filter = no_filter
        self.residual = residual

        self.setup_filter()

    def setup_filter(self, Cs=None, Cn=None, **inv_kwargs):
        """
        Setup the filter matrix given self.Cs, self.Cn
        and self.rcond. This takes pseudo-inv of C_data
        and sets self.G (signal map prediction matrix)
        and self.V (signal variance matrix)

        Parameters
        ----------
        Cs : tensor, optional
            Set self.Cs as Cs if provided, otherwise keep self.Cs
        Cn : tensor, optional
            Set self.Cn as Cn if provided, otherwise keep self.Cn
        inv_kwargs : dict, optional
            Kwargs to send to linalg.invert_matrix() for
            inverting self.C
        """
        # G = S [S + N]^-1
        self.Cs = self.Cs if Cs is None else Cs
        self.Cn = self.Cn if Cn is None else Cn
        self.C = self.Cs + self.Cn
        self.C_inv = linalg.invert_matrix(self.C, **inv_kwargs)
        self.C_inv = self.C_inv.to(self.dtype).to(self.device)
        self.Cs = self.Cs.to(self.dtype).to(self.device)

        self.set_GV()

    def set_GV(self):
        """
        Setup filtering matrices G and the variance matrix V
        given self.Cs and self.C_inv
        """
        self.G = self.Cs @ self.C_inv
        self.V = self.Cs - self.Cs @ self.C_inv @ self.Cs.T.conj()

    def predict(self, inp, Cs=None):
        """
        Given input data, form the prediction
        of the signal

        y_map = G @ y_inp

        Note that its covariance is held as self.V

        Parameters
        ----------
        inp : tensor or dataset.TensorData subclass
            Data to filter
        Cs : tensor, optional
            Square (or rectangular) matrix holding the
            covariance of the estimated signal for each
            (x^prime, x), of shape (N_x_new, N_x),
            where N_x are the number of data samples,
            and N_x_new are the number of points to estimate
            the signal. By default, x_new = x such that
            Cs is square and stored as self.Cs. Passing
            a Cs here allows one to estimate the signal at
            new points along the data sampling axis.

        Returns
        -------
        tensor or dataset
        """
        if isinstance(inp, dataset.TensorData):
            out = inp.copy()
            out.data = self.predict(out.data)
            return out

        # assume inp is a tensor from here
        G = self.G if Cs is None else Cs @ self.C_inv

        ein = self.ein.copy()
        ein = ein[:inp.ndim]
        ein[self.dim] = 'j'
        ein = ''.join(ein)
        y = torch.einsum("ij,{}->{}".format(ein, ein.replace('j','i')), G, inp)

        return y

    def forward(self, inp, Cs=None, **kwargs):
        """
        Filter the input and return
        """
        if self.no_filter:
            return inp

        if isinstance(inp, np.ndarray):
            inp = torch.as_tensor(inp)

        elif isinstance(inp, dataset.TensorData):
            out = inp.copy()
            out.data = self.forward(inp.data, **kwargs)
            return out

        # assume inp is a tensor from here
        y_filt = self.predict(inp, Cs=Cs)

        if self.residual:
            y_filt = inp - y_filt

        return y_filt


def rbf_cov(x, ls, amp=1, x2=None, dtype=None, device=None):
    """
    Generate a Gaussian (aka RBF) covariance

    amp * exp(-.5 dx^2 / ls^2)

    Parameters
    ----------
    x : tensor
        Independent axis labels of the data
        e.g. frequencies, times, etc
    ls : float
        Length-scale of the covariance (i.e. 1 / filter half-width)
        in units of [x]
    amp : float, optional
        Multiplicative variance, default is 1
    x2 : tensor, optional
        Second set of independent axis labels, generating
        a non-square covariance matrix.

    Returns
    -------
    tensor
    """
    x = torch.atleast_2d(x)
    x2 = x if x2 is None else torch.atleast_2d(x2)
    cov = amp * torch.exp(-.5 * (x2.T - x)**2 / ls**2)
    cov = cov.to(device)
    if dtype is not None:
        cov = cov.to(dtype)

    return cov


def exp_cov(x, ls, amp=1, x2=None, dtype=None, device=None):
    """
    Generate an exponential covariance

    amp * exp(-dx / ls)

    Parameters
    ----------
    x : tensor
        Independent axis labels of the data
        e.g. frequencies, times, etc
    ls : float
        Length-scale of the covariance (i.e. 1 / filter half-width)
        in units of [x]
    amp : float, optional
        Multiplicative variance, default is 1
    x2 : tensor, optional
        Second set of independent axis labels, generating
        a non-square covariance matrix.

    Returns
    -------
    tensor
    """
    x = torch.atleast_2d(x)
    x2 = x if x2 is None else torch.atleast_2d(x2)
    cov = amp * torch.exp(-torch.abs(x2.T - x) / ls)
    cov = cov.to(device)
    if dtype is not None:
        cov = cov.to(dtype)

    return cov


def sinc_cov(x, ls, amp=1, x2=None, dtype=None, device=None):
    """
    Generate a Sinc covariance

    Parameters
    ----------
    x : tensor
        Independent axis labels of the data
        e.g. frequencies, times, etc
    ls : float
        Length-scale of the covariance (i.e. 1 / filter half-width)
        in units of [x]
    amp : float, optional
        Multiplicative variance, default is 1
    x2 : tensor, optional
        Second set of independent axis labels, generating
        a non-square covariance matrix.

    Returns
    -------
    tensor
    """
    x = torch.atleast_2d(x)
    x2 = x if x2 is None else torch.atleast_2d(x2)
    cov = amp * torch.sinc((x2.T - x) / ls)
    cov = cov.to(device)
    if dtype is not None:
        cov = cov.to(dtype)

    return cov


def phasor_mat(x, shift, neg=True, x2=None, dtype=None, device=None):
    """
    Generate a complex phasor matrix

    Parameters
    ----------
    x : tensor
        Independent axis labels of the data
        e.g. frequencies, times, etc
    shift : float
        Amount of complex shift to apply to a covariance,
        in units of 1 / [x].
    neg : bool, optional
        Sign convention for Fourier term. If True will use
        exp(-2j) otherwise will use exp(2j)
    x2 : tensor, optional
        Second set of independent axis labels, generating
        a non-square covariance matrix.

    Returns
    -------
    tensor
    """
    x = torch.atleast_2d(x)
    x2 = x if x2 is None else torch.atleast_2d(x2)
    coeff = 2j * np.pi
    if neg:
        coeff *= -1
    cov = torch.exp(coeff * (x2.T - x) * shift)
    cov = cov.to(device)
    if dtype is not None:
        cov = cov.to(dtype)

    return cov


def gauss_sinc_cov(x, gauss_ls, sinc_ls, high_prec=False):
    """
    Convolution of a Gaussian and Sinc covariance function
    See appendix A2 of arxiv:1608.05854
    """
    raise NotImplementedError
    sinc_ls = sinc_ls * (2 / np.pi)

    arg = gauss_ls / np.sqrt(2) / sinc_ls
    Xc = X / gauss_ls / np.sqrt(2)

    dists = pdist(Xc, metric="euclidean")
    K = func(dists, arg)
    K = squareform(K)
    np.fill_diagonal(K, 1)

    if high_prec:
        import mpmath
        fn = lambda z: mpmath.exp(-z**2) * (mpmath.erf(arg + 1j*z) + mpmath.erf(arg - 1j*z)).real
        K = 0.5 * np.asarray(np.frompyfunc(fn, 1, 1)(dists), dtype=float) / special.erf(arg)

    else:
        K = 0.5 * np.exp(-dists**2) / special.erf(arg) \
            * (special.erf(arg + 1j*dists) + special.erf(arg - 1j*dists))
        # replace nans with zero: in this limit, you should use high_prec
        # but this is a faster approximation
        K[np.isnan(K)] = 0.0

    return K


def gen_cov_modes(cov, N=None, rcond=None, device=None, dtype=None):
    """
    Given a hermitian covariance matrix, take its SVD and return the top
    N modes, or all modes with singular value at least rcond above the
    max singular value

    Parameters
    ----------
    cov : tensor
        A covariance matrix of shape (M, M) to decompose
    N : int, optional
        Take the top N modes where N <= M
    rcond : float, optional
        Take all modes that have singular value > S.max() * rcond
    device : str, optional
        Push modes to a new device
    dtype : type, optional
        Cast modes to a new data type

    Returns
    -------
    A : tensor
        Forward model matrix of shape (M, N)
    evals : tensor
        Eigenvalues of cov
    """
    assert N is None or rcond is None, "cannot provide both N and rcond"
    evals, A = torch.linalg.eigh(cov)
    A = A.flip([1])
    evals = evals.flip([0])

    if N is not None:
        A = A[:, :N]

    elif rcond is not None:
        A = A[:, evals >= evals.max() * rcond]

    A = A.to(device)
    if dtype is not None:
        A = A.to(dtype)

    return A, evals

