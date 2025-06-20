"""
Module for visibility and map filtering
"""
import torch
import numpy as np
from scipy import special

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


class MatFilter(BaseFilter):
    """
    A generic matrix filtering of the form

    y_filt = G @ y

    if residual is True then the output is

    y_filt = y - G @ y
    """
    def __init__(self, G=None, dim=-1, dtype=None, device=None,
                 residual=False, input_idx=None,
                 inplace=False, name=None, attrs=None):
        """
        Parameters
        ----------
        G : tensor
            Filtering matrix of shape (N_pred_samples, N_data_samples)
        dim : int
            Dimension of input data to apply filter
        dtype : torch dtype, optional
            This is the data type of the input data to-be filtered.
        residual : bool, optional
            If True, subtract MAP estimate of signal from data to form
            the residual, otherwise simply return its MAP estimate (default)
        input_idx : tensor, optional
            Indices of the input to modify along dim
            with the filtered output. Default is all indices.
        inplace : bool, optional
            If True, edit input tensor in place in forward(),
            otherwise make a copy (default).
        name : str, optional
            Name of the filter
        """
        _attrs = attrs
        attrs = ['G', 'input_idx']
        if _attrs is not None:
            attrs += _attrs
        super().__init__(dim=dim, name=name, attrs=attrs)
        self.G = torch.as_tensor(G, device=device) if G is not None else G
        self.dtype = dtype
        self.ein = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.residual = residual
        self.input_idx = input_idx
        self.inplace = inplace

    def setup_filter(self, G=None):
        """
        Setup the filter matrix
        """
        self.G = torch.as_tensor(G, device=device) if G is not None else self.G

    def predict(self, y, **kwargs):
        """
        Filter the input data as

        y_filt = G @ y

        Parameters
        ----------
        y : tensor or dataset.TensorData subclass
            Data to filter

        Returns
        -------
        tensor or dataset
        """
        if isinstance(y, dataset.TensorData):
            out = y.copy()
            out.data = self.predict(out.data)
            return out

        # assume y is a tensor from here
        # get G matrix
        G = self.G

        # index it if necessary
        if hasattr(self, "_idx") and self._idx is not None:
            G = G[self._rowidx, self._idx]

        ein = self.ein.copy()
        ein = ein[:y.ndim]
        ein[self.dim] = 'j'
        ein = ''.join(ein)
        y_filt = torch.einsum("ij,{}->{}".format(ein, ein.replace('j','i')), G, y)

        return y_filt

    def forward(self, y, **kwargs):
        """
        Filter the input and return
        """
        if isinstance(y, np.ndarray):
            y = torch.as_tensor(y)

        elif isinstance(y, dataset.TensorData):
            out = y.copy(copydata=False, copymeta=False)
            out.data = self.forward(y.data, **kwargs)
            return out

        else:
            # tensor
            if not self.inplace:
                y = y.clone()

        # assume y is a tensor from here
        y_filt = self.predict(y, **kwargs)

        # get indexing if needed
        if self.input_idx is not None:
            idx = [slice(None) for i in range(y.ndim)]
            idx[self.dim] = self.input_idx
        else:
            idx = slice(None)

        if self.residual:
            y[idx] -= y_filt
        else:
            if self.input_idx is not None:
                y[idx] = y_filt
            else:
                y = y_filt

        return y

    def set_G_idx(self, idx=None, rowidx=None):
        """
        Set indexing of G before applying to input

        Parameters
        ----------
        idx : tensor or slice object, optional
            Pick out these elements from rows and cols of G.
        rowidx : list or slice object, optional
            If provided, treat idx as column indexing of G,
            and this as row indexing of G, otherwise use
            idx for column and row indexing (default)
        """
        if idx is not None:
            if isinstance(idx, slice):
                pass
            else:
                idx = torch.atleast_2d(torch.as_tensor(idx))
        self._idx = idx

        rowidx = rowidx if rowidx is not None else idx
        if rowidx is not None:
            if isinstance(rowidx, slice):
                pass
            else:
                rowidx = torch.atleast_2d(torch.as_tensor(rowidx)).T
        self._rowidx = rowidx


class GPFilter(MatFilter):
    """
    A Gaussian Process filter.
    MAP prediction of a noisy signal (e.g. Wiener filter)

    y_filt = G y

    where

    G = C_signal^pred [C_signal + C_noise]^-1
    
    where C_signal is the covariance of the signal
    to remove from the data, and C_noise is the
    covariance of the remaining parts of the data
    (can include thermal noise and whatever other
    terms in the data).
    """
    def __init__(self, Cs, Cn, Cs_cross=None, Cs_pred=None, dim=-1,
                 dtype=None, device=None, residual=False, input_idx=None,
                 inplace=False, name=None, inv='pinv', hermitian=True, rcond=1e-15, eps=None):
        """
        Parameters
        ----------
        Cs : tensor
            Square covariance of signal you want to estimate.
            of shape (N_pred_samples, N_data_samples)
        Cn : tensor
            Square covariance of the noise (and other things)
            of shape (N_data_samples, N_data_samples)
        Cs_cross : tensor, optional
            Cross-covariance of signal between
            (prediction points, data points). Default is Cs.
        Cs_pred : tensor, optional
            Auto-covariance of signal between
            (prediction points, prediction points). Default is Cs.
        dim : int, optional
            Dimension of input data to apply filter
        dtype : torch dtype, optional
            This is the data type of the input data to-be filtered.
        residual : bool, optional
            If True, subtract MAP estimate of signal from data to form
            the residual, otherwise simply return its MAP estimate (default)
        input_idx : tensor, optional
            Indices of the input to modify along dim
            with the filtered output. Default is all indices.
        name : str, optional
            Name of the filter
        inv : str, optional
            Inversion type, default is pinv
        hermitian : bool, optional
            If input covariances are real-symmetric or complex-hermitian.
            Generally this is true, unless one applies a 
            complex phasor in front of Cs.
        rcond : float, optional
            rcond parameter when taking pinv of C_data
        eps : float, optional
            Reguarlization parameter for inversion of C_data
        """
        attrs = ['Cs', 'Cn', 'C', 'C_inv', 'G', 'V', 'input_idx']
        super().__init__(dim=dim, name=name, attrs=attrs, input_idx=input_idx, inplace=inplace)
        self.Cs = torch.as_tensor(Cs, device=device)
        self.Cn = torch.as_tensor(Cn, device=device)
        self.Cs_pred = torch.as_tensor(Cs_pred, device=device) if Cs_pred is not None else Cs_pred
        self.Cs_cross = torch.as_tensor(Cs_cross, device=device) if Cs_cross is not None else Cs_cross
        self.dtype = dtype
        self.ein = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.residual = residual
        self.rcond = rcond
        self.hermitian = hermitian
        self.eps = eps
        self.inv = inv

        self.setup_filter()

    def setup_filter(self, Cs=None, Cn=None, Cs_pred=None, Cs_cross=None, inv=None,
                     hermitian=None, rcond=None, eps=None):
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
        Cs : tensor, optional
            Set self.Cs as Cs if provided, otherwise keep self.Cs
        kwargs : dict, optional
            Kwargs to send to linalg.invert_matrix() for
            inverting self.C. Overwrites current defaults if provided
        """
        # G = S [S + N]^-1
        self.Cs = self.Cs if Cs is None else Cs
        self.Cn = self.Cn if Cn is None else Cn
        self.Cs_pred = self.Cs_pred if Cs_pred is None else Cs_pred
        self.Cs_cross = self.Cs_cross if Cs_cross is None else Cs_cross
        self.C = self.Cs + self.Cn
        self.inv = self.inv if inv is None else inv
        self.hermitian = self.hermitian if hermitian is None else hermitian
        self.rcond = self.rcond if rcond is None else rcond
        self.eps = self.eps if eps is None else eps
        self.C_inv = linalg.invert_matrix(self.C, inv=self.inv, hermitian=self.hermitian,
                                          rcond=self.rcond, eps=self.eps)
        self.C_inv = self.C_inv.to(self.dtype).to(self.device)
        self.Cs = self.Cs.to(self.dtype).to(self.device)
        if self.Cs_pred is not None:
            self.Cs_pred = self.Cs_pred.to(self.dtype).to(self.device)
        if self.Cs_cross is not None:
            self.Cs_cross = self.Cs_cross.to(self.dtype).to(self.device)

        self.set_GV()

    def set_GV(self):
        """
        Setup filtering matrices G and the variance matrix V
        given self.Cs and self.C_inv
        """
        Cs = self.Cs if self.Cs_cross is None else self.Cs_cross
        Cs_pred = self.Cs if self.Cs_pred is None else self.Cs_pred
        self.G = Cs @ self.C_inv
        self.V = Cs_pred - Cs @ self.C_inv @ Cs.T.conj()


class LstSqFilter(MatFilter):
    """
    A least squares filter
    """
    def __init__(self, G, dim=-1, device=None, dtype=None,
                 residual=True, name=None):
        """
        Parameters
        ----------
        G : tensor
            Filtering matrix of shape (N_pred_samples, N_data_samples)
        dim : int
            Dimension of input data to apply filter
        dtype : torch dtype, optional
            This is the data type of the input data to-be filtered.
        residual : bool, optional
            If True, subtract MAP estimate of signal from data to form
            the residual, otherwise simply return its MAP estimate (default)
        name : str, optional
            Name of the filter
        """
        attrs = ['G']
        super().__init__(dim=dim, name=name, attrs=attrs)
        self.G = torch.as_tensor(G, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.residual = residual
        
    def setup_filter(self, G=None):
        """
        Set filtering matrix
        """
        self.G = torch.as_tensor(G, device=device, dtype=dtype) if G is not None else self.G


class WedgeFilter:
    """
    A baseline-dependent frequency filter
    (i.e. a wedge filter).
    """
    def __init__(self, filters, filt2bls, inplace=False):
        """
        Parameters
        ----------
        filters : list
            List of BaseFilter subclasses
        filt2bls : dict
            Dictionary mapping int index in self.filters
            to all the baselines in the input VisData that
            it will filter. If input is a tensor, the values
            should be the indices of the input tensor
            along the Nbls axis that it will filter.
        inplace : bool, optional
            If True, edit input data inplace.
        """
        self.filters = filters
        self.filt2bls = filt2bls
        self.inplace = inplace

    def __call__(self, vd):
        is_VD = isinstance(vd, dataset.VisData)
        if is_VD:
            vout = vd.copy(copydata=True)
        else:
            vout = vd.clone()

        for i, bls in self.filt2bls.items():
            if is_VD:
                filt = self.filters[i](vout.get_data(bl=bls, squeeze=False))
                vout.set(bls, filt, arr='data')
            else:
                vout[:, :, bls] = self.filters[i](vout[:, :, bls])

        return vout


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


def gauss_sinc_cov(x, gauss_ls, sinc_ls, x2=None,
                   dtype=None, device=None, high_prec=True):
    """
    A Gaussian-convolved Sinc covariance model,
    or a top-hat truncated Gaussian Fourier space kernel.

    Convolution of a Gaussian and Sinc covariance function
    See appendix A2 of arxiv:1608.05854

    Parameters
    ----------
    x : tensor
        Sampling points of covariance
    gauss_ls : float
        Gaussian length scale
    sinc_ls : float
        Sinc length scale
    x2 : tensor, optional
        Second set of independent axis labels, generating
        a non-square covariance matrix.
    dtype : torch.dtype, optional
        Output dtype
    device : torch.device, optional
        Output device
    high_prec : bool, optional
        If True use mpmath arbitrary precision
        library, otherwise use numpy.

    Returns
    -------
    tensor
    """
    sinc_ls = sinc_ls / np.pi

    # get distances
    arg = gauss_ls / np.sqrt(2) / sinc_ls
    xc = x / gauss_ls / np.sqrt(2)
    x2c = xc if x2 is None else x2 / gauss_ls / np.sqrt(2)
    dists = (x2c[:, None] - xc[None, :])

    # get unique dists
    ud, ui = torch.unique(dists, return_inverse=True)

    if high_prec:
        import mpmath
        fn = lambda z: mpmath.exp(-z**2) * (mpmath.erf(arg + 1j*z) + mpmath.erf(arg - 1j*z)).real
        K = 0.5 * torch.as_tensor(np.asarray(np.frompyfunc(fn, 1, 1)(ud.numpy()), dtype=float))
        K /= special.erf(arg)

    else:
        K = (0.5 * torch.exp(-ud**2) / special.erf(arg) \
            * (special.erf(arg + 1j*ud) + special.erf(arg - 1j*ud))).real
        # replace nans with zero: in this limit, you should use high_prec
        # but this is a faster approximation
        K[torch.isnan(K)] = 0.0

    # expand back to NxM
    cov = K[ui]

    # fill dx = 0 if needed
    cov[torch.isclose(dists, torch.tensor(0.), atol=1e-7)] = 1.0

    if dtype is not None:
        cov = cov.to(dtype)
    if device is not None:
        cov = cov.to(device)

    return cov


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

