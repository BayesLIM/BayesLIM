"""
linear_model.py : tools for linear models of the type
y = Ax
"""
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy import special as scispc
import copy

from . import utils, linalg
from .utils import _float, _cfloat


class LinearModel:
    """
    A linear model of

        y = Ax
    """
    def __init__(self, linear_mode, dim=0, coeff=None, diag=False, idx=None,
                 out_dtype=None, out_shape=None, meta=None, **kwargs):
        """
        Parameters
        ----------
        linear_model : str
            The kind of model to generate: ['custom', 'poly', 'fourier'].
            See utils.gen_linear_A.
        dim : int, optional
            The dimension of the input params tensor to sum over.
        coeff : tensor, optional
            A tensor of params.shape to multiply with params before
            forward transform (e.g. alm_mult for sph. harm.)
        diag : bool, optional
            If True, assume A is diagonal and use element-wise product.
            (only for linear_mode='custom'). Only stores diagonal of A.
        idx : tensor, optional
            This indexes the input params along dim, if provided.
            Note that if coeff is provided, the order of operations is
            (params * coeff)[idx]
        out_dtype : dtype, optional
            Cast output of forward as this dtype if desired
        out_shape : tuple, optional
            Reshape output to this shape if desired
        meta : dict, optional
            Additional metadata to attach to self as self.meta
        kwargs : dict
            keyword arguments for utils.gen_linear_A()
        """
        self.linear_mode = linear_mode
        self.dim = dim
        self.coeff = coeff
        self.idx = idx
        self.out_dtype = out_dtype
        self.out_shape = out_shape
        self.meta = meta if meta is not None else {}

        if self.linear_mode in ['poly']:
            if kwargs.get('whiten', False):
                # if using whiten, get x0, dx parameters now
                # and store in kwargs for later
                _, x0, dx = utils.prep_xarr(kwargs.get('x'),
                                            d0=kwargs.get('d0', None),
                                            logx=kwargs.get('logx', False),
                                            whiten=kwargs.get('whiten', False),
                                            x0=kwargs.get('x0', None),
                                            dx=kwargs.get('dx', None))

                if not kwargs.get('x0', None):
                    kwargs['x0'] = x0
                if not kwargs.get('dx', None):
                    kwargs['dx'] = dx

        self.kwargs = kwargs
        self.A = gen_linear_A(linear_mode, **kwargs)
        self.freqs = None
        if linear_mode == 'fourier':
            _, self.freqs = gen_fourier_A(kwargs.get('x'),
                                          Ndeg=kwargs.get('Ndeg', None),
                                          device=kwargs.get('device', None),
                                          fft_norm=kwargs.get('fft_norm', 'ortho'))
        self.diag = diag
        if diag and self.A.ndim == 2:
            self.A = torch.diag(self.A)
        self.device = self.A.device

    def forward(self, params, A=None, coeff=None):
        """
        Forward pass parameter tensor through design matrix

        Parameters
        ----------
        params : tensor
            Parameter tensor to forward model
        A : tensor, optional
            Use this (Nsamples, Nfeatures) design
            matrix instead of self.A when taking
            forward model. If self.diag = True,
            this should be of shape (Nsamples,)
        coeff : tensor, optional
            Use this tensor with params.shape instead
            of self.coeff and multiply by params before
            forward transform (e.g. alm_mult for sph. harm.)

        Returns
        -------
        tensor
        """
        # get matrices and metadata
        A = A if A is not None else self.A
        coeff = coeff if coeff is not None else self.coeff
        idx = self.idx if hasattr(self, 'idx') else None
        if coeff is not None:
            params = params * coeff
        ndim = params.ndim

        # index params if needed
        if idx is not None:
            params = torch.index_select(params, self.dim, idx)

        # forward model
        if self.diag:
            # if A is diagonal, this is just a element-wise product
            if len(A) > 1:
                pshape = [-1 if i == self.dim else 1 for i in range(ndim)]
                A = A.reshape(pshape)
            out = A * params
        elif self.dim in [ndim - 2, -2] or ndim == 1:
            # trivial matmul
            out = A @ params
        elif self.dim in [ndim, -1]:
            # trivial matmul of A-transpose
            out = params @ A.T
        else:
            # otherwise would moveax, dot, moveax
            # so let's just use einsum
            t1 = 'ab'
            assert ndim <= 8
            t2 = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'][:ndim]
            t2[self.dim] = 'b'
            t2 = ''.join(t2)
            out = t2.replace('b', 'a')
            out = torch.einsum("{},{}->{}".format(t1, t2, out), A, params)

        if self.out_dtype is not None:
            out = out.to(self.out_dtype)

        if self.out_shape is not None:
            out = out.reshape(self.out_shape)

        return out

    def __call__(self, params, A=None):
        return self.forward(params, A=A)

    def least_squares(self, y, **kwargs):
        """
        Estimate a params tensor from the data vector

        Parameters
        ----------
        y : tensor
            Data vector to use in estimating a
            new params tensor
        kwargs : dict
            keyword arguments for linalg.least_squares()

        Returns
        -------
        tensor
        """
        if y.dtype != self.A.dtype:
            y = y.to(self.A.dtype)
        A = self.A
        if self.diag:
            if len(A) != y.shape[self.dim]:
                A = A.expand(y.shape[self.dim])
            A = torch.diag(A)
        return linalg.least_squares(A, y, dim=self.dim, **kwargs)

    def generate_A(self, x, **interp1d_kwargs):
        """
        Generate a new A matrix at new x values.
        If linear_mode is 'custom', then we interpolate
        the existing A, otherwise we generate
        a new A using the existing setup parameters.

        Parameters
        ----------
        x : tensor
            New x values to generate A
        kwargs : dict
            Kwargs for scipy interp1d(), used
            if linear_mode is custom.

        Returns
        -------
        tensor
        """
        if self.linear_mode == 'custom':
            # perform interpolation of existing A
            A = interp1d(self.kwargs['x'], self.A.cpu().numpy(),
                         axis=0, **interp1d_kwargs)(x)
            A = torch.as_tensor(A).to(self.device)
        else:
            kwargs = copy.deepcopy(self.kwargs)
            kwargs['x'] = x
            A = gen_linear_A(self.linear_mode, **kwargs)
            A = A.to(self.device)

        return A

    def push(self, device):
        """
        Push items to new device
        """
        dtype = isinstance(device, torch.dtype)
        self.A = utils.push(self.A, device)
        if self.coeff is not None:
            self.coeff = utils.push(self.coeff, device)
        if not dtype:
            self.device = device
            if hasattr(self, 'idx') and self.idx is not None:
                self.idx = utils.push(self.idx, device)


class MultiLM:
    """
    Multiple linear models of the type

        y = Ax

    for multiple dimensions of a single tensor.
    """
    def __init__(self, LM):
        """
        Parameters
        ----------
        LM : list of LinearModel objects
            Multiple LinearModel objects to apply
            to a single input tensor
        """
        self.LM = LM

    def __call__(self, params, **kwargs):
        return self.forward(params, **kwargs)

    def forward(self, params, **kwargs):
        for i, LM in enumerate(self.LM):
            params = LM(params, **kwargs)

        return params

    def least_squares(self, y, **kwargs):
        for i, LM in enumerate(self.LM):
            y = LM.least_squares(y, **kwargs)

        return y

    def push(self, device):
        for i, LM in enumerate(self.LM):
            LM.push(device)


class DictLM:
    """
    A dictionary of linear models for different
    parameter names, takes parameter name and tensor as
    """
    def __init__(self, LMs):
        """
        Parameters
        ----------
        LMs : dict
            Keys are the parameter names, e.g. 'rime.sky.eor.params'
            and values are LinearModel objects
        """
        self.LMs = LMs
        self.device = list(self.LMs.values())[0].device

    def __call__(self, name, params, **kwargs):
        return self.forward(name, params, **kwargs)

    def forward(self, name, params, **kwargs):
        """
        Parameters
        ----------
        name : str
            Name of parameter in self.LMs
        params : tensor

        Returns
        -------
        tensor
        """
        assert name in self.LMs

        if name in self.LMs:
            return self.LMs[name](params, **kwargs)
        else:
            return params

    def least_squares(self, name, y, **kwargs):
        return self.LMs[name].least_squares(y, **kwargs)

    def push(self, device):
        for i, LM in self.LMs.items():
            LM.push(device)
        self.device = list(self.LMs.values())[0].device


def gen_linear_A(linear_mode, A=None, x=None, d0=None, logx=False,
                 whiten=True, x0=None, dx=None, Ndeg=None, basis='direct',
                 qr=False, device=None, dtype=None, fft_norm='ortho', **kwargs):
    """
    Generate a linear mapping design matrix A of shape
    (Nsamples, Nfeatures)

    Parameters
    ----------
    linear_mode : str
        One of ['poly', 'custom', 'fourier']
    A : tensor, optional
        (mode='custom') Linear mapping of shape (Nsamples, Nfeatures)
    x : tensor, optional
        (mode='poly', 'fourier') sample values
    d0 : float, optional
        (mode='poly') divide x by d0 before any other operation
        if provided (for whitening)
    logx : bool, optional
        If True, take logarithm of x before generating
        A matrix (mode='poly')
    whiten : bool, optional
        (mode='poly') whiten samples
    x0 : float, optional
        (mode='poly') center x by x0
    dx : float, optional
        (mode='poly') scale x by 1/dx after centering
    Ndeg : int, optional
        (mode='poly', 'fourier') Number of poly degrees or Fourier terms
    basis : str, optional
        (mode='poly') poly basis
    qr : bool, optional
        (mode='poly') If True, re-orthogonalize poly modes
    device : str, optional
        Device to push A to
    dtype : type, optional
        data type to cast A to. Default is utils._float() or
        the existing dtype of A if passed.
    fft_norm : str, optional
        (mode='fourier') The norm kwarg of torch.fft.fft

    Returns
    -------
    A : tensor
        Design matrix of shape (Nsamples, Nfeatures)
    """
    if dtype is None:
        if A is not None:
            dtype = A.dtype
        else:
            dtype = utils._float()
    if linear_mode == 'poly':
        A = gen_poly_A(x, Ndeg, basis=basis, d0=d0, logx=logx, whiten=whiten,
                       x0=x0, dx=dx, qr=qr)
    elif linear_mode == 'custom':
        assert A is not None
        A = torch.as_tensor(A)
    elif linear_mode == 'fourier':
        A, _ = gen_fourier_A(x, Ndeg=Ndeg, device=device, fft_norm=fft_norm)
    else:
        raise NameError("linear_mode {} not recognized".format(linear_mode))

    A = torch.atleast_1d(A).to(dtype).to(device)

    return A


def gen_fourier_A(x, Ndeg=None, device=None, fft_norm='ortho'):
    """
    Generate a complex Fourier series matrix A of shape
    (Nsamples, Ndeg)

    Parameters
    ----------
    x : tensor
        Independent axis sample values
    Ndeg : int
        Number of Fourier modes to keep, default
        is all modes. For Ndeg != len(x) this keeps
        the "Ndeg" central Fourier modes
    device : str, optional
        Device to push A to
    fft_norm : str, optional
        FFT norm convention

    Returns
    -------
    A : tensor
    freqs : tensor
    """
    # get modes
    A = torch.fft.fftshift(torch.fft.fft(torch.eye(len(x)), dim=-1, norm=fft_norm), dim=-1)
    freqs = torch.fft.fftshift(torch.fft.fftfreq(len(x), torch.as_tensor(x[1] - x[0])))

    # cut down to central Ndeg terms
    if Ndeg is not None:
        N = A.shape[1]//2 - Ndeg//2
        A = A[:, N:N+Ndeg]
        freqs = freqs[N:N+Ndeg]

    return A, freqs


def gen_poly_A(x, Ndeg, device=None, basis='direct', d0=None,
               logx=False, whiten=True, x0=None, dx=None, qr=False):
    """
    Generate design matrix (A) for polynomial of Ndeg across x,
    with coefficient ordering

    .. math::

        y = Ax = a_0 * x^0 + a_1 * x^1 + a_2 * x^2 + \ldots

    Parameters
    ----------
    x : ndarray
        vector of independent axis values
    Ndeg : int
        Polynomial degree
    device : str, optional
        device to send A matrix to before return
    basis : str, optional
        Polynomial basis to use.
        ['direct', 'legendre', 'chebyshevt', 'chebyshevu']
        direct (default) is a standard polynomial (x^0 + x^1 + ...)
    d0 : float, optional
        Divide x by x0 before any other operation if provided
    logx : bool, optional
        If True, take log of x before generating A matrix or whitening.
    whiten : bool, optional
        If True, center (i.e. subtract mean) and scale (i.e. range of -1, 1) x.
        Useful when using orthogonal polynomial bases
    x0 : float, optional
        If whiten, use this centering instead of x.mean()
    dx : float, optional
        If whiten, use this scaling instead of (x-x0).max()
    qr : bool, optional
        If True, use QR factorization to make resultant A strictly
        orthogonal (i.e. gram-schmidt). Note this makes A effectively
        the same as basis='legendre', whiten=True, regardless of 
        the chosen basis or x0, dx.

    Returns
    -------
    A : tensor
        Polynomial design matrix (Nx, Ndeg)
    """
    x, _, _ = utils.prep_xarr(x, d0=d0, logx=logx, whiten=whiten, x0=x0, dx=dx)

    # setup the polynomial
    if basis == 'direct':
        A = np.vstack([x**i for i in range(Ndeg)]).T
    elif basis == 'legendre':
        A = np.vstack([scispc.eval_legendre(i, x) for i in range(Ndeg)]).T
    elif basis == 'chebyshevt':
        A = np.vstack([scispc.eval_chebyt(i, x) for i in range(Ndeg)]).T
    elif basis == 'chebyshevu':
        A = np.vstack([scispc.eval_chebyu(i, x) for i in range(Ndeg)]).T
    elif basis == 'laguerre':
        A = np.vstack([scispc.eval_laguerre(i, x) for i in range(Ndeg)]).T
    else:
        raise NameError("didn't recognize basis {}".format(basis))

    if qr:
        A = np.linalg.qr(A)[0]

    A = torch.as_tensor(A, dtype=_float(), device=device)

    return A


