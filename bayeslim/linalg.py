"""
Linear algebra and related utility functions
"""
import numpy as np
import torch

from .utils import _float, _cfloat, D2R, viewreal, viewcomp


def cmult(a, b):
    """
    Complex multiplication of two real-valued torch
    tensors in "2-real" form, or of shape (..., 2)
    where the last axis indexes the real and imag
    component of the tensor, respectively.

    Parameters
    ----------
    a : tensor
        In 2-real form
    b : tensor
        In 2-real form

    Returns
    -------
    tensor
        Complex product of a and b in 2-real form
    """
    return viewreal(viewcomp(a) * viewcomp(b))


def cdiv(a, b):
    """
    Complex division (a / b) of two real-valued torch
    tensors in "2-real" form, or of shape (..., 2)
    where the last axis indexes the real and imag
    component of the tensor, respectively.

    Parameters
    ----------
    a : tensor
        In 2-real form
    b : tensor
        In 2-real form

    Returns
    -------
    tensor
        Complex division of a / b in 2-real form
    """
    return viewreal(viewcomp(a) / viewcomp(b))


def cconj(z):
    """
    Complex conjugate of a real-valued torch
    tensor in "2-real" form, or of shape (..., 2)
    where the last axis indexes the real and imag
    component of the tensor, respectively.

    Parameters
    ----------
    z : tensor
        In 2-real form

    Returns
    -------
    tensor
        Complex conjugate of z in 2-real form
    """
    return viewreal(viewcomp(z).conj())


def ceinsum(equation, *operands):
    """
    A shallow wrapper around torch.einsum,
    taking 2-real operands and returning
    2-real output.

    Parameters
    ----------
    equation : str
        A torch.einsum equation
    operands : tensor
        torch tensors to operate on in 2-real form

    Returns
    -------
    tensor
        Output of einsum in 2-real form
    """
    raise NotImplementedError("pytorch doesn't yet support complex autodiff for this")
    operands = (viewcomp(op) for op in operands)
    return viewreal(torch.einsum(equation, *operands))


def cinv(z):
    """
    Take the inverse of z
    across the first two axes

    Parameters
    ----------
    z : tensor
        torch tensor in 2-real form

    Returns
    -------
    tensor
        inverse of z in 2-real form
    """
    return viewreal(torch.inverse(viewcomp(z).T).T)


def diag_matmul(a, b):
    """
    Multiply two diagonal 1x1 or 2x2 matrices manually.
    This is generally faster than matmul or einsum
    for large, high dimensional stacks of 2x2 matrices.
    This drops the off-diagonal components of a and b.

    !! Note: this specifically ignores the off-diagonal for 2x2 matrices !!
    If you need off-diagonal components, you are
    better off using torch.matmul or torch.einsum directly.
    If you know off-diagonals are zero and are static, you can
    just use element-wise multiplication a * b.

    Parameters
    ----------
    a, b : tensor
        of shape (Nax, Nax, ...), where Nax = 1 or 2

    Returns
    -------
    c : tensor
        of shape (Nax, Nax, ...)
    """
    if a.shape[0] == 1:
        # 1x1: trivial
        return a * b
    elif a.shape[0] == 2:
        # 2x2
        c = torch.zeros_like(a)
        c[0, 0] = a[0, 0] * b[0, 0]
        c[1, 1] = a[1, 1] * b[1, 1]
        return c
    else:
        raise ValueError("only 1x1 or 2x2 tensors")


def diag_inv(a):
    """
    Invert a diagonal 1x1 or 2x2 matrix manually.
    This is only beneficial for 2x2 matrices where
    you want to drop the off-diagonal terms.

    Parameters
    ----------
    a : tensor
        of shape (Nax, Nax, ...), where Nax = 1 or 2

    Returns
    -------
    c : tensor
        of shape (Nax, Nax, ...)
    """
    if a.shape[0] == 1:
        # 1x1: trivial
        return 1 / a
    elif a.shape[0] == 2:
        # 2x2
        c = torch.zeros_like(a)
        c[0, 0] = 1 / a[0, 0]
        c[1, 1] = 1 / a[1, 1]
        return c
    else:
        raise ValueError("only 1x1 or 2x2 tensors")

def angle(z):
    """
    Compute phase of the 2-real tensor z

    Parameters
    ----------
    z : tensor
        In 2-real form

    Returns
    -------
    float or ndarray
        Phase of z in radians
    """
    return torch.angle(viewcomp(z))


def abs(z):
    """
    Take the abs of a 2-real tensor z.

    Parameters
    ----------
    z : tensor
        In 2-real form

    Returns
    -------
    tensor
        The amplitude of the input 2-real tensor
        projected into the complex plane with
        zero phase

    """
    zabs = torch.clone(z)
    zabs[..., 0] = torch.linalg.norm(z, axis=-1)
    zabs[..., 1] = 0
    return zabs


def apply_phasor(z, phi):
    """
    Apply a complex phasor to z

    Parameters
    ----------
    z : tensor
        In 2-real form
    phi : float
        Phase of phasor in radians

    Returns
    -------
    tensor
        z in 2-real form with phi applied
    """
    return viewreal(viewcomp(z) * np.exp(1j * phi))


def project_out_phase(z, avg_axis=None, select=None):
    """
    Compute and project out the phase of z

    Parameters
    ----------
    z : tensor
        In 2-real form
    avg_axis : int, optional
        Average z along avg_axis before computing
        its phase. Default is None.
    select : list, optional
        Use this to index z after any averaging
        before computing the phase.
        E.g.: select = [slice(None), slice(0, 1)].
        Note that this indexing must keep z's dimensionality.
        Default is None.

    Returns
    -------
    tensor
        z in 2-real form with phase projected out
    """
    if avg_axis is not None:
        za = torch.mean(z, axis=avg_axis, keepdim=True)
    else:
        za = z
    if select is not None:
        za = z[select]
    z_phs = angle(za)

    return apply_phasor(z, -z_phs)


def ones(*args, **kwargs):
    """
    Construct a 2-real tensor of ones

    Parameters
    ----------
    shape : tuple
        Shape of tensor

    Returns
    -------
    tensor
        A 2-real tensor full of ones

    Notes
    -----
    keyword arguments passed to torch.ones
    """
    ones = torch.ones(*args, **kwargs)
    ones[..., 1] = 0
    return ones


def cmatmul(a, b):
    """
    Perform 1x1 or 2x2 matrix multiplication
    along the first two axes of a and b
    in 2-real form. Note: this is slow
    compared to torch.einsum, but doesn't need
    to cast to complex

    Parameters
    -----------
    a : tensor
        In 2-real form with shape of b
    b : tensor
        In 2-real form with shape of a

    Returns
    -------
    tensor
        Matrix multiplication of a and b along
        their 0th and 1st axes
    """
    # determine if 1x1 or 2x2 matmul
    assert b.shape[0] == b.shape[1] == a.shape[0] == a.shape[1]
    assert a.shape[0] in [1, 2]
    twodim = True if a.shape[0] == 2 else False

    if not twodim:
        # 1x1 matmul is trivial
        return cmult(a, b)
    else:
        # 2x2 matmul
        c = torch.zeros_like(a)

        # upper left real
        c[0, 0, ..., 0] = a[0, 0, ..., 0] * b[0, 0, ..., 0] - a[0, 0, ..., 1] * b[0, 0, ..., 1] \
                          + a[0, 1, ..., 0] * b[1, 0, ..., 0] - a[0, 1, ..., 1] * b[1, 0, ..., 1]

        # upper left imag
        c[0, 0, ..., 1] = a[0, 0, ..., 0] * b[0, 0, ..., 1] + a[0, 0, ..., 1] * b[0, 0, ..., 0] \
                          + a[0, 1, ..., 0] * b[1, 0, ..., 1] + a[0, 1, ..., 1] * b[1, 0, ..., 0]

        # upper right real
        c[0, 1, ..., 0] = a[0, 0, ..., 0] * b[0, 1, ..., 0] - a[0, 0, ..., 1] * b[0, 1, ..., 1] \
                          + a[0, 1, ..., 0] * b[1, 1, ..., 0] - a[0, 1, ..., 1] * b[1, 1, ..., 1]

        # upper right imag
        c[0, 1, ..., 1] = a[0, 0, ..., 0] * b[0, 1, ..., 1] + a[0, 0, ..., 1] * b[0, 1, ..., 0] \
                          + a[0, 1, ..., 0] * b[1, 1, ..., 1] + a[0, 1, ..., 1] * b[1, 1, ..., 0]

        # lower left real
        c[1, 0, ..., 0] = a[1, 0, ..., 0] * b[0, 0, ..., 0] - a[1, 0, ..., 1] * b[0, 0, ..., 1] \
                          + a[1, 1, ..., 0] * b[1, 0, ..., 0] - a[1, 1, ..., 1] * b[1, 0, ..., 1]

        # lower left imag
        c[1, 0, ..., 1] = a[1, 0, ..., 0] * b[0, 0, ..., 1] + a[1, 0, ..., 1] * b[0, 0, ..., 0] \
                          + a[1, 1, ..., 0] * b[1, 0, ..., 1] + a[1, 1, ..., 1] * b[1, 0, ..., 0]

        # lower right real
        c[1, 1, ..., 0] = a[1, 0, ..., 0] * b[0, 1, ..., 0] - a[1, 0, ..., 1] * b[0, 1, ..., 1] \
                          + a[1, 1, ..., 0] * b[1, 1, ..., 0] - a[1, 1, ..., 1] * b[1, 1, ..., 1]

        # lower right imag
        c[1, 1, ..., 1] = a[1, 0, ..., 0] * b[0, 1, ..., 1] + a[1, 0, ..., 1] * b[0, 1, ..., 0] \
                          + a[1, 1, ..., 0] * b[1, 1, ..., 1] + a[1, 1, ..., 1] * b[1, 1, ..., 0]


    return c


def least_squares(A, y, dim=0, Ninv=None, norm='inv', pinv=True, rcond=1e-15, eps=0):
    """
    Solve a linear equation via generalized least squares.
    For the linear system of equations

    .. math ::
        
        y = A x

    with parameters x, the least squares solution is

    .. math ::

        \hat{x} = D A.T N^{-1} y

    where the normalization matrix is defined

    .. math ::

        D = (A.T N^{-1} A + \epsilon I)^{-1}

    If A is large this can be sped up by moving A and y
    to the GPU first.

    Parameters
    ----------
    A : tensor
        Design matrix, mapping parameters to outputs
        of shape (N, M)
    y : tensor
        Observation vector or matrix, of shape (M, ...)
    dim : int, optional
        Dimension in y to multiply into A. Default is 0th axis.
    Ninv : tensor, optional
         Inverse of noise matrix used for weighting
         of shape (N, N), or just its diagonal of shape (N,).
         Default is identity matrix.
    norm : str, optional
        Normalization type, [None, 'inv', 'diag']
        None : no normalization, assume D is identity
        'inv' : invert A.T Ninv A
        'diag' : take inverse of diagonal of A.T Ninv A
    pinv : bool, optional
        Use pseudo inverse if inverting A (default).
        Can also specify regularization parameter instead.
    rcond : float, optional
        rcond parameter for taking pseudo-inverse
    eps : float, optional
        Regularization parameter (default is None)

    Returns
    -------
    x : tensor
        estimated parameters
    D : tensor
        derived normalization matrix, depending
        on choice of pinv and eps
    """
    # Note that we actually solve the transpose of x
    # and then re-transpose. 
    # i.e. x = y.T @ Ninv @ A @ (A.T @ Ninv A)^{-1}
    # this is for broadcasting when y is multi-dim

    # move y axis
    y = y.moveaxis(dim, -1)

    # get un-normalized estimate of x
    if Ninv is not None:
        if Ninv.ndim == 2:
            # Ninv is a matrix
            y = y @ Ninv
        else:
            # Ninv is diagonal
            y = y * Ninv

    x = y @ A.conj()

    # get normalization matrix
    if norm == 'inv':
        # invert to get D
        if Ninv is None:
            Dinv = A.T.conj() @ A
        else:
            if Ninv.ndim == 2:
                # Ninv is matrix
                Dinv = A.T.conj() @ Ninv @ A
            else:
                # Ninv is diagonal
                Dinv = (A.T.conj() * Ninv) @ A
        # add regularization if desired
        Dinv += torch.eye(len(Dinv)) * eps
        # invert
        if pinv:
            D = torch.pinverse(Dinv, rcond=rcond)
        else:
            D = torch.inverse(Dinv)
        x = x @ D

    elif norm == 'diag':
        # just invert diagonal to get D
        if Ninv is None:
            Dinv = (torch.abs(A)**2).sum(dim=0)
        else:
            if Ninv.ndim == 2:
                # Ninv is a matrix
                Dinv = torch.diag(A.T.conj() @ Ninv @ A)
            else:
                # Ninv is diagonal
                Dinv = (Ninv * torch.abs(A.T)**2).T.sum(dim=0)
        D = 1 / Dinv
        x = x * D

    # return axis to dim
    x.moveaxis(-1, dim)

    return x, D




