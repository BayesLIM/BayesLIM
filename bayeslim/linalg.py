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
    across the last two axes
    (excluding the last 2-real axis)

    Parameters
    ----------
    z : tensor
        torch tensor in 2-real form

    Returns
    -------
    tensor
        inverse of z in 2-real form
    """
    return viewreal(torch.inverse(viewcomp(z)))


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


def cholesky_inverse(A, check_errors=True):
    """
    Take the inverse of a M x M matrix A using
    its cholesky factorization (assuming A is positive definite).
    This is generally faster than linalg.pinv for symmetric,
    full-rank matrices.

    Note that this uses torch.linalg.cholesky_ex rather than
    torch.linalg.cholesky, which is faster but doesn't
    error if A is not positive definite!

    Parameters
    ----------
    A : tensor
        Positive definite matrix of shape (M, M)
    check_errors : bool, optional
        If True, check for positive definiteness of
        input A.

    Returns
    -------
    Ainv : tensor
        Inverse of A of shape (M, M)
    L : tensor
        Lower triangular cholesky factorization
        of A
    """
    if A.ndim == 1:
        Ainv = 1 / A
        L = torch.sqrt(A)
    else:
        L = torch.linalg.cholesky_ex(A, check_errors=check_errors).L
        I = torch.eye(len(L), dtype=L.dtype, device=L.device)
        Linv = torch.linalg.solve_triangular(L, I, upper=False)
        Ainv = Linv.T @ Linv

    return Ainv, L


def invert_matrix(A, inv='pinv', rcond=1e-15, hermitian=False, eps=None,
                  driver=None):
    """
    Invert a matrix A. Note, you may not need to invert this matrix if
    it is part of a linear system solving for x:
        y = Ax
    in which case see least_squares()

    Parameters
    ----------
    A : tensor
        A 2D matrix to invert.
        If A.ndim == 1 we return 1 / A
        If A.ndim > 2 we compute the inversion for
        the first two dimensions for each element
        in the higher dimensions. e.g. for A.ndim == 3:
        invert_matrix(A[:,:,0]), invert_matrix(A[:,:,1]), ...
    inv : str, optional
        The options are
        'inv'   : torch.linalg.inv
        'pinv'  : torch.linalg.pinv, kwargs rcond, hermitian
        'chol'  : cholesky-inverse
        'lstsq' : least-squares inverse, kwargs rcond, driver
        'diag'  : just invert the diagonal
    rcond : float, optional
        Relative condition to use in singular-value based inversion
    hermitian : bool, optional
        Whether A is symmetric
    eps : float, optional
        Diagonal * eps regularization to add to A before inversion.
        Only used for 'inv', 'pinv', 'chol', 'lstsq'.
        Note this edits the input A inplace.
    driver : str, optional
        Least-squares routine to use, see torch.linalg.lstsq()

    Returns
    -------
    tensor
    """
    if inv == 'diag' or A.ndim == 1:
        if A.ndim == 1:
            return 1.0 / A
        else:
            return torch.diag(1.0 / torch.diag(A))

    # get inversion method
    def inverse(mat, inv=inv, rcond=rcond, eps=eps, hermitian=hermitian, driver=driver):
        """
        mat is assumed to be 2D (N, N) matrix
        """
        if eps is not None:
            mat.diagonal().add_(eps)
        if inv == 'inv':
            return torch.linalg.inv(mat)
        elif inv == 'pinv':
            return torch.linalg.pinv(mat, rcond=rcond, hermitian=hermitian)
        elif inv == 'chol':
            return cholesky_inverse(mat)[0]
        elif inv == 'lstsq':
            return torch.linalg.lstsq(mat, torch.eye(len(mat), dtype=mat.dtype, device=mat.device),
                                      rcond=rcond, driver=driver).solution
        else:
            raise NameError("didn't recognize inv='{}'".format(inv))

    def recursive_inv(A, iA, inverse=inverse):
        if A.ndim > 2:
            for i in range(A.shape[-1]):
                recursive_inv(A[:, :, i], iA[:, :, i])
            return
        iA[:, :] = inverse(A)

    iA = torch.zeros_like(A)
    recursive_inv(A, iA)

    return iA


def least_squares(A, y, dim=0, mode='matrix', norm='inv', pinv=True,
                  eps=0, rcond=1e-15, hermitian=True, D=None, preconj=False,
                  pretran=False, driver=None, Ninv=None, Ndiag=True):
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

    Notes
    -----
    mode = 'matrix' works well when the number of
        Nvariables is small, i.e., computational
        scaling is largely independent of y batch dimensions,
        but has poor scaling for Nvariables.
    mode = 'lstsq' works well when Nvariables is large,
        and has good scaling with Nvariables but has
        bad scaling for Nbatch dimensions.

    Parameters
    ----------
    A : tensor
        Design matrix, mapping parameters to outputs
        of shape (Nsamples, Nvariables)
    y : tensor
        Observation tensor of shape (..., Nsamples, ...)
        where ... are batch dimensions
    dim : int, optional
        Dimension in y to multiply into A. Default is 0th axis.
    mode : str, optional
        Whether to use iterative least-squares to solve normal equations for x_hat,
        or to compute D via matrix operations to get x_hat. Default is matrix mode.
        options = ['matrix', 'lstsq']
    norm : str, optional
        Normalization type for matrix mode, [None, 'inv', 'diag']
            None : no normalization, assume D is identity
            'inv' : invert A.T Ninv A
            'pinv' : use pseudo-inverse to inver A.T Ninv A
            'diag' : take inverse of diagonal of A.T Ninv A
    pinv : bool, optional
        Use pseudo inverse if norm = 'inv' (same as inv='pinv').
        Can also specify regularization parameter instead.
    eps : float, optional
        Regularization parameter (default is None)
        to add to A while taking inverse.
    rcond : float, optional
        rcond parameter for taking pseudo-inverse
    hermitian : bool, optional
        Hermitian parameter for pseudo-inverse
    D : tensor, optional
        Normalization matrix to use instead of
        the one created internally, dictated by
        choice of norm. D must still conform
        to expected shape given norm.
    preconj : bool, optional
        Assume A has already been pre-conjugated,
        in which case Aconj = A and A = A.conj().
        This is only helpful when pre-computing
        D (such that A.T.conj() A is not needed)
        and passing a large A on the GPU where
        both A and A.conj() cannot fit in memory.
    pretran : bool, optional
        Assume A has been pre-transposed,
        i.e. instead of shape (Nsamples. Nvariables)
        it is (Nvariables, Nsamples).
    driver : str, optional
        If norm = 'lstsq', this is
        driver option for torch.linalg.lstsq()
    Ninv : tensor, optional
         Inverse of noise matrix used for weighting
         of shape (N, N) if Ndiag==True, or just the
         inverse of its diagonal with Ninv.ndim == 1
         or Ninv.ndim == y.ndim
    Ndiag : bool, optional
        If True, Ndiag represents diagonal of N,
        otherwise it's assumed to be NxN matrix.

    Returns
    -------
    xhat : tensor
        estimated parameters
    D : tensor
        derived normalization matrix, depending
        on choice of norm and eps
    """
    assert y.ndim <= 8
    # sum over 'i' for y tensor
    A_ein = 'ij' if not pretran else 'ji'
    A_ein2 = A_ein.replace('j', 'k')
    y_ein = ['a','b','c','d','e','f','g','h'][:y.ndim]
    y_ein[dim] = 'i'
    y_ein = ''.join(y_ein)

    # weight the data vector by inverse covariance
    if Ninv is not None:
        if not Ndiag:
            # turn Ninv into cholesky if in lstsq mode
            if mode == 'lstsq': Ninv = torch.linalg.cholesky(Ninv)
            # Ninv is a matrix
            y = torch.einsum("ki,{}->{}".format(y_ein, y_ein.replace('i', 'k')), Ninv, y)
        else:
            # Ninv is diagonal
            if mode == 'lstsq':
                Ninv = torch.sqrt(Ninv)

            # weight y by Ninv
            if Ninv.ndim == 1:
                shape = [1 for i in range(y.ndim)]
                shape[ndim] = y.shape[dim]
                y = Ninv.reshape(shape) * y
            else:
                y = Ninv * y

    # check if we are in lstsq mode
    if mode == 'lstsq':
        if pretran:
            A = A.T
        if preconj:
            A = A.conj()
        # weight A by Ninv cholesky if needed
        if Ninv is not None:
            if not Ndiag:
                A = L @ A
            else:
                A = A * Ninv[:, None]

        # make sure A has enough dims
        if A.ndim < y.ndim:
            A = A.reshape(torch.Size([1]*(y.ndim - A.ndim)) + A.shape)
        # make sure dim of y is in -2 dim
        if y.ndim > 1:
            y = y.moveaxis(dim, -2)

        # now do solve
        xhat = torch.linalg.lstsq(A, y, driver=driver).solution
        if y.ndim > 1:
            xhat = xhat.moveaxis(-2, dim)

        return xhat, None

    # otherwise assume we are in matrix mode
    assert mode == 'matrix'

    # get A.T y: un-normalized hat(x)
    if preconj:
        Aconj = A
    else:
        Aconj = A.conj()
    xhat = torch.einsum("{},{}->{}".format(A_ein, y_ein, y_ein.replace('i', 'j')),
                        Aconj, y)

    # compute normalization matrix: (A.T Ninv A)^-1, multiply it into xhat
    if norm in ['inv', 'pinv', 'chol']:
        if D is None:
            if preconj:
                A = A.conj()
            # get A.T Ninv A
            if Ninv is None:
                Dinv = torch.einsum("{},{}->jk".format(A_ein, A_ein2),
                                    Aconj, A)
            else:
                if not Ndiag:
                    # Ninv is a matrix
                    Dinv = torch.einsum("{},il,{}->jk".format(A_ein, A_ein2.replace('i', 'l')),
                                        Aconj, Ninv, A)
                else:
                    # Ninv is diagonal
                    assert Ninv.ndim == 1
                    Dinv = torch.einsum("{},i,{}->jk".format(A_ein, A_ein2),
                                        Aconj, Ninv, A)

            if torch.is_complex(Dinv):
                Dinv = Dinv.real

            # invert
            if norm == 'inv' and pinv:
                norm = 'pinv'
            D = invert_matrix(Dinv, inv=norm, rcond=rcond, eps=eps, hermitian=hermitian)

        # apply D to un-normalized xhat
        xhat = torch.einsum("kj,{}->{}".format(y_ein.replace('i', 'j'), y_ein.replace('i', 'k')),
                        D.to(xhat.dtype), xhat)

    elif norm == 'diag':
        if D is None:
            if preconj:
                A = A.conj()
            # just invert diagonal to get D
            if Ninv is None:
                Dinv = A.norm(dim=0 if not pretran else 1).pow(2)
            else:
                if not Ndiag:
                    # Ninv is a matrix
                    Dinv = torch.einsum("{},il,{}->jk".format(A_ein, A_ein2.replace('i', 'l')),
                                        Aconj, Ninv, A)
                    Dinv = torch.diag(Dinv)
                else:
                    # Ninv is diagonal
                    if Ninv.ndim == 1:
                        Dinv = (Ninv[:, None] * torch.abs(A)**2).sum(dim=0 if not pretran else 1)
                    else:
                        Dinv = torch.einsum("{},{}->{}".format(y_ein, A_ein, y_ein.replace('i', 'j')),
                            Ninv, A.abs().pow(2),
                        )
            if torch.is_complex(Dinv):
                Dinv = Dinv.real
            D = 1 / Dinv

        # apply D to un-normalized xhat
        if D.ndim == 1:
            shape = [1 for i in range(xhat.ndim)]
            shape[dim] = len(D)
            D = D.reshape(shape)
        xhat = D * xhat

    else:
        # no normalization
        D = torch.ones(A.shape[1 if not pretran else 0])

    ### LEGACY
    # Note that we actually solve the transpose of x
    # and then re-transpose. 
    # i.e. x = y.T @ Ninv @ A @ (A.T @ Ninv A)^{-1}
    # this is for broadcasting when y is multi-dim
    # move y axis
    #y = y.moveaxis(dim, -1)
    # get un-normalized estimate of x
    #if Ninv is not None:
    #    if Ninv.ndim == 2:
    #        # Ninv is a matrix
    #        y = y @ Ninv
    #    else:
    #        # Ninv is diagonal
    #        y = y * Ninv
    #x = y @ A.conj()
    # get normalization matrix
    #if norm == 'inv':
    #    # invert to get D
    #    if Ninv is None:
    #        Dinv = A.T.conj() @ A
    #    else:
    #        if Ninv.ndim == 2:
    #            # Ninv is matrix
    #            Dinv = A.T.conj() @ Ninv @ A
    #        else:
    #            # Ninv is diagonal
    #            Dinv = (A.T.conj() * Ninv) @ A
    #    # add regularization if desired
    #    if eps > 0:
    #        Dinv += torch.eye(len(Dinv), device=Dinv.device, dtype=Dinv.dtype) * eps
    #    if torch.is_complex(Dinv):
    #        Dinv = Dinv.real
    #    # invert
    #    if pinv:
    #        D = torch.pinverse(Dinv, rcond=rcond)
    #    else:
    #        D = torch.inverse(Dinv)
    #    x = x @ D.to(x.dtype)
    #elif norm == 'diag':
    #    # just invert diagonal to get D
    #    if Ninv is None:
    #        Dinv = (torch.abs(A)**2).sum(dim=0)
    #    else:
    #        if Ninv.ndim == 2:
    #            # Ninv is a matrix
    #            Dinv = torch.diag(A.T.conj() @ Ninv @ A)
    #        else:
    #            # Ninv is diagonal
    #            Dinv = (Ninv * torch.abs(A.T)**2).T.sum(dim=0)
    #    if torch.is_complex(Dinv):
    #        Dinv = Dinv.real
    #    D = 1 / Dinv
    #    x = x * D.to(x.dtype)
    #else:
    #    D = np.eye(A.shape[1])
    ## return axis to dim
    #x = x.moveaxis(-1, dim)
    ### LEGACY

    return xhat, D

