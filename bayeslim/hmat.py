"""
Module for computing, manipulating, inverting,
and factorizing (sparse) matrix representations,
notably for hessian and inverse-hessian matrices
"""
from abc import abstractmethod
import numpy as np
import torch

from . import utils, paramdict, linalg


class BaseMat(object):

    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def mat_vec_mul(self, vec, transpose=False, **kwargs):
        pass

    @abstractmethod
    def mat_mat_mul(self, mat, transpose=False, **kwargs):
        pass

    @abstractmethod
    def to_dense(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def push(self, device):
        pass

    @abstractmethod
    def scalar_mul(self, scalar):
        pass

    @abstractmethod
    def diagonal(self):
        pass

    @abstractmethod
    def to_transpose(self):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __rmul__(self, other):
        pass

    @abstractmethod
    def __imul__(self, other):
        pass

    @abstractmethod
    def least_squares(self, y, **kwargs):
        pass

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)


class DenseMat(BaseMat):
    """
    A dense representation of a rectangular matrix
    """
    def __init__(self, H):
        """
        Parameters
        ----------
        H : tensor
            Dense 2D tensor
        """
        if H.ndim == 1:
            H = H[:, None]
        self._shape = H.shape
        self.H = H
        self._complex = torch.is_complex(H)
        self.dtype = H.dtype
        self.device = H.device

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        """
        Matrix-vector multiplication

        Parameters
        ----------
        vec : tensor
            Must have length of self.shape[1]
        transpose : bool, optional
            If True, take (complex) transpose of self.H
            before multiplication, in which case vec
            must have length of self.shape[0]
        out : tensor, optional
            Put results into this tensor

        Returns
        -------
        tensor
        """
        H = self.H.T.conj() if transpose else self.H
        if not self._complex and torch.is_complex(vec):
            result = torch.complex(H @ vec.real, H @ vec.imag)
        else:
            result = H @ vec
        if out is not None:
            out[:] += result
            result = out

        return result

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        """
        Matrix-matrix multiplication

        Parameters
        ----------
        mat : tensor
            Must have shape of (self.shape[1], M)
            where M is any integer if right == False
            otherwise of shape (M, self.shape[0])
        transpose : bool, optional
            If True, take (complex) transpose of self.H
            before multiplication
        out : tensor, optional
            Put results into this tensor

        Returns
        -------
        tensor
        """
        H = self.H.T.conj() if transpose else self.H
        if not self._complex and torch.is_complex(mat):
            result = torch.complex(H @ mat.real, H @ mat.imag)
        else:
            result = H @ mat
        if out is not None:
            out[:] += result
            result = out

        return result

    def to_dense(self, transpose=False):
        """
        Return a dense form of the matrix
        """
        H = self.H
        H = H.T.conj() if transpose else H
        return H

    def to_transpose(self):
        return TransposedMat(self)

    def __call__(self, vec, **kwargs):
        if vec.ndim == 1:
            return self.mat_vec_mul(vec, **kwargs)
        else:
            return self.mat_mat_mul(vec, **kwargs)

    def push(self, device):
        """
        Push object to a new device (or dtype)
        """
        self.H = utils.push(self.H, device)
        self.device = self.H.device
        self.dtype = self.H.dtype
        self._complex = torch.is_complex(self.H)

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        self.H *= scalar

    def diagonal(self):
        return self.H.diagonal()

    def least_squares(self, y, **kwargs):
        """
        Solve the y = Ax problem for x given y.

        Parameters
        ----------
        y : tensor
            Output of self(x)
        kwargs : kwargs for ba.linalg.least_squares()
        """
        return linalg.least_squares(self.H, y, **kwargs)

    def __mul__(self, other):
        return DenseMat(self.H * other)

    def __rmul__(self, other):
        return DenseMat(other * self.H)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class DiagMat(BaseMat):
    """
    A diagonal (or scalar) representation of a matrix.
    This can also be a scalar matrix by passing
    diag as a single element.
    """
    def __init__(self, size, diag):
        """
        Parameters
        ----------
        size : int
            Number of elements along the diagonal
        diag : tensor
            The diagonal elements. For a scalar matrix
            this can be a len-1 tensor.
        """
        self.size = size
        self.diag = diag
        self._complex = torch.is_complex(diag)
        self.dtype = diag.dtype
        self.device = diag.device

    @property
    def shape(self):
        return (self.size, self.size)

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        """
        Matrix-vector multiplication
        """
        diag = self.diag
        if transpose and self._complex:
            diag = diag.conj()
        result = diag * vec
        if out is not None:
            out[:] += result
            result = out

        return result

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        """
        Matrix-matrix multiplication

        Parameters
        ----------
        mat : tensor
            Must have shape of (self.size, M)
            where M is any integer

        Returns
        -------
        tensor
        """
        diag = self.diag
        if transpose and self._complex:
            diag = diag.conj()
        result = diag[:, None] * mat
        if out is not None:
            out[:] += result
            result = out

        return result

    def __call__(self, vec, **kwargs):
        if vec.ndim == 1:
            return self.mat_vec_mul(vec)
        else:
            return self.mat_mat_mul(vec)

    def to_dense(self, transpose=False, **kwargs):
        """
        Return a dense representation of the matrix
        """
        diag = torch.atleast_1d(self.diag)
        if len(diag) < self.size:
            diag = torch.ones(self.size, dtype=self.dtype, device=self.device) * self.diag

        if transpose and self._complex:
            diag = diag.conj()

        return torch.diag(diag)

    def to_transpose(self):
        return TransposedMat(self)

    def push(self, device):
        """
        Push object to a new device or dtype
        """
        self.diag = utils.push(self.diag, device)
        self.dtype = self.diag.dtype
        self.device = self.diag.device

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        self.diag *= scalar

    def diagonal(self):
        return self.diag

    def least_squares(self, y, **kwargs):
        """
        Solve the y = Ax problem for x given y.

        Parameters
        ----------
        y : tensor
            Output of self(x)
        """
        return DiagMat(self.size, 1/self.diag)(y)

    def __mul__(self, other):
        return DiagMat(self.size, self.diag * other)

    def __rmul__(self, other):
        return DiagMat(self.size, other * self.diag)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class HadamardMat(BaseMat):
    """
    A dense, Ndim matrix that takes an element-wise product
    (aka Hadamard product) with an incoming matrix
    """
    def __init__(self, H):
        """
        Parameters
        ----------
        H : tensor
            Dense, ndim tensor
        """
        self._shape = H.shape
        self.H = H
        self._complex = torch.is_complex(H)
        self.dtype = H.dtype
        self.device = H.device

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        """
        Technically this operation doesn't exist, but we will
        assume vec is a matrix of the same shape as self
        """
        return self.mat_mat_mul(vec, transpose=transpose, out=out, **kwargs)

    def mat_mat_mul(self, mat, transpose=False, out=None, square=False, **kwargs):
        """
        This is a hadamard product with mat, not a matrix multiplication

        Parameters
        ----------
        mat : tensor
            A matrix of the same shape as self.H
        square : bool, optional
            Square self before multiplying
        """
        H = self.H.T.conj() if transpose else self.H
        if square:
            H = H**2
        result = H * mat
        if out is not None:
            out[:] += result
            result = out

        return result

    def to_dense(self, transpose=False):
        """
        Return a dense form of the matrix
        """
        H = self.H
        H = H.T.conj() if transpose else H
        return H

    def to_transpose(self):
        return TransposedMat(self)

    def __call__(self, mat, **kwargs):
        return self.mat_mat_mul(mat, **kwargs)

    def push(self, device):
        """
        Push object to a new device (or dtype)
        """
        self.H = utils.push(self.H, device)
        self.device = self.H.device
        self.dtype = self.H.dtype
        self._complex = torch.is_complex(self.H)

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        self.H *= scalar

    def diagonal(self):
        if self.H.ndim == 1:
            return self.H
        else:
            return self.H.diagonal()

    def least_squares(self, y, **kwargs):
        """
        Solve the y = Ax problem for x given y.

        Parameters
        ----------
        y : tensor
            Output of self(x)
        """
        return HadamardMat(1/self.H)(y)

    def __mul__(self, other):
        return HadamardMat(self.H * other)

    def __rmul__(self, other):
        return HadamardMat(other * self.H)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class TriangMat(BaseMat):
    """
    A (square) triangular matrix representation. Note only
    the lower or upper triangular is actually stored.
    """
    def __init__(self, L, lower=True):
        """
        Parameters
        ----------
        L : tensor
            A 2D matrix with dense lower (upper)
            components, or a 1D array indexed with
            torch.tril_indices()
        lower : bool, optional
            If True, L is represented as lower triangular,
            otherwise upper triangular
        """
        self.device = L.device
        if L.ndim == 1:
            # triangular component is raveled
            N = int(np.sqrt(8*b + 1) - 1) // 2
            shape = (N, N)
            if lower:
                self.idx = torch.tril_indices(*shape, device=self.device)
            else:
                self.idx = torch.triu_indices(*shape, device=self.device)

        else:
            # passed as 2D matrix
            shape = L.shape
            if lower:
                self.idx = torch.tril_indices(*shape, device=self.device)
            else:
                self.idx = torch.triu_indices(*shape, device=self.device)
            L = L[self.idx[0], self.idx[1]]

        self.L = L
        self.dtype = L.dtype
        self.lower = lower
        self._shape = shape
        self._diag_idx = torch.where(self.idx[0] == self.idx[1])[0]
        self._complex = torch.is_complex(L)

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        """
        Matrix-vector product

        Parameters
        ----------
        vec : tensor

        Returns
        -------
        tensor
        """
        # get dense representation
        L = self.to_dense(transpose=transpose)

        # peform matmul
        if not self._complex and torch.is_complex(vec):
            result = torch.complex(L @ vec.real, L @ vec.imag)
        else:
            result = L @ vec

        # copy to out if needed
        if out is not None:
            out[:] += result
            result = out

        return result

    def mat_mat_mul(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

    def to_dense(self, transpose=False):
        """
        Return dense 2D Tensor
        """
        H = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        H[self.idx[0], self.idx[1]] = self.L
        if transpose:
            H = H.T.conj()

        return H

    def to_transpose(self):
        return TransposedMat(self)

    def push(self, device):
        self.L = utils.push(self.L, device)
        if not isinstance(device, torch.dtype):
            self.idx = utils.push(self.idx, device)
            self._diag_idx = utils.push(self._diag_idx, device)
        self.dtype = self.L.dtype
        self.device = self.L.device

    def scalar_mul(self, scalar):
        self.L *= scalar

    def diagonal(self):
        return self.L[self.diag_idx]

    def least_squares(self, y, **kwargs):
        """
        Solve the y = Ax problem for x given y.

        Parameters
        ----------
        y : tensor
            Output of self(x)
        kwargs : kwargs for ba.linalg.least_squares()
        """
        return linalg.least_squares(self.to_dense(), y, **kwargs)

    def __mul__(self, other):
        return TriangMat(self.L * other, lower=self.lower)

    def __rmul__(self, other):
        return TriangMat(self.L * other, lower=self.lower)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class SparseMat(BaseMat):
    """
    An outer-product representation of a rectangular matrix

        M = a I + U V
    """
    def __init__(self, shape, U, V=None, Hdiag=None, hermitian=False):
        """
        Parameters
        ----------
        shape : tuple
            2-tuple of (Nrows, Ncols) of matrix to model
        U : tensor
            The left-hand eigenmodes of shape (Nrows, Nmodes)
        V : tensor, optional
            The right-hand eigenmodes of shape (Ncols, Nmodes),
            if hermitian, will throw this away and use U.T instead
        Hdiag : tensor, optional
            The starting diagonal of the matrix, default is zeros.
            This should have ndim=1 and length of min(Nrows, Ncols).
        hermitian : bool, optional
            If the underlying matrix being modeled can be assumed
            to be hermitian, in which case we only store U and assume
            V = U.T.conj() = U^-1
        """
        self._shape = shape
        self.Hdiag = Hdiag
        self.U = U
        self._complex = torch.is_complex(U)
        self.device = U.device
        self.dtype = U.dtype
        if hermitian:
            self.V = None
        else:
            self.V = V
        self.hermitian = hermitian

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        """
        Matrix-vector multiplication

        Parameters
        ----------
        vec : tensor
            Must have length of self.shape[1]
        transpose : bool, optional
            If True, take (complex) transpose of self.U
            before multiplication, in which case vec
            must have length of self.shape[0]
        out : tensor, optional
            Put result into this tensor

        Returns
        -------
        tensor
        """
        if not transpose:
            # get right-hand modes
            V = self.V if not self.hermitian else self.U.T
            V = V.conj() if self.hermitian and self._complex else V

            # get left-hand modes
            U = self.U

        else:
            # get right-hand modes
            V = self.U.T
            V = V.conj() if self._complex else V

            # get left-hand modes
            U = self.V.T if not self.hermitian else self.U
            U = U.conj() if self.hermitian and self._complex else U

        # multiply by left-hand and right-hand modes
        assert V is not None
        if not self._complex and torch.is_complex(vec):
            result = torch.complex(U @ (V @ vec.real), U @ (V @ vec.imag))
        else:
            result = U @ (V @ vec)

        if self.Hdiag is not None:
            N = len(self.Hdiag)
            result[:N] += self.Hdiag * vec[:N]

        if out is not None:
            out[:] += result
            result = out

        return result

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        """
        Matrix-matrix multiplication

        Parameters
        ----------
        mat : tensor
            Must have shape (self.shape[1], M)
        transpose : bool, optional
            If True, take (complex) transpose of self.U
            before multiplication, in which case mat
            must have shape of (self.shape[0], M)
        out : tensor, optional
            Put result into this tensor

        Returns
        -------
        tensor
        """
        if not transpose:
            # get right-hand modes
            V = self.V if not self.hermitian else self.U.T
            V = V.conj() if self.hermitian and self._complex else V

            # get left-hand modes
            U = self.U

        else:
            # get right-hand modes
            V = self.U.T
            V = V.conj() if self._complex else V

            # get left-hand modes
            U = self.V.T if not self.hermitian else self.U
            U = U.conj() if self.hermitian and self._complex else U

        # multiply by left-hand and right-hand modes
        assert V is not None
        if not self._complex and torch.is_complex(mat):
            result = torch.complex(U @ (V @ mat.real), U @ (V @ mat.imag))
        else:
            result = U @ (V @ mat)

        if self.Hdiag is not None:
            N = len(self.Hdiag)
            result[:N] += self.Hdiag[:, None] * mat[:N]

        if out is not None:
            out[:] += result
            result = out

        return result

    def to_dense(self, transpose=False):
        """
        Return a dense form of the matrix
        """
        V = self.U.T if self.hermitian else self.V
        V = V.conj() if self.hermitian and self._complex else V
        out = self.U @ V
        if self.Hdiag is not None:
            N = len(self.Hdiag)
            out[:N] += self.Hdiag

        out = out.T if transpose else out
        out = out.conj() if transpose and self._complex else out

        return out

    def to_transpose(self):
        return TransposedMat(self)

    def __call__(self, vec, **kwargs):
        if vec.ndim == 1:
            return self.mat_vec_mul(vec, **kwargs)
        else:
            return self.mat_mat_mul(vec, **kwargs)

    def push(self, device):
        """
        Push object to a new device (or dtype)
        """
        self.U = utils.push(self.U, device)
        if self.V is not None:
            self.V = utils.push(self.V, device)
        if self.Hdiag is not None:
            self.Hdiag = utils.push(self.Hdiag, device)
        self._complex = torch.is_complex(self.U)
        self.dtype = self.U.dtype
        self.device = self.U.device

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        self.U *= scalar
        if self.Hdiag is not None:
            self.Hdiag *= scalar

    def diagonal(self):
        N = min(self._shape)
        if self.Hdiag is not None:
            diag = self.Hdiag
        else:
            diag = torch.zeros(N, device=self.device, dtype=self.dtype)

        U = self.U
        Vt = self.V.T if self.V is not None else self.U.conj()
        diag = diag + (U[:N, :] * Vt[:N, :]).sum(1)

        return diag

    def __mul__(self, other):
        U = self.U
        V = self.V
        Hdiag = self.Hdiag
        if V is not None:
            V = V * other
        else:
            if isinstance(other, torch.Tensor):
                U = other[:, None] * U
            else:
                U = U * other
        if Hdiag is not None:
            Hdiag = Hdiag * other
        return SparseMat(self.shape, U, V=V, Hdiag=Hdiag,
                         hermitian=self.hermitian)

    def least_squares(self, y, **kwargs):
        """
        Solve the y = Ax problem for x given y
        using Woodbury matrix formula:
        (A + UV)^-1 = Ainv - Ainv U (I + V Ainv U)^-1 V Ainv

        Parameters
        ----------
        y : tensor
            Output of self(x)
        """
        # first get A^-1 in rectangular form
        Ainv = self.Hdiag.pow(-1).diag()

        # get right-hand modes
        V = self.V if not self.hermitian else self.U.T.conj()

        # get left-hand modes
        U = self.U

        # get inner inverse
        I = torch.eye(len(V))
        inv = torch.linalg.pinv(I + V @ Ainv @ U)

        # return inverted y
        return (Ainv - Ainv @ U @ inv @ V @ Ainv) @ y

    def __rmul__(self, other):
        U = self.U
        V = self.V
        Hdiag = self.Hdiag
        if isinstance(other, torch.Tensor):
            U = other[:, None] * U
        else:
            U = U * other
        if Hdiag is not None:
            Hdiag = Hdiag * other
        return SparseMat(self.shape, U, V=V, Hdiag=Hdiag,
                         hermitian=self.hermitian)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class ZeroMat(BaseMat):
    """
    A zero matrix
    """
    def __init__(self, shape, dtype=None, device=None):
        """
        Parameters
        ----------
        shape : tuple
            Holds (Nrows, Ncols) of the zero matrix
        """
        self._shape = shape
        self.dtype = dtype if dtype is not None else utils._float()
        self.device = device

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        size = self.shape[0] if not transpose else self.shape[1]
        result = torch.zeros(size, device=self.device, dtype=vec.dtype)
        if out is not None:
            out[:] += result
            result = out

        return result

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        size = self.shape[0] if not transpose else self.shape[1]
        result = torch.zeros((size, mat.shape[1]), device=self.device, dtype=mat.dtype)
        if out is not None:
            out[:] += result
            result = out

        return result

    def __call__(self, vec, **kwargs):
        if vec.ndim == 1:
            return self.mat_vec_mul(vec, **kwargs)
        else:
            return self.mat_mat_mul(vec, **kwargs)

    def to_dense(self, transpose=False):
        shape = self.shape if not transpose else self.shape[::-1]
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def to_transpose(self):
        return TransposedMat(self)

    def push(self, device):
        if isinstance(device, torch.dtype):
            self.dtype = device
        else:
            self.device = device

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        pass

    def diagonal(self):
        return torch.zeros(min(self._shape), dtype=self.dtype, device=self.device)

    def __mul__(self, other):
        return ZeroMat(self.shape, device=self.device, dtype=self.dtype)

    def __rmul__(self, other):
        return ZeroMat(self.shape, device=self.device, dtype=self.dtype)

    def __imul__(self, other):
        return self


class OneMat(BaseMat):
    """
    A ones matrix filled with any scalar value. Note this is not identity or
    a diagonal matrix! For that see DiagMat().
    """
    def __init__(self, shape, scalar=1.0, dtype=None, device=None):
        """
        Parameters
        ----------
        shape : tuple
            Holds (Nrows, Ncols) of a scalar value
        scalar : float
            The value of the matrix elements
        """
        self._shape = shape
        self.scalar = scalar
        self.dtype = dtype if dtype is not None else utils._float()
        self.device = device

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        vsum = vec.sum(0) * self.scalar
        result = torch.ones(self.shape[0], device=self.device, dtype=vec.dtype) * vsum
        if out is not None:
            out[:] += result
            result = out

        return result

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        msum = mat.sum(dim=0, keepdims=True) * self.scalar
        result = torch.ones(self.shape[0], mat.shape[1], device=self.device, dtype=mat.dtype) * msum
        if out is not None:
            out[:] += result
            result = out

        return result

    def __call__(self, vec, **kwargs):
        if vec.ndim == 1:
            return self.mat_vec_mul(vec, **kwargs)
        else:
            return self.mat_mat_mul(vec, **kwargs)

    def to_dense(self, transpose=False):
        shape = self.shape if not transpose else self.shape[::-1]
        return torch.ones(shape, dtype=self.dtype, device=self.device) * self.scalar

    def to_transpose(self):
        return TransposedMat(self)

    def push(self, device):
        if isinstance(device, torch.dtype):
            self.dtype = device
        else:
            self.device = device

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        self.scalar *= scalar

    def diagonal(self):
        return torch.ones(min(self._shape), dtype=self.dtype, device=self.device) * self.scalar

    def __mul__(self, other):
        if isinstance(other, OneMat):
            return OneMat(self.shape, self.scalar * other.scalar, device=self.device, dtype=self.dtype)
        else:
            return OneMat(self.shape, self.scalar * other, device=self.device, dtype=self.dtype)

    def __rmul__(self, other):
        if isinstance(other, OneMat):
            return OneMat(self.shape, self.scalar * other.scalar, device=self.device, dtype=self.dtype)
        else:
            return OneMat(self.shape, self.scalar * other, device=self.device, dtype=self.dtype)

    def __imul__(self, other):
        if isinstance(other, OneMat):
            return self * other.scalar
        else:
            return self * other


class TransposedMat(BaseMat):
    """
    A shallow wrapper around a *Mat object to
    transpose its attributes and methods.
    Note that this doesn't copy any data
    and the data from the parent object is mutable.
    """
    def __init__(self, matobj):
        """
        Parameters
        ----------
        matobj : *Mat object
            A DiagMat, DenseMat, ... object
            to transpose
        """
        if isinstance(matobj, torch.Tensor):
            matobj = DenseMat(matobj)
        self._matobj = matobj
        self.dtype = matobj.dtype
        self.device = matobj.device

    @property
    def shape(self):
        return self._matobj.shape[::-1]

    def mat_vec_mul(self, vec, transpose=False, **kwargs):
        """
        Matrix-vector multiplication
        """
        return self._matobj.mat_vec_mul(vec, transpose=not transpose, **kwargs)

    def mat_mat_mul(self, mat, transpose=False, **kwargs):
        """
        Matrix-matrix multiplication
        """
        return self._matobj.mat_mat_mul(mat, transpose=not transpose, **kwargs)

    def __call__(self, vec, **kwargs):
        if vec.ndim == 1:
            return self.mat_vec_mul(vec, **kwargs)
        else:
            return self.mat_mat_mul(vec, **kwargs)

    def to_dense(self, transpose=False):
        """
        Return a dense representation of the matrix
        """
        return self._matobj.to_dense(transpose=not transpose)

    def to_transpose(self):
        return TransposedMat(self)

    def push(self, device):
        """
        Push object to a new device or dtype
        """
        self._matobj.push(device)
        self.dtype = self._matobj.dtype
        self.device = self._matobj.device

    def __repr__(self):
        return "<TransposedMat({})>".format(str(self._matobj))

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        scalar = torch.as_tensor(scalar)
        if scalar.is_complex():
            scalar = scalar.conj()
        self._matobj.scalar_mul(scalar)

    def diagonal(self):
        return self._matobj.diagonal()

    def __mul__(self, other):
        return TransposedMat(self._matobj * other)

    def __rmul__(self, other):
        return TransposedMat(other * self._matobj)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class PartitionedMat(BaseMat):
    """
    A rectangular (possibly symmetric) matrix that has been partitioned into
    on-diagonal blocks and their correponding off-diagonal
    blocks, using DenseMat, DiagMat, SparseMat.

    Assume the matrix A and its product with p can be decomposed as

        | x  x  x                |  | x  |
        | x A11 x       A12      |  | p1 |
        | x  x  x                |  | x  |
        |          x  x  x  x  x |  | x  |
        |          x  x  x  x  x |  | x  |
        |   A21    x  x A22 x  x |  | p2 |
        |          x  x  x  x  x |  | x  |
        |          x  x  x  x  x |  | x  |

    we create two "columns" for A which are summed to get
    the output vector. E.g. the first column computes
    [A11 @ p1, A21 @ p1] and the second computes
    [A12 @ p2, A22 @ p2]. Each component (A11, A12, A22)
    can be stored as DenseMat, DiagMat, SparseMat.
    Note if symmetric=True then A21 is just a transpose of A12.

    The user should feed a blocks dictionary which holds
    as a key each unique component tuple, e.g. (1, 1), (2, 2), (1, 2), ...,
    and its value is a *Mat object.
    If an off-diagonal component is missing it is assumed zero.
    If an on-diagonal component is missing its col & row is zero.
    """
    def __init__(self, blocks, symmetric=True):
        """
        Setup the matrix columns given a blocks dictionary
        e.g. of the form
        {
        (1, 1) : DenseMat
        (1, 2) : SparseMat
        (2, 2) : DiagMat
        (2, 3) : SparseMat
        (3, 3) : DiagMat
        }
        where (1, 2) is an on-diagonal block and (1, 2) is an off-diagonal block.
        Non-existant off-diagonal blocks treated as zero, non-existant on-diagonal
        blocks are ignored completely.
        Sets the self.matcols list and self.vec_idx list.

        Parameters
        ----------
        blocks : dict
            A dictionary holding the various independent blocks of the matrix.
            with (i, j) tuple key and BaseMat value
        symmetric : bool, optional
            If True (default), then blocks should only hold one of the
            off-diagonal components per unique (i, j) combination.
            I.e. you should only provide (1,2) and not (2,1), and
            (1,3) and not (3,1), and so on.
            If False, you should provide all off-diagonal components,
            otherwise missing ones are assumed ZeroMat.
        """
        # type check
        for k, v in blocks.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 1:
                    blocks[k] = DiagMat(len(v), v)
                else:
                    blocks[k] = DenseMat(v)

        # get all the on-diagonal matrices
        ondiag_keys = sorted([k for k in blocks if len(set(k)) == 1])

        # get paritioned matrix metadata from on-diagonal blocks
        self._Ncols = len(ondiag_keys)
        Nrows = sum([blocks[k].shape[0] for k in ondiag_keys])
        Ncols = sum([blocks[k].shape[1] for k in ondiag_keys])
        self._shape = (Nrows, Ncols)
        self.dtype = blocks[ondiag_keys[0]].dtype
        self.device = blocks[ondiag_keys[0]].device
        self.symmetric = symmetric

        self.matcols, self.diagmats, self.vec_idx = [], [], []
        size = 0
        # iterate over each major column object
        for i, k in enumerate(ondiag_keys):
            # append block to diagmats
            self.diagmats.append(blocks[k])

            # get indexing for a vector dotted into this matrix column
            self.vec_idx.append(slice(size, size+blocks[k].shape[1]))
            size += blocks[k].shape[1]

            # get all the theoretical sub-blocks in this vertical column
            block_keys = [(j[0], k[1]) for j in ondiag_keys]

            # now fill a list with these matrix objects
            mats = []
            for j, bk in enumerate(block_keys):
                if bk in blocks:
                    # append this block to mats
                    mats.append(blocks[bk])

                elif bk[::-1] in blocks and self.symmetric:
                    # transpose this block b/c self is symmetric
                    mats.append(TransposedMat(blocks[bk[::-1]]))

                else:
                    # make this a ZeroMat
                    shape = (blocks[(bk[0],bk[0])].shape[0], blocks[k].shape[1])
                    blocks[bk] = ZeroMat(shape, dtype=self.dtype, device=self.device)
                    mats.append(blocks[bk])

            # now append the entire MatColumn to matcols
            self.matcols.append(MatColumn(mats))

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        """
        Return the matrix multiplied by a vector

        Parameters
        ----------
        vec : tensor
            Vector to take product with
        transpose : bool, optional
            Transpose this column before mat-vec
        out : tensor, optional
            Put results into this tensor
        
        Returns
        -------
        tensor
        """
        if transpose:
            return self.to_transpose()(vec, out=out, **kwargs)

        shape = (self.shape[0],) + vec.shape[1:]
        result = torch.zeros(shape, dtype=vec.dtype, device=self.device)
        for i, matcol in enumerate(self.matcols):
            result += matcol(vec[self.vec_idx[i]])

        if out is not None:
            out[:] += result
            result = out

        return result

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        """
        Return the matrix multiplied by a matrix

        Parameters
        ----------
        mat : tensor
            ndim=2 matrix of shape (self.shape[1], M)
        transpose : bool, optional
            Transpose this column before mat-mat
        out : tensor, optional
            Put results into this tensor

        Returns
        -------
        tensor
        """
        return self.mat_vec_mul(mat, transpose=transpose, out=out, **kwargs)

    def __call__(self, vec, transpose=False, **kwargs):
        return self.mat_vec_mul(vec, transpose=transpose, **kwargs)

    def to_dense(self, transpose=False, **kwargs):
        """
        Return a dense representation of the matrix
        """
        if transpose:
            return self.to_transpose().to_dense()
        out = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        size = 0
        for i, matcol in enumerate(self.matcols):
            out[:, size:size + matcol.shape[1]] = matcol.to_dense()
            size += matcol.shape[1]

        return out

    def to_transpose(self):
        """
        Return a transposed copy of self, having re-ordered
        the ColMat objects such that the object is transposed

        Returns
        -------
        ParitionedMat object
        """
        blocks = {}
        for i, matcol in enumerate(self.matcols):
            for j, mat in enumerate(matcol.mats):
                blocks[(i+1, j+1)] = TransposedMat(mat)

        return PartitionedMat(blocks, symmetric=self.symmetric)

    def push(self, device):
        """
        Push partitioned matrix to a new device or dtype
        """
        for matcol in self.matcols:
            matcol.push(device)
        self.dtype = self.matcols[0].mats[0].dtype
        self.device = self.matcols[0].mats[0].device

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        for matcol in self.matcols:
            matcol.scalar_mul(scalar)

    def diagonal(self):
        return torch.cat([b.diagonal() for b in self.diagmats])

    def least_squares(self, y, **kwargs):
        """
        Solve the y = Ax problem for x given y.

        Note: WIP. Currently only uses the block
        diagonals for solving the inverse problem.

        Parameters
        ----------
        y : tensor
            Output of self(x)
        kwargs : dict, kwargs for ba.linalg.least_squares()
        """
        x = []
        for idx, mat in zip(self.vec_idx, self.diagmats):
            x.append(mat.least_squares(y[idx], **kwargs))

        return torch.cat(x)

    def __mul__(self, other):
        blocks = {}
        for i, matcol in enumerate(self.matcols):
            for j, mat in enumerate(matcol.mats):
                blocks[(j+1, i+1)] = mat * other
        return PartitionedMat(blocks, symmetric=self.symmetric)

    def __rmul__(self, other):
        blocks = {}
        for i, matcol in enumerate(self.matcols):
            for j, mat in enumerate(matcol.mats):
                blocks[(j+1, i+1)] = other * mat
        return PartitionedMat(blocks, symmetric=self.symmetric)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class SolveMat(BaseMat):
    """
    A representation of the inverse matrix product,
    where the matrix-vector product is solved
    via least squares, or if A is triangular via
    forward substituion. Solve the following for x

        Ax = b

    if chol == True, then assume we are solving

        A A.T x = b

    where A is a lower-tri Cholesky factor.
    """
    def __init__(self, A, tri=False, lower=True, chol=False):
        """
        Parameters
        ----------
        A : tensor
            The linear model matrix
        tri : bool, optional
            If True, treat A as triangular
        lower : bool, optional
            If True (and if tri), assume A is lower triangular
            else upper triangular
        chol : bool, optional
            If True (and if tri), assume input is the cholesky,
            in which case we do forward and
            backward substitution to solve the system
        """
        if isinstance(A, BaseMat):
            A = A.to_dense()
        self.A = A
        self._shape = self.A.shape
        self.device = A.device
        self.dtype = A.dtype
        self.tri = tri
        self.lower = lower
        self.chol = chol
        if chol: assert self.tri, "If passing A as chol, it must also be triangular"

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, chol=None, **kwargs):
        """
        Parameters
        ----------
        vec : tensor
            Vector to take linear solution against
        transpose : bool, optional
            If True, transpose self.A before solving system
        out : tensor, optional
            Put result into this tensor
        chol : bool, optional
            If passed, use this value of chol instead of self.chol.
            Default is to use self.chol

        Returns
        -------
        tensor
        """
        chol = chol if chol is not None else self.chol
        A = self.A if not transpose else self.A.T.conj()
        lower = self.lower if not transpose else not self.lower
        if self.tri:
            # A is triangular
            ndim = vec.ndim
            if ndim == 1: vec = vec[:, None]

            # do forward sub
            result = self._solve_tri(A, vec, upper=not lower)

            # check if we need to do backward sub
            if chol:
                result = self._solve_tri(A.T.conj(), result, upper=lower)

            if ndim == 1:
                result = result.squeeze()
        else:
            # generic solve
            result = self._solve(A, vec)

        if out is not None:
            out[:] += result
            result = out

        return result

    def _solve_tri(self, A, B, upper=False, **kwargs):
        """
        shallow wrapper around torch.linalg.solve_triangular
        handling complex inputs
        """
        if torch.is_complex(B) and not torch.is_complex(A):
            # handle complex input
            rB = B[:, None] if B.ndim == 1 else B
            rB = torch.cat([rB.real, rB.imag], dim=-1)
            out = torch.linalg.solve_triangular(A, rB, upper=upper, **kwargs)
            out = torch.complex(out[:, :out.shape[1]//2], out[:, out.shape[1]//2:])
            if B.ndim == 1:
                out = out[:, 0]
            return out
        else:
            return torch.linalg.solve_triangular(A, B, upper=upper, **kwargs)

    def _solve(self, A, B, **kwargs):
        """
        shallow wrapper around torch.linalg.solve
        handling complex inputs
        """
        if torch.is_complex(B) and not torch.is_complex(A):
            # handle complex input
            rB = B[:, None] if B.ndim == 1 else B
            rB = torch.cat([rB.real, rB.imag], dim=-1)
            out = torch.linalg.solve(A, rB, **kwargs)
            out = torch.complex(out[:, :out.shape[1]//2], out[:, out.shape[1]//2:])
            if B.ndim == 1:
                out = out[:, 0]
            return out
        else:
            return torch.linalg.solve(A, B, **kwargs)

    def mat_mat_mul(self, mat, **kwargs):
        """
        Same as mat_vec_mul
        """
        return self.mat_vec_mul(mat, **kwargs)

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

    def push(self, device):
        self.A = utils.push(self.A, device)
        if isinstance(device, torch.dtype):
            self.device = device

    def to_dense(self, **kwargs):
        return self(torch.eye(self.shape[1], device=self.device, dtype=self.dtype), **kwargs)

    def to_transpose(self):
        if self.tri:
            # if triangular, need to change self.lower arg
            return SolveMat(self.A.T.conj(), tri=self.tri, lower=not self.lower, chol=self.chol)
        else:
            return TransposedMat(self)

    def scalar_mul(self, scalar):
        self.A /= scalar

    def diagonal(self):
        return self.to_dense().diagonal()

    def __mul__(self, other):
        return SolveMat(self.A / other, tri=self.tri, lower=self.lower, chol=self.chol)

    def __rmul__(self, other):
        return SolveMat(self.A / other, tri=self.tri, lower=self.lower, chol=self.chol)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class MatColumn(BaseMat):
    """
    A series of matrix objects that have the 
    same Ncols but differing Nrows. E.g.

    | M1 |
    | M2 |
    | M3 |

    """
    def __init__(self, mats):
        """"
        Parameters
        ----------
        mats : list
            A list of BaseMat subclasses that represent
            a single column of a partitioned matrix.
            Each object must have the same Ncols.
        """
        self.mats = mats

        self.idx = []
        Nrows = 0
        Ncols = self.mats[0].shape[1]
        for m in self.mats:
            assert Ncols == m.shape[1]
            self.idx.append(slice(Nrows, Nrows+m.shape[0]))
            Nrows += m.shape[0]
        self._shape = (Nrows, Ncols)

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        result = []
        for i, mat in enumerate(self.mats):
            if transpose:
                result.append(mat(vec[self.idx[i]], transpose=transpose))
            else:
                _out = None if out is None else out[self.idx[i]]
                result.append(mat(vec, out=_out))

        if out is None:
            if transpose:
                out = sum(result)
            else:
                out = torch.cat(result, dim=0)
        else:
            if transpose:
                out[:] += sum(result)

        return out

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        return self.mat_vec_mul(mat, transpose=transpose, out=out, **kwargs)

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

    def push(self, device):
        for m in self.mats:
            m.push(device)

    def __repr__(self):
        return "<MatColumn of shape {}>".format(self.shape)

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        for mat in self.mats:
            mat.scalar_mul(scalar)

    def to_dense(self, transpose=False):
        out = torch.cat([m.to_dense() for m in self.mats], dim=0)
        if transpose:
            out = out.T
            if torch.is_complex(out):
                out = out.conj()

        return out

    def to_transpose(self):
        return MatRow([TransposedMat(m) for m in self.mats])

    def __mul__(self, other):
        return MatColumn([m * other for m in self.mats])

    def __rmul__(self, other):
        return MatColumn([other * m for m in self.mats])

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class MatRow(BaseMat):
    """
    A series of matrix objects that have the 
    same Nrows but differing Cols. E.g.

    | M1 M2 M3|

    """
    def __init__(self, mats):
        """"
        Parameters
        ----------
        mats : list
            A list of BaseMat subclasses that represent
            a single row of a partitioned matrix.
            Each object must have the same Nrows.
        """
        self.mats = mats

        self.idx = []
        Nrows = self.mats[0].shape[0]
        Ncols = 0
        for m in self.mats:
            assert Nrows == m.shape[0]
            self.idx.append(slice(Ncols, Ncols+m.shape[1]))
            Ncols += m.shape[1]
        self._shape = (Nrows, Ncols)

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        result = []
        for i, mat in enumerate(self.mats):
            if transpose:
                _out = None if out is None else out[self.idx[i]]
                result.append(mat(vec, out=_out, transpose=transpose))
            else:
                result.append(mat(vec[self.idx[i]]))

        if out is None:
            if transpose:
                out = torch.cat(result, dim=0)
            else:
                out = sum(result)

        else:
            if not transpose:
                out[:] += sum(result)

        return out

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        return mat_vec_mul(mat, transpose=transpose, out=out, **kwargs)

    def __call__(self, vec, **kwargs):
        return mat_vec_mul(vec, **kwargs)

    def push(self, device):
        for m in self.mats:
            m.push(device)

    def __repr__(self):
        return "<MatRow of shape {}>".format(self.shape)

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        for mat in self.mats:
            mat.scalar_mul(scalar)

    def to_dense(self, transpose=False):
        out = torch.cat([m.to_dense() for m in self.mats], dim=1)
        if transpose:
            out = out.T
            if torch.is_complex(out):
                out = out.conj()

        return out

    def to_transpose(self):
        return MatColumn([TransposedMat(m) for m in self.mats])

    def __mul__(self, other):
        return MatRow([m * other for m in self.mats])

    def __rmul__(self, other):
        return MatRow([other * m for m in self.mats])

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class MatSum:
    """
    A series of matrix objects that have the
    same (Nrows, Ncols), whose mat-vec products should be computed
    and then summed.
    """
    def __init__(self, mats):
        """"
        Parameters
        ----------
        mats : list
            A list of *Mat objects that represent
            a matrix sum
        """
        self.mats = mats

    def mat_vec_mult(self, vec, **kwargs):
        return torch.sum([m(vec, **kwargs) for m in self.mats], dim=0)

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mult(vec, **kwargs)

    def to_dense(self, sum=True, transpose=False):
        out = [m.to_dense(transpose=transpose) for m in self.mats]
        if sum:
            out = torch.sum(out, dim=0)
        else:
            out = torch.tensor(out)
        return out

    def push(self, device):
        for m in self.mats:
            m.push(device)

    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
        for mat in self.mats:
            mat.scalar_mul(scalar)

    def __mul__(self, other):
        return MatSum([m * other for m in self.mats])

    def __rmul__(self, other):
        return MatSum([other * m for m in self.mats])

    def __imul__(self, other):
        self.scalar_mul(other)
        return self


class MatDict:
    """
    A mirror class to ParamDict, holding *Mat objects
    as values instead of parameter tensors.
    """
    def __init__(self, mats):
        """
        Parameters
        ----------
        mats : dict
            Dictionary of *Mat objects as values
            with str keys
        """
        self.mats = mats
        self._setup()

    def _setup(self):
        self.devices = {k: self.mats[k].device for k in self.keys()}

    def keys(self):
        return list(self.mats.keys())

    def values(self):
        return list(self.mats.values())

    def items(self):
        return list(self.mats.items())

    def push(self, device):
        """
        Push mats to device. Can feed
        device as a dictionary which will push
        mats to different devices
        """
        if isinstance(device, dict):
            for k in device:
                self.mats[k].push(device[k])
        else:
            for k in self.mats:
                self.mats[k].push(device)
        self._setup()

    def to_dense(self, transpose=False):
        """
        Get dense copies of matrices and return as a ParamDict

        Parameters
        ----------
        transpose : bool, optional
            If True, transpose the matrix
        """
        return paramdict.ParamDict({k: self.mats[k].to_dense(transpose=transpose) for k in self.keys()})

    def mat_vec_mul(self, vec, **kwargs):
        """
        Perform matrix-vector product on a ParamDict vector object
        and return a ParamDict object

        Parameters
        ----------
        vec : Paramdict

        Returns
        -------
        ParamDict
        """
        out = {}
        for k in self.keys():
            if k in vec:
                v = vec[k] if vec[k].ndim == 1 else vec[k].view(-1)
                o = self.mats[k].mat_vec_mul(v)
                out[k] = o if vec[k].ndim == 1 else o.view(vec[k].shape)

        return paramdict.ParamDict(out)

    def __getitem__(self, key):
        return self.mats[key]

    def __setitem__(self, key, val):
        self.mats[key] = val

    def update(self, other):
        for key in other:
            self.__setitem__(key, other[key])
        self._setup()

    def __iter__(self):
        return (p for p in self.mats)


class HierMat:
    """
    A hierarchically nested set of 2x2
    block matrices (e.g. HODLR)

    | 0,0  0,1 |
    | 1,0  1,1 |

    which can be indexed as
    H[0] or H[(0, 0)] for the first block and
    H[(0, 1)] for the off-diagonals
    """
    def __init__(self, A00, A11, A01=None, A10=None, sym=False, scalar=None):
        """
        Initialize the 2x2 blocks of the matrix

        Parameters
        ----------
        A00 : HierMat, BaseMat or tensor
            The upper diagonal
        A11 : HierMat, BaseMat, or tensor
            The lower diagonal
        A01 : HierMat, BaseMat or tensor
            Upper off-diagonal, default is zeros
        A10 : HierMat, BaseMat or tensor
            Lower off-diagonal, default is zeros,
            will use A01 if sym==True and vice-versa
        sym : bool, optional
            Whether this matrix is symmetric
        scalar : float, optional
            A float to multiply output by
        """
        # wrap tensors with DenseMat if needed
        A00 = DenseMat(A00) if isinstance(A00, torch.Tensor) else A00
        A11 = DenseMat(A11) if isinstance(A11, torch.Tensor) else A11
        A01 = DenseMat(A01) if isinstance(A01, torch.Tensor) else A01
        A10 = DenseMat(A10) if isinstance(A10, torch.Tensor) else A10

        # check A01 and A10 if sym
        if sym:
            # use transposed version if needed
            if A01 is None and A10 is not None:
                A01 = TransposedMat(A10)
            if A10 is None and A01 is not None:
                A10 = TransposedMat(A01) 

        self.A00 = A00
        self.A11 = A11
        self.A01 = A01
        self.A10 = A10

        if A01 is not None:
            assert A01.shape[0] == A00.shape[0]
            assert A01.shape[1] == A11.shape[1]
        if A10 is not None:
            assert A10.shape[0] == A11.shape[0]
            assert A10.shape[1] == A00.shape[1]

        self.dtype = self.A00.dtype
        self.device = self.A00.device
        self.sym = sym
        self.scalar = scalar

        self._shape0 = A00.shape
        self._shape1 = A11.shape
        shape = (A00.shape[0] + A11.shape[0], A00.shape[1] + A11.shape[1])
        self._shape = shape

        # these are indexing arrays for mat-vec products
        self._idx0 = (slice(self._shape0[0]), slice(self._shape0[1]),)
        self._idx1 = (slice(self._shape0[0], self._shape0[0]+self._shape1[0]),
                      slice(self._shape0[1], self._shape0[1]+self._shape1[1]))

    @property
    def shape(self):
        return self._shape

    def diagonal(self, return_tensor=True):
        """
        Parameters
        ----------
        return_tensor : bool, optional
            If True return a tensor, otherwise return
            a list
        """
        diag = []
        if isinstance(self.A00, HierMat):
            diag.extend(self.A00.diagonal(False))
        elif isinstance(self.A00, torch.Tensor):
            diag.append(self.A00.diagonal())
        elif isinstance(self.A00, BaseMat):
            diag.append(self.A00.diagonal())

        if isinstance(self.A11, HierMat):
            diag.extend(self.A11.diagonal(False))
        elif isinstance(self.A11, torch.Tensor):
            diag.append(self.A11.diagonal())
        elif isinstance(self.A11, BaseMat):
            diag.append(self.A11.diagonal())

        if return_tensor:
            diag = torch.cat(diag)

        if self.scalar is not None:
            if return_tensor:
                diag *= self.scalar
            else:
                for d in diag:
                    d *= self.scalar

        return diag

    def __getitem__(self, idx):
        if idx in [0, (0, 0)]:
            return self.A00
        elif idx in [1, (1, 1)]:
            return self.A11
        elif idx == (0, 1):
            return self.A01
        elif idx == (1, 0):
            return self.A10

    def push(self, device):
        self.scalar = utils.push(self.scalar, device)
        self.A00 = utils.push(self.A00, device)
        self.A11 = utils.push(self.A11, device)
        if self.A01 is not None:
            self.A01 = utils.push(self.A01, device)
        if self.A10 is not None:
            self.A10 = utils.push(self.A10, device)
        self.device = self.A00.device

    def mat_vec_mul(self, vec, out=None, **kwargs):
        """
        Perform HierMat-vector product recursively

        Parameters
        ----------
        vec : tensor
            Tensor of shape (self.shape[1], ...)
        out : tensor, optional
            Insert output into this tensor

        Returns
        -------
        tensor
        """
        # first column
        out00 = self.A00(vec[self._idx0[1]], out=None if out is None else out[self._idx0[0]])
        out10 = None
        if self.A10 is not None:
            out10 = self.A10(vec[self._idx0[1]], out=None if out is None else out[self._idx1[0]])

        # second column
        out11 = self.A11(vec[self._idx1[1]], out=None if out is None else out[self._idx1[0]])
        out01 = None
        if self.A01 is not None:
            out01 = self.A01(vec[self._idx1[1]], out=None if out is None else out[self._idx0[0]])

        if out is None:
            if out01 is not None:
                out00 += out01
            if out10 is not None:
                out11 += out10
            out = torch.cat([out00, out11])

        if self.scalar is not None:
            out *= self.scalar

        return out

    def to_transpose(self):
        """
        Return a transposed version of self
        """
        A10t = None if self.A01 is None else self.A01.to_transpose() 
        A01t = None if self.A10 is None else self.A10.to_transpose()
        Ht = HierMat(A00=self.A00.to_transpose(), A11=self.A11.to_transpose(),
                            A10=A10t, A01=A01t, sym=self.sym, scalar=self.scalar)

        return Ht

    def to_dense(self, transpose=False):
        """
        Note this is very inefficient because, unlike self.diagonal(),
        we have to continually re-init tensors at every
        nested level. not recommended for high performance cases.
        """
        # first get A00 and A11
        A00 = self.A00.to_dense()
        A11 = self.A11.to_dense()
        if self.A01 is not None:
            A01 = self.A01.to_dense()
        else:
            A01 = torch.zeros((A00.shape[0], A11.shape[1]), dtype=A00.dtype, device=A00.device)
        if self.A10 is not None:
            A10 = self.A10.to_dense()
        else:
            A10 = torch.zeros((A11.shape[0], A00.shape[1]), dtype=A00.dtype, device=A00.device)

        # now combine
        H = torch.cat([
            torch.cat([A00, A01], dim=1),
            torch.cat([A10, A11], dim=1),
            ], dim=0)

        if self.scalar is not None:
            H *= self.scalar

        if transpose:
            H = H.T

        return H

    def scalar_mul(self, scalar):
        """
        Multiply matrix representation by a scalar
        """
        if self.scalar is None:
            self.scalar = torch.tensor(1.0, device=self.device)
        self.scalar *= scalar

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

    def to_SolveHierMat(self, lower=True, trans_solve=False):
        """
        Convert self to a SolveHierMat and return (not inplace).
        Assumes that self is a Cholesky matrix.
        """
        scalar = 1 / self.scalar if self.scalar is not None else None
        if scalar is not None and trans_solve:
            scalar = scalar**2
        H = SolveHierMat(self.A00, self.A11, A01=self.A01, A10=self.A10,
                         lower=lower, trans_solve=trans_solve, scalar=scalar)
        return H

    def least_squares(self, y, **kwargs):
        """
        Solve the y = Mx problem for x given y
        using Banaschiewicz inversion formula
        for a partitioned matrix:

            | A B |
        M = | C D |

        where A and D are invertible, then

               | A^-1 + A^-1 B S_a^-1 C A^-1,  -A^-1 B S_a^-1 |
        M^-1 = | -S_a^-1 C A^-1,                       S_a^-1 |

        S_a = D - C A^-1 B

        Note: work in progress. Currently we only do implicit
        matrix-vector product of M^-1 if M is diagonal. If B
        or C is non-zero then we do a solve against self.to_dense().

        Parameters
        ----------
        y : tensor
            Output of self(x)
        kwargs : dict, kwargs for ba.linalg.least_squares()
        """
        if not self.A10 and not self.A01:
            # setup solves against diagonal
            x0 = self.A00.least_squares(y[self._idx0[1]], **kwargs)
            x1 = self.A11.least_squares(y[self._idx1[1]], **kwargs)

            return torch.cat([x0, x1])

        else:
            # TODO: do all of the implicit matrix-inverse vector products!
            # Need to figure out how to do S_a^-1 y implicitly...

            # B or C is present, so for now we just do a solve against
            # the dense matrix...
            M = self.to_dense()
            return torch.linalg.lstsq(M, y)

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)

    def __repr__(self):
        return "{}\n| {}, {} |\n| {}, {} |".format(self, self.A00, self.A01, self.A10, self.A11)


class SolveHierMat(HierMat):
    """
    A subclass of HierMat used specifically for
    HODLR representation of Cholesky forms.

    Given a lower-tri L, solve for z

        L z = x
    """
    def __init__(self, A00, A11, A01=None, A10=None, lower=True, trans_solve=False, scalar=None):
        """
        Setup a hierarchical cholesky matrix where on-diagonal
        are cholesky factors and off-diagonal are tensors or any sparse
        representation

        Parameters
        ----------
        A00 : tensor, SolveMat or SolveHierMat
            Upper on-diagonal block. Note if passed as tensor, this
            must be a dense tensor which will be turned into SolveMat
        A11 : tensor, SolveMat or SolveHierMat
            Lower on-diagonal block. Same type comment as above.
        A01 : tensor or BaseMat, optional
            Upper off-diagonal. This can be a sparse tensor.
        A10 : tensor or BaseMat, optional
            Lower off-diagonal. This can be a sparse tensor.
        lower : bool, optional
            If True, assume cholesky is lower triangular
        trans_solve : bool, optional
            If True, perform transpose solve after initial solve.
            This is needed for solving A x = b for x given A = L L^T,
            as opposed to just solving L x = b for x. 
            This is equivalent to the chol=True kwarg of SolveMat
        scalar : float, optional
            A float to multiply output by
        """
        if A00.__class__ == HierMat:
            A00 = A00.to_SolveHierMat(lower=lower, trans_solve=False)
        if A11.__class__ == HierMat:
            A11 = A11.to_SolveHierMat(lower=lower, trans_solve=False)
        if isinstance(A00, BaseMat) and not isinstance(A00, SolveMat):
            A00 = A00.to_dense()
        if isinstance(A11, BaseMat) and not isinstance(A11, SolveMat):
            A11 = A11.to_dense()
        if isinstance(A00, torch.Tensor):
            A00 = SolveMat(A00, tri=True, lower=lower, chol=False)
        if isinstance(A11, torch.Tensor):
            A11 = SolveMat(A11, tri=True, lower=lower, chol=False)
        super().__init__(A00, A11, A01, A10, sym=False, scalar=scalar)
        self.lower = lower
        self.trans_solve = trans_solve
        self._T = None

    def mat_vec_mul(self, vec, out=None, trans_solve=None, **kwargs):
        """
        Matrix-vector product using linear solves

        Parameters
        ----------
        vec : tensor
        out : tensor, optional
            Add output to this tensor
        trans_solve : bool, optional
            Use this value of trans_solve as opposed to self.trans_solve

        Returns
        -------
        tensor
        """
        # forward substitution
        if self.lower:
            # first solve L_00 z_0 = v_0
            v_0 = vec[self._idx0[1]]
            z_0 = self[0](v_0, out=None if out is None else out[self._idx0[0]], trans_solve=False)

            # next solve L_11 z_1 = v_1 - L_10 z_0
            v_1 = vec[self._idx1[1]]
            if self[(1, 0)] is not None:
                v_1 = v_1 - self[(1, 0)](z_0)
            z_1 = self[1](v_1, out=None if out is None else out[self._idx1[0]], trans_solve=False)

        # backward substitution
        else:
            # first solve L_11 z_1 = v_1
            v_1 = vec[self._idx1[1]]
            z_1 = self[1](v_1, out=None if out is None else out[self._idx1[0]], trans_solve=False)

            # then solve L_00 z_0 = v_0 - L01 z_1
            v_0 = vec[self._idx0[1]]
            if self[(0, 1)] is not None:
                v_0 = v_0 - self[(0, 1)](z_1)
            z_0 = self[0](v_0, out=None if out is None else out[self._idx0[0]], trans_solve=False)

        if out is None:
            out = torch.cat([z_0, z_1])

        if self.scalar is not None:
            out *= self.scalar

        # perform transpose solve if needed (i.e. for solving L L^T x = b)
        trans_solve = trans_solve if trans_solve is not None else self.trans_solve
        if trans_solve:
            out = self.to_transpose()(out, out=torch.zeros_like(out), trans_solve=False)

        return out

    def to_transpose(self):
        """
        Return a transposed version of self
        """
        if self._T is not None:
            return self._T
        A10t = None if self.A01 is None else self.A01.to_transpose() 
        A01t = None if self.A10 is None else self.A10.to_transpose()
        Ht = SolveHierMat(A00=self.A00.to_transpose(), A11=self.A11.to_transpose(),
                          A10=A10t, A01=A01t, lower=not self.lower, scalar=self.scalar,
                          trans_solve=self.trans_solve)
        if self._T is None:
            # cache this for repeated calls
            self._T = Ht

        return Ht

    def scalar_mul(self, scalar):
        """
        Multiply (inverse) matrix representation by a scalar
        """
        if self.scalar is None:
            self.scalar = torch.tensor(1.0, device=self.device)
        self.scalar *= scalar
        if self._T is not None:
            self._T.scalar_mul(scalar)


def make_hodlr(mat, indices, trisolve=False, lower=True,
               Nrank=None, rcond=None, sparse_tol=None):
    """
    Construct a hierarchical HODLR matrix

    Parameters
    ----------
    mat : tensor or dict
        This is the matrix to sub-divide into a HODLR form
    indices : list of slice or tensor
    trisolve : bool, optional
        If True, treat mat as triangular and return
        a SolveHierMat, otherwise return a HierMat
    lower : bool, optional
        If True (and if trisolve), treat mat as lower
        triangular

    Returns
    -------
    (Solve)HierMat
    """
    raise NotImplementedError

