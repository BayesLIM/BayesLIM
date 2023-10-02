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
        H = self.H.T if transpose else self.H
        H = H.conj() if transpose and self._complex else H
        result = H @ vec
        if out is not None:
            out[:] = result
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
        H = self.H.T if transpose else self.H
        H = H.conj() if transpose and self._complex else H
        result = H @ mat
        if out is not None:
            out[:] = result
            result = out

        return result

    def to_dense(self, transpose=False):
        """
        Return a dense form of the matrix
        """
        H = self.H
        H = H.T if transpose else H
        H = H.conj() if transpose and self._complex else H
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

    def __mul__(self, other):
        return DenseMat(self.H * other)

    def __rmul__(self, other):
        return DenseMat(other * self.H)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)


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
            out[:] = result
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
            out[:] = result
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

    def __mul__(self, other):
        return DiagMat(self.size, self.diag * other)

    def __rmul__(self, other):
        return DiagMat(self.size, other * self.diag)

    def __imul__(self, other):
        self.scalar_mul(other)
        return self

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)


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
            to be hermitian.
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
        result = U @ (V @ vec)

        if self.Hdiag is not None:
            N = len(self.Hdiag)
            result[:N] += self.Hdiag * vec[:N]

        if out is not None:
            out[:] = result
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
        result = U @ (V @ mat)

        if self.Hdiag is not None:
            N = len(self.Hdiag)
            result[:N] += self.Hdiag[:, None] * mat[:N]

        if out is not None:
            out[:] = result
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

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)


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
        result = torch.zeros(size, device=self.device, dtype=self.dtype)
        if out is not None:
            out[:] = result
            result = out

        return result

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        size = self.shape[0] if not transpose else self.shape[1]
        result = torch.zeros((size, mat.shape[1]), device=self.device, dtype=self.dtype)
        if out is not None:
            out[:] = result
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

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)


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
        result = torch.ones(self.shape[0], device=self.device, dtype=self.dtype) * vsum
        if out is not None:
            out[:] = result
            result = out

        return result

    def mat_mat_mul(self, mat, transpose=False, out=None, **kwargs):
        msum = mat.sum(dim=0, keepdims=True) * self.scalar
        result = torch.ones(self.shape[0], mat.shape[1]) * msum
        if out is not None:
            out[:] = result
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

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)


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
        return self._matobj.mat_vec_mul(vec, transpose=transpose==False, **kwargs)

    def mat_mat_mul(self, mat, transpose=False, **kwargs):
        """
        Matrix-matrix multiplication
        """
        return self._matobj.mat_mat_mul(mat, transpose=transpose==False, **kwargs)

    def __call__(self, vec, **kwargs):
        if vec.ndim == 1:
            return self.mat_vec_mul(vec, **kwargs)
        else:
            return self.mat_mat_mul(vec, **kwargs)

    def to_dense(self, transpose=False):
        """
        Return a dense representation of the matrix
        """
        return self._matobj.to_dense(transpose=transpose==False)

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

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)


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
            out[:] = result
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

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)


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

    def mat_vec_mul(self, vec, transpose=False, out=None, **kwargs):
        """
        Parameters
        ----------
        vec : tensor
            Vector to take linear solution against
        transpose : bool, optional
            If True, transpose self.A before solving system
        out : tensor, optional
            Put result into this tensor

        Returns
        -------
        tensor
        """
        A = self.A if not transpose else self.A.T.conj()
        if self.tri:
            # A is triangular
            ndim = vec.ndim
            if ndim == 1: vec = vec[:, None]
            
            # do forward sub
            result = torch.linalg.solve_triangular(A, vec, upper=not self.lower)

            # check if we need to do backward sub
            if self.chol:
                result = torch.linalg.solve_triangular(A.T.conj(), result, upper=self.lower)

            if ndim == 1:
                result = result.squeeze()
        else:
            # generic solve
            result = torch.linalg.solve(A, vec)

        if out is not None:
            out[:] = result
            result = out

        return result

    def mat_mat_mul(self, mat, transpose=False, **kwargs):
        """
        Same as mat_vec_mul
        """
        return self.mat_vec_mul(mat, transpose=transpose, **kwargs)

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

    def push(self, device):
        self.A = utils.push(self.A, device)
        if isinstance(device, torch.dtype):
            self.device = device

    def to_dense(self, **kwargs):
        return self(torch.eye(self.shape[1], device=self.device))

    def to_transpose(self):
        if self.tri:
            # if triangular, need to change self.lower arg
            return SolveMat(self.A.T.conj(), tri=self.tri, lower=self.lower==False, chol=self.chol)
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

    def __str__(self):
        return "<{} ({}x{})>".format(self.__class__.__name__, *self.shape)


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
                out[:] = sum(result)

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
                out[:] = sum(result)

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

    def mat_vec_mul(self, vec):
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
    def __init__(self, A00, A11, A01=None, A10=None, sym=False):
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
        self.A00 = utils.push(self.A00, device)
        self.A11 = utils.push(self.A11, device)
        if self.A01 is not None:
            self.A01 = utils.push(self.A01, device)
        if self.A10 is not None:
            self.A10 = utils.push(self.A10, device)

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

        return out

    def to_transpose(self):
        """
        Return a transposed version of self
        """
        A10t = None if self.A01 is None else self.A01.to_transpose() 
        A01t = None if self.A10 is None else self.A10.to_transpose()
        Ht = HierMat(A00=self.A00.to_transpose(), A11=self.A11.to_transpose(),
                            A10=A10t, A01=A01t, sym=self.sym)

        return Ht

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

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
    def __init__(self, A00, A11, A01=None, A10=None, lower=True):
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
        """
        if isinstance(A00, BaseMat) and not isinstance(A00, SolveMat):
            A00 = A00.to_dense()
        if isinstance(A11, BaseMat) and not isinstance(A11, SolveMat):
            A11 = A11.to_dense()
        if isinstance(A00, torch.Tensor):
            A00 = SolveMat(A00, tri=True, lower=lower, chol=False)
        if isinstance(A11, torch.Tensor):
            A11 = SolveMat(A11, tri=True, lower=lower, chol=False)
        super().__init__(A00, A11, A01, A10, sym=False)
        self.lower = lower

    def mat_vec_mul(self, vec, out=None, **kwargs):
        """
        Matrix-vector product using linear solves

        Parameters
        ----------
        vec : tensor

        Returns
        -------
        tensor
        """
        # forward substitution
        if self.lower:
            # first solve L_00 z_0 = v_0
            v_0 = vec[self._idx0[1]]
            z_0 = self[0](v_0, out=None if out is None else out[self._idx0[0]])

            # next solve L_11 z_1 = v_1 - L_10 z_0
            v_1 = vec[self._idx1[1]]
            if self[(1, 0)] is not None:
                v_1 = v_1 - self[(1, 0)](z_0)
            z_1 = self[1](v_1, out=None if out is None else out[self._idx1[0]])

        # backward substitution
        else:
            # first solve L_11 z_1 = v_1
            v_1 = vec[self._idx1[1]]
            z_1 = self[1](v_1, out=None if out is None else out[self._idx1[0]])

            # then solve L_00 z_0 = v_0 - L01 z_1
            v_0 = vec[self._idx0[1]]
            if self[(0, 1)] is not None:
                v_0 = v_0 - self[(0, 1)](z_1)
            z_0 = self[0](v_0, out=None if out is None else out[self._idx0[0]])

        if out is None:
            out = torch.cat([z_0, z_1])

        return out

    def to_transpose(self):
        """
        Return a transposed version of self
        """
        A10t = None if self.A01 is None else self.A01.to_transpose() 
        A01t = None if self.A10 is None else self.A10.to_transpose()
        Ht = SolveHierMat(A00=self.A00.to_transpose(), A11=self.A11.to_transpose(),
                          A10=A10t, A01=A01t, lower=self.lower==False)

        return Ht


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


