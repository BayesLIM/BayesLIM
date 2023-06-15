"""
Module for computing, storing, and updating
Hessians and inverse Hessians
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
    def mat_vec_mul(self, vec, transpose=False):
        pass

    @abstractmethod
    def to_dense(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def push(self, device):
        pass

    @abstractmethod
    def scalar_mul(self, scalar):
        """
        Multiply a scalar into the matrix
        inplace

        Parameters
        ----------
        scalar : float
        """
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

    def mat_vec_mul(self, vec, transpose=False):
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

        Returns
        -------
        tensor
        """
        H = self.H.T if transpose else self.H
        H = H.conj() if transpose and self._complex else H
        return H @ vec

    def to_dense(self, transpose=False):
        """
        Return a dense form of the matrix
        """
        H = self.H
        H = H.T if transpose else H
        H = H.conj() if transpose and self._complex else H
        return H

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

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
        self.dtype = diag.dtype
        self.device = diag.device

    @property
    def shape(self):
        return (self.size, self.size)

    def mat_vec_mul(self, vec, **kwargs):
        """
        Matrix-vector multiplication
        """
        return self.diag * vec

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec)

    def to_dense(self, **kwargs):
        """
        Return a dense representation of the matrix
        """
        diag = torch.atleast_1d(self.diag)
        if len(diag) < self.size:
            diag = torch.ones(self.size, dtype=self.dtype, device=self.device) * self.diag

        return torch.diag(diag)

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
            The starting diagonal of the matrix, default is zeros
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

    def mat_vec_mul(self, vec, transpose=False):
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
        out = U @ (V @ vec)

        if self.Hdiag is not None:
            N = len(self.Hdiag)
            out[:N] += self.Hdiag * vec[:N]

        return out

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

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

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

    def mat_vec_mul(self, vec, transpose=False):
        size = self.shape[0] if not transpose else self.shape[1]
        return torch.zeros(size, device=self.device, dtype=self.dtype)

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

    def to_dense(self, transpose=False):
        shape = self.shape if not transpose else self.shape[::-1]
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

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
        return self._matobj.mat_vec_mul(vec, transpose=transpose==False)

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec, **kwargs)

    def to_dense(self, transpose=False):
        """
        Return a dense representation of the matrix
        """
        return self._matobj.to_dense(transpose=transpose==False)

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


class PartitionedMat(BaseMat):
    """
    A square matrix that has been partitioned into
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
    Note A21 is just a transpose of A12.

    The user should feed a blocks dictionary which holds
    as a key each unique component str (e.g. '11', '22', '12', '33', ...)
    and its value is a *Mat object.
    If an off-diagonal component is missing it is assumed zero.
    """
    def __init__(self, blocks):
        """
        Setup the matrix columns given a blocks dictionary
        e.g. of the form
        {
        '11' : DenseMat
        '12' : SparseMat
        '22' : DiagMat
        '23' : SparseMat
        '33' : DiagMat
        }
        where '11' is an on-diagonal block and '12' is an off-diagonal block.
        Non-existant off-diagonal blocks treated as zero, non-existant on-diagonal
        blocks are ignored completely.
        Sets the self.matcols list and self.vec_idx list.

        Parameters
        ----------
        blocks : dict
            A dictionary holding the various independent blocks of the matrix.
        """
        # get all the on-diagonal matrices
        ondiag_keys = sorted([k for k in blocks if k[0] == k[1]])

        # get paritioned matrix metadata from on-diagonal blocks
        self._Ncols = len(ondiag_keys)
        length = sum([blocks[k].shape[0] for k in ondiag_keys])
        self._shape = (length, length)
        self.dtype = blocks[ondiag_keys[0]].dtype
        self.device = blocks[ondiag_keys[0]].device

        self.matcols, self.vec_idx = [], []
        size = 0
        # iterate over each major column object
        for i, k in enumerate(ondiag_keys):
            # get indexing for a vector dotted into this matrix column
            self.vec_idx.append(slice(size, size+blocks[k].shape[0]))
            size += blocks[k].shape[0]

            # get all the theoretical sub-blocks in this vertical column
            block_keys = ["{}{}".format(j[0], k[1]) for j in ondiag_keys]

            # now fill a list with these matrix objects
            mats, opts = [], []
            for j, bk in enumerate(block_keys):
                # if this is an off-diag and its not here, make it a zeromat
                if bk not in blocks and bk[::-1] not in blocks:
                    shape = (blocks["{}{}".format(bk[0],bk[0])].shape[0], blocks[k].shape[0])
                    blocks[bk] = ZeroMat(shape, dtype=self.dtype, device=self.device)

                # append this block to mats
                if bk in blocks:
                    mats.append(blocks[bk])
                else:
                    mats.append(TransposedMat(blocks[bk[::-1]]))

            # now append the entire MatColumn to matcols
            self.matcols.append(MatColumn(mats))

    @property
    def shape(self):
        return self._shape

    def mat_vec_mul(self, vec, **kwargs):
        """
        Return the matrix multiplied by a vector

        Parameters
        ----------
        vec : tensor
        
        Returns
        -------
        tensor
        """
        out = torch.zeros(len(vec), dtype=vec.dtype, device=vec.device)
        for i, matcol in enumerate(self.matcols):
            out += matcol(vec[self.vec_idx[i]])

        return out

    def __call__(self, vec, **kwargs):
        return self.mat_vec_mul(vec)

    def to_dense(self, **kwargs):
        """
        Return a dense representation of the matrix
        """
        out = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        size = 0
        for i, matcol in enumerate(self.matcols):
            out[:, size:size + matcol.shape[1]] = matcol.to_dense()
            size += matcol.shape[1]

        return out

    def push(self, device):
        """
        Push partitioned matrix to a new device or dtype
        """
        for matcol in self.matcols:
            matcol.push(device)
        self.dtype = self.matcols[0].dtype
        self.device = self.matcols[0].device

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


class MatColumn:
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
            A list of *Mat objects that represent
            a single column of a partitioned matrix
        """
        self.mats = mats

        Nrows = 0
        Ncols = self.mats[0].shape[1]
        for m in self.mats:
            assert Ncols == m.shape[1]
            Nrows += m.shape[0]
        self._shape = (Nrows, Ncols)

    @property
    def shape(self):
        return self._shape

    def __call__(self, vec, **kwargs):
        return torch.cat([m(vec) for m in self.mats], dim=0)

    def to_dense(self, transpose=False):
        out = torch.cat([m.to_dense() for m in self.mats], dim=0)
        if transpose:
            out = out.T
            if torch.is_complex(out):
                out = out.conj()

        return out

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

    def mat_vec_mult(self, vec, transpose=False):
        return torch.sum([m(vec, transpose=transpose) for m in self.mats], dim=0)

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


def invert_hessian(hess, inv='pinv', diag=False, idx=None, rm_thresh=1e-15, rm_fill=1e-15,
                   rm_offdiag=False, rcond=1e-15, eps=None, hermitian=True, return_hess=False):
    """
    Invert a Hessian (Fisher Information) matrix (H) to get a covariance
    matrix

    Parameters
    ----------
    hess : tensor or ParamDict
        The Hessian matrix (see optim.compute_hessian)
    inv : str, optional
        If not diag, this is the inversion method. One of
        'pinv' : use pseudo-inverse, takes kwargs: rcond, hermitian
        'lstsq' : use least-squares, takes kwargs: rcond
        'chol' : use cholesky
        'diag' : just invert the diagonal component
    diag : bool, optional
        If True, the input hess tensor represents the diagonal
        of the Hessian, regardless of its shape or ndim.
    idx : array or slice object, optional
        Only used if diag=False. Grab these indices of the 2D hess
        matrix before inverting. Output covariance has rm_fill in
        the diagonal of non-inverted components
    rm_thresh : float, optional
        For diagonal elements of hess below this
        value, truncate these row/columns before inversion.
        If passing idx, rm_thresh operates after applying idx.
    rm_fill : float, optional
        For row/columns that are truncated by rm_thresh,
        this fills the diagonal of the output covariance
    rm_offdiag : bool, optional
        If True, remove the off-diagonal components of hess if
        it has any.
    rcond : float, optional
        rcond parameter for pinverse
    eps : float, optional
        Small value to add to diagonal of hessian (only if diag=False or rm_offdiag=False)
    hermitian : bool, optional
        Hermitian parameter for torch.pinverse
    return_hess : bool, optional
        If True, return downselected Hessian matrix
    
    Returns
    -------
    tensor
    """
    if isinstance(hess, paramdict.ParamDict):
        cov = {}
        for k in hess:
            cov[k] = invert_hessian(hess[k], diag=diag, idx=idx, eps=eps,
                                    rm_offdiag=rm_offdiag, hermitian=hermitian,
                                    rm_thresh=rm_thresh, rm_fill=rm_fill)
        return paramdict.ParamDict(cov)

    if diag:
        # assume hessian holds diagonal, can be any shape
        cov = torch.ones_like(hess, device=hess.device, dtype=hess.dtype)
        s = hess > rm_thresh
        cov[s] = 1 / hess[s]
        cov[~s] = rm_fill
        if return_hess:
            cov = hess

        return cov

    else:
        # assume hessian is 2D
        if rm_offdiag:
            hess = torch.diag(hess.diag())

        H = hess

        # get idx array
        if idx is None:
            idx = np.arange(len(H))
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(H)
            if stop < 0: stop = len(H)
            step = idx.step if idx.step is not None else 1
            idx = np.arange(start, stop, step)
        elif isinstance(idx, (list, tuple)):
            idx = np.asarray(idx)

        # combine idx with rm_thresh
        good_idx = np.where(H.diagonal() > rm_thresh)[0]
        idx = np.array([i for i in idx if i in good_idx])

        # select out indices
        H = H[idx[:, None], idx[None, :]]

        if return_hess:
            return H

        # add eps if desired, do it not inplace here, as oppossed to in invert_matrix
        if eps is not None:
            H = H + eps * torch.eye(len(H), dtype=H.dtype, device=H.device)

        # take inverse to get cov
        C = linalg.invert_matrix(H, inv=inv, rcond=rcond, hermitian=hermitian)

        # fill cov with shape of hess
        cov = torch.eye(len(hess), device=hess.device, dtype=hess.dtype) * rm_fill
        cov[idx[:, None], idx[None, :]] = C

        return cov

