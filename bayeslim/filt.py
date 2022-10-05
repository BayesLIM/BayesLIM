"""
Module for visibility and map filtering
"""
import torch
import numpy as np

from . import utils, dataset


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
    Subtract the MAP prediction
    of the signal you want to filter out.

    y_filt = y - y_map

    where

    y_map = C_signal C_data^-1 y

    where C_data is the full data covariance
    and C_signal is the covariance of the
    signal we want to remove from the data.
    """
    def __init__(self, C_signal, C_data, dim=0, no_filter=False,
                 rcond=1e-15, dtype=None, device=None, name=None):
        """
        Parameters
        ----------
        C_signal : tensor
            Square covariance of signal you want to remove
            of shape (N_pred_samples, N_data_samples)
        C_data : tensor
            Square covariance of the data
            of shape (N_data_samples, N_data_samples)
        dim : int
            Dimension of input data to apply filter
        no_filter : bool, optional
            If True, don't filter the input data and
            return as-is
        rcond : float, optional
            rcond parameter when taking pinv of C_data
        dtype : torch dtype, optional
            This is the data type of the input data to-be filtered.
        name : str, optional
            Name of the filter
        """
        attrs = ['C_signal', 'C_data', 'C_data_inv', 'R', 'V']
        super().__init__(dim=dim, name=name, attrs=attrs)
        self.C_signal = torch.as_tensor(C_signal, device=device)
        self.C_data = torch.as_tensor(C_data, device=device)
        self.dtype = dtype
        self.rcond = rcond
        self.setup_filter()
        self.ein = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.no_filter = no_filter

    def setup_filter(self):
        """
        Setup the filter matrix given self.C_signal, self.C_data
        and self.rcond. This takes pseudo-inv of C_data
        and sets self.R (signal map prediction matrix)
        and self.V (signal variance matrix)
        """
        self.C_data_inv = torch.linalg.pinv(self.C_data,
                                            rcond=self.rcond,
                                            hermitian=True)
        Cs, Cinv = self.C_signal, self.C_data_inv
        # get signal prediction matrix
        self.R = Cs @ Cinv
        self.R = self.R.to(self.dtype).to(self.device)
        # get signal variance matrix
        self.V = Cs - Cs @ Cinv @ Cs.T.conj()

    def predict(self, inp):
        """
        Given input data, form the prediction
        of the signal

        y_map = R @ y_inp

        Note that its covariance is held as self.V

        Parameters
        ----------
        inp : tensor or dataset.TensorData subclass

        Returns
        -------
        tensor or dataset
        """
        if isinstance(inp, dataset.TensorData):
            out = inp.copy()
            out.data = self.predict(out.data)
            return out

        # assume inp is a tensor from here
        ein = self.ein.copy()
        ein = ein[:inp.ndim]
        ein[self.dim] = 'j'
        ein = ''.join(ein)
        y = torch.einsum("ij,{}->{}".format(ein, ein.replace('j','i')), self.R, inp)

        return y

    def forward(self, inp, **kwargs):
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
        y_filt = inp - self.predict(inp)

        return y_filt
