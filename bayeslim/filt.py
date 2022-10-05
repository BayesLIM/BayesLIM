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

    y_map = G y

    where

    G = C_signal [C_signal + C_noise]^-1

    if generalized_form = False or

    G = [C_signal^-1 + C_noise^-1]^-1 C_noise^-1

    if generalized_form = True, where C_signal is
    the covariance of the signal and C_noise is the
    covariance of the remaining parts of the data
    (can include thermal noise and whatever other
    terms in the data).
    """
    def __init__(self, Cs, Cn, dim=0, hermitian=True, no_filter=False,
                 generalized_form=False, rcond=1e-15, dtype=None,
                 device=None, name=None):
        """
        Parameters
        ----------
        Cs : tensor
            Square covariance of signal you want to remove
            of shape (N_pred_samples, N_data_samples)
        Cn : tensor
            Square covariance of the noise
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
        generalized_form : bool, optional
            If True, use alternative filtering
            matrix (see above)
        rcond : float, optional
            rcond parameter when taking pinv of C_data
        dtype : torch dtype, optional
            This is the data type of the input data to-be filtered.
        name : str, optional
            Name of the filter
        """
        attrs = ['Cs', 'Cn', 'Cs_inv', 'Cn_inv' 'C_inv', 'R', 'V']
        super().__init__(dim=dim, name=name, attrs=attrs)
        self.Cs = torch.as_tensor(Cs, device=device)
        self.Cn = torch.as_tensor(Cn, device=device)
        self.generalized_form = generalized_form
        self.dtype = dtype
        self.rcond = rcond
        self.setup_filter()
        self.ein = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.hermitian = hermitian
        self.no_filter = no_filter

    def setup_filter(self):
        """
        Setup the filter matrix given self.C_signal, self.C_data
        and self.rcond. This takes pseudo-inv of C_data
        and sets self.G (signal map prediction matrix)
        and self.V (signal variance matrix)
        """
        pinv = lambda x: torch.linalg.pinv(x, rcond=self.rcond, hermitian=self.hermitian)
        if self.generalized_form:
            # G = [S^-1 + N^-1]^-1 N^-1
            self.Cs_inv = pinv(self.Cs)
            self.Cn_inv = pinv(self.Cn)
            self.C_inv = pinv(self.Cs_inv + self.Cn_inv)
            self.G = self.Cs @ se
            self.V = self.Cs - self.Cs @ pinv(self.Cs + self.Cn) @ self.Cs.T.conj()

        else:
            # G = S [S + N]^-1
            self.C_inv = pinv(self.Cs + self.Cn)
            self.G = self.Cs @ self.C_inv
            self.V = self.Cs - self.Cs @ self.C_inv @ self.Cs.T.conj()

        self.G = self.G.to(self.dtype).to(self.device)

    def predict(self, inp):
        """
        Given input data, form the prediction
        of the signal

        y_map = G @ y_inp

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
        y = torch.einsum("ij,{}->{}".format(ein, ein.replace('j','i')), self.G, inp)

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
