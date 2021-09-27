"""
Module for ParamDict
"""
import numpy as np
import torch


class ParamDict:
    """
    An object holding a dictionary of model parameters.
    """
    def __init__(self, params):
        """
        Parameters
        ----------
        params : dict
            Dictionary of parameters with str keys
            and tensor values
        """
        self.params = params
        self._setup()

    def _setup(self):
        self.devices = {k: self.params[k].device for k in self.keys()}

    def keys(self):
        return list(self.params.keys())

    def values(self):
        return list(self.params.values())

    def items(self):
        return list(self.params.items())

    def __mul__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] * other.params[k] for k in self.keys()})
        else:
            return ParamDict({k: self.params[k] * other for k in self.keys()})

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys():
                self.params[k] *= other.params[k]
        else:
            for k in self.keys():
                self.params[k] *= other
        return self

    def __div__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] / other.params[k] for k in self.keys()})
        else:
            return ParamDict({k: self.params[k] / other for k in self.keys()})

    def __rdiv__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: other.params[k] / self.params[k] for k in self.keys()})
        else:
            return ParamDict({k: other / self.params[k] for k in self.keys()})
        return self

    def __idiv__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys():
                self.params[k] /= other.params[k]
        else:
            for k in self.keys():
                self.params[k] /= other

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __add__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] + other.params[k] for k in self.keys()})
        else:
            return ParamDict({k: self.params[k] + other for k in self.keys()})

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys():
                self.params[k] += other.params[k]
        else:
            for k in self.keys():
                self.params[k] += other
        return self

    def __sub__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] - other.params[k] for k in self.keys()})
        else:
            return ParamDict({k: self.params[k] - other for k in self.keys()})

    def __rsub__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: other.params[k] - self.params[k] for k in self.keys()})
        else:
            return ParamDict({k: other - self.params[k] for k in self.keys()})

    def __isub__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys():
                self.params[k] -= other.params[k]
        else:
            for k in self.keys():
                self.params[k] -= other
        return self

    def __neg__(self):
        return ParamDict({k: -self.params[k] for k in self.keys()})

    def __pow__(self, alpha):
        return ParamDict({k: self.params[k]**alpha for k in self.keys()})

    def __iter__(self):
        return (p for p in self.params)

    def clone(self):
        """clone object"""
        return ParamDict({k: self.params[k].detach().clone() for k in self.keys()})

    def copy(self):
        """copy object"""
        return ParamDict({k: torch.nn.Parameter(self.params[k].detach().clone()) for k in self.keys()})

    def detach(self):
        """detach object"""
        return ParamDict({k: self.params[k].detach() for k in self.keys()})

    def update(self, other):
        for key in other:
            self.__setitem__(key, other[key])
        self._setup()

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val

    def write_pkl(self, fname, overwrite=False):
        """
        Write ParamDict to .pkl file

        Parameters
        ----------
        fname : str
            Path to output .pkl filename
        overwrite : bool, optional
            If True overwrite fname if it exists
        """
        from bayeslim import io
        pd = ParamDict({k: self.params[k].detach() for k in self.keys()})
        io.write_pkl(fname, pd, overwrite=overwrite)

    @staticmethod
    def read_pkl(fname, force_cpu=False):
        """
        Load .pkl file and return ParamDict object

        Parameters
        ----------
        fname : str
            .pkl file to load as ParamDict
        force_cpu : bool, optional
            Force tensors onto CPU, even if they were
            written from a GPU

        Returns
        -------
        ParamDict object
        """
        from bayeslim import io
        pd = io.read_pkl(fname)
        if force_cpu:
            for k in pd.keys():
                pd.params[k] = pd.params[k].cpu()
        pd._setup()

        return pd

