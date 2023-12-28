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

    def push(self, device, inplace=True, copy=True):
        """
        Push params to device. Can feed
        device as a dictionary which will push
        params to different devices. If not inplace,
        make a copy and return

        Parameters
        ----------
        device : str
            Device to push all keys
        inplace : bool, optional
            If True (default) perform inplace, otherwise
            make a clone/copy and return
        copy : bool, optional
            Only used if inplace=False.
            If True, make a copy (detach and clone), otherwise
            just clone, which keeps it in graph but eliminates
            upstream mutability. If True and if key is a Parameter,
            new tensor is also a Parameter, otherwise new tensor
            is a leaf.
        """
        if inplace:
            obj = self
        else:
            if copy:
                obj = self.copy()
            else:
                obj = self.clone()

        for k in obj.params:
            d = device if not isinstance(device, dict) else device[k]
            obj.params[k] = utils.push(obj.params[k], d)

        obj._setup()

        if not inplace:
            return obj

    def clone(self, **kwargs):
        """clone object params"""
        return ParamDict({k: self.params[k].clone(**kwargs) for k in self.keys()})

    def copy(self):
        """copy (detach and clone) object. preserves requires_grad"""
        out = ParamDict({})
        for k in self.keys():
            p = self.params[k]
            if p.requires_grad:
                out[k] = torch.nn.Parameter(p.detach().clone())
            else:
                out[k] = p.detach().clone()
        return out

    def detach(self):
        """detach object (don't clone)"""
        return ParamDict({k: self.params[k].detach() for k in self.keys()})

    def ones(self):
        """Return a cloned object filled with ones"""
        pdict = self.clone()
        for k in pdict:
            pdict[k][:] = 1.0
        return pdict

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
        io.write_pkl(fname, self.clone(), overwrite=overwrite)

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

    def operator(self, func, args=(), inplace=False):
        """
        Apply a function to each tensor value in self
        and return the ParamDict object
        e.g. 
            ParamDict.operator(torch.log)
            ParamDict.operator(lambda x: torch.mean(x, dim=0))

        One can also feed additional arguments that are passed
        to the func, including other ParamDict objects
        which are iterated with the same keys as self.
        func kwargs can be handled using lambda as shown above.

        Parameters
        ----------
        func : callable
            Function to call on each tensor in self
        args : iteratble, optional
            Additional arguments to pass to func, note
            that self is treated as first argument.
        inplace : bool, optional
            Apply operation inplace and return None

        Returns
        -------
        ParamDict object
        """
        if inplace:
            for k in self.keys():
                _args = (a if not isinstance(a, (dict, ParamDict)) else a[k] for a in args)
                self[k] = func(self[k], *_args)
        else:
            out = {}
            for k in self.keys():
                _args = (a if not isinstance(a, (dict, ParamDict)) else a[k] for a in args)
                out[k] = func(self[k], *_args)

            return ParamDict(out)

    def __mul__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] * other[k] for k in self.keys()})
        else:
            return ParamDict({k: self.params[k] * other for k in self.keys()})

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys():
                self.params[k] *= other[k]
        else:
            for k in self.keys():
                self.params[k] *= other
        return self

    def __matmul__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] @ other[k] for k in self.keys()})
        else:
            return ParamDict({k: self.params[k] @ other for k in self.keys()})

    def __rmatmul__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: other[k] @ self.params[k] for k in self.keys()})
        else:
            return ParamDict({k: other @ self.params[k] for k in self.keys()})

    def __imatmul__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys():
                self.params[k] @= other[k]
        else:
            for k in self.keys():
                self.params[k] @= other
        return self

    def __div__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] / other[k] for k in self.keys()})
        else:
            return ParamDict({k: self.params[k] / other for k in self.keys()})

    def __rdiv__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: other[k] / self.params[k] for k in self.keys()})
        else:
            return ParamDict({k: other / self.params[k] for k in self.keys()})

    def __idiv__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys():
                self.params[k] /= other[k]
        else:
            for k in self.keys():
                self.params[k] /= other
        return self

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __add__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] + other[k] for k in self.keys()})
        else:
            return ParamDict({k: self.params[k] + other for k in self.keys()})

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys():
                self.params[k] += other[k]
        else:
            for k in self.keys():
                self.params[k] += other
        return self

    def __sub__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] - other[k] for k in self.keys()})
        else:
            return ParamDict({k: self.params[k] - other for k in self.keys()})

    def __rsub__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: other[k] - self.params[k] for k in self.keys()})
        else:
            return ParamDict({k: other - self.params[k] for k in self.keys()})

    def __isub__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys():
                self.params[k] -= other[k]
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


def model2pdict(model, parameters=True, clone=False, prefix=None):
    """
    Build ParamDict from a model.
    Note that dict values are just pointers to the
    actual parameter tensors in the model,
    unless clone is True

    Parameters
    ----------
    model : utils.Module subclass
    parameters : bool, optional
        If True, only return model params
        that are Parameter objects, otherwise
        return all model params tensors
    clone : bool, optional
        If True, detach and clone the tensors.
        Default is False.
    prefix : str, optional
        Prefix for any assigned attributes

    Returns
    -------
    ParamDict object
    """
    d = {}
    prefix = prefix if prefix is not None else ''
    # append params from this module if exists
    if hasattr(model, 'params'):
        if not parameters or model.params.requires_grad:
            ## TODO: this won't work for ArrayModel antpos params
            params = model.params
            if clone:
                params = params.detach().clone()
            d[prefix + 'params'] = params

    for child in model.named_children():
        key, mod = child
        # add sub modules as well
        key = '{}{}.'.format(prefix, key)
        sub_d = model2pdict(mod, parameters=parameters, clone=clone, prefix=key)
        d.update(sub_d)

    return ParamDict(d)
