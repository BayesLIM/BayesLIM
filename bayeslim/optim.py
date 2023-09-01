"""
Optimization module
"""
import numpy as np
import torch
import os
from collections import OrderedDict
import time
import pickle
from collections.abc import Iterable

from . import utils, paramdict, linalg
from .dataset import VisData, MapData, TensorData


class BaseLogPrior:
    """
    A base LogPrior object with universal
    functionality for all priors
    """
    def __init__(self, index=None, func=None, fkwargs=None, attrs=None):
        """
        Parameters
        ----------
        index : tuple, optional
            Use this to index the forward-pass
            params before computing prior
        func : callable, optional
            Use this to manipulate the forward-pass
            params before computing the prior.
            This occurs after indexing.
        fkwargs : dict, optional
            keyword arguments for func if needed
        attrs : list, optional
            List of self.attribute names attached to push
            to device when using push() e.g. ['lower_bound', ...]
        """
        self.index = index
        self.func = func
        self.fkwargs = fkwargs if fkwargs is not None else {}
        self.attrs = attrs if attrs is not None else []

    def _index_func(self, params):
        if self.index is not None:
            params = params[self.index]
        if self.func is not None:
            params = self.func(params, **self.fkwargs)

        return params

    def forward(self, params):
        raise NotImplementedError

    def __call__(self, params):
        return self.forward(params)

    def push(self, device):
        dtype = isinstance(device, torch.dtype)
        # push index if needed
        if not dtype:
            if self.index is not None:
                index = []
                for idx in self.index:
                    if isinstance(idx, (torch.Tensor, np.ndarray)):
                        idx = torch.as_tensor(idx, device=device)
                    index.append(idx)
                self.index = tuple(index)

        # push specified attributes
        for attr in self.attrs:
            if hasattr(self, attr):
                a = getattr(self, attr)
                setattr(self, attr, utils.push(a, device))


class LogUniformPrior(BaseLogPrior):
    """
    log uniform prior
    """
    def __init__(self, lower_bound, upper_bound,
                 index=None, func=None, fkwargs=None):
        """
        Parameters
        ----------
        lower_bound, upper_bound : tensors
            Tensors holding the min and max of the allowed
            parameter value. Tensors should broadcast with
            the (indexed) params input
        index : slice or tuple of slice objects
            indexing of params tensor before computing prior.
            default is no indexing.
        func : callable, optional
            pass params through this func after indexing
        fkwargs : dict, optional
            optional kwargs for func
        """
        super().__init__(index, func, fkwargs,
                         attrs=['lower_bound', 'upper_bound'])
        self.lower_bound = torch.as_tensor(lower_bound)
        self.upper_bound = torch.as_tensor(upper_bound)
        self.norm = torch.sum(torch.log(1/(self.upper_bound - self.lower_bound)))

    def forward(self, params):
        """
        Evalute uniform logprior on a parameter tensor.

        Parameters
        ----------
        params : tensor
            Parameter tensor to evaluate prior on.
        """
        # index input params tensor if necessary
        params = self._index_func(params)

        # out of bounds if sign of residual is equal
        lower_sign = torch.sign(self.lower_bound - params)
        upper_sign = torch.sign(self.upper_bound - params)

        # aka if sum(abs(sign)) == 2
        out_of_bounds = torch.abs(lower_sign + upper_sign) == 2

        # the returns below are awkward b/c we need to
        # preserve the output's graph connection to params
        if out_of_bounds.any():
            # out of bounds
            return torch.sum(params) * -np.inf
        else:
            # not out of bounds
            lp = torch.sum(params)
            return lp / lp * self.norm


class LogTaperedUniformPrior(BaseLogPrior):
    """
    Log of an edge-tapered uniform prior, constructed by
    multiplying two mirrored tanh functions and taking log.

    .. math::
        
        c &= \alpha/(b_u - b_l) \\
        P &= \tanh(c(x-b_l))\cdot\tanh(-c(x-b_u))

    """
    def __init__(self, lower_bound=None, upper_bound=None, kind='sigmoid',
                 alpha=10000., index=None, func=None, fkwargs=None):
        """
        Parameters
        ---------
        lower_bound, upper_bound : tensors, optional
            Tensors holding the min and max of the allowed
            parameter value. Tensors should broadcast with
            the (indexed) params input. If only one is provided,
            then the prior is defined as open-ended on the other bound.
        kind : str, optional
            Kind of taper to apply, either ['sigmoid', 'tanh'].
            Sigmoid is defined on all R and thus does not enacting strict
            lower and upper bounds, but is well-defined and differentiable
            over all R, whereas tanh has a hard cutoff at lower and upper bounds.
        alpha : tensor, optional
            A scaling coefficient that determines the amount of tapering.
            The scaling coefficient is determined as alpha / (upper - lower)
            where good values for alpha are 1 < alpha < 100000
            alpha -> inf, less edge taper (like a tophat)
            alpha -> 1, more edge taper (like an inverted quadratic)
            If one of the upper or lower bound is not provided,
            then dbound = upper - lower = 1.0
        index : slice or tuple of slice objects
            indexing of params tensor before computing prior.
            default is no indexing.
        func : callable, optional
            pass params through this func after indexing
        fkwargs : dict, optional
            optional kwargs for func
        """
        super().__init__(index, func, fkwargs,
                         attrs=['coeff', 'lower_bound', 'upper_bound'])
        assert lower_bound is not None or upper_bound is not None
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        if self.lower_bound is not None:
            self.lower_bound = torch.as_tensor(self.lower_bound)
        if self.upper_bound is not None:
            self.upper_bound = torch.as_tensor(self.upper_bound)
        self.alpha = torch.as_tensor(alpha)
        if self.upper_bound is not None and self.lower_bound is not None:
            self.dbound = self.upper_bound - self.lower_bound
        else:
            self.dbound = 1.0
        self.coeff = self.alpha / self.dbound
        self.kind = kind

    def forward(self, params):
        """
        Evaluate log tapered uniform prior on params

        Parameters
        ----------
        params : tensor
        """
        # index input params tensor if necessary
        params = self._index_func(params)

        if self.kind == 'sigmoid':
            func = torch.sigmoid
        elif self.kind == 'tanh':
            func = torch.tanh

        prob = 1.0
        if self.lower_bound is not None:
            prob = prob * func(self.coeff * (params - self.lower_bound))
        if self.upper_bound is not None:
            prob = prob * func(-self.coeff * (params - self.upper_bound))

        return torch.sum(torch.log(prob))


class LogGaussPrior(BaseLogPrior):
    """
    log Gaussian prior. L2 norm regularization
    """
    def __init__(self, mean, cov, sparse_cov=True, side='both',
                 density=True, index=None, func=None, fkwargs=None):
        """
        mean and cov must match shape of params, unless
        sparse_cov == False, in which case cov is 2D matrix
        dotted into params.ravel()

        Parameters
        ----------
        mean : tensor
            mean tensor, broadcasting with (indexed) params shape
        cov : tensor
            covariance tensor, broadcasting with params
        sparse_cov : bool, optional
            If True, cov is the diagonal of the covariance matrix
            with shape matching (indexed) data. Otherwise, cov
            is a 2D matrix dotted into (indexed) params.ravel()
        side : str, optional
            Half or double-sided Gaussian
            'lower' : for y > mean, set residual to zero
            'upper' : for y < mean, set residual to zero
            'both'  : standard gaussian behavior (default)
            Normalization is standard in all cases.
        density : bool, optional
            If True, normalize by PDF area (i.e. prob density)
            otherwise normalize by peak
        index : slice or tuple of slice objects
            indexing of params tensor before computing prior.
            default is no indexing.
        func : callable, optional
            pass params through this func after indexing
        fkwargs : dict, optional
            optional kwargs for func
        """
        super().__init__(index, func, fkwargs,
                         attrs=['mean', 'icov'])
        self.mean = torch.atleast_1d(torch.as_tensor(mean))
        self.cov = torch.atleast_1d(torch.as_tensor(cov))
        self.sparse_cov = sparse_cov
        self.side = side
        self.density = density
        self.compute_icov()

    def forward(self, params):
        """
        Evaluate log Gaussian prior

        Parameters
        ----------
        params : tensor
            Parameter tensor
        """
        params = self._index_func(params)
        res = params - self.mean
        if self.side == 'upper':
            res[res < 0] = 0
        elif self.side == 'lower':
            res[res > 0] = 0
        if self.sparse_cov:
            res = torch.abs(res) if torch.is_complex(res) else res
            chisq = torch.sum(res**2 * self.icov)
        else:
            res = res.ravel()
            res2 = res.conj() if torch.is_complex(res) else res
            chisq = torch.sum(res @ self.icov @ res2)

        out = -0.5 * chisq.real
        if self.density:
            out -= self.norm

        return out

    def compute_icov(self, **kwargs):
        """
        Takes self.cov and computes and sets self.icov

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments for linalg.invert_matrix() if
            not sparse_cov. hermitian=True is hard-coded.
        """
        if self.sparse_cov:
            self.icov = 1. / self.cov
            self.logdet = torch.sum(torch.log(self.cov))
            self.ndim = sum(self.cov.shape)
        else:
            self.icov = linalg.invert_matrix(self.cov, hermitian=True, **kwargs)
            self.logdet = torch.slogdet(self.cov).logabsdet
            self.ndim = len(self.cov)
        self.norm = 0.5 * (self.ndim * torch.log(torch.tensor(2*np.pi)) + self.logdet)
        self.icov = self.icov.to(self.mean.device)


class LogLaplacePrior(BaseLogPrior):
    """
    Log Laplacian Prior. L1 norm regularization
    """
    def __init__(self, mean, scale, side='both', density=True,
                 index=None, func=None, fkwargs=None):
        """
        mean and scale must match shape of params, or be scalars

        .. math::

            \log P(y|m,s) = -|y-m|/s

        The derivative of the residual at zero is defined
        to be zero.

        Parameters
        ----------
        mean : tensor
            mean tensor, broadcasting with (indexed) params shape
        scale : tensor
            scale tensor, broadcasting with params
        side : str, optional
            Half or double-sided exponential
            'lower' : for y > mean, set residual to zero
            'upper' : for y < mean, set residual to zero
            'both'  : standard Laplacian behavior (default)
            Normalization is standard in all cases.
        density : bool, optional
            If True, normalize by PDF area (i.e. prob density)
            otherwise normalize by peak
        index : slice or tuple of slice objects
            indexing of params tensor before computing prior.
            default is no indexing.
        func : callable, optional
            pass params through this func after indexing
        fkwargs : dict, optional
            optional kwargs for func
        """
        super().__init__(index, func, fkwargs,
                         attrs=['mean', 'scale'])
        self.mean = torch.atleast_1d(torch.as_tensor(mean))
        self.scale = torch.atleast_1d(torch.as_tensor(scale))
        self.norm = torch.sum(torch.log(2*self.scale))
        self.side = side
        self.density = density

    def forward(self, params):
        """
        Evaluate log Laplacian prior

        Parameters
        ----------
        params : tensor
            Parameter tensor
        """
        params = self._index_func(params)
        res = params - self.mean

        if self.side == 'upper':
            res[res < 0] = 0
        elif self.side == 'lower':
            res[res > 0] = 0

        out = -torch.sum(torch.abs(res) / self.scale)
        if self.density:
            out -= self.norm

        return out


class LogProb(utils.Module):
    """
    The (negative) log posterior density, or the (negative)
    of the log likelihood plus the log prior, which is
    proportional to the log posterior up to a constant.

    This object handles both the likelihood and
    (optionally) any priors. It assumes a Gaussian
    likelihood of the form

    .. math::

        -log(L) &= \frac{1}{2}(d - \mu)^T \Sigma^{-1} (d - \mu)\\
                &+ \frac{1}{2}\log|\Sigma| + \frac{n}{2}\log(2\pi)
    """
    def __init__(self, model, target, start_inp=None, cov_parameter=False,
                 prior_dict=None, device=None, compute='post', negate=True,
                 grad_type='accumulate'):
        """
        Parameters
        ----------
        model : utils.Module or utils.Sequential subclass
            THe forward model that holds all parameters
            as attributes (i.e. model.named_parameters())
        target : dataset object
            Dataset object holding (mini-batch) data tensors
            of shape (Npol, Npol, Nbls, Ntimes, Nfreqs).
            This should have the same Nbatch as the model.
            This should also hold the inverse covariance,
            otherwise icov is assumed to be ones.
        start_inp : dataset object or iterable object, optional
            Starting input to model. This should have the same
            Nbatch as the model.
        cov_parameter : bool, optional
            If fed a covariance, this makes the inverse covariance
            a parameter in the fit. (not yet implemented)
        prior_dict : dict, optional
            A dictionary with keys of each model parameter name (str)
            accessed from self.model, and values as their logprior
            callables to evaluate. Default is to evaluate
            priors attached to each Module, but if prior_dict is provided
            it supercedes. E.g. {'params': logprior_fn} will evaluate
            logprior_fn(self.model.params)
        device : str, optional
            Set the device for this object
        compute : str, optional
            Distribution to compute out of ['post', 'like', 'prior'],
            regardless of what is attached to the object. This can be
            used to sample from either the posterior, likelihood, or prior,
            with the same API one would use to sample from the posterior.
        negate : bool, optional
            Return the negative log posterior for grad descent (default).
            Otherwise return the log posterior for MCMC.
        grad_type : str, optional
            Gradient type ['accumulate', 'stochastic'].
            If accumulate, then iterate over batches
        """
        super().__init__()
        self.model = model
        assert isinstance(target, torch.utils.data.Dataset)
        self.target = target
        self.start_inp = start_inp
        if cov_parameter:
            raise NotImplementedError
        self.cov_parameter = cov_parameter
        self.device = device
        self.prior_dict = prior_dict
        self.compute = compute
        self.negate = negate
        self.closure_eval = 0
        self.grad_type = grad_type

        # set empty grad mod parameters
        self.set_grad_mod()

        # clear prior cache
        self.clear_prior_cache()

        # set no main params by default
        self.set_main_params()

        # iterate over modules to check for overlapping _names
        # which makes conflicts in prior evaluation
        names = []
        overlap = False
        for mod in self.model.modules():
            if mod.name in names:
                overlap = True
            names.append(mod.name)
        if overlap:
            print("Warning: overlapping module names in model could" \
                  " lead to conflicts in prior evaluation")

    def set_main_params(self, model_params=None, LM=None, set_p0=False):
        """
        Setup a single main parameter tensor that automatically
        interfaces with specified submodule tensors in models. This
        is used for functionality that requires all parameters to
        be on a single tensor (e.g. Hessian computation).
        This can also be used for funcs that need all parameters
        on a single device, although this can also be handled
        easily by moving each mod.params to a single device but
        keeping their response functions on a separate device.

        For any call to self, the values from self.main_params are
        copied over to submodules before forward model evaluation.

        Note: assumes all params are of utils._float() dtype.

        Parameters
        ----------
        model_params : list, optional
            List of submodules params tensors with "prob.model"
            as base to collect and stack in main_params.
            E.g. ['sky.params', 'cal.params']
            If None, main_params is set as None.
            You can also index each params by passing
            a 2-tuple as (str, index), e.g.
            [
            ('sky.params', (range(10), 0, 0)),
            ('beam.params', [(0, [0, 1, 2]), (1, [1, 2])],
            ...,
            ]
            Note that you can also feed two indexing tuples
            (must be wrapped in a list!) which will index the
            params tensor multiple times in that order.
        main_LM : LinearModel object or BaseMat subclass
            This is a linear model object that can act as a 
            preconditioner to main_params. It is an R^N -> R^N
            mapping that maps main_params to its expected
            from given its various sub-params components.
            This is acted upon in the self.send_main_params() call.
            In principle, this should be the lower Cholesky
            of the covariance matrix (i.e. inverse Hessian).
            This is set as self._main_LM.
        set_p0 : bool, optional
            If True, assign main_params values to main_p0 (non-parameter)
            and set main_params as zeros (parameter). In this case,
            main_params + main_p0 is done before sending to sub-modules.
            Note: unlike other modules, if main_LM is assigned then the
            order of operations is main_LM(main_params) + main_p0.
        """
        # if main_params already exists, turn keys back into Parameters
        if hasattr(self, "_main_indices") and self._main_indices is not None:
            for param in self._main_indices:
                self.model.set_param(param)

        # create blank main_params
        self.main_params = None
        self.main_p0 = None
        self._main_indices = None
        self._main_shapes = None
        self._main_devices = None
        self._main_index = None
        self._main_LM = LM
        self._main_set_p0 = set_p0
        self._main_N = None
        if model_params is not None:
            # setup main_params metadata
            N = 0
            self._main_indices = {}
            self._main_shapes = {}
            self._main_devices = {}
            self._main_index = {}
            for param in model_params:
                # get (multi-)indexing if desired
                if isinstance(param, str):
                    idx = None
                else:
                    param, idx = param

                if idx is None or not isinstance(idx, list):
                    if idx is None:
                        # no indexing,take the whole tensor
                        p = self.model[param].detach()
                    else:
                        # single indexing, take part of tensor
                        p = self.model[param][idx].detach()

                    # get shapes
                    shape = p.shape
                    numel = shape.numel()
                    indices = slice(N, N + numel)
                    device = p.device
                    N += numel

                else:
                    # this is mult-indexing
                    shape = []
                    indices = []
                    for _idx in idx:
                        p = self.model[param][_idx].detach()
                        p_shape = p.shape
                        shape.append(p_shape)
                        numel = p_shape.numel()
                        indices.append(slice(N, N + numel))
                        device = p.device
                        N += numel

                # append metadata
                self._main_indices[param] = indices
                self._main_shapes[param] = shape
                self._main_devices[param] = device
                self._main_index[param] = idx

            self._main_N = N

            # collect values from leaf tensors and insert to main_params
            self.collect_main_params()

            # this sends values back to leaf tensors making them leaf views
            self.send_main_params()

    def collect_main_params(self, inplace=True):
        """
        Take existing values of submodule params and using metadata like
        _main_indices, ..., collect values and assign as self.main_params.
        Note that if self.main_p0 is not None then the values are set
        as main_p0 (non-parameter) and self.main_params is set as zeros.

        Parameters
        ----------
        inplace : bool, optional
            If True (default) collect sub-params
            and update self.main_params inplace. Otherwise
            collect sub-params and return tensor.
        """
        if len(self._main_indices) > 0:
            params = torch.zeros(self._main_N, dtype=utils._float(), device=self.device)
            for k in self._main_indices:
                idx, indices = self._main_index[k], self._main_indices[k]
                if idx is None:
                    params[indices] = self.model[k].detach().to(self.device).to(utils._float()).ravel()
                else:
                    if not isinstance(idx, list):
                        # single index
                        params[indices] = self.model[k][idx].detach().to(self.device).to(utils._float()).ravel()
                    else:
                        for _idx, _inds in zip(idx, indices):
                            # multi-index
                            params[_inds] = self.model[k][_idx].detach().to(self.device).to(utils._float()).ravel()

            if not inplace:
                return params

            if self._main_set_p0:
                self.main_p0 = params
                self.main_params = torch.nn.Parameter(torch.zeros_like(params))
            else:
                self.main_p0 = None
                self.main_params = torch.nn.Parameter(params)

            # this sends main_params back to leaf tensors, making them leaf views
            self.send_main_params()

    def send_main_params(self, inplace=True, main_params=None,
                         fill=None, main_p0=None):
        """
        Take existing value of self.main_params and using
        _main_indices, _main_shapes, _main_types,_main_index,
        send its values to the relevant submodule params.

        Parameters
        ----------
        inplace : bool, optional
            If True (default) send main_params to sub-params
            on the module. Otherwise return the re-shaped tensors
            in a dictionary.
        main_params : tensor, optional
            Use this main_params tensor instead of self.main_params
            when sending to sub-params. Default is self.main_params
        fill : float, optional
            If None (default) keep un-indexed elements in params
            with their existing values. Otherwise, fill them with
            this value before returning.
        main_p0 : tensor, optional
            Use this tensor instead of self.main_p0. Note that
            if main_LM exists, the order of operations is
            main_LM(main_params) + main_p0
        """
        # get main_params
        main_params = main_params if main_params is not None else self.main_params
        main_p0 = main_p0 if main_p0 is not None else self.main_p0

        if main_params is not None:
            # pass it through LM if desired
            if self._main_LM is not None:
                main_params = self._main_LM(main_params)

            # sum with p0 if desired
            if main_p0 is not None:
                main_params = main_params + main_p0

            # setup holding container needed for inplace = False
            if not inplace:
                # use a dummy Python3 class object to set params
                class Obj:
                    def __getitem__(self, k):
                        klist = k.split('.')
                        if len(klist) == 1:
                            return getattr(self, klist[0])
                        else:
                            return getattr(self, klist[0])['.'.join(klist[1:])]
                model = Obj()
            else:
                # otherwise use self.model
                model = self.model

            for pname in self._main_indices:
                if not inplace:
                    # create sub-objects for Obj class if needed
                    pname_list = pname.split('.')
                    _model = model
                    for j, pn in enumerate(pname_list):
                        if j == len(pname_list) - 1:
                            setattr(_model, pn, self.model[pname].detach().clone())
                        else:
                            if not hasattr(_model, pn):
                                setattr(_model, pn, Obj())
                            _model = getattr(_model, pn)

                # get metadata
                inds = self._main_indices[pname]
                idx = self._main_index[pname]
                shape = self._main_shapes[pname]
                device = self._main_devices[pname]

                if not isinstance(inds, list):
                    # turn single or no indexing into multi-index form
                    inds, idx, shape = [inds], [idx], [shape]

                for i, (_inds, _idx, _shape) in enumerate(zip(inds, idx, shape)):
                    value = main_params[_inds]
                    value = value.reshape(_shape)
                    value = value.to(device)

                    # only fill if this is first index of this param
                    # only add if this isn't first index of this param
                    # only clobber existing param if first index
                    utils.set_model_attr(model, pname, value, idx=_idx,
                                         clobber_param=True if i == 0 else False,
                                         no_grad=False,
                                         fill=fill if i == 0 else None,
                                         add=i != 0)

            if not inplace:
                # collect dictionary of params and return
                return {k: model[k] for k in self._main_indices}

    @property
    def Nbatch(self):
        """get total number of batches in model"""
        if hasattr(self.model, 'Nbatch'):
            return self.model.Nbatch
        else:
            return 1

    @property
    def batch_idx(self):
        """return current batch index in model"""
        if hasattr(self.model, 'batch_idx'):
            return self.model.batch_idx
        else:
            return 0

    def set_batch_idx(self, idx):
        """Set the current batch index"""
        if hasattr(self.model, 'set_batch_idx'):
            self.model.set_batch_idx(idx)
        elif idx > 0:
            raise ValueError("No method set_batch_idx and requested idx > 0")

    def get_batch_data(self, idx=None):
        """
        Get target and input for the minibatch index idx.

        Parameters
        ----------
        idx : int, optional
            The minibatch index to run, if self.prob
            is batched. Default is self.batch_idx.
            Otherwise just evaluate prob.
            If passed also sets self.batch_idx.

        Returns
        -------
        target object, starting input object
        """
        if idx is not None:
            self.set_batch_idx(idx)
        target = self.target[self.batch_idx]
        inp = None if self.start_inp is None else self.start_inp[self.batch_idx]

        return target, inp

    def forward_chisq(self, idx=None, sum_chisq=True):
        """
        Compute and return chisquare
        by evaluating the forward model and comparing
        against the target data

        Parameters
        ----------
        idx : int, optional
            The minibatch index to run, if self.prob
            is batched. Default is self.batch_idx.
            Otherwise just evaluate prob.
            If passed also sets self.batch_idx.
        sum_chisq : bool, optional
            If True, sum the chisquare over
            the target data axes and return a scalar,
            otherwise returns a tensor of chisq values

        Returns
        -------
        tensor
            Chisquare
        tensor
            Residual of prediction with target
        """
        ## TODO: allow for kwarg dynamic icov for cosmic variance

        # get batch data
        target, inp = self.get_batch_data(idx)

        # copy over main_params if needed
        if self.main_params is not None:
            self.send_main_params()

        # forward pass model
        out = self.model(inp, prior_cache=self.prior_cache)
        if isinstance(out, (VisData, MapData, TensorData)):
            out = out.data
        prediction = out.to(self.device)

        # compute residual
        res = prediction - target.data

        # get inverse covariance
        if hasattr(target, 'icov'):
            icov = target.icov
            cov_axis = target.cov_axis
        else:
            icov = None
            cov_axis = None

        # evaluate chi square
        chisq = apply_icov(res, icov, cov_axis)
        if sum_chisq:
            chisq = torch.sum(chisq)
        if torch.is_complex(chisq):
            chisq = chisq.real

        return chisq, res

    def forward_like(self, idx=None, **kwargs):
        """
        Compute log (Gaussian) likelihood
        by evaluating the chisquare of the forward model
        compared to the target data

        Parameters
        ----------
        idx : int, optional
            The minibatch index to run, if self.prob
            is batched. Default is self.batch_idx.
            Otherwise just evaluate prob.
            If passed also sets self.batch_idx.
        """
        # evaluate chisq for this batch index
        chisq, res = self.forward_chisq(idx)

        # get target and inp data for this batch index
        target, inp = self.get_batch_data(idx)

        # use it to get log likelihood normalization
        if hasattr(target, 'icov') and target.icov is not None:
            like_norm = 0.5 * (target.cov_ndim * torch.log(torch.tensor(2*np.pi)) + target.cov_logdet)
        else:
            like_norm = 0

        # form loglikelihood
        loglike = -0.5 * chisq - like_norm

        if self.negate:
            return -loglike
        else:
            return loglike

    def forward_prior(self, **kwargs):
        """
        Compute log prior given state of params tensors
        """
        # send over main_params if compute = 'prior'
        if self.compute == 'prior' and self.main_params is not None:
            self.send_main_params()

        # evaluate log prior
        logprior = torch.as_tensor(0.0)
        if self.prior_dict is not None:
            # use prior_dict
            for p_key, p_obj in self.prior_dict.items():
                if isinstance(p_obj, (tuple, list)):
                    for p_ob in p_obj:
                        logprior += p_ob(self.model[p_key])
                else:
                    logprior += p_obj(self.model[p_key])

        else:
            if len(self.prior_cache) == 0:
                # populate prior_cache if it is empty
                for _, mod in self.model.named_modules():
                    if hasattr(mod, 'params'):
                        mod.eval_prior(self.prior_cache)

            # add priors
            for k in self.prior_cache:
                logprior = logprior + self.prior_cache[k]

        # clear prior cache
        self.clear_prior_cache()

        # return log prior
        if self.negate:
            return -logprior
        else:
            return logprior

    def forward(self, idx=None, **kwargs):
        """
        Compute log posterior (up to a constant).
        Note that the value of self.negate determines
        if output is log posterior or negative log posterior.

        Parameters
        ----------
        idx : int, optional
            The minibatch index to run, if self.prob
            is batched. Default is self.batch_idx.
            Otherwise just evaluate prob.
            If passed also sets self.batch_idx.
        """
        assert self.compute in ['post', 'like', 'prior']
        prob = torch.as_tensor(0.0, device=self.device)

        # evaluate and add likelihood
        if self.compute in ['post', 'like']:
            prob = prob + self.forward_like(idx, **kwargs)

        # evalute and add prior
        if self.compute in ['post', 'prior']:
            if self.compute == 'prior':
                # if compute is prior, clear any attached graph tensors
                # in all modules such they are regenerated during this
                # forward call (e.g. beam.R.beam_cache)
                for mod in self.modules():
                    mod.clear_graph_tensors()

            # evaluate prior
            prob = prob + self.forward_prior(**kwargs)

        self.clear_prior_cache()

        return prob

    def __call__(self, idx=None, **kwargs):
        """
        Evaluate forward model given starting input, and
        compute posterior given target for a particular
        minibatch index.

        Parameters
        ----------
        idx : int, optional
            The minibatch index to run, if self.prob
            is batched. Default is self.batch_idx.
            Otherwise just evaluate prob.
            If passed also sets self.batch_idx.
        """
        return self.forward(idx, **kwargs)

    def closure(self):
        """
        Function for evaluating the model, performing
        backprop, and returning output given self.grad_type
        """
        self.closure_eval += 1
        if torch.is_grad_enabled():
            self.zero_grad()

        # if accumulating, run all minibatches and backprop
        if self.grad_type == 'accumulate':
            loss = 0
            for i in range(self.Nbatch):
                out = self(i)
                if out.requires_grad:
                    out.backward()
                loss = loss + out.detach()
            loss = loss / self.Nbatch

        # if stochastic, just run current batch, then backprop
        elif self.grad_type == 'stochastic':
            out = self()
            if out.requires_grad:
                out.backward()
            loss = out.detach()

        # modify gradients if desired
        self.grad_modify()

        return loss

    def set_grad_mod(self, grad_mods=None, alpha=1.0):
        """
        Setup parameter gradient modification given mod type.
        Required kwargs are "value" and "mod_type",
        optional includes "index" and others

        The following modification types are allowed
        "clamp" : set |params.grad[index]| > value to zero.
        "replace" : set params.grad[index] to value.
        "isolate" : multiply gradients by dynamic range
            with respect to largest gradient, i.e.
            params.grad *= (params.grad/params.grad.max(dim))**value.
            This has the effect of "isolating" the total gradient
            along its steepest dimension.
        "clip" : keep top N entries in params.grad[index]
            along "dim" axis and set all others to zero,
            where N = value. "dim" is optional kwarg.
            This can be thought of as a form of binary
            isolation.
        "mult" : multiply gradient by value

        Parameters
        ----------
        grad_mods : list of tuples, optional
            List of parameter names and mod dictionaries, e.g.
            [("model.module1.params",
                {"value" : float or tensor(...),
                 "index" : (slice(None), slice(None), (0, 1, 2), ...),
                 "mod_type" : "clamp",
                 }
             ),
             ("model.module2.params",
                {...}
             )]
        alpha : float, optional
            Overall factor to multiply all
            mod "values" by
        """
        self.grad_mods = grad_mods
        self.alpha = alpha

    def grad_modify(self):
        """
        Modify parameter gradients
        """
        if self.grad_mods is not None:
            for param, mod in self.grad_mods:
                grad = self[param].grad
                idx = mod.get('index', slice(None))
                value = mod.get('value') * self.alpha
                mod_type = mod.get("mod_type")
                if grad is not None:
                    if mod_type == 'clamp':
                        # set grad to zero when outside bounds
                        gidx = grad[idx]
                        out_bounds = (gidx < -value) | (gidx > value)
                        gidx[out_bounds] = 0.0
                        grad[idx] = gidx

                    elif mod_type == 'replace':
                        # set grad to value
                        grad[idx] = value

                    elif mod_type == 'isolate':
                        dim = mod.get('dim', None)
                        abs_grad = torch.abs(grad[idx])
                        if dim is None:
                            gmax = torch.max(abs_grad)
                        else:
                            gmax = torch.max(abs_grad, dim=dim, keepdims=True).values
                        grad[idx] *= (abs_grad / gmax)**value

                    elif mod_type == 'clip':
                        # keep N strongest gradients along dim (N = value)
                        dim = mod.get('dim')
                        asort = torch.argsort(torch.abs(grad[idx]), dim=dim,
                                              descending=True)
                        grad[idx] *= (asort <= value)

                    elif mod_type == 'mult':
                        # multiply gradients by value
                        grad[idx] *= value

    def push(self, device):
        """
        Transfer target data to device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype: self.device = device
        for d in self.target.data:
            d.push(device)

    def clear_prior_cache(self):
        """
        Clear the self.prior_cache dictionary
        """
        if hasattr(self, 'prior_cache'):
            del self.prior_cache
        self.prior_cache = {}

    def _set_icov(self, icov):
        """
        LEGACY
        Set inverse covariance as self.icov
        and compute likelihood normalizations

        Parameters
        ----------
        icov : tensor, optional
            Inverse covariance of the target data.
            See optim.apply_icov() for shape details.
            Default is to compute it from self.cov.
            Note that self.icov is used for the loglike,
            and self.cov is used for normalization. The
            two should therefore be consistent.
        """
        ### LEGACY ###
        # in order to make icov a parameter, needs to be attached to prob, not
        # target.data!
        raise NotImplementedError
        # push cov to device is needed
        if self.cov.device is not self.device:
            self.cov = self.cov.to(self.device)

        # compute likelihood normalization from self.cov
        self.like_ndim = sum(self.target.data.shape)
        if self.cov_axis is None:
            self.like_logdet = torch.sum(torch.log(self.cov))
        elif self.cov_axis == 'full':
            self.like_logdet = torch.slogdet(self.cov).logabsdet
        else:
            self.like_logdet = 0
            for i in range(self.cov.shape[2]):
                for j in range(self.cov.shape[3]):
                    for k in range(self.cov.shape[4]):
                        for l in range(self.covh.shape[5]):
                            self.like_logdet += torch.slogdet(self.cov[:, :, i, j, k, l]).logabsdet
        self.like_norm = 0.5 * (self.like_ndim * torch.log(torch.tensor(2*np.pi)) + self.like_logdet)

        # set icov
        if icov is not None:
            # use utils.push in case icov is a parameter
            self.icov = utils.push(icov, self.device)
        else:
            # compute icov and set it
            self.icov = utils.push(compute_icov(self.cov, self.cov_axis), self.device)

        if self.parameter:
            self.icov = torch.nn.Parameter(self.icov.detach().clone())


class Trainer:
    """Object for training a model wrapped with
    the LogProb posterior class"""
    def __init__(self, prob, opt=None, track=False, track_params=None):
        """
        Parameters
        ----------
        prob : LogProb object
        opt : torch.optim.Optimizer class object or instance
        track : bool, optional
            If True, track value of all parameters and append to
            a chain during optimization.
        track_params : list, optional
            If track, this is a list of params to track. Default
            is prob.named_params(), but can also track non-parameter
            tensors as well.
        """
        self.prob = prob
        self._epoch_loss = []
        self._epoch_times = []
        self.track = track
        self.set_opt(opt)

        if prob.grad_type == 'accumulate':
            self.Nbatch = 1
        elif prob.grad_type == 'stochastic':
            self.Nbatch = prob.Nbatch

        self.chain = {}
        if self.track:
            self.init_chain(track_params)

    def init_chain(self, track_params=None):
        """
        Setup chain lists for tracking

        Parameters
        ----------
        track_params : list, optional
            List of attributes of prob to track.
            e.g. ['model.beam.params', ...]. Default is to use
            1. prob.main_params if available, otherwise
            2. prob.model.named_parameters
        """
        self.chain = {}
        if track_params is not None:
            for k in track_params:
                self.chain[k] = []

        else:
            if self.prob.main_params is not None:
                for k in self.prob._main_indices:
                    self.chain['model.' + k] = []
            else:
                for k in self.prob.model.named_parameters():
                    self.chain['model.' + k[0]] = []

    def set_opt(self, opt, *args, **kwargs):
        """
        Set optimizer. If opt is a class object,
        this will instantiate it as

        .. code-block:: python

            self.opt = opt(self.prob.parameters(), *args, **kwargs)

        otherwise if it is a instance just use it as is.

        Parameters
        ----------
        opt : torch optimizer class or class-instance
            Optimizer to use in the run. Setting the
            optimizer will not destroy self.loss array
        """
        if opt is None:
            return
        if isinstance(opt, type):
            # this is a class object
            self.opt = opt(self.prob.parameters(), *args, **kwargs)
        else:
            # this is a class instsance
            self.opt = opt

    def train(self, Nepochs=1, Nreport=None):
        """
        Train the model. Loss is stored in self.loss,
        and parameter values (if track) stored in self.chain

        Parameters
        ----------
        Nepochs : int
            Number of training epochs

        Returns
        -------
        info : dict
            information about the run
        """
        train_start = time.time()

        for epoch in range(Nepochs):
            epoch_start = time.time()
            if Nreport is not None:
                if (epoch > 0) and (epoch % Nreport == 0):
                    print("epoch {}, {:.1f} sec".format(epoch, time.time() - start))

            # zero grads
            self.opt.zero_grad()

            # iterate over minibatches
            L = 0
            for i in range(self.Nbatch):
                # append current state
                if self.track:
                    for k in self.chain:
                        self.chain[k].append(self.prob[k].detach().clone())

                # evaluate forward model from current state
                # backprop, and make a step
                L += self.opt.step(self.prob.closure)

            # append batch-averaged loss
            self._epoch_loss.append(L / self.Nbatch)
            # append time info
            total_time = 0. if len(self._epoch_times) == 0 else self._epoch_times[-1]
            self._epoch_times.append(total_time + (time.time() - epoch_start))

        info = dict(duration=time.time() - train_start)

        return info

    def get_chain(self, name=None, idx=None):
        """
        Extract and return chain history
        if tracking

        Parameters
        ----------
        name : str, optional
            Return just one param.
        idx : int, optional
            Pick out a single epoch from the
            chain, otherwise retuan all epochs.

        Returns
        -------
        tensor or dict
            Chain for given name
        """
        assert self.track
        if name is not None:
            chain = self.chain[name]
            if idx is None:
                chain = torch.stack(chain)
            else:
                chain = chain[idx]

        else:
            if idx is None:
                chain = {k: torch.stack(c) for k, c in self.chain.items()}
            else:
                chain = {k: c[idx] for k, c in self.chain.items()}

        return chain

    def revert_chain(self, Nepochs):
        """
        If tracking, step backwards in the chain
        N times and populate params with chain state,
        popping the last N steps in the chain.

        Note: the current state of the model is not
        the last entry in the chain, so moving back
        one step in model history corresponds to taking
        the last entry in the current chain.

        Parameters
        ----------
        Nepochs : int
            Number of epochs to revert back to (>0)
        """
        if self.track:
            if Nepochs > 0:
                # cycle through chain and update params
                for k in self.chain:
                    with torch.no_grad():
                        self.prob[k] = self.chain[k][-Nepochs]
                        self.chain[k] = self.chain[k][:-Nepochs]

                # pop other lists
                self._epoch_loss = self._epoch_loss[:-Nepochs]
                self._epoch_times = self._epoch_times[:-Nepochs]

                # collect updates into main_params if used
                self.prob.collect_main_params()

    @property
    def loss(self):
        return torch.as_tensor(self._epoch_loss)

    @property
    def times(self):
        return torch.as_tensor(self._epoch_times)


def apply_icov(data, icov, cov_axis, mode='vis'):
    """
    Apply inverse covariance to data

    .. math ::

        data^{\dagger} \cdot \Sigma^{-1} \cdot data

    Parameters
    ----------
    data : tensor
        data tensor to apply to icov
    icov : tensor
        inverse covariance to apply to data
    cov_axis : str
        data axis over which covariance is modeled
        [None, 'bl', 'time', 'freq', 'full', 'pix']
        See Notes
    mode : str, optional
        Either ['vis' (default), 'map'].
        If 'vis' and cov_axis is not None or full
        icov is assumed to be ndim = 6 and
        data of shape (Npol, Npol, Nbls, Ntimes, Nfreqs)
        If 'map' and cov_axis is not None or full
        icov is assumed to be ndim = 5 and
        data of shape (Npol, 1, Nfreqs, Npix)

    Returns
    -------
    tensor

    Notes
    -----
    cov_axis : None
        icov matches the shape of data and represents
        the inverse of the covariance diagonal
    cov_axis : 'full'
        icov is 2D of shape (data.size, data.size) and
        represents the full inv cov of data.ravel()
    For the following, data is assumed to be of shape
    (Npol, Npol, Nbls, Ntimes, Nfreqs) for mode = 'vis' or
    (Npol, 1, Nfreqs, Npix) for mode = 'map'
    cov_axis : 'bl'
        icov is shape (Nbl, Nbl, Npol, Npol, Ntimes, Nfreqs)
    cov_axis : 'time'
        icov is shape (Ntimes, Ntimes, Npol, Npol, Nbls, Nfreqs)
    cov_axis : 'freq'
        mode = 'vis'
            icov is shape (Nfreqs, Nfreqs, Npol, Npol, Nbls, Ntimes)
        mode = 'map'
            icov is shape (Nfreqs, Nfreqs, Npol, 1, Npix)
    cov_axis : 'pix'
        icov is shape (Npix, Npix, Npol, 1, Nfreqs)
    """
    if cov_axis is None:
        # icov is just diagonal
        if icov is None:
            out = data.conj() * data
        else:
            out = data.conj() * data * icov
    elif cov_axis == 'full':
        # icov is full inv cov
        out = data.ravel().conj() @ icov @ data.ravel()
    elif cov_axis == 'bl':
        # icov is along bls
        out = torch.einsum("ijklm,kk,ijklm->ijlm", d.conj(), icov, d)
    elif cov_axis == 'time':
        # icov is along times
        out = torch.einsum("ijklm,ll,ijklm->ijkm", d.conj(), icov, d)
    elif cov_axis == 'freq':
        # icov is along freqs
        if mode == 'vis':
            out = torch.einsum("ijklm,mm,ijklm->ijkl", d.conj(), icov, d)
        elif mode == 'map':
            out = torch.einsum("ijkl,kk,ijkl->ijl", d.conj(), icov, d)
    elif cov_axis == 'pix':
        # icov is along Npix of map
        out = torch.einsum("ijkl,ll,ijkl->ijk", d.conj(), icov, d)

    return out


def cov_get_diag(cov, cov_axis, mode='vis', shape=None):
    """
    Get the diagonal of a covariance and reshape
    into appropriate data shape, depending
    on cov_axis and mode

    Parameters
    ----------
    cov : tensor
        The covariance or inv. covariance tensor
    cov_axis : str
        The covariance type.
        None   : cov is variance w/ shape of data
        'full' : cov is (N, N) matrix
        'bl'   : cov is (Nbl, Nbl, Npol, Npol, Ntimes, Nfreqs)
        'time' : cov is (Ntimes, Ntimes, Npol, Npol, Nbls, Nfreqs)
        'freq' :
            mode = 'vis' : cov is (Nfreqs, Nfreqs, Npol, Npol, Nbls, Ntimes)
            mode = 'map' : cov is (Nfreqs, Nfreqs, Npol, 1, Npix)
        'pix'  : cov is (Npix, Npix, Npol, 1, Nfreqs)
        It is assumed that data is shape
        mode = 'vis' : (Npol, Npol, Nbls, Ntimes, Nfreqs)
        mode = 'map' : (Npol, 1, Nfreqs, Npix)
        See optim.apply_cov() for more details. 
    mode : str, optional
        Either ['vis', 'map'], whether this cov is for a VisData
        or MapData object.
    shape : tuple, optional
        Only needed for cov_axis = 'full', this is the
        shape of the data.

    Returns
    -------
    tensor
    """
    N = len(cov)
    if cov_axis is None:
        return cov
    elif cov_axis == 'full':
        diag = cov.diagonal()
        return diag.reshape(shape)
    elif cov_axis == 'bl':
        diag = cov[range(N), range(N)]
        return diag.moveaxis(0, 2)
    elif cov_axis == 'time':
        diag = cov[range(N), range(N)]
        return diag.moveaxis(0, 3)
    elif cov_axis == 'freq':
        diag = cov[range(N), range(N)]
        return diag.moveaxis(0, 4)
    elif cov_axis == 'pix':
        diag = cov[range(N), range(N)]
        return diag.moveaxis(0, -1)
    else:
        raise NameError("didn't recognize cov_axis {}".format(cov_axis))


def compute_icov(cov, cov_axis, inv='pinv', **kwargs):
    """
    Compute the inverse covariance. Shallow wrapper
    around linalg.invert_matrix().

    Parameters
    ----------
    cov : tensor
        data covariance. See optim.apply_icov() for shapes
    cov_axis : str
        covariance type. See optim.apply_icov() for options.
        If input covariance is dense, it is assumed that its
        dense dimensions are the first two dimensions, and
        extra dimensions are batch dimensions.
    inv : str, optional
        The kind of inversion method. See linalg.invert_matrix()
    kwargs : dict, optional
        Keyword arguments for linalg.invert_matrix()

    Returns
    -------
    tensor
    """
    # set inversion function
    if cov_axis is None:
        # this is just diagonal
        icov = 1 / cov
    elif cov_axis == 'full':
        # invert full covariance
        icov = linalg.invert_matrix(cov, inv=inv, **kwargs)

    return icov


def _hessian(func, inputs, vectorize=False, N=None):
    """
    A close copy of torch.autograd.functional.hessian but
    with some added functionality
    
    Parameters
    ----------
    func : callable
        A function that takes inputs and returns a scalar
    inputs : tensor
        Inputs to func, over which to compute hessian
    vectorize : bool, optional
        Kwarg to torchautograd.functional.hessian()
    N : int, optional
        If provided, only compute hessian[:N, :] component
        of the hessian, such that output is non-square.

    Returns
    -------
    tensor
    """
    def jac_func(*inputs, N=N):
        jac = torch.autograd.functional.jacobian(func, inputs, create_graph=True)
        if N is not None:
            jac = (jac[0][:N],)
        return jac

    res = torch.autograd.functional.jacobian(jac_func, inputs, vectorize=vectorize)
    return res[0]


def compute_hessian(prob, pdict, rm_offdiag=False, Npdict=None, vectorize=False):
    """
    Compute Hessian of prob with respect to params.
    Note that this edits params in prob inplace!

    Parameters
    ----------
    prob : optim.LogProb object
        Log posterior object
    pdict : ParamDict object
        Holding parameters of prob for which to compute hessian,
        and the values at which to compute it
    rm_offdiag : bool, optional
        If True, only keep the diagonal of the Hessian and
        reshape to match input params shape.
    Npdict : dict, optional
        N parameter, see optim._hessian(), for each key in pdict
    vectorize : bool, optional
        kwarg for torch.autograd.functional.hessian

    Returns
    -------
    ParamDict object
        Hessian of prob
    """
    # get all leaf variables on prob
    named_params = prob.named_params

    # unset all named params
    prob.unset_param(named_params)

    # iterate over keys in pdict
    hess = paramdict.ParamDict({})
    for param in pdict:
        # setup func
        inp = pdict[param]
        shape = inp.shape
        _N = None if Npdict is None else Npdict[param]
        N2 = shape.numel()
        N1 = _N if _N is not None else N2
        def func(x):
            utils.set_model_attr(prob, param, x, clobber_param=True)
            return prob()
        # iterate over batches
        for i in range(prob.Nbatch):
            prob.set_batch_idx(i)
            h = _hessian(func, inp, N=_N, vectorize=vectorize).reshape(N1, N2)
            if rm_offdiag:
                h = h.diag().reshape(shape)
            if i == 0:
                H = h
            else:
                H += h

        hess[param] = H

        # unset param
        prob.unset_param(param)

    # make every key in named_params a leaf variable again
    prob.set_param(named_params)

    return hess


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


def main_params_index(prob, main_index, subset_index):
    """
    Take a LogProb object and its main_index dictionary
    and index a subset of its elements.

    Parameters
    ----------
    prob : LobProb object
        A LogProb with set_main_params() activated
    main_index : dict
        The prob._main_index dictionary
    subset_index : dict
        Similar in structure to main_index, except
        now values index the tensor params[main_index[param]]

    Returns
    -------
    list
    """
    def update(p, main_idx, subset_idx, start):
        if main_index is not None:
            p = p[main_idx]
        _idx = np.arange(p.numel()).reshape(p.shape) + start
        start += _idx.max() + 1
        if subset_idx is not None:
            _idx = _idx[subset_idx]
        else:
            return start
        if len(_idx) == 0:
            return start
        if isinstance(_idx, (int, np.integer)):
            _idx = [_idx]
        else:
            _idx = _idx.ravel()
        idx.extend(_idx)
        return start

    idx = []
    start = 0
    for k, v in main_index.items():
        p = prob.model[k]
        s_idx = subset_index[k] if k in subset_index else None
        if isinstance(v, list):
            # v holds multiple indexing tuple
            for i, _v in enumerate(v):
                start = update(p, _v, s_idx[i], start)
        else:
            # v is just a single indexing tuple
            start = update(p, v, s_idx, start)

    return idx

