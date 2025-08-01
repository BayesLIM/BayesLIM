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
import copy

from . import utils, paramdict, linalg, linear_model
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
    def __init__(self, mean, cov, diag_cov=True, side='both',
                 density=True, index=None, func=None, fkwargs=None):
        """
        mean and cov must match shape of params, unless
        diag_cov == False, in which case cov is 2D matrix
        dotted into params.ravel()

        Parameters
        ----------
        mean : tensor
            mean tensor, broadcasting with (indexed) params shape
        cov : tensor
            covariance tensor, broadcasting with params
        diag_cov : bool, optional
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
        self.diag_cov = diag_cov
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
        if self.diag_cov:
            res_sq = (res * res.conj()).real if torch.is_complex(res) else res**2
            chisq = torch.sum(res_sq * self.icov)
        else:
            res = res.ravel()
            chisq = torch.sum(res @ self.icov @ res.conj())

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
            not diag_cov. hermitian=True is hard-coded.
        """
        if self.diag_cov:
            self.icov = 1. / self.cov
            self.logdet = torch.sum(torch.log(self.cov))
            self.ndim = self.cov.numel()
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
                 grad_type='accumulate', complex_circular=True):
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
        complex_circular : bool, optional
            If True, assume the likelihood residuals are
            complex circularly symmetric, which changes the
            log-likelihood normalization.
            In this case, the covariance is assumed to be
            Cz = Cx + Cy, where z = x + jy is the circ. sym. vector.
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
        self.complex_circular = complex_circular

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

        Parameters
        ----------
        model_params : list, optional
            List of submodules params tensors to collect and
            stack in main_params, with "self.model" basename.
            E.g. ['rime.sky.params', 'cal.params']
            If None, main_params is set as None.
            You can also index each params by passing
            a 2-tuple as (str, index), e.g.
            [
            ('rime.sky.params', (range(10), 0, 0)),
            ('rime.beam.params', [(0, [0, 1, 2]), (1, [1, 2])],
            ...,
            ]
            Note that you can also feed two indexing tuples
            (must be wrapped in a *list*!) which will index the
            params tensor multiple times in that order.
            Note you can also end the tuple with an additional
            string which will be the shorthand for the parameter.
            E.g. ('rime.sky.params', None, 'sky') is called 'sky'.
        LM : LinearModel or DictLM or BaseMat subclass
            This is a linear model object that can act as a 
            preconditioner to main_params. It is an R^N -> R^N
            mapping that acts as preconditioner on main_params.
            This is acted upon in the self.send_main_params() call.
            In principle, this could be the lower Cholesky
            of the covariance matrix, or an NxN matrix holding
            the covariance eigenvectors.
            This is set as self._main_LM. If this is a DictLM object
            then this acts on each reshaped sub-parameter.
        set_p0 : bool, optional
            If True, assign main_params values to main_p0 (non-parameter)
            and set main_params as zeros (parameter). In this case,
            main_params + main_p0 is done before sending to sub-modules.
            Note: unlike other modules, if main_LM is assigned then the
            order of operations is main_LM(main_params) + main_p0.
        """
        # if main_params already exists, turn keys back into Parameters
        if hasattr(self, "_main_names") and self._main_names is not None:
            for k, pname in self._main_names.items():
                self.model.set_param(pname)

        # create blank main_params
        self.main_params = None
        self.main_p0 = None
        self._main_indices = None
        self._main_shapes = None
        self._main_devices = None
        self._main_index = None
        self._main_names = None
        self._main_dtypes = None
        self._main_dtype = None
        if LM is not None and not utils.check_devices(LM.device, self.device):
            LM = copy.deepcopy(LM)
            LM.push(self.device)
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
            self._main_names = {}
            self._main_dtypes = {}
            for param in model_params:
                # get (multi-)indexing if desired
                if isinstance(param, str):
                    idx = None
                    name = param
                else:
                    # assume param is a tuple
                    if len(param) == 2:
                        param, idx = param
                        name = param
                    else:
                        param, idx, name = param

                if idx is None or not isinstance(idx, list):
                    if idx is None:
                        # no indexing, take the whole tensor
                        p = self.model[param].detach()
                    else:
                        # ensure idx is tensor on param device
                        device = self.model[param].device
                        idx = tuple(utils._idx2ten(i, device=device) for i in idx)
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
                    # ensure idx holds tensors on param's device
                    device = self.model[param].device
                    idx = [tuple(utils._idx2ten(i, device=device) for i in _idx) for _idx in idx]
                    for _idx in idx:
                        # index param
                        p = self.model[param][_idx].detach()
                        p_shape = p.shape
                        shape.append(p_shape)
                        numel = p_shape.numel()
                        indices.append(slice(N, N + numel))
                        device = p.device
                        N += numel

                # append metadata
                self._main_indices[name] = indices
                self._main_shapes[name] = shape
                self._main_devices[name] = device
                self._main_index[name] = idx
                self._main_names[name] = param
                self._main_dtypes[name] = p.dtype

            self._main_N = N

            # determine main_dtype: this should be highest resolution dtype of all dtypes
            # i.e. complex128 > float64, float64 > float32
            for i, dtype in enumerate(self._main_dtypes.values()):
                if i == 0:
                    self._main_dtype = dtype
                else:
                    if utils._float_resol[dtype] > utils._float_resol[self._main_dtype]:
                        self._main_dtype = dtype

            # collect values from leaf tensors and insert to main_params
            self.collect_main_params()

            # send Parameters back to leaf tensors making them leaf views
            self.send_main_params()

    def sort_main_params(self, new_main_indices, incomplete=False):
        """
        Given a previously defined main_params, resort the
        elements of self.main_params to an arbitrary order
        and create a new self._main_indices.

        Parameters
        ----------
        new_main_indices : dict
            The new main_indices dictionary, to replace
            self._main_indices. Note this must conform
            to shape of the existing self._main_indices
            e.g. if _main_indices = {'a': [0,1,2,3], 'b': [4,5,6,7]}
            a possible input is {'a': [0, 4, 1, 5], 'b': [2, 6, 3, 7]}
            such that the even idx of 'a' and 'b' are stored at the front
            of the tensors, and the odd indices are pushed to the back.
        incomplete : bool, optional
            If True, new_main_indices only holds the indices
            *for each param* you want stored at the front of
            the tensor, with all other elements to be pushed to
            the back. I.e. the proper main_indices is computed for you.
            e.g. if _main_indices = {'a': [0,1,2,3], 'b': [4,5,6,7]}
            a possible input corresponding to the correct input
            shown in the above parameter doc would be
            {'a': [0, 2], 'b': [0, 2]}, which would then be transformed
            as {'a': [0, 4, 1, 5], 'b': [2, 6, 3, 7]}.
        """
        # make sure this is located on self.device
        new_main_indices = copy.deepcopy(new_main_indices)

        # determine if we need to complete the dict
        if incomplete:
            # get the length of params moved to the front
            def count(v):
                length = 0
                if v is None:
                    pass
                elif isinstance(v, slice):
                    start = v.start if v.start is not None else 0
                    stop = v.stop
                    step = v.step if v.step is not None else 1
                    length = (stop - start) // step
                else:
                    length = len(v)

                return length

            N = 0
            for k, v in new_main_indices.items():
                if isinstance(v, list):
                    for _v in v:
                        N += count(_v)
                else:
                    N += count(v)

            # now iterate through params and make new indexing tensors
            def new_indexing_tensor(old, new, n1, n2):
                """n1 is start of first half, n2 is start of second half"""
                # get length of the old
                old_len = count(old)
                # get new in integer form
                new = utils._slice2tensor(new)
                new = new if new is not None else []
                # make new indexing tensor starting from n2
                idx = torch.arange(n2, n2 + old_len)
                n2 += old_len - len(new)
                # now iterate through new and insert n1 indices
                # and push forward idx when doing so
                for n in new:
                    idx[n] = n1
                    idx[n+1:] -= 1
                    n1 += 1

                return idx, n1, n2

            n1, n2 = 0, N
            for k, new in new_main_indices.items():
                old = self._main_indices[k]
                if isinstance(new, list):
                    idx = []
                    for _new, _old in zip(new, old):
                        _idx, n1, n2 = new_indexing_tensor(_old, _new, n1, n2)
                        idx.append(_idx)
                    new_main_indices[k] = idx

                else:
                    idx, n1, n2 = new_indexing_tensor(old, new, n1, n2)
                    new_main_indices[k] = idx

        # now iterate over completed indices and make sure then are tensors on self.device
        for k, v in new_main_indices.items():
            if isinstance(v, list):
                new_main_indices[k] = [utils._idx2ten(_v, device=self.device) for _v in v]
            else:
                new_main_indices[k] = utils._idx2ten(v, device=self.device)

        # create a new main_params tensor
        main_params = torch.zeros_like(self.main_params)
        if self.main_p0 is not None:
            main_p0 = torch.zeros_like(self.main_p0)

        # iterate over params
        for k, new in new_main_indices.items():
            old = self._main_indices[k]
            if isinstance(new, list):
                for _new, _old in zip(new, old):
                    main_params[_new] = self.main_params.data[_old]
                    if self.main_p0 is not None:
                        main_p0[_new] = self.main_p0.data[_old]

            else:
                main_params[new] = self.main_params.data[old]
                if self.main_p0 is not None:
                    main_p0[new] = self.main_p0.data[old]

        # assign
        self.main_params = torch.nn.Parameter(main_params)
        if self.main_p0 is not None:
            self.main_p0 = main_p0
        self._main_indices = new_main_indices

    def collect_main_params(self, inplace=True):
        """
        Take existing values of submodule params and using metadata like
        _main_indices, ..., collect values and assign as self.main_params.
        Note that if self._main_set_p0 == True then the values are set
        as main_p0 (non-parameter) and self.main_params is set as zeros.

        Parameters
        ----------
        inplace : bool, optional
            If True (default) collect sub-params
            and update self.main_params inplace. Otherwise
            collect sub-params and return tensor.
        """
        if self._main_indices is not None and len(self._main_indices) > 0:
            params = torch.zeros(self._main_N, dtype=self._main_dtype, device=self.device)
            for k in self._main_indices:
                name = self._main_names[k]
                idx, indices = self._main_index[k], self._main_indices[k]
                if idx is None:
                    params[indices] = self.model[name].detach().to(self.device).to(self._main_dtype).flatten()
                else:
                    if not isinstance(idx, list):
                        # single index
                        params[indices] = self.model[name][idx].detach().to(self.device).to(self._main_dtype).flatten()
                    else:
                        for _idx, _inds in zip(idx, indices):
                            # multi-index
                            params[_inds] = self.model[name][_idx].detach().to(self.device).to(self._main_dtype).flatten()

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

    def send_main_params(self, main_params=None, inplace=True,
                         fill=None, main_p0=None):
        """
        Take existing value of self.main_params and using
        _main_indices, _main_shapes, _main_types, _main_index,
        send its values to the relevant submodule params.

        Parameters
        ----------
        main_params : tensor, optional
            Use this main_params tensor instead of self.main_params
            when sending to sub-params. Default is self.main_params
        inplace : bool, optional
            If True (default) send main_params to sub-params
            on the module. Otherwise return the re-shaped tensors
            in a dictionary.
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

            # pass it through LM if desired (note this doesn't apply to main_p0)
            # if main_LM is a DictLM, wait until we reshape each parameter
            if self._main_LM is not None and not isinstance(self._main_LM, linear_model.DictLM):
                main_params = self._main_LM(main_params)

            # sum with p0 if desired. if DictLM do this later
            if main_p0 is not None and not isinstance(self._main_LM, linear_model.DictLM):
                main_params = main_params + main_p0

            for name, pname in self._main_names.items():
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
                inds = self._main_indices[name]
                idx = self._main_index[name]
                shape = self._main_shapes[name]
                device = self._main_devices[name]
                dtype = self._main_dtypes[name]

                if not isinstance(inds, list):
                    # turn single or no indexing into multi-index form
                    inds, idx, shape = [inds], [idx], [shape]

                for i, (_inds, _idx, _shape) in enumerate(zip(inds, idx, shape)):
                    # index main_params for this pname
                    value = main_params[_inds]
                    value = value.reshape(_shape)

                    # pass through DictLM if needed
                    if isinstance(self._main_LM, linear_model.DictLM):
                        value = self._main_LM(name, value)
                        # sum with main_p0 if needed
                        if main_p0 is not None:
                            value = value + main_p0[_inds].reshape(_shape)

                    if not utils.check_devices(value.device, device):
                        value = value.to(device)

                    if value.dtype != dtype:
                        value = value.to(dtype)

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
                return {k: model[v] for k, v in self._main_names.items()}

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

    @batch_idx.setter
    def batch_idx(self, val):
        """Set the current batch index"""
        if hasattr(self.model, 'batch_idx'):
            self.model.batch_idx = val
        elif val > 0:
            raise ValueError("No attr batch_idx and requested idx > 0")

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
            self.batch_idx = idx
        target = self.target[self.batch_idx]
        inp = None if self.start_inp is None else self.start_inp[self.batch_idx]

        return target, inp

    def forward_chisq(self, idx=None, main_params=None, sum_chisq=True, **kwargs):
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
        main_params : tensor, optional
            If passed, use this main_params instead of self.main_params,
            if self.set_main_params() has been run.
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

        # get batch data and icov before forward pass (asynchronous load)
        data = target.get_data()

        if hasattr(target, 'icov'):
            icov = target.get_icov()
            cov_axis = target.cov_axis
        else:
            icov = None
            cov_axis = None

        # if batch_idx == 0, clear prior cache
        if self.batch_idx == 0:
            self.clear_prior_cache()

        # copy over main_params if needed
        main_params = main_params if main_params is not None else self.main_params
        if main_params is not None:
            self.send_main_params(main_params=main_params)

        # forward pass model
        prediction = self.model(inp, prior_cache=self.prior_cache)

        if isinstance(prediction, (VisData, MapData, TensorData)):
            prediction = prediction.data

        if not utils.check_devices(prediction.device, self.device):
            prediction = prediction.to(self.device)

        # compute residual
        res = prediction - data

        # evaluate chi square
        chisq = apply_icov(res, icov, cov_axis)
        if sum_chisq:
            chisq = torch.sum(chisq)
        if torch.is_complex(chisq):
            chisq = chisq.real

        return chisq, res

    def forward_like(self, idx=None, main_params=None, **kwargs):
        """
        Compute log (Gaussian) likelihood
        by evaluating the chisquare of the forward model
        compared to the target data

        Parameters
        ----------
        idx : int, optional
            The minibatch index to run, if self.prob
            is batched. Default is to run self.batch_idx.
            If passed also sets self.batch_idx.
        main_params : tensor, optional
            If passed, use this main_params instead of self.main_params,
            if self.set_main_params() has been run.
        """
        # evaluate chisq for this batch index
        chisq, res = self.forward_chisq(idx, main_params=main_params)

        # get target and inp data for this batch index
        target, inp = self.get_batch_data()

        # use it to get log likelihood normalization
        if hasattr(target, 'icov') and target.icov is not None:
            if self.complex_circular:
                # L(z) = exp(-z^H Cz^-1 z) / (pi^n det(Cz))
                like_norm = target.cov_ndim * torch.log(torch.tensor(np.pi)) + target.cov_logdet
            else:
                # L(x) = exp(-.5 x^H Cx^-1 x) / ((2pi)^n det(Cx)^1/2)
                like_norm = 0.5 * (target.cov_ndim * torch.log(torch.tensor(2*np.pi)) + target.cov_logdet)
        else:
            like_norm = 0

        # form loglikelihood
        if self.complex_circular:
            loglike = -chisq - like_norm
        else:
            loglike = -0.5 * chisq - like_norm

        if self.negate:
            return -loglike
        else:
            return loglike

    def forward_prior(self, idx=None, main_params=None, **kwargs):
        """
        Compute log prior given state of params tensors.
        Note that we assume that the prior is the same
        for all batch indices, so we only evaluate and return
        it when batch_idx == 0 to avoid double counting it.

        Parameters
        ----------
        idx : int, optional
            The minibatch index to run, if self.prob
            is batched. Default is to run self.batch_idx.
            If passed also sets self.batch_idx.
        main_params : tensor, optional
            If passed, use this main_params instead of self.main_params,
            if self.set_main_params() has been run.
        """
        if idx is not None:
            self.batch_idx = idx

        main_params = main_params if main_params is not None else self.main_params
        # send over main_params if compute = 'prior'
        if self.compute == 'prior' and main_params is not None:
            self.send_main_params(main_params=main_params)

        # clear existing prior if batch_idx == 0 and compute == 'prior'
        if self.compute == 'prior' and self.batch_idx == 0:
            self.clear_prior_cache()

        # evaluate log prior
        logprior = torch.zeros(1, device=self.device)
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
                logprior += self.prior_cache[k]

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
            Note: prior is only evaluated at batch_idx = 0.
        """
        assert self.compute in ['post', 'like', 'prior']
        if idx is not None:
            self.batch_idx = idx

        prob = torch.zeros(1, device=self.device)

        # evaluate and add likelihood
        if self.compute in ['post', 'like']:
            prob = self.forward_like(**kwargs)

        # evalute and add prior
        if self.compute in ['post', 'prior'] and self.batch_idx == 0:
            # only evaluate prior for batch_idx=0 to avoid double counting
            if self.compute == 'prior':
                # if compute is prior, clear any attached graph tensors
                # in all modules such they are regenerated during this
                # forward call (e.g. beam.R.beam_cache).
                # if compute == 'post' then this has already been done
                for mod in self.modules():
                    mod.clear_graph_tensors()

            # evaluate prior
            pr = self.forward_prior(**kwargs)
            prob = pr if prob is None else prob + pr

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
        return self.forward(idx=idx, **kwargs)

    def closure(self, **kwargs):
        """
        Function for evaluating the model, performing
        backprop, and returning output given self.grad_type.
        kwargs are to out.backward(**kwargs)
        """
        self.closure_eval += 1
        if torch.is_grad_enabled():
            self.zero_grad()

        # if accumulating, run all minibatches and backprop
        if self.grad_type == 'accumulate':
            loss = 0
            for i in range(self.Nbatch):
                self.batch_idx = i
                out = self()
                if out.requires_grad:
                    out.backward(**kwargs)
                loss = loss + out.detach()
            loss = loss / self.Nbatch
            self.batch_idx = 0

        # if stochastic, just run current batch, then backprop
        elif self.grad_type == 'stochastic':
            out = self()
            if out.requires_grad:
                out.backward(**kwargs)
            loss = out.detach()

        # modify gradients if desired
        self.grad_modify()

        # clear prior cache
        self.clear_prior_cache()

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
        if self.start_inp is not None:
            for d in self.start_inp.data:
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

    def clear_graph_tensors(self):
        """
        Clear non-leaf tensors with requires_grad (i.e. ones that
        may be stale), specifically the end-result of send_main_params()
        """
        if self._main_names is not None:
            for name, pname in self._main_names.items():
                self.model[pname] = self.model[pname].detach()


class DistributedLogProb(utils.Module):
    """
    A distributed set of individual LogProb objects working
    in a data-parallel manner (i.e. assuming they are on separate
    devices but have the same model structure). Each LogProb must have
    a main_params tensor defined and have identical downsteam
    model structure, with the only difference being the dataset
    they are predicting and comparing against.

    Note: this assumes all devices are on the same
    node (i.e. devices can pass information to each other directly).
    For multi-node distributed training see pytorch DDP.

    Note: assumes GPU parallelism is across data not parameters,
    thus when evaluating the prior, we only compute it
    for one GPU (the zeroth), and the other GPUs we
    only compute the likelihood
    """
    def __init__(self, probs, device=None, check=True):
        """
        Parameters
        ----------
        probs : list of LogProb
            Holding each LogProb model to evaluate in parallel
        device : str, optional
            Device to hold the master main_params tensor.
            Default is the device of the first LogProb in probs
        """
        super().__init__()
        self.probs = probs
        if check:
            self.check()
        self.device = device if device is not None else probs[0].device
        self.collect_main_params()

        # make sure we aren't double counting the prior
        if self.probs[0].compute == 'post':
            for prob in self.probs[1:]:
                prob.compute = 'like'

    def check(self):
        """
        Run basic checks on LogProb objs
        """
        for i, prob in enumerate(self.probs):
            assert isinstance(prob, LogProb)
            assert hasattr(prob, 'main_params')
            assert prob.main_params.is_leaf
            if i == 0:
                main_params = prob.main_params
                Nbatch = prob.Nbatch
            else:
                assert main_params.shape == prob.main_params.shape
                assert Nbatch == prob.Nbatch

    @property
    def devices(self):
        return [prob.device for prob in self.probs]

    def set_main_params(self, *args, **kwargs):
        """
        Set main_params on sub-prob objects
        and set main_params on this object on self.device
        """
        for prob in self.probs:
            prob.set_main_params(*args, **kwargs)

        self.collect_main_params()

    def sort_main_params(self, new_main_indices, incomplete=False):
        """
        Resort elements of main_params, see LogProb.sort_main_params()
        for details
        """
        self.main_params = None
        self._main_indices = None
        self._main_index = None
        self._main_names = None
        for prob in self.probs:
            prob.sort_main_params(new_main_indices, incomplete=incomplete)

        self.collect_main_params()

    def collect_main_params(self, **kwargs):
        """
        Shallow wrapper around prob.collect_main_params(),
        and then collect onto self.main_params
        """
        self.main_params = None
        for prob in self.probs:
            prob.collect_main_params()
        self._main_indices = copy.deepcopy(self.probs[0]._main_indices)
        self._main_index = self.probs[0]._main_index
        self._main_names = self.probs[0]._main_names
        if self.probs[0].main_params is not None:
            self.main_params = torch.nn.Parameter(self.probs[0].main_params.data.to(self.device))
            
            for k, v in self._main_indices.items():
                if isinstance(v, list):
                    self._main_indices[k] = [utils._idx2ten(_v, self.device) for _v in v]
                else:
                    self._main_indices[k] = utils._idx2ten(v, self.device)

    def send_main_params(self, main_params=None, send_probs=False, **kwargs):
        """
        Copy self.main_params to each object in self.prob.
        Note this does not also call self.probs[i].send_main_params().
        Note this does not preserve graph from self.main_params, each
        self.probs[i].main_params is itself a torch.nn.Parameter object

        Parameters
        ----------
        main_params : tensor, optional
            If provided, overwrite self.main_params with
            this tensor.
        send_probs : bool, optional
            If True, also call self.probs[i].send_main_params().
            Default is False because this is already done
            in the normal forward() call.
        """
        ### TODO: instead of copying Parameter to probs,
        ### just keep the graph anchored on gpu0 and send as
        ### main_params.to(gpu2), and therefore don't need
        ### to "collect_gradients" b/c its done automatically
        ### (will this improve weak scaling?)
        ### (will this appreciably increase VRAM on gpu0?)
        with torch.no_grad():
            if main_params is not None:
                self.main_params = torch.nn.Parameter(main_params)
            for prob, device in zip(self.probs, self.devices):
                prob.main_params = torch.nn.Parameter(self.main_params.to(device))
                if send_probs:
                    prob.send_main_params()

    def get_main_params(self, add_p0=False):
        """
        Get a copy of self.main_params and, if available and
        if add_p0, add main_p0 from the first object in self.probs
        """
        main_params = None
        if hasattr(self, 'main_params') and self.main_params is not None:
            main_params = self.main_params.data.clone()
            prob = self.probs[0]
            if add_p0 and prob.main_p0 is not None:
                main_params += prob.main_p0.clone().to(self.device)

        return main_params

    def closure(self, **kwargs):
        """
        Function for evaluating the model, performing
        backprop, and returning output given self.grad_type
        for each LogProb in self.probs
        """
        if torch.is_grad_enabled():
            self.zero_grad()

        # send over main_params to self.probs
        self.send_main_params()

        # evaluate closures in parallel
        loss = []
        for i, prob in enumerate(self.probs):
            if self.compute == 'prior' and i > 0: continue  # don't double count prior
            loss.append(prob.closure(**kwargs))

        # collect gradients if grad enabled
        if torch.is_grad_enabled():
            for i, prob in enumerate(self.probs):
                if self.compute == 'prior' and i > 0: continue  # don't double count prior
                if i == 0:
                    self.main_params.grad = prob.main_params.grad.to(self.device)
                else:
                    self.main_params.grad += prob.main_params.grad.to(self.device)

        return sum([l.to(self.device) for l in loss])

    @property
    def closure_eval(self):
        return self.probs[0].closure_eval

    @property
    def Nbatch(self):
        """get total number of batches in model"""
        return self.probs[0].Nbatch

    @property
    def batch_idx(self):
        """return current batch index in model"""
        return self.probs[0].batch_idx

    @batch_idx.setter
    def batch_idx(self, val):
        """Set the current batch index"""
        for prob in self.probs:
            prob.batch_idx = val

    @property
    def compute(self):
        return self.probs[0].compute

    @compute.setter
    def compute(self, val):
        for i, prob in enumerate(self.probs):
            if val == 'post' and i > 0:
                val = 'like'
            prob.compute = val

    @property
    def negate(self):
        return self.probs[0].negate

    @negate.setter
    def negate(self, val):
        for prob in self.probs:
            prob.negate = val

    @property
    def grad_type(self):
        return self.probs[0].grad_type

    def push(self, device):
        """
        Push main_params to a new device
        """
        self.main_params = utils.push(self.main_params, device)
        self.device = device

    def clear_graph_tensors(self):
        """
        Clear non-leaf tensors with requires_grad on all this object
        and all LogProb sub-modules (i.e. graph tensors that are stale after
        a backward pass)
        """
        for prob in self.probs:
            prob.clear_graph_tensors()
            for mod in prob.modules():
                mod.clear_graph_tensors()


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
                for _, k in self.prob._main_names.items():
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
        icov is shape (Npol, Npol, Ntimes, Nfreqs, Nbl, Nbl)
    cov_axis : 'time'
        icov is shape (Npol, Npol, Nbls, Nfreqs, Ntimes, Ntimes)
    cov_axis : 'freq'
        mode = 'vis'
            icov is shape (Npol, Npol, Nbls, Ntimes, Nfreqs, Nfreqs)
        mode = 'map'
            icov is shape (Npol, 1, Npix, Nfreqs, Nfreqs)
    cov_axis : 'pix'
        icov is shape (Npol, 1, Nfreqs, Npix, Npix)
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
        out = torch.einsum("ijklm,ijlmkn,ijnlm->ijlm", d.conj(), icov, d)
    elif cov_axis == 'time':
        # icov is along times
        out = torch.einsum("ijklm,ijkln,ijknm->ijkm", d.conj(), icov, d)
    elif cov_axis == 'freq':
        # icov is along freqs
        if mode == 'vis':
            out = torch.einsum("ijklm,ijklmn,ijkln->ijkl", d.conj(), icov, d)
        elif mode == 'map':
            out = torch.einsum("ijkl,ijlkn,ijnl->ijl", d.conj(), icov, d)
    elif cov_axis == 'pix':
        # icov is along Npix of map
        out = torch.einsum("ijkl,ijkln,ijkn->ijk", d.conj(), icov, d)

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
        'bl'   : cov is (Npol, Npol, Ntimes, Nfreqs, Nbl, Nbl)
        'time' : cov is (Npol, Npol, Nbls, Nfreqs, Ntimes, Ntimes)
        'freq' :
            mode = 'vis' : cov is (Npol, Npol, Nbls, Ntimes, Nfreqs, Nfreqs)
            mode = 'map' : cov is (Npol, 1, Npix, Nfreqs, Nfreqs)
        'pix'  : cov is (Npol, 1, Nfreqs, Npix, Npix)
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
    diag = cov.diagonal(dim1=-2, dim2=-1)
    if cov_axis == 'full':
        return diag.reshape(shape)
    elif cov_axis == 'bl':
        return diag.moveaxis(-1, 2)
    elif cov_axis == 'time':
        return diag.moveaxis(-1, 3)
    elif cov_axis == 'freq':
        if mode == 'vis':
            return diag
        elif mode == 'map':
            return diag.moveaxis(-1, 2)
    elif cov_axis == 'pix':
        return diag
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
        icov = 1 / cov.clip(1e-40)
    elif cov_axis == 'full':
        # invert full covariance
        icov = linalg.invert_matrix(cov, inv=inv, hermitian=True, **kwargs)

    return icov


def compute_hessian(prob, params, Nstart=None, Nrows=None, vectorize=False,
                    rm_offdiag=False, grad_real=True, cast2real=False, out_ftype=None):
    """
    Compute the hessian for a LogProb object. Note this
    purposely avoids torch.autograd.functional.hessian
    because it doesn't seem to parallelize well with
    how DistributedLogProb handles multi-GPU case.

    Returns a ParamDict object or a list of such
    with str keys and tensor values.

    Parameters
    ----------
    prob : LogProb, DistributedLogProb, list
        LogProb forward model for computing hessian
        with respect to. Can be a list of independent
        LogProb objects on different devices
    params : str or list
        Name of parameters with prob as root
        (e.g. model.rime.beam.params or main_params)
        to compute hessian for. Can also be a list of
        parameter names.
    Nstart : ParamDict, int, optional
        Starting row of the hessian to compute, default is zeroth row.
    Nrows : ParamDict, int, optional
        Number of rows of the hessian to compute, default is all rows.
    vectorize : bool, optional
        Not implemented yet. See torch.autograd.grad()
    rm_offdiag : bool, optional
        Get rid of off diagonal components when computing
        the hessian. Note that due to autograd limitations,
        we still have to compute the off diagonal, but
        we just truncate them afterwards to save space.
    grad_real : bool, optional
        If True and if params is complex-valued, only compute
        the hessian from the real component of the gradient.
        Otherwise, compute the hessian from the imaginary
        component of the gradient.
    cast2real : bool, optional
        If True and if the params is complex, only store the
        real component of the hessian matrix. If grad_real == False,
        then only store the imaginary component. Otherwise, return
        the complex hessian as-is.
    out_ftype : dtype object, optional
        Cast hessian from default float type to this float type
        (e.g. float64 -> float32). Default is to keep default type.
        Note: to convert from complex->float use cast2real=True.

    Returns
    -------
    ParamDict or list
    """
    # type check prob
    if isinstance(prob, LogProb):
        probs = [prob]
    elif isinstance(prob, DistributedLogProb):
        probs = prob.probs
    elif isinstance(prob, list):
        probs = prob
    else:
        raise ValueError

    # check if computing prior, in which case only keep one prob
    if probs[0].compute == 'prior':
        probs = probs[:1]

    # type check params
    if isinstance(params, (str, np.str_)):
        params = [params]

    # type check Nrows
    if isinstance(Nrows, (int, np.integer)):
        Nrows = {k: Nrows for k in params}
    if isinstance(Nstart, (int, np.integer)):
        Nstart = {k: Nstart for k in params}

    # generate list of pdict for each LogProb
    pdict = [paramdict.ParamDict({p: prob[p] for p in params}) for prob in probs]

    ## TODO: see torch.autograd.functional.jacobian for how to implement
    ## vectorize=True using autograd.grad(..., is_grad_batched=True)
    if vectorize:
        raise NotImplementedError

    # setup placeholders for hessians
    hess = [paramdict.ParamDict({}) for j in range(len(probs))]

    # get max Nbatch across all probs
    Nbatch = max([prob.Nbatch for prob in probs])

    # iterate over params (move up to innermost loop? will this increase memory too much?)
    for param in params:
        # get shape of the output hessian (not necessarily square)
        ntot = pdict[0][param].numel()
        nstart = Nstart[param] if Nstart is not None else 0
        nend = Nrows[param] + nstart if Nrows is not None else ntot
        nend = min([nend, ntot])
        nrows = nend - nstart

        # get dtype of the hessian
        if out_ftype is None:
            dtype = pdict[0][param].dtype
            if torch.is_complex(pdict[0][param]) and cast2real:
                dtype = utils._cfloat2float[dtype]
        else:
            dtype = out_ftype

        # initialize empty hessian for this param and every prob object
        for j, prob in enumerate(probs):
            if rm_offdiag:
                hess[j][param] = torch.zeros(nrows, device=prob.device, dtype=dtype)
            else:
                hess[j][param] = torch.zeros(nrows, ntot, device=prob.device, dtype=dtype)

        # iterate over Nbatch
        for i in range(Nbatch):

            # iterate over LogProb objects and get gradients of output
            grads = {j: None for j in range(len(probs))}
            for j, prob in enumerate(probs):
                # perform forward pass for each LogProb if this batch_idx exists
                if i < prob.Nbatch:
                    prob.batch_idx = i
                    out = prob()

                    # get flat gradient
                    grads[j] = torch.autograd.grad(out, pdict[j][param], create_graph=True, allow_unused=True)[0].flatten()

            # compute nrows of the hessian
            # note: this is where vectorize=True would go...
            # note: this particular FOR loop ordering (first n, then probs) is what makes
            # this parallelize well over multiple probs, not sure why...
            z = 0
            for k in range(nstart, nend):
                for j, prob in enumerate(probs):
                    # skip if no gradient computed
                    if grads[j] is None: continue

                    # compute a row of the hessian
                    if torch.is_complex(pdict[j][param]) and not grad_real:
                        g = grads[j][k].imag

                    else:
                        g = grads[j][k].real

                    # compute hessian row
                    h = torch.autograd.grad(g, pdict[j][param], retain_graph=k < nend - 1, allow_unused=True)[0].flatten()

                    if rm_offdiag:
                        h = h[k]

                    if cast2real:
                        if torch.is_complex(pdict[j][param]) and not grad_real:
                            h = h.imag
                        else:
                            h = h.real

                    if out_ftype is not None:
                        h = h.to(out_ftype)

                    # add this row to the hessian
                    hess[j][param][z] += h

                z += 1

    # reshape if rm_offdiag
    if rm_offdiag and Nstart is None and Nrows is None:
        for j, h in enumerate(hess):
            for param in h:
                h[param] = h[param].reshape(pdict[j][param].shape)

    return hess


def invert_hessian(hess, inv='pinv', diag=False, idx=None, rm_thresh=None, rm_fill=1e-15,
                   rm_offdiag=False, rcond=1e-15, eps=None, hermitian=True):
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
        Default is not removal.
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


    rm_thresh = rm_thresh if rm_thresh is not None else -1e50
    if diag:
        # assume hessian holds diagonal, can be any shape
        cov = torch.ones_like(hess, device=hess.device, dtype=hess.dtype)
        s = hess > rm_thresh
        cov[s] = 1 / hess[s]
        cov[~s] = rm_fill

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
            idx = utils._slice2tensor(idx)
        else:
            idx = utils._idx2tensor(idx)

        # combine idx with rm_thresh
        good_idx = np.where(H.diagonal() > rm_thresh)[0]
        idx = np.array([i for i in idx if i in good_idx])

        # select out indices
        H = H[idx[:, None], idx[None, :]]

        # add eps if desired, do it not inplace here, as oppossed to in invert_matrix
        if eps is not None:
            H = H + eps * torch.eye(len(H), dtype=H.dtype, device=H.device)

        # take inverse to get cov
        C = linalg.invert_matrix(H, inv=inv, rcond=rcond, hermitian=hermitian)

        # fill cov with shape of hess
        cov = torch.eye(len(hess), device=hess.device, dtype=hess.dtype) * rm_fill
        cov[idx[:, None], idx[None, :]] = C

        return cov


def mask_hessian(hess, thresh=1e0):
    """
    Given a hessian (or any square matrix), remove
    rows and columns that have a diagonal value less than thresh.

    Parameters
    ----------
    hess : tensor
        Matrix to mask
    thresh : float, optional
        Threshold for diagonal values, below which
        the rows/cols are truncated

    Returns
    -------
    masked_hess : tensor
        New masked tensor
    mask : tensor
        True where unmasked, False where masked
    """
    mask = hess.diagonal() >= thresh
    masked_hess = hess[mask, :][:, mask]

    return masked_hess, mask


def unmask_hessian(hess, mask, val=1e0, maskleft=False):
    """
    Given a masked hessian and the mask, reconstruct
    the initial matrix with identity rows/cols along
    the masked regions, with diagonal = val.
    Output is not a view of input.

    Parameters
    ----------
    hess : tensor
        Masked matrix
    mask : tensor
        Boolean mask, True where unmasked, see mask_hessian()
    val : float, optional
        Value to insert along diagonal of masked rows/cols
    maskleft : bool, optional
        If True, only unmask the left side of hess. Default
        is to unmask both sides. If True, val is not used.

    Returns
    -------
    hess : tensor
    """
    # get indexing vector
    mask = torch.as_tensor(mask)
    idx = torch.arange(len(mask))
    for i in torch.where(~mask)[0]:
        idx[i] = -1
        idx[i+1:] -= 1
    idx += 1

    # zeropad
    I = torch.zeros(hess.shape[0], 1, device=hess.device, dtype=hess.dtype)
    hess = torch.cat([I, hess], dim=1)
    I = torch.zeros(1, hess.shape[1], device=hess.device, dtype=hess.dtype)
    hess = torch.cat([I, hess], dim=0)

    # index reshape
    if maskleft:
        hess = hess[idx]
    else:
        hess = hess[idx, :][:, idx]

        # insert val along masked diagonal
        idx = torch.where(~mask)[0]
        hess[idx, idx] = val
    
    return hess


def main_params_index(prob, param, sub_index=None, params=None):
    """
    Take a LogProb object and its main_index dictionary
    and return an indexing of a subset of its prob.main_params.

    Parameters
    ----------
    prob : LobProb object
        A LogProb with set_main_params() activated
    param : str
        The param name of prob._main_index to sub-index
    sub_index : tuple, optional
        Given p = prob[param][idx] where idx is
        prob._main_index[param], select a further subset
        of p[sub_index] if provided, otherwise use the full
        shape of p. Note if idx is a list, then sub_index
        must also be a list. Default is to use the full
        parameter size.
    params : list, optional
        List of parameter names to iterate over when creating
        the indexing tensor. Default is to iterate over all
        in the order of prob._main_index. If e.g. you want to
        get indexing tensors for a subset of these, or in
        a different order, then use this kwarg.

    Returns
    -------
    tensor
    """
    # assert has _main_index
    assert prob._main_index is not None
    assert prob._main_shapes is not None
    assert param in prob._main_index
    params = params if params is not None else list(prob._main_index.keys())

    def select(prob, param, main_index, sub_index):
        pname = prob._main_names[param]
        p = prob.model[pname]
        if main_index is not None:
            p = p[main_index]

        idx = torch.arange(p.numel(), device=prob.device)

        if sub_index is not None:
            idx = idx.reshape(p.shape)[sub_index].ravel()

        return idx

    # iterate over all sub-parameters
    start = 0
    for k in params:
        v = prob._main_index[k]
        if k == param:
            # we are selecting this parameter
            if isinstance(v, list):
                idx = []
                for i in range(len(v)):
                    idx.extend(select(prob, param, v[i], sub_index[i] if sub_index is not None else None))
                idx = torch.as_tensor(idx, device=prob.device)

            else:
                idx = select(prob, param, v, sub_index)

            idx += start

            # don't need to iterate through main_index any further
            break

        else:
            # not selecting this parameter, so
            # push start forward by its size
            if isinstance(v, list):
                for i in range(len(v)):
                    start += prob._main_shapes[k][i].numel()
            else:
                start += prob._main_shapes[k].numel()

    return idx


def main_params_kron_inv_hess(prob, hess, param, method='chol', **inv_kwargs):
    """
    Compute the Kronecker factorization of the inverse hessian
    matrix for a given sub-parameter in main_params

    Parameters
    ----------
    prob : LogProb object
        with main_params set
    hess : tensor
        A hessian matrix of shape
        (main_params, main_params)
    param : str
        A parameter string in prob._main_index
    method : str, optional
        ['chol', 'svd'] how to take the decomposition.
        'chol' : take inverse of hess sub-block, then cholesky
        'svd'  : take svd of hess sub-block, use u @ 1/s^.5
    inv_kwargs : dict, optional
        kwargs for taking inv when method = 'chol'

    Returns
    -------
    tensor
    """
    assert param in prob._main_index

    # down select hess to this param
    hidx = prob._main_indices[param]
    hess = hess[hidx, :][:, hidx]

    # now get param indexing
    idx = prob._main_index[param]
    shape = prob.model[param][idx].shape
    N = np.prod(shape[1:])

    if method == 'chol':
        cov = linalg.invert_matrix(hess[:N, :N], **inv_kwargs)
        L = torch.linalg.cholesky(cov)
    elif method == 'svd':
        u, s, v = torch.linalg.svd(hess[:N, :N])
        L = u @ torch.diag(1 / s**.5)

    L = L.sum(1) / L.shape[1]**.5

    kron = torch.kron(torch.eye(shape[0]), L).T

    return kron
