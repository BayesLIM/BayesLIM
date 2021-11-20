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

from . import utils, paramdict
from .dataset import VisData, MapData, TensorData


class LogUniformPrior:
    """
    log uniform prior
    """
    def __init__(self, lower_bound, upper_bound, index=None):
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
        """
        self.lower_bound = torch.as_tensor(lower_bound)
        self.upper_bound = torch.as_tensor(upper_bound)
        self.norm = torch.sum(torch.log(1/(self.upper_bound - self.lower_bound)))
        self.index = index

    def forward(self, params):
        """
        Evalute uniform logprior on a parameter tensor.

        Parameters
        ----------
        params : tensor
            Parameter tensor to evaluate prior on.
        """
        # index input params tensor if necessary
        if self.index is not None:
            params = params[self.index]

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

    def __call__(self, params):
        return self.forward(params)


class LogTaperedUniformPrior:
    """
    Log of an edge-tapered uniform prior, constructed by
    multiplying two mirrored tanh functions and taking log.

    .. math ::
        
        c &= \alpha/(b_u - b_l) \\
        P &= \tanh(c(x-b_l))\cdot\tanh(-c(x-b_u))

    """
    def __init__(self, lower_bound, upper_bound, kind='sigmoid',
                 alpha=100., index=None):
        """
        Parameters
        ---------
        lower_bound, upper_bound : tensors
            Tensors holding the min and max of the allowed
            parameter value. Tensors should broadcast with
            the (indexed) params input
        kind : str, optional
            Kind of taper to apply, either ['sigmoid', 'tanh'].
            Sigmoid is defined on all R and thus does not enacting strict
            lower and upper bounds, but is well-defined and differentiable
            over all R, whereas tanh has a hard cutoff at lower and upper bounds.
        alpha : tensor, optional
            A scaling coefficient that determines the amount of tapering.
            The scaling coefficient is determined as alpha / (upper - lower)
            where good values for alpha are 1 < alpha < 1000
            alpha -> 1000, less edge taper (like a tophat)
            alpha -> 1, more edge taper (like an inverted quadratic)
        index : slice or tuple of slice objects
            indexing of params tensor before computing prior.
            default is no indexing.
        """
        self.lower_bound = torch.as_tensor(lower_bound)
        self.upper_bound = torch.as_tensor(upper_bound)
        self.alpha = torch.as_tensor(alpha)
        self.index = index
        self.dbound = self.upper_bound - self.lower_bound
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
        if self.index is not None:
            params = params[self.index]

        if self.kind == 'sigmoid':
            func = torch.sigmoid
        elif self.kind == 'tanh':
            func = torch.tanh

        prob = func(self.coeff * (params - self.lower_bound)) \
                          * func(-self.coeff * (params - self.upper_bound))
        return torch.sum(torch.log(prob))

    def __call__(self, params):
        return self.forward(params)


class LogGaussPrior:
    """
    log Gaussian prior
    """
    def __init__(self, mean, cov, sparse_cov=True, index=None):
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
        index : slice or tuple of slice objects
            indexing of params tensor before computing prior.
            default is no indexing.
        """
        self.mean = torch.as_tensor(mean)
        self.cov = torch.as_tensor(cov)
        self.sparse_cov = sparse_cov
        self.index = index
        if self.sparse_cov:
            self.icov = 1 / self.cov
            self.logdet = torch.sum(torch.log(self.cov))
            self.ndim = sum(self.cov.shape)
        else:
            self.icov = torch.linalg.pinv(self.cov)
            self.logdet = torch.slogdet(self.cov)
            self.ndim = len(self.cov)
        self.norm = 0.5 * (self.ndim * torch.log(torch.tensor(2*np.pi)) + self.logdet)

    def forward(self, params):
        """
        Evaluate log Gaussian prior

        Parameters
        ----------
        params : tensor
            Parameter tensor
        """
        if self.index is not None:
            params = params[self.index]
        res = params - self.mean
        if self.sparse_cov:
            chisq = torch.sum(res**2 * self.icov)
        else:
            res = res.ravel()
            chisq = torch.sum(res @ self.icov @ res)

        return -0.5 * chisq - self.norm

    def __call__(self, params):
        return self.forward(params)


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
        self.named_params = [k[0] for k in self.named_parameters()]
        self.prior_dict = prior_dict
        self.compute = compute
        self.negate = negate
        self.closure_eval = 0
        self.grad_type = grad_type

        # clear prior cache
        self.clear_prior_cache()

        # set no main params by default
        self.set_main_params()

    def set_main_params(self, model_params=None):
        """
        Setup a single main parameter tensor that automatically
        interfaces with specified submodule tensors in models. This
        is used for optimizers that require all parameters on a single
        device. Sets self.main_params.

        For any call to self, the values from self.main_params are
        copied over to submodules before forward model evaluation.

        Note: assumes all params are of utils._float() dtype.

        Parameters
        ----------
        model_params : list, optional
            List of submodules params tensors (with model as root) to
            collect sort into main_params.
            E.g. ['rime.sky.params', 'cal.params']
            If None, main_params is set as None.
        """
        self.main_params = None
        self._main_indices = None
        self._main_shapes = None
        self._main_devices = None
        if model_params is not None:
            N = 0
            params = []
            self._main_indices = {}
            self._main_shapes = {}
            self._main_devices = {}
            for param in model_params:
                # get parameter and metadata
                p = self.model[param].detach().to('cpu').to(utils._float())
                shape = p.shape
                numel = shape.numel()
                device = p.device

                # append metadata
                params.append(p.ravel())
                self._main_indices[param] = slice(N, N + numel) 
                self._main_shapes[param] = shape
                self._main_devices[param] = device
                N += numel

            self.main_params = torch.nn.Parameter(torch.cat(params))
            self.send_main_params()

    def send_main_params(self):
        """
        Take existing value of self.main_params and using
        _main_indices, _main_shapes and _main_types, and
        send its values to the relevant submodule params.
        """
        if self.main_params is not None:
            for param in self._main_indices:
                # get shaped parameter on device
                value = self.main_params[self._main_indices[param]]
                value = value.reshape(self._main_shapes[param])
                value = value.to(self._main_devices[param])
                utils.set_model_attr(self.model, param, value,
                                     clobber_param=True, no_grad=False)

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

    def forward_chisq(self, idx=None):
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
        if hasattr(target, 'icov') and target.icov is not None:
            icov = target.icov
            cov_axis = target.cov_axis
        else:
            icov = torch.ones_like(res, device=res.device)
            cov_axis = None

        # evaluate chi square
        chisq = torch.sum(apply_icov(res, icov, cov_axis))
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
        prob = torch.as_tensor(0.0)

        # evaluate and add likelihood
        if self.compute in ['post', 'like']:
            prob = prob + self.forward_like(idx, **kwargs)

        # evalute and add prior
        if self.compute in ['post', 'prior']:
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
            return loss / self.Nbatch

        # if stochastic, just run current batch, then backprop
        elif self.grad_type == 'stochastic':
            out = self()
            if out.requires_grad:
                out.backward()
            return out.detach()

    def push(self, device):
        """
        Transfer target data to device
        """
        self.device = device
        for d in self.target.data:
            d.push(device)

    def clear_prior_cache(self):
        """
        Clear the self.prior_cache dictionary
        """
        if hasattr(self, 'prior_cache'):
            del self.prior_cache
        self.prior_cache = {}

    def set_icov(self, icov):
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
    def __init__(self, prob, opt, track=False):
        """
        Parameters
        ----------
        prob : LogProb object
        opt : torch.Optimizer object
        track : bool, optional
            If True, track value of all prob.parameters and append to
            a chain during optimization.
        """
        self.prob = prob
        self.opt = opt
        self.loss = []
        self.closure_eval = 0
        self.track = track

        if prob.grad_type == 'accumulate':
            self.Nbatch = 1
        elif prob.grad_type == 'stochastic':
            self.Nbatch = prob.Nbatch

        self.chain = {}
        if prob.main_params is not None:
            for k in prob._main_indices:
                self.chain['model.' + k] = []
        else:
            for k in prob.named_parameters():
                self.chain[k[0]] = []

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
        start = time.time()

        for epoch in range(Nepochs):
            if Nreport is not None:
                if (epoch > 0) and (epoch % Nreport == 0):
                    print("epoch {}, {:.1f} sec".format(epoch, time.time() - start))

            # zero grads
            self.opt.zero_grad()

            # iterate over minibatches
            _loss = 0
            for i in range(self.Nbatch):
                # append current params
                if self.track:
                    for k in self.chain:
                        self.chain[k].append(self.prob[k].detach().clone())

                # evaluate forward model, backprop, make a step
                _loss += self.opt.step(self.prob.closure)

            # append batch-averaged loss
            self.loss.append(_loss / self.Nbatch) 

        time_elapsed = time.time() - start
        info = dict(duration=time_elapsed)

        return info

    def get_chain(self, name=None):
        """
        Extract and return chain history

        Parameters
        ----------
        name : str, optional
            Return just one param.

        Returns
        -------
        tensor or dict
            Chain for given name
        """
        if name is not None:
            return torch.stack(self.chain[name])
        else:
            return {k: torch.stack(c) for k, c in self.chain.items()}


def apply_icov(data, icov, cov_axis):
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
        [None, 'bl', 'time', 'freq', 'full']
        See Notes

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
    (Npol, Npol, Nbls, Ntimes, Nfreqs). See VisData for
    more details.
    cov_axis : 'bl'
        icov is shape (Nbl, Nbl, Npol, Npol, Ntimes, Nfreqs)
    cov_axis : 'time'
        icov is shape (Ntimes, Ntimes, Npol, Npol, Nbls, Nfreqs)
    cov_axis : 'freq'
        icov is shape (Nfreqs, Nfreqs, Npol, Npol, Nbls, Ntimes)
    """
    if cov_axis is None:
        # icov is just diagonal
        out = data.conj() * data * icov
    elif cov_axis == 'full':
        # icov is full inv cov
        out = data.ravel().conj() @ icov @ data.ravel()
    elif cov_axis == 'bl':
        # icov is along bls
        d = data.moveaxis(2, 0)
        out = d.T.conj() @ icov @ d
    elif cov_axis == 'time':
        # icov is along times
        d = data.moveaxis(3, 0)
        out = d.T.conj() @ icov @ d
    elif cov_axis == 'freq':
        # icov is along freqs
        d = data.moveaxis(4, 0)
        out = d.T.conj() @ icov @ d

    return out


def compute_icov(cov, cov_axis, pinv=True, rcond=1e-15):
    """
    Compute the inverse covariance

    Parameters
    ----------
    cov : tensor
        data covariance. See optim.apply_icov() for shapes
    cov_axis : str
        covariance type. See optim.apply_icov() for options
    pinv : bool, optional
        Use pseudo inverse, otherwise use direct inverse
    rcond : float, optional
        rcond kwarg for pinverse

    Returns
    -------
    tensor
    """
    # set inversion function
    inv = lambda x: torch.pinverse(x, rcond=rcond) if pinv else torch.inverse
    if cov_axis is None:
        # this is just diagonal
        icov = 1 / cov
    elif cov_axis == 'full':
        # invert full covariance
        icov = inv(cov)
    else:
        # cov is 6-dim, only invert first two dims
        icov = torch.zeros_like(cov)
        for i in range(cov.shape[2]):
            for j in range(cov.shape[3]):
                for k in range(cov.shape[4]):
                    for l in range(cov.shape[5]):
                        icov[:, :, i, j, k, l] = inv(cov[:, :, i, j, k, l])

    return icov



def compute_hessian(prob, params, keep_diag=False, **kwargs):
    """
    Compute Hessian of prob with respect to params.
    Note that this edits params in prob inplace!

    Parameters
    ----------
    prob : optim.LogProb object
        Log posterior object
    params : ParamDict object
        Holding parameters of prob for which to compute hessian,
        and the values at which to compute it
    keep_diag : bool, optional
        If True, only keep the diagonal of the Hessian and
        reshape to match input params shape.
    **kwargs : kwargs for autograd.functional.hessian

    Returns
    -------
    ParamDict object
        Hessian of prob
    """
    # TODO: enable Hessian between params
    hess = paramdict.ParamDict({})
    for param in params:
        # setup func
        inp = params[param]
        shape = inp.shape
        N = shape.numel()
        def func(x):
            utils.set_model_attr(prob, param, x, clobber_param=True)
            return prob()
        h = torch.autograd.functional.hessian(func, inp, **kwargs).reshape(N, N)
        if keep_diag:
            h = h.diag().reshape(shape)
        hess[param] = h

    return hess

