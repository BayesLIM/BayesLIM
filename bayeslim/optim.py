"""
Optimization module
"""
import numpy as np
import torch
import os
from collections import OrderedDict
import time
import pickle

from . import utils, io


class Sequential(utils.Module):
    """
    A minimal mirror of torch.nn.Sequential without the
    iterators and with added features (inherits from utils.Module)

    Instantiation takes a parameter dictionary as
    input and updates model before evaluation. e.g.

    .. code-block:: python

        S = Sequential(OrderedDict(model1=model1, model2=model2))

    where evaluation order is S(params) -> model2( model1( params ) )

    Note that the keys of the parameter dictionary
    must conform to nn.Module.named_parameters() syntax.
    """
    def __init__(self, models):
        """
        Parameters
        ----------
        models : OrderedDict
            Models to evaluate in sequential order.
        """
        super().__init__()
        # get ordered list of model names
        self._models = list(models)
        # assign models as sub modules
        for name, model in models.items():
            self.add_module(name, model)

    def forward(self, inp=None, pdict=None):
        """
        Evaluate model in sequential order,
        optionally updating all parameters beforehand

        Parameters
        ----------
        inp : tensor or VisData
            optional input to first model
        pdict : ParamDict
            Parameter dictionary with keys
            conforming to nn.Module.get_parameter
            syntax, and values as tensors
        """
        # update parameters of module and sub-modules
        if pdict is not None:
            for k in pdict:
                param = self[k]
                is_parameter = isinstance(param, torch.nn.Parameter)
                self[k] = utils.push(pdict[k], param.device, is_parameter)

        for name in self._models:
            inp = self.get_submodule(name)(inp)
        return inp


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
            chisq = 0.5 * torch.sum(res**2 * self.icov)
        else:
            res = res.ravel()
            chisq = 0.5 * torch.sum(res @ self.icov @ res)

        return -chisq - self.norm

    def __call__(self, params):
        return self.forward(params)


class LogProb(utils.Module):
    """
    The (negative) log posterior density, or the (negative)
    of the log likelihood plus the log prior, which is
    proportional to the log posterior up to a constant.

    This object handles both the likelihood and
    (optionally) the priors

    Negative log Gaussian likelihood

    .. math::

        -log(L) &= \frac{1}{2}(d - \mu)^T \Sigma^{-1} (d - \mu)\\
                &+ \frac{1}{2}\log|\Sigma| + \frac{n}{2}\log(2\pi) 
    """
    def __init__(self, model, target, cov, cov_axis, parameter=False,
                 param_list=None, prior_list=None, device=None,
                 compute='post', negate=True):
        """
        Parameters
        ----------
        model : utils.Module or optim.Sequential subclass
            THe forward model that holds all parameters
            as attributes (i.e. model.named_parameters())
        target : tensor
            Data tensor of shape
            (Npol, Npol, Nbls, Ntimes, Nfreqs)
        cov : tensor
            Covariance of the target data.
            See optim.apply_icov() for shape details.
        cov_axis : str
            This specifies the kind of covariance. See optim.apply_icov()
            for details.
        parameter : bool, optional
            If fed a covariance, this makes the inverse covariance
            a parameter in the fit. (not yet implemented)
        param_list : list
            List of all param names (str) to pull
            from model and to evaluate their prior
        prior_list : list
            List of Prior callables (e.g. LogGaussPrior)
            for each element in param_list. For no prior on a
            parameter, pass element as None. A param name
            str may be repeated, e.g. in case multiple priors
            are used for a single params tensor but with different
            indexing.
        device : str, optional
            Set the device for this object
        compute : str, optional
            Distribution to compute out of ['post', 'like', 'prior'],
            regardless of what is attached to the object. This can be
            used to sample from either the posterior, likelihood, or prior,
            with the same API one would use to sample from the posterior.
        negate : bool, optional
            Return the negative log posterior (for gradient descent minimization).
            Otherwise return the log posterior (for MCMC)
        """
        super().__init__()
        self.model = model
        self.target = target
        if parameter:
            raise NotImplementedError
        self.parameter = parameter
        self.device = device
        self.cov = cov
        self.cov_axis = cov_axis
        self.icov = None
        self.param_list = param_list
        self.prior_list = prior_list
        self.compute = compute
        self.negate = negate

    def set_icov(self, icov=None):
        """
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

    def forward_like(self, inp=None):
        """
        Compute negative log likelihood after a pass through model

        Parameters
        ----------
        inp : tensor, optional
            Starting input to self.model
        """
        ## TODO: allow for kwarg dynamic cov for cosmic variance

        # forward pass model
        out = self.model(inp)
        prediction = out.data.to(self.device)

        # compute residual
        res = prediction - self.target.data

        # evaluate negative log likelihood
        chisq = 0.5 * torch.sum(apply_icov(res, self.icov, self.cov_axis))

        loglike = -chisq - self.like_norm
        if self.negate:
            return -loglike
        else:
            return loglike

    def forward_prior(self):
        """
        Compute negative log prior given
        state of model parameters
        """
        # evaluate negative log prior
        logprior = 0
        if self.param_list is not None and self.prior_list is not None:
            for param, prior in zip(self.param_list, self.prior_list):
                logprior += prior(utils.get_model_attr(self.model, param))

        if self.negate:
            return -logprior
        else:
            return logprior

    def forward(self, inp=None):
        """
        Compute negative log posterior (up to a constant)

        Parameters
        ----------
        inp : tensor, optional
            Starting input to self.model
            for likelihood evaluation
        """
        assert self.compute in ['post', 'like', 'prior']
        prob = 0

        if self.compute in ['post', 'like']:
            prob += self.forward_like(inp)

        if self.compute in ['post', 'prior']:
            prob += self.forward_prior()

        return prob


def train(model, opt, Nepochs=1, loss=[], closure=None, verbose=True):
    """
    Train a Sequential model

    Parameters
    ----------
    model : optim.Sequential object
        A sky / visibility / instrument model that ends
        with evaluation of the negative log-posterior
    opt : torch.nn.optimization object
    Nepochs : int, optional
        Number of epochs to run

    Returns
    -------
    dict : convergence info
    """
    start = time.time()

    # iterate over epochs
    for epoch in range(Nepochs):
        if verbose:
            if epoch % 100 == 0:
                print('Epoch {}/{}'.format(epoch, Nepochs))

        opt.zero_grad()
        out = model()
        out.backward()
        # check for nan gradients
        opt.step(closure)
        loss.append(out.detach().clone())
    time_elapsed = time.time() - start
    info = dict(duraction=time_elapsed, loss=loss)

    return info


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
    For the following, data is of shape
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
        out = torch.abs(data)**2 * icov
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
        data covariance. See optim.apply_cov() for shapes
    cov_axis : str
        covariance type. See optim.apply_cov() for options
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

