"""
Optimization module
"""
import numpy as np
import torch

from . import utils


class LogGaussLikelihood(torch.nn.Module):
    """
    Negative log Gaussian likelihood

    .. math::

        -log(L) &= \frac{1}{2}(d - \mu)^T \Sigma^{-1} (d-\mu)\\
                &+ \frac{1}{2}\log|\Sigma| + \frac{n}{2}\log(2\pi) 
    """
    def __init__(self, target, cov, sparse_cov=True, parameter=False, device=None):
        """
        Parameters
        ----------
        target : tensor
            Data tensor
        cov : tensor
            Covariance of target data.
            Can be an nD tensor with the same
            shape as target, in which case
            this is just the diagonal of the cov mat
            reshaped as the target (sparse_cov=True)
            Can also be a 2D tensor holding the on and
            off diagonal covariances, in which case
            target must be a flat column tensor (sparse_cov=False)
        parameter : bool, optional
            If fed a covariance, this makes the inverse covariance
            a parameter in the fit.
        device : str, optional
            Set the device for this object
        """
        super().__init__()
        self.target = target
        self.parameter = parameter
        self.device = device
        self.sparse_cov = sparse_cov
        self.set_icov(cov)

    def set_icov(self, cov):
        """
        Set inverse covariance as self.icov

        Parameters
        ----------
        cov : tensor
            Covariance of target data.
            Can be an nD tensor with the same
            shape as target, in which case
            this is just the diagonal of the cov mat
            reshaped as the target (sparse_cov=True)
            Can also be a 2D tensor holding the on and
            off diagonal covariances, in which case
            target must be a flat column tensor (sparse_cov=False)
        """
        self.cov = utils.push(cov, self.device)
        if self.sparse_cov:
            self.icov = 1 / self.cov
            self.logdet = torch.sum(torch.log(self.cov))
            self.ndim = sum(self.cov.shape)
        else:
            self.icov = torch.linalg.pinv(self.cov)
            self.logdet = torch.slogdet(self.cov)
            self.ndim = len(self.cov)
        self.norm = 0.5 * (self.ndim * torch.log(torch.tensor(2*np.pi)) + self.logdet)
        if self.parameter:
            self.icov = torch.nn.Parameter(self.icov.detach().clone())

    def forward(self, prediction):
        """
        Compute negative log likelihood given prediction

        Parameters
        ----------
        prediction : tensor
            Forward model prediction, must match target shape
            If covariance is a matrix, not a vector, then
            prediction must have shape (Ndim, Nfeatures),
            where the cov mat is Ndim x Ndim, and Nfeatures
            is to be summed over after forming chi-square.
        """
        ## TODO: allow for kwarg dynamic cov for cosmic variance
        # compute residual
        res = torch.abs(prediction - self.target)

        # get negative log likelihood
        if self.sparse_cov:
            chisq = 0.5 * torch.sum(res**2 * self.icov)
        else:
            chisq = 0.5 * torch.sum(res @ self.icov @ res)

        return chisq + self.norm


class LogPrior(torch.nn.Module):
    """
    Negative log prior. See LogUniformPrior and LogGaussPrior
    for examples of prior callables.
    """
    def __init__(self, model, params, priors):
        """
        To set an unbounded prior (i.e. no prior)
        pass params as an empty list.

        Parameters
        ----------
        model : nn.Module or rime.Sequential
            The forward model object
        params : list
            List of parameter strings conforming
            to model.named_parameters() syntax
        priors : dict
            Dictionary of logprior callables
            with keys matching params and values
            as logprior functions.
        """
        super().__init__()
        self.model = model
        self.params = params
        self.priors = priors

    def forward(self):
        """
        Extract current parameter values from model,
        evaluate their logpriors and sum
        """
        logprior = 0
        for p in self.params:
            param = self.model.get_parameter(p)
            logprior += self.priors[p](param)

        return logprior

    def __call__(self):
        return self.forward()


class LogProb(torch.nn.Module):
    """
    The log probabilty density of the likelihood times the prior,
    which is proportional to the log posterior up to a constant.
    """
    def __init__(self, target, loglike, logprior=None):
        """
        Parameters
        ----------
        loglike : LogGaussLikelihood object
        logprior : LogPrior object
        """
        super().__init__()
        self.target = target
        self.loglike = loglike
        self.logprior = logprior

    def forward(self, prediction):
        """
        Evaluate the negative log likelihood and prior

        Parameters
        ----------
        prediction : tensor
            Model prediction to evaluate in likelihood
        """
        logprob = self.loglike(prediction)
        if self.logprior is not None:
            logprob += self.logprior()
        return  logprob


class LogUniformPrior:
    """
    Negative log uniform prior
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Parameters
        ----------
        lower_bound, upper_bound : tensors
            Tensors holding the min and max of the allowed
            parameter value. Tensors should match input params
            in shape.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.norm = torch.sum(torch.log(1/(upper_bound - lower_bound)))

    def forward(self, params):
        """
        Evalute uniform logprior on a parameter tensor.

        Parameters
        ----------
        params : tensor
            Parameter tensor to evaluate prior on.
        """
        # out of bounds if sign of residual is equal
        lower_sign = torch.sign(self.lower_bound - params)
        upper_sign = torch.sign(self.upper_bound - params)
        # aka if sum(abs(sign)) == 2
        out = torch.abs(lower_sign + upper_sign) == 2
        if out.any():
            return np.inf
        return self.norm

    def __call__(self, params):
        return self.forward(params)


class LogGaussPrior:
    """
    Negative log Gaussian prior
    """
    def __init__(self, mean, cov, sparse_cov=True):
        """
        mean and cov must match shape of params, unless
        sparse_cov == False, in which case cov is 2D matrix
        dotted into params.ravel()

        Parameters
        ----------
        mean : tensor
            Parameter mean tensor, matching param
        cov : tensor
            Parameter covariance tensor
        sparse_cov : bool, optional
            If True, cov is the diagonal of the covariance
            else ,cov is a 2D matrix dotted into params.ravel()
        """
        self.mean = mean
        self.cov = cov
        self.sparse_cov = sparse_cov
        if self.sparse_cov:
            self.icov = 1 / self.cov
            self.logdet = torch.sum(torch.log(cov))
            self.ndim = sum(cov.shape)
        else:
            self.icov = torch.linalg.pinv(cov)
            self.logdet = torch.slogdet(cov)
            self.ndim = len(cov)
        self.norm = 0.5 * (self.ndim * torch.log(torch.tensor(2*np.pi)) + self.logdet)

    def forward(self, params):
        """
        Evaluate negative log Gaussian prior

        Parameters
        ----------
        params : tensor
            Parameter tensor
        """
        res = params - self.mean
        if self.sparse_cov:
            chisq = 0.5 * torch.sum(res**2 * self.icov)
        else:
            res = res.ravel()
            chisq = 0.5 * torch.sum(res @ self.icov @ res)

        return chisq + self.norm

    def __call__(self, params):
        return self.forward(params)


class ParamDict:
    """
    An object holding a dictionary of model parameters.
    """
    def __init__(self, params):
        self.params = params
        self.keys = list(self.params.keys())

    def __mul__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] * other.params[k] for k in self.keys})
        else:
            return ParamDict({k: self.params[k] * other for k in self.keys})

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys:
                self.params[k] *= other.params[k]
        else:
            for k in self.keys:
                self.params[k] *= other
        return self

    def __div__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] / other.params[k] for k in self.keys})
        else:
            return ParamDict({k: self.params[k] / other for k in self.keys})

    def __rdiv__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: other.params[k] / self.params[k] for k in self.keys})
        else:
            return ParamDict({k: other / self.params[k] for k in self.keys})
        return self

    def __idiv__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys:
                self.params[k] /= other.params[k]
        else:
            for k in self.keys:
                self.params[k] /= other

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __add__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] + other.params[k] for k in self.keys})
        else:
            return ParamDict({k: self.params[k] + other for k in self.keys})

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys:
                self.params[k] += other.params[k]
        else:
            for k in self.keys:
                self.params[k] += other
        return self

    def __sub__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: self.params[k] - other.params[k] for k in self.keys})
        else:
            return ParamDict({k: self.params[k] - other for k in self.keys})

    def __rsub__(self, other):
        if isinstance(other, ParamDict):
            return ParamDict({k: other.params[k] - self.params[k] for k in self.keys})
        else:
            return ParamDict({k: other - self.params[k] for k in self.keys})

    def __isub__(self, other):
        if isinstance(other, ParamDict):
            for k in self.keys:
                self.params[k] -= other.params[k]
        else:
            for k in self.keys:
                self.params[k] -= other
        return self

    def __neg__(self):
        return ParamDict({k: -self.params[k] for k in self.keys})

    def __pow__(self, alpha):
        return ParamDict({k: self.params[k]**alpha for k in self.keys})

    def __iter__(self):
        return (p for p in self.params)

    def clone(self):
        """clone object"""
        return ParamDict({k: self.params[k].clone() for k in self.keys})

    def copy(self):
        """copy object"""
        return ParamDict({k: torch.nn.Parameter(self.params[k].detach().clone()) for k in self.keys})

    def detach(self):
        """detach object"""
        return ParamDict({k: self.params[k].detach() for k in self.keys})

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val

    def __repr__(self):
        return self.params.__repr__()


