"""
Optimization module
"""
import numpy as np
import torch

from . import utils


class LogGaussLikelihood(torch.nn.modules.loss._Loss):
    """
    Un-normalized negative log Gaussian
    likelihood wrapper around torch _Loss object
    """
    def __init__(self, cov=None, device=None):
        """
        Parameters
        ----------
        cov : tensor
            Covariance of target data.
            Can be an nD tensor with the same
            shape as target, in which case
            this is just the diagonal of the cov mat
            reshaped as the target.
            Can also be a 2D tensor holding the on and
            off diagonal covariances, in which case
            target must be a flat column tensor.
        device : str, optional
            Set the device for this object
        """
        super(LogGaussLikelihood, self).__init__()
        self.cov = None
        self.icov = None
        self.device = device
        if cov is not None:
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
            reshaped as the target.
            Can also be a 2D tensor holding the on and
            off diagonal covariances, in which case
            target must be a flat column tensor.
        """
        self.cov = cov.to(self.device)
        self.cov_mat = cov.ndim > 1
        if self.cov_mat:
            self.icov = torch.linalg.pinv(cov)
        else:
            self.icov = 1 / self.cov

    def forward(self, prediction, target, cov=None):
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
        target : tensor
            Data tensor
        cov : tensor, optional
            Dynamically assigned covariance of target, which
            recomputes self.icov. This is used for when
            the cov is a parameter to be fitted.
            Default is to use self.icov.
        """
        # if passed a new covariance, compute icov
        if cov is not None:
            # this is for when covariance is a parameter
            self.set_icov(cov)

        # compute residual
        res = prediction - target

        # get negative log likelihood
        if self.cov_mat:
            return 0.5 * torch.sum(res.T @ self.icov @ res)
        else:
            return 0.5 * torch.sum(res**2 * self.icov)


class LogUniformPrior(torch.nn.modules.loss._Loss):
    """
    Log uniform prior wrapper around torch _Loss object
    """
    def __init__(self, param_bounds):
        """

        Parameters
        ----------
        param_bounds : list of tensors
            List of tensors of shape (2, Nparam) holding the 
            min and max of the allowed parameter value.
            Each tensor in the list corresponds to a params
            tensor in the input list to forward(param_list).
        """
        self.bounds = param_bounds

    def forward(self, param_list):
        """
        Evalute uniform logprior on a parameter list
        that matches the ordering of param_bounds.

        Parameters
        ----------
        param_list : list of tensors
            List of tensors holding parameter
            values
        """
        logprior = torch.tensor(0)
        for i, p in enumerate(param_list):
            # out of bounds if sign of res is equal
            sign = torch.sign(self.bounds[i] - p)
            # aka if sum(abs(sign)) == 2
            out = torch.sum(torch.abs(sign), axis=0) == 2
            if out.any():
                logprior += np.inf
                break

        return logprior


class LogGaussPrior(torch.nn.modules.loss._Loss):
    """
    Negative log Gaussian prior
    """
    def __init__(self, param_means, param_sigs):
        """
        Parameters
        ----------
        param_means : list of tensors
            List of Gaussian mean tensors matching
            shape of elements in param_list
        param_sigs : list of tensors
            List of Gaussian stand. dev. tensors
            matching shape of elements in param_list
        """
        self.means = param_means
        self.sigs = param_sigs

    def forward(self, param_list):
        """
        Evaluate negative log Gaussian prior

        Parameters
        ----------
        param_list : list of tensors
            List of parameter tensors matching
            ordering of means and sigs
        """
        logprior = torch.tensor(0)
        for i, p in enumerate(param_list):
            logprior += torch.sum(0.5 * (p - self.means[i])**2 / self.sigs[i]**2)

        return logprior


class LogProb(torch.nn.modules.loss._Loss):
    """
    The log probabilty density of the likelihood times the prior,
    which is proportional to the log posterior up to a constant.
    """

    def __init__(self, loglike, logprior):
        """
        Parameters
        ----------
        loglike : LogLikelihood object
        logprior : LogPrior object
        """
        self.loglike = loglike
        self.logprior = logprior

    def forward(self, prediction, target, param_list, cov=None):
        """
        Evaluate the negative log likelihood and prior

        Parameters
        ----------
        prediction : tensor
            Model mean prediction
        target : tensor
            Data vector
        param_list : list of tensors
            List of parameter tensors, matching
            ordering of the distributions parameters
            in loglike and logprior
        cov : tensor
            Data covariance if it is parameter dependent
        """
        return self.loglike.forward(prediction, target, cov=cov) \
               + self.logprior(param_list)



