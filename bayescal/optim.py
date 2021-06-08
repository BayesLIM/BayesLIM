"""
Optimization module
"""
import numpy as np
import torch

from . import utils


class LogGaussLikelihood(torch.nn.modules.loss._Loss):
    """
    Negative log Gaussian likelihood
    """

    def __init__(self, cov):
        """
        """
        super(LogGaussLikelihood, self).__init__()
        self.set_icov(cov)

    def set_icov(self, cov):
        """
        Set inverse covariance (self.icov)

        Parameters
        ----------
        cov : tensor
            Covariance of target data.
            Can be an nD tensor with the same
            shape as target, in which case
            this is just the diagonal of the cov mat
            reshaped as the target.
            Can also be a 2D tensor holding on and
            off diagonal covariances, in which case
            target must be a flat column tensor.
        """
        self.cov = cov
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
        target : tensor
            Data tensor
        cov : tensor, optional
            Dynamically assigned covariance of target, which
            recomputes self.icov. This is used for when
            the cov is a parameter to be fitted.
        """
        # if passed a new covariance, compute icov
        if cov is not None:
            # this is for when covariance is a parameter
            self.set_icov(cov)

        # compute residual
        res = prediction - target
        
        # get negative log likelihood
        if self.cov_mat:
            nll = 0.5 * torch.sum(res @ self.icov @ res.T)
        else:
            nll = 0.5 * torch.sum(res**2 * self.icov)

        return nll


class LogUniformPrior:

    def __init__(self, *args, **kwargs):
        """
        """
        pass


class LogGaussPrior:

    def __init__(self, *args):
        """
        """
        pass


class LogProb:
    """
    The log probabilty density of the likelihood times the prior,
    which is proportional to the log posterior up to a constant
    """

    def __init__(self, *args, **kwargs):
        """
        """
        pass








