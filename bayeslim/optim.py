"""
Optimization module
"""
import numpy as np
import torch
import os

from . import utils, io


class Sequential(torch.nn.Sequential):
    """
    A shallow wrapper around torch.nn.Sequential.
    The forward call takes a parameter dictionary as
    input and updates model before evaluation. e.g.

    .. code-block:: python

        S = Sequential(OrderedDict(model1=model1, model2=model2))

    where evaluation order is S(params) -> model2( model1( params ) )

    Note that the keys of the parameter dictionary
    must conform to nn.Module.named_parameters() syntax.
    """
    def __init__(self, models, starting_input=None):
        """
        Parameters
        ----------
        models : OrderedDict
            Models to evaluate in sequential order.
        starting_input : tensor, optional
            If the first model in the sequence needs
            an input, specify it here.
        """
        super().__init__(models)
        self.starting_input = starting_input

    def __getitem__(self, name):
        return io.get_model_attr(self, name)

    def __setitem__(self, name, value):
        io.set_model_attr(self, name, value)

    def forward(self, pdict=None):
        """
        Evaluate model in sequential order,
        optionally updating all parameters beforehand

        Parameters
        ----------
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

        # evaluate models
        inp = self.starting_input
        for i, model in enumerate(self):
            inp = model(inp)

        return inp


class LogGaussLikelihood(utils.Module):
    """
    Negative log Gaussian likelihood

    .. math::

        -log(L) &= \frac{1}{2}(d - \mu)^T \Sigma^{-1} (d - \mu)\\
                &+ \frac{1}{2}\log|\Sigma| + \frac{n}{2}\log(2\pi) 
    """
    def __init__(self, target, cov, cov_axis, parameter=False, device=None):
        """
        Parameters
        ----------
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
        device : str, optional
            Set the device for this object
        """
        super().__init__()
        self.target = target
        if parameter:
            raise NotImplementedError
        self.parameter = parameter
        self.device = device
        self.cov = cov
        self.cov_axis = cov_axis
        self.icov = None

    def set_icov(self, icov=None):
        """
        Set inverse covariance as self.icov

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
        self.ndim = sum(self.target.shape)
        if self.cov_axis is None:
            self.logdet = torch.sum(torch.log(self.cov))
        elif self.cov_axis == 'full':
            self.logdet = torch.slogdet(self.cov).logabsdet
        else:
            self.logdet = 0
            for i in range(self.cov.shape[2]):
                for j in range(self.cov.shape[3]):
                    for k in range(self.cov.shape[4]):
                        for l in range(self.covh.shape[5]):
                            self.logdet += torch.slogdet(self.cov[:, :, i, j, k, l]).logabsdet
        self.norm = 0.5 * (self.ndim * torch.log(torch.tensor(2*np.pi)) + self.logdet)

        # set icov
        if icov is not None:
            # use utils.push in case icov is a parameter
            self.icov = utils.push(icov, self.device)
        else:
            # compute icov and set it
            self.icov = utils.push(compute_icov(self.cov, self.cov_axis), self.device)

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
        res = (prediction - self.target).to(self.device)

        # get negative log likelihood
        chisq = 0.5 * torch.sum(apply_icov(res, self.icov, self.cov_axis))

        return chisq + self.norm


class LogPrior(utils.Module):
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
        model : torch.nn.Module subclass
            THe forward model that holds all parameters
            as attributes (i.e. model.named_parameters())
        params : list of str
            List of all params tensors to pull dynamically
            from model and to evaluate their prior
        priors : dict
            Dictionary of logprior objects, with keys
            matching syntax of params and values as callables.
            If a prior doesn't exist for a parameter it is assumed
            unbounded.
        """
        super().__init__()
        self.model = model
        self.params = params
        self.priors = priors

    def forward(self):
        """
        Evaluate log priors for all params and sum.
        """
        logprior = 0
        for p in self.params:
            if p in self.priors:
                logprior += self.priors[p](io.get_model_attr(self.model, p))

        return logprior

    def __call__(self):
        return self.forward()


class LogProb(utils.Module):
    """
    The negative log posterior density: the likelihood times the prior,
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
            Parameter mean tensor, matching params shape
        cov : tensor
            Parameter covariance tensor
        sparse_cov : bool, optional
            If True, cov is the diagonal of the covariance
            else, cov is a 2D matrix dotted into params.ravel()
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
        self.keys = list(self.params.keys())
        self.devices = {k: self.params[k].device for k in self.keys}

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

    def update(self, other):
        for key in other:
            self.__setitem__(key, other[key])
        self._setup()

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val

    def __repr__(self):
        return 'ParamDict\n{}'.format(self.params.__repr__())

    def write_npy(self, fname, overwrite=False):
        """
        Write ParamDict to .npy file

        Parameters
        ----------
        fname : str
            Path to output .npy filename
        overwrite : bool, optional
            If True overwrite fname if it exists
        """
        if not os.path.exists(fname) or overwrite:
            # reinstantiate with detached params
            pd = ParamDict({k: self.params[k].detach() for k in self.keys})
            np.savez(fname, pd)
        else:
            print("{} exists, not overwriting".format(fname))

    @staticmethod
    def load_npy(fname, force_cpu=False):
        """
        Load .npy file and return ParamDict object

        Parameters
        ----------
        fname : str
            .npy file to load
        force_cpu : bool, optional
            Force tensors onto CPU, even if they were
            written from a GPU

        Returns
        -------
        ParamDict object
        """
        # load pd, by default they are loaded into the cpu
        pd = np.load(fname, allow_pickle=True).item()
        if force_cpu:
            for k in pd.keys:
                pd.params[k] = pd.params[k].cpu()
        pd._setup()

        return pd


def train(model, opt, Nepochs=1):
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
    # setup
    train_loss = []

    # iterate over epochs
    for epoch in range(Nepochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, Nepochs))
            print('-' * 10)

        opt.zero_grad()
        out = model()
        out.backward()
        opt.step()
        train_loss.append(out)
    time_elapsed = time.time() - start
    info = dict(train_loss=train_loss, valid_loss=valid_loss, train_acc=train_acc, valid_acc=valid_acc,
                optimizer=opt)

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
        out = data**2 * icov
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

