"""
Sampler module
"""
import numpy as np
import torch
import os
import glob
from datetime import datetime
import json

from . import utils, optim, io
from .paramdict import ParamDict


class SamplerBase:
    """
    Base sampler class
    """
    def __init__(self, x0):
        """
        Parameters
        ----------
        x0 : ParamDict
            Starting parameters
        outdir : str, optional
            A directory path for checkpoint saving
            parameters and models of this run.
        """
        self.x = x0.clone().copy()
        self.accept_ratio = 1.0
        self._acceptances = []
        self.chain = {k: [] for k in x0.keys()}
        # this contains all lists, which is needed
        # when writing the chain to file and loading it
        self.lists = ['_acceptances']

    def step(self):
        """overload this method in a Sampler subclass.
        Should make an MCMC step and either accept or reject it,
        update self.x accordingly and return an acceptance bool
        """
        raise NotImplementedError

    def sample(self, Nsample, Ncheck=None, outfile=None, description=''):
        """
        Sample and append to chain
 
        Parameters
        ----------
        Nsamples : int
            Number of samples to run
        Ncheck : int, optional
            When i_sample % Ncheck == 0, checkpoint
            the chain to outfile.
        outfile : str, optional
            If checkpointing, this is the .npz filepath 
            to write the chain
        """
        for i in range(Nsample):
            accept = self.step()
            self._acceptances.append(accept)
            for k in self.x.keys():
                self.chain[k].append(utils.tensor2numpy(self.x[k], clone=True))
            if Ncheck is not None:
                if i > 0 and i % Ncheck == 0:
                    assert outfile is not None
                    self.write_chain(outfile, overwrite=True, description=description)
        self.accept_ratio = sum(self._acceptances) / len(self._acceptances)

    def get_chain(self, keys=None):
        if keys is None:
            keys = self.chain.keys()
        elif isinstance(keys, str):
            keys = [keys]
        return {k: torch.as_tensor(self.chain[k]) for k in keys}

    def _write_chain(self, outfile, attrs=[], overwrite=False, description=''):
        """
        Write chain to an npz file

        Parameters
        ----------
        outfile : str
            Path to output npz file
        attrs : list of str, optional
            List of additional attributes to write to file.
            Base attrs are chain, _acceptances, x, accept_ratio
        overwrite : bool, optional
            Overwrite if file exists
        description : str, optional
            Description of run
        """
        attrs  = ['chain', '_acceptances', 'accept_ratio', 'x'] + attrs
        self.accept_ratio = sum(self._acceptances) / len(self._acceptances)
        fexists = os.path.exists(outfile)
        if not fexists or overwrite:
            write_dict = {}
            for attr in attrs:
                if attr in self.lists:
                    # special treatment for list to preserve it upon read
                    # wrap it in its own dict
                    write_dict[attr] = {attr: getattr(self, attr)}
                else:
                    write_dict[attr] = getattr(self, attr)
            write_dict['description'] = "Written UTC: {}\n{}\n{}".format(datetime.utcnow(),
                                                                         '-'*40, description)
            np.savez(outfile, **write_dict)

        else:
            print("{} exists, not overwriting...".format(outfile))

    def write_chain(self, outfile, overwrite=False, description=''):
        """
        Write chain to an npz file

        Parameters
        ----------
        outfile : str
            Path to output npz file
        overwrite : bool, optional
            Overwrite if file exists
        """
        # overload this function in Sampler subclass
        self._write_chain(outfile, overwrite=overwrite, description=description)

    def load_chain(self, infile):
        """
        Read a chain from npz file and attach attributes to object.
        This overwrites all attributes in self from the infile

        Parameters
        ----------
        infile : str
            Filepath to npz file written by self.write_chain
        """
        with np.load(infile, allow_pickle=True) as f:
            for key in f:
                if key not in ['description']:
                    if key in self.lists:
                        # special treatment for lists
                        setattr(self, key, f[key].item()[key])
                    else:
                        setattr(self, key, f[key].item())


class HMC(SamplerBase):
    """
    Hamiltonian Monte Carlo sampler.

    Note that the mass matrix only informs the momentum
    sampling: the kinetic energy always uses an identity mass matrix.
    """
    def __init__(self, potential_fn, x0, eps, cov=None, sparse_cov=True, Nstep=10):
        """
        Parameters
        ----------
        potential_fn : Potential object or callable
            Takes parameter vector x and 
            returns potential scalar and potential
            gradient wrt x.
        x0 : ParamDict
            Starting value for parameters
        eps : ParamDict
            Size of position step in units of x
        cov : ParamDict, optional
            Covariance matrix to inform momentum sampling
        sparse_cov : bool, optional
            If True, cov represents the diagonal
            of the cov matrix. Otherwise, it is
            a 2D tensor for each param.ravel()
            in x0.
        Nstep : int, optional
            Number of leapfrog updates per step
        """
        super().__init__(x0)
        self.potential_fn = potential_fn
        self.fn_evals = 0
        self.Nstep = Nstep
        if isinstance(eps, torch.Tensor):
            eps = ParamDict({k: eps for k in x0})
        self._U = np.inf # starting potential energy
        self.p = None    # ending momentum
        self.eps = eps
        self.set_cov(cov, sparse_cov)

    def set_cov(self, cov=None, sparse_cov=None, rcond=1e-15):
        """
        Set the parameter covariance, aka the
        inverse mass matrix, used to define the kinetic energy.
        Also sets the cholesky of the mass matrix, used for
        sampling the momenta.
        Default is sparse, unit variance for all parameters.

        Parameters
        ----------
        cov : ParamDict, optional
            Covariance matrix for each parameter in ParamDict
        sparse_cov : bool or ParamDict, optional
            If True, cov represents just the variance,
            otherwise it represents full covariance
        rcond : float, optional
            rcond parameter for pinverse of cov to get mass matrix
        """
        ## TODO: allow for structured covariances between parameters
        if cov is None:
            cov = ParamDict({k: torch.ones_like(self.x[k]) for k in self.x})
            sparse_cov = True
        if isinstance(sparse_cov, bool):
            sparse_cov = {k: sparse_cov for k in self.x}

        # set the cholesky of the mass matrix
        self.chol = {}
        for k in cov:
            if sparse_cov[k]:
                self.chol[k] = torch.sqrt(1 / cov[k])
            else:
                try:
                    self.chol[k] = torch.linalg.cholesky(torch.pinverse(cov[k], rcond=rcond))
                except RuntimeError:
                    # error in taking cholesky, so just default to using diagonal
                    self.chol[k] = torch.sqrt(1 / cov[k].diagonal().reshape(self.x[k].shape))
                    sparse_cov[k] = True
                    cov[k] = cov[k].diagonal().reshape(self.x[k].shape)

        self.cov = cov
        self.sparse_cov = sparse_cov

    def K(self, p):
        """
        Compute the kinetic energy given state of
        momentum p

        Parameters
        ----------
        p : ParamDict or tensor
            Momentum tensor

        Returns
        -------
        scalar
            kinetic energy
        """
        if isinstance(p, torch.Tensor):
            K = torch.sum(p**2 / 2)
        else:
            K = 0
            for k in p:
                if self.sparse_cov[k]:
                    K += torch.sum(self.cov[k] * p[k]**2 / 2)
                else:
                    prav = p[k].ravel()
                    K += prav @ self.cov[k] @ prav / 2

        return K

    def dUdx(self, x):
        """
        Compute potential and derivative of
        potential given x. Append potential and
        derivative to self.U, self.gradU, respectively
        """
        self._U, self._gradU = self.potential_fn(x)
        self.fn_evals += 1
        return self._gradU

    def draw_momentum(self):
        """
        Draw from a mean-zero, unit variance normal and
        multiply by mass matrix cholesky if available
        """
        p = {}
        for k in self.x:
            x = self.x[k]
            N = x.shape.numel()
            momentum = torch.randn(N, device=x.device)
            if self.sparse_cov[k]:
                p[k] = self.chol[k] * momentum.reshape(x.shape)
            else:
                p[k] = (self.chol[k] @ momentum).reshape(x.shape)

        return ParamDict(p)

    def step(self, sample_p=True):
        """
        Make a HMC step with metropolis update

        Parameters
        ----------
        sample_p : bool, optional
            If True (default) randomly re-sample momentum
            variables given mass matrix. If False, use
            existing self.p to begin (not standard, and
            only used for tracking an HMC trajectory)
        """
        # sample momentum vector and get starting energies
        if sample_p:
            p = self.draw_momentum()
        else:
            p = self.p.copy()
        K_start = self.K(p)
        U_start = self._U

        # copy temporary position tensor
        q = self.x.copy()

        # run leapfrog steps from current position
        q_new, p_new = leapfrog(q, p, self.dUdx, self.eps, self.Nstep,
                                invmass=self.cov, sparse_mass=self.sparse_cov)

        # get final energies
        K_end = self.K(p_new)
        U_end = self._U

        # evaluate metropolis acceptance
        prob = torch.exp(K_start + U_start - K_end - U_end)
        accept = np.random.rand() < prob

        if accept:
            self.x = q_new
            self.p = p_new
            ## TODO: save new Ugrad and pass to leapfrog
            ## to save 1 call to dUdx per iteration
        else:
            self._U = U_start

        return accept

    def optimize_cov(self, Nback=None, sparse_cov=True, robust=False):
        """
        Try to compute the covariance of self.x given 
        recent sampling history of Nback most-recent samples.

        Parameters
        ----------
        Nback : int, optional
            Number of samples starting from the back of the chain
            to use. Default is all samples.
        sparse_cov : bool, optional
            Compute diagonal (True) or full covariance (False)
            of covariance matrix
        robust : bool, optional
            Use robust measure of the variance if sparse_cov is True.
        """
        Cov = ParamDict({})
        for k, chain in self.chain.items():
            if Nback is None:
                Nback = len(chain) 
            device = self.x[k].device
            dtype = self.x[k].dtype
            c = np.array(chain)[-Nback:].T
            if sparse_cov:
                if robust:
                    cov = torch.tensor(np.median(np.abs(c - np.median(c, axis=1)), axis=1),
                                        dtype=dtype, device=device)
                    cov = (invm * 1.42)**2
                else:
                    cov = torch.tensor(np.var(c, axis=1), dtype=dtype, device=device)
            else:
                cov = torch.tensor(np.cov(c), dtype=dtype, device=device)
            Cov[k] = cov

        self.set_cov(Cov, sparse_cov=sparse_cov)

    def write_chain(self, outfile, overwrite=False, description=''):
        """
        Write chain to an npz file

        Parameters
        ----------
        outfile : str
            Path to output npz file
        overwrite : bool, optional
            Overwrite if file exists
        """
        if isinstance(self.potential_fn, Potential):
            model_tree = json.dumps(io.get_model_description(self.potential_fn)[1], indent=2)
            description = "{}\n{}\n{}".format(model_tree, '-'*40, description)
        self._write_chain(outfile, overwrite=overwrite,
                          attrs=['fn_evals', '_H'], description=description)


class NUTS(HMC):
    """
    No-U Turn sampler for HMC
    """
    def __init__(self, potential_fn, x0, eps, mass=None, sparse_mass=True):
        """
        Parameters
        ----------
        potential_fn : Potential object or callable
            Takes parameter vector x and 
            returns potential scalar and potential
            gradient wrt x.
        x0 : ParamDict
            Starting value for parameters
        eps : ParamDict
            Size of position step in units of x
        mass : ParamDict, optional
            Mass matrix. Default is unit matrix.
        sparse_mass : bool, optional
            If True, mass represents the diagonal
            of the mass matrix. Otherwise, it is
            a 2D tensor for each param.ravel()
            in x0.
        """
        raise NotImplementedError

    def build_tree(self, ):
        pass

    def step(self):
        """
        Make a HMC step with metropolis update
        """
        # sample momentum vector
        p = self.draw_momentum()

        # copy temporary position tensor
        q = self.x.copy()

        # run leapfrog steps from current position
        q_new, p_new = leapfrog(q, p, self.dUdx, self.eps, self.Nstep)

        # evaluate metropolis acceptance
        H_new = self.K(p_new) + self._U
        prob = torch.exp(self._H - H_new)
        accept = np.random.rand() < prob

        if accept:
            self._H = H_new.detach()
            self.x = q_new

        return accept


class Potential(utils.Module):
    """
    The potential field, or the negative log posterior
    up to a constant.
    """
    def __init__(self, prob, param_name=None):
        """
        Parameters
        ----------
        prob : optim.LogProb object
            The full forward model ending in the log posterior.
        param_name : str, optional
            If feeding forward(x) with x as a Tensor,
            this is the param attached to self.prob
            to update.
        """
        super().__init__()
        self.prob = prob
        self.param_name = param_name

    def forward(self, x=None, **kwargs):
        """
        Evalute the potential function at 
        parameter value x and return 
        the potential and its gradient

        Parameters
        ----------
        x : ParamDict or tensor
            Update the model with these param
            values

        Returns
        -------
        U : tensor
            Potential value
        gradU : ParamDict
            Potential gradient
        """
        # update params
        if x is not None:
            if isinstance(x, ParamDict):
                self.update(x)
            else:
                self[self.param_name] = torch.as_tensor(x)

        # zero gradients
        self.prob.zero_grad()

        # evaluate model and perform backprop
        U = self.prob.closure()

        # collect gradients
        gradU = ParamDict({k: self[k].grad.clone() for k in self.named_params})

        return U, gradU

    def __call__(self, x=None, **kwargs):
        return self.forward(x)


def leapfrog(q, p, dUdq, eps, N, invmass=1, sparse_mass=True):
    """
    Perform N leapfrog steps for position and momentum
    states.

    Parameters
    ----------
    q : tensor or ParamDict
        Position tensor which requires grad.
    p : tensor or ParamDict
        Momentum tensor, must be on same device
        as q.
    dUdq : callable
        Potential energy gradient at q.
    eps : tensor, scalar, or ParamDict
        Step size in units of q, if a tensor must
        be on q device
    N : int
        Number of leapfrog steps
    invmass : tensor, scalar, ParamDict, optional
        scalar or diagonal of inverse mass matrix
        if a tensor, must be on q device
    sparse_mass : bool, ParamDict, optional
        If True, invmass is inverse of cov diagonal
        else, invmass is full inv cov

    Returns
    -------
    tensor, tensor
        Updated position and momentum tensors
    """
    ## TODO: incorporate data split (Neal+2011 Sec 5.)
    ## TODO: incorporate friction term (Chen+2014 SGHMC)
    ## TODO: allow for more frequent update of "fast" parameters
    if isinstance(q, ParamDict):
        if isinstance(sparse_mass, bool):
            sparse_mass = {k: sparse_mass for k in q}
        if isinstance(eps, (float, int, torch.Tensor)):
            eps = ParamDict({k: eps for k in q})
        if isinstance(invmass, (float, int, torch.Tensor)):
            invmass = ParamDict({k: torch.ones_like(p[k]) * invmass for k in q})

    # momentum half step
    p -= dUdq(q) * (eps / 2)

    # iterate over steps
    for i in range(N):
        with torch.no_grad():
            def pos_step(q, eps, invmass, p, sparse_mass):
                # position full step on tensors
                if sparse_mass:
                    q += (eps * invmass) * p
                else:
                    q += eps * (invmass @ p.ravel()).reshape(p.shape)

            if isinstance(q, torch.Tensor):
                pos_step(q, eps, invmass, p, sparse_mass)

            elif isinstance(q, ParamDict):
                for k in q:
                    pos_step(q[k], eps[k], invmass[k], p[k], sparse_mass[k])

        if i != (N - 1):
            # momentum full step
            p -= dUdq(q) * eps

    # momentum half step
    p -= dUdq(q) * (eps / 2)

    # return position, negative momentum
    return q, p

