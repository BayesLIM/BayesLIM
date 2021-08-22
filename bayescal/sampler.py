"""
Sampler module
"""
import numpy as np
import torch
import os
import glob
from datetime import datetime
import json

from . import utils, optim
from .optim import ParamDict


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
        self.chain = {k: [] for k in x0.keys}
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
            for k in self.x.keys:
                self.chain[k].append(utils.tensor2numpy(self.x[k], clone=True))
            if Ncheck is not None:
                if i > 0 and i % Ncheck == 0:
                    assert outfile is not None
                    self.write_chain(outfile, overwrite=True, description=description)
        self.accept_ratio = sum(self._acceptances) / len(self._acceptances)

    def get_chain(self):
        return {k: torch.as_tensor(self.chain[k]) for k in self.chain.keys()}

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
    Hamiltonian Monte Carlo sampler
    """
    def __init__(self, potential_fn, x0, eps, mass, sparse_mass=True, Nstep=10):
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
        mass : ParamDict
            Mass matrix
        sparse_mass : bool, optional
            If True, mass represents the diagonal
            of the mass matrix. Otherwise, it is
            a 2D tensor for each param.ravel()
            in x0.
        Nstep : int, optional
            Number of leapfrog updates per step
        """
        super().__init__(x0)
        self.potential_fn = potential_fn
        self.fn_evals = 0
        self.Nstep = Nstep
        self.sparse_mass = sparse_mass
        if isinstance(mass, torch.Tensor):
            mass = ParamDict({k: mass for k in x0})
        self.mass = mass
        invmass = {}
        for k in mass:
            if sparse_mass:
                invmass[k] = 1 / mass[k]
            else:
                invmass[k] = torch.pinverse(mass[k])
        self.invmass = ParamDict(invmass)
        if isinstance(eps, torch.Tensor):
            eps = ParamDict({k: eps for k in x0})
        self.eps = eps
        self._H = np.inf   # starting energy

    def K(self, p):
        """
        Compute the kinetic energy given state of p

        Parameters
        ----------
        p : tensor
            Momentum tensor
        """
        if self.sparse_mass:
            K = p**2 * (self.invmass / 2)
            K = sum([sum(K[k]) for k in K])
        else:
            K = 0
            for k in p:
                prav = p[k].ravel()
                K += prav @ self.invmass[k] @ prav / 2

        return K

    def dUdx(self, x):
        """
        Compute potential and derivative of
        potential given x. Append potential and
        derivative to self.U, self.gradU.
        """
        self._U, self._gradU = self.potential_fn(x)
        self.fn_evals += 1
        return self._gradU

    def draw_momentum(self):
        """
        Draw a mean-zero random momentum vector from the mass matrix
        """
        mn = torch.distributions.multivariate_normal.MultivariateNormal
        invm = self.invmass
        p = {}
        for k in invm:
            x = self.x[k]
            if self.sparse_mass:
                p[k] = torch.randn(x.shape, device=x.device) * torch.sqrt(invm[k])
            else:
                N = x.shape.numel()
                p[k] = mn(torch.zeros(N, dtype=x.dtype), invm[k]).sample().reshape(x.shape).to(x.device)
        return ParamDict(p)

    def step(self):
        """
        Make a HMC step with metropolis update
        """
        # sample momentum vector
        p = self.draw_momentum()

        # copy temporary position tensor
        q = self.x.copy()

        # run leapfrog steps from current position
        q_new, p_new = leapfrog(q, p, self.dUdx, self.eps, self.Nstep,
                                invmass=self.invmass, sparse_mass=self.sparse_mass)

        # evaluate metropolis acceptance
        H_new = self.K(p_new) + self._U
        prob = torch.exp(self._H - H_new)
        accept = np.random.rand() < prob

        if accept:
            self._H = H_new.detach()
            self.x = q_new

        return accept

    def optimize_mass(self, Nback=None, sparse_mass=True, robust=False):
        """
        Optimize the mass matrix given
        sampling history in self.chain.
        Note that currently mass matrix is diagonal.

        Parameters
        ----------
        Nback : int, optional
            Number of samples starting
            from the back of the chain
            to use. Default is all samples.
        sparse_mass : bool, optional
            Compute diagonal (True) or full covariance (False)
            of mass matrix
        robust : bool, optional
            Use robust measure of the variance.
        """
        self.sparse_mass = sparse_mass
        for k, chain in self.chain.items():
            if Nback is None:
                Nback = len(chain) 
            device = self.x[k].device
            dtype = self.x[k].dtype
            c = np.array(chain)[-Nback:].T
            if sparse_mass:
                if robust:
                    invm = torch.tensor(np.median(np.abs(c - np.median(c, axis=1)), axis=1),
                                        dtype=dtype, device=device)
                    ivnm = (invm * 1.42)**2
                else:
                    invm = torch.tensor(np.var(c, axis=1), dtype=dtype, device=device)
                m = 1 / invm
            else:
                invm = torch.tensor(np.cov(c), dtype=dtype, device=device)
                m = torch.pinverse(invm)
            self.mass[k] = m
            self.invmass[k] = invm

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
            model_tree = json.dumps(utils.get_model_description(self.potential_fn)[1], indent=2)
            description = "{}\n{}\n{}".format(model_tree, '-'*40, description)
        self._write_chain(outfile, overwrite=overwrite,
                          attrs=['fn_evals', '_H'], description=description)


class NUTS(HMC):
    """
    No-U Turn sampler
    """
    def __init__(self, ):
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
        q_new, p_new = leapfrog(q, p, self.dUdx, self.eps, self.Nstep,
                                invmass=self.invmass, sparse_mass=self.sparse_mass)

        # evaluate metropolis acceptance
        H_new = self.K(p_new) + self._U
        prob = torch.exp(self._H - H_new)
        accept = np.random.rand() < prob

        if accept:
            self._H = H_new.detach()
            self.x = q_new

        return accept


class Potential(torch.nn.Module):
    """
    The HMC potential, holding the full forward model
    """
    def __init__(self, model):
        """
        Parameters
        ----------
        model : Module object or Sequential object
            The full forward model ending in the log posterior,
            which takes a ParamDict and returns the log post. (aka potential)
        """
        super().__init__()
        self.model = model
        self.named_params = list(dict(self.model.named_parameters()).keys())

    def forward(self, x=None):
        """
        Evalute the potential function at 
        parameter value x and return 
        the potential and its gradient

        Parameters
        ----------
        x : ParamDict
            Parameter values to evaluate
            the forward model at

        Returns
        -------
        U : tensor
            Potential value
        gradU : ParamDict
            Potential gradient
        """
        # zero gradients
        self.model.zero_grad()
        ## TODO: allow for gradient accumulation here
        ## (e.g. iterations over bl groups)
        # evaluate model
        U = self.model(x)
        # run reverse AD
        U.backward()
        # collect gradients
        gradU = ParamDict({k: self.model.get_parameter(k).grad.clone() for k in self.named_params})
        return U, gradU

    def __call__(self, x=None):
        return self.forward(x)


def leapfrog(q, p, dUdq, eps, N, invmass=1, sparse_mass=False):
    """
    Perform N leapfrog steps for position and momentum
    states.

    Parameters
    ----------
    q : tensor
        Position tensor which requires grad.
    p : tensor
        Momentum tensor, must be on same device
        as q.
    dUdq : callable
        Potential energy gradient at q.
    eps : tensor or scalar
        Step size in units of q, if a tensor must
        be on q device
    N : int
        Number of leapfrog steps
    invmass : tensor or scalar, optional
        scalar or diagonal of inverse mass matrix
        if a tensor, must be on q device
    sparse_mass : bool, optional
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
    # momentum half step
    p -= dUdq(q) * (eps / 2)

    # iterate over steps
    for i in range(N):
        with torch.no_grad():
            # position full step
            if sparse_mass:
                q += (eps * invmass) * p
            else:
                for k in q:
                    q[k] += eps[k] * (invmass[k] @ p[k].ravel()).reshape(p[k].shape)

        if i != (N - 1):
            # momentum full step
            p -= dUdq(q) * eps

    # momentum half step
    p -= dUdq(q) * (eps / 2)

    # return negative momentum
    return q, -p

