"""
Sampler module
"""
import numpy as np
import torch
import os
import glob

from . import utils, optim
from .optim import ParamDict


class SamplerBase:
    """
    Base sampler class
    """
    def __init__(self, x0, outdir=None):
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
        self.outdir = outdir

    def step(self):
        """overload this method in a Sampler subclass.
        Should make an MCMC step and either accept or reject it,
        update self.x accordingly and return an acceptance bool
        """
        raise NotImplementedError

    def sample(self, Nsample):
        """
        Sample and append to chain

        Parameters
        ----------
        Nsamples : int
            Number of samples to run
        """
        for i in range(Nsample):
            accept = self.step()
            self._acceptances.append(accept)
            for k in self.x.keys:
                self.chain[k].append(utils.tensor2numpy(self.x[k], clone=True))
        self.accept_ratio = sum(self._acceptances) / len(self._acceptances)

    def get_chain(self):
        return {k: torch.as_tensor(self.chain[k]) for k in self.chain.keys()}

    def _checkpoint(self, attrs=[], overwrite=False):
        """
        Save the current state of the chain, parameters,
        and forward model to self.outdir as mcmc_run00.npz
        """
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        # save chain
        cfiles = sorted(glob.glob(self.outdir + '/mcmc_run*npz'))
        if len(cfiles) == 0:
            run_ind = 0
        else:
            run_ind = int(cfiles[-1].split('_')[-1][3:5]) + 1
        outfile = self.outdir + "/mcmc_run{:02d}.npz".format(run_ind)
        kwargs = {attr: getattr(self, attr) for attr in attrs}
        np.savez(outfile, chain=chain, accept_ratio=self.accept_ratio,
                 _acceptances=self._acceptances, x=self.x, outdir=self.outdir, **kwargs)


class HMC(SamplerBase):
    """
    Hamiltonian Monte Carlo sampler
    """
    def __init__(self, potential_fn, x0, eps, mass, Nstep=10):
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
            Diagonal of mass matrix for sampling the momentum
        Nstep : int, optional
            Number of leapfrog updates per step
        """
        super().__init__(x0)
        self.potential_fn = potential_fn
        self.fn_evals = 0
        self.Nstep = Nstep
        if isinstance(mass, torch.Tensor):
            mass = ParamDict({k: mass for k in x0})
        self.mass = mass
        self.invmass = 1 / mass
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
        K = p**2 * (self.invmass / 2)
        return sum([sum(K[k]) for k in K])

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
        invmass = self.invmass
        p = ParamDict({k: torch.randn(self.x[k].shape) * torch.sqrt(invmass[k]) for k in invmass.keys})

        return p

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
                                invmass=self.invmass)

        # evaluate metropolis acceptance
        H_new = self.K(p_new) + self._U
        prob = torch.exp(self._H - H_new)
        accept = np.random.rand() < prob

        if accept:
            self._H = H_new.detach()
            self.x = q_new

        return accept

    def optimize_mass(self):
        """
        Optimize the mass matrix given
        sampling history in self.chain

        Parameters
        ----------

        """
        raise NotImplementedError

    def from_file(self, fname):
        """
        Load an HMC sampler from file
        """
        raise NotImplementedError
        with np.load(fname) as f:
            pass

        # instantiate
        H = HMC(potent_fn, x0, eps, Nstep=Nstep, mass=mass)
        # update attrs
        H.chain
        H._acceptances 
        H.accept_ratio 

        return H


class NUTS(HMC):
    """
    No-U Turn sampler
    """
    def __init__(self, ):
        raise NotImplementedError


class Potential:
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
        # evaluate model
        U = self.model(x)
        # run reverse AD
        U.backward()
        # collect gradients
        gradU = ParamDict({k: self.model.get_parameter(k).grad.clone() for k in self.named_params})
        return U, gradU

    def __call__(self, x=None):
        return self.forward(x)


def leapfrog(q, p, dUdq, eps, N, invmass=1):
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
            q += p * (eps * invmass)
        if i != (N - 1):
            # momentum full step
            p -= dUdq(q) * eps

    # momentum half step
    p -= dUdq(q) * (eps / 2)

    # return negative momentum
    return q, -p

