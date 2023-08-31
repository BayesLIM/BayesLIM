"""
Sampler module
"""
import numpy as np
import torch
import os
import glob
from datetime import datetime
import json
import copy

from . import utils, optim, io, linalg
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
        """
        self.x = x0.clone().copy()
        self.accept_ratio = 1.0
        self._acceptances = []
        self.chain = {k: [] for k in x0.keys()}
        self.Uchain = []
        # this contains all list attrs, which is needed
        # when writing the chain to file and loading it
        self._lists = ['_acceptances', 'Uchain']

    def step(self):
        """overload this method in a Sampler subclass.
        Should make an MCMC step and either accept or reject it,
        update self.x accordingly and return an acceptance bool
        and the probability
        """
        raise NotImplementedError

    def append_chain(self, q, U=None):
        """
        Append a positional ParamDict to self.chain

        Parameters
        ----------
        q : ParamDict
            Position to append to self.chain
        U : float, optional
            Potential energy to append to self.Uchain
        """
        for k in q.keys():
            self.chain[k].append(utils.tensor2numpy(q[k], clone=True))
        self.Uchain.append(U)

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
            accept, prob = self.step()
            self._acceptances.append(accept)

            # append parameter values to chain
            self.append_chain(self.x, U=self._U)

            # write chain if needed
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
            Base attrs are chain, _acceptances, x, accept_ratio, Uchain
        overwrite : bool, optional
            Overwrite if file exists
        description : str, optional
            Description of run
        """
        attrs  = ['chain', '_acceptances', 'accept_ratio', 'x', 'Uchain'] + attrs
        self.accept_ratio = sum(self._acceptances) / len(self._acceptances)
        fexists = os.path.exists(outfile)
        if not fexists or overwrite:
            write_dict = {}
            for attr in attrs:
                if attr in self._lists:
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
                    if key in self._lists:
                        # special treatment for lists
                        setattr(self, key, f[key].item()[key])
                    else:
                        setattr(self, key, f[key].item())

    def clear_chain(self, N=None):
        """
        Clear the oldest N entries from the chain.
        Default is clear the whole chain.

        Parameters
        ----------
        N : int, optional
            Clear the oldest N entries in the chain.
            Default is all entries
        """
        for k in self.chain:
            Nclear = N if N is not None else len(self.chain[k])
            self.chain[k] = self.chain[k][Nclear:]
        self.Uchain = self.Uchain[Nclear:]
        self._divergences = [(d[0]-Nclear, d[1]) for d in self._divergences]


class HMC(SamplerBase):
    """
    Hamiltonian Monte Carlo sampler.

    The step size, eps, and the number of steps must be
    chosen a priori. A good starting point is eps ~ [0.1, 0.9]
    and Nstep = max[1 / eps, 5] assuming the a (cov)variance
    is provided.
    """
    def __init__(self, potential_fn, x0, eps, cov=None, sparse_cov=True, Nstep=10,
                 dHmax=1000, record_divergences=False):
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
            Covariance matrix to inform momentum sampling.
            You can also pass the Hessian via self.set_cov().
        sparse_cov : bool, optional
            If True, cov represents the diagonal
            of the cov matrix. Otherwise, it is
            a 2D tensor for each param.ravel()
            in x0.
        Nstep : int, optional
            Number of leapfrog updates per step
        dHmax : float, optional
            Maximum allowable change in Hamiltonian for a
            step, in which case the trajectory is deemed divergent.
            In this case, the trajectory restarts from a random position
            in the chain and resamples the momentum.
        record_divergences : bool, optional
            If True, record metadata about divergences as they
            appear in self._divergences
        """
        super().__init__(x0)
        self._lists += ['_divergences']
        self.potential_fn = potential_fn
        self.fn_evals = 0
        self.Nstep = Nstep
        self.dHmax = dHmax
        self.record_divergences = record_divergences
        self._divergences = []  # [(chain_sample, final x, final p), ]
        if isinstance(eps, (torch.Tensor)):
            eps = ParamDict({k: eps for k in x0})
        self._U = np.inf # starting potential energy
        self._gradU = None
        self.p = None    # ending momentum
        self.eps = eps
        self.set_cov(cov=cov, sparse_cov=sparse_cov)

    def set_cov(self, cov=None, hess=None, sparse_cov=True, inv='chol', rcond=1e-15):
        """
        Set the parameter covariance, aka the inverse mass matrix,
        used to define the kinetic energy.
        Also sets the cholesky of the mass matrix, used for
        sampling the momenta.
        Default is sparse, unit variance for all parameters.
        Can also pass the Hessian instead of the covariance.

        Parameters
        ----------
        cov : ParamDict, optional
            Covariance matrix for each parameter in ParamDict
            aka the inverse mass matrix
        hess : ParamDict, optional
            Hessian matrix for each parameter in ParamDict.
            Pass either hess or cov. This is aka the mass matrix.
        sparse_cov : bool or ParamDict, optional
            If True, cov represents just the variance,
            otherwise it represents full covariance
        inv : str, optional
            How to take the inverse of a dense mass matrix or inv. mass matrix.
            'chol' : use cholesky_inverse
            'pinv' : use pseudo-inverse
        rcond : float, optional
            rcond parameter for pinverse of cov to get mass matrix

        Results
        -------
        self.cov : ParamDict
            Adopted covariance (inv mass matrix) of parameters
        self.chol : ParamDict
            Choleksy (lower) factorization of mass matrix (hessian) of parameters
        self.logdetM : float
            Log determinant of mass matrices
        self.spare_cov : ParamDict
            Whether covariance is sparse for each parameter
        """
        ## TODO: allow for structured covariances between parameters
        assert not ((cov is not None) and (hess is not None)), "cannot pass both cov and hess"
        if isinstance(sparse_cov, bool):
            sparse_cov = {k: sparse_cov for k in self.x}

        # if only passed covariance, compute hessian
        if cov is not None:
            # get inverse of covariance
            hess = ParamDict({})
            for k in cov:
                if inv == 'chol':
                    hess[k] = linalg.cholesky_inverse(cov[k])[0]
                elif inv == 'inv':
                    hess[k] = torch.linalg.pinv(cov[k], hermitian=True, rcond=rcond)
                else:
                    raise NameError("didn't recognize inv={}".format(inv))

        # if passed nothing, assign hess as identity
        elif cov is None and hess is None:
            # assign hess as identity (assign as None)
            hess = ParamDict({})
            for k, sparse in sparse_cov.items():
                hess[k] = None

        # now get cholesky of hessian
        self.sparse_cov = sparse_cov
        self.chol = ParamDict({})
        self.cov = ParamDict({})
        self.logdetM = torch.tensor(0.)
        for k in hess:
            if hess[k] is None:
                _cov, _L = None, None
            elif sparse_cov[k]:
                _cov, _L = linalg.cholesky_inverse(hess[k].ravel())
                _cov, _L = _cov.reshape(hess[k].shape), _L.reshape(hess[k].shape)
            else:
                # if this errors out, try adding a small regularizing scalar matrix to hess (i.e. Tikhonov)
                _cov, _L = linalg.cholesky_inverse(hess[k])
            self.chol[k] = _L
            if cov is None:
                # assign cov if not passed
                if hess[k] is None:
                    self.cov[k] = None
                elif inv == 'chol':
                    self.cov[k] = _cov
                elif inv == 'pinv':
                    self.cov[k] = torch.linalg.pinv(hess[k], hermitian=True, rcond=rcond)
                else:
                    raise NameError("didn't recognize inv={}".format(inv))
            else:
                # passed cov in function call
                self.cov[k] = cov[k]

            if hess[k] is None:
                pass
            elif sparse_cov[k]:
                self.logdetM += torch.log(torch.prod(hess[k]))
            else:
                self.logdetM += torch.log(torch.linalg.det(hess[k]))

    def K(self, p):
        """
        Compute the kinetic energy given state of
        momentum p. Uses state of self.cov (i.e. inv-mass)
        to scale momenta.

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
                if self.cov[k] is None:
                    K = torch.sum(p[k]**2 / 2)
                elif self.sparse_cov[k]:
                    K += torch.sum(self.cov[k] * p[k]**2 / 2)
                else:
                    prav = p[k].ravel()
                    K += prav @ self.cov[k] @ prav / 2

        return K + self.logdetM

    def is_divergent(self, H_start, H_end):
        """
        Assess whether a trajectory has diverged
        based on initial and final Hamiltonian values,
        and self.dHmax threshold

        Parameters
        ----------
        H_start : float
            Starting Hamiltonian energy level (K_start + U_start)
        H_end : float
            Ending Hamiltonian energy level (K_end + U_end)

        Returns
        -------
        bool
        """
        return (H_end - H_start) > self.dHmax

    def dUdx(self, x, Ucache=None, **kwargs):
        """
        Compute potential and derivative of
        potential given x. Append potential and
        derivative to self.U, self.gradU, respectively

        Parameters
        ----------
        Ucache : list, optional
            If provided, append potential value
        """
        self._U, self._gradU = self.potential_fn(x)
        self.fn_evals += 1
        if Ucache is not None:
            Ucache.append(self._U)

        return self._gradU

    def draw_momentum(self):
        """
        Draw from a mean-zero, unit variance normal and
        multiply by M^1/2 aka the mass matrix cholesky

        Returns
        -------
        p : ParamDict
            Random Gaussian momenta scaled by choleksy
        """
        p = {}
        for k in self.x:
            x = self.x[k]
            N = x.shape.numel()
            momentum = torch.randn(N, device=x.device)
            if self.chol[k] is None:
                p[k] = momentum.reshape(x.shape)
            elif self.sparse_cov[k]:
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

        Returns
        -------
        bool
            If the sampler accepted the step
        float
            The computed acceptance probability
        """
        # copy temporary position tensor
        q = self.x.copy()

        # sample momentum vector and get starting energies
        if sample_p:
            p = self.draw_momentum()
        else:
            p = self.p.clone()
        K_start = self.K(p)
        if self._U is None:
            self.dUdx(q)
        U_start = self._U
        H_start = K_start + U_start

        # run leapfrog steps from current position inplace!
        dUdq0 = self._gradU
        leapfrog(q, p, self.dUdx, self.eps, self.Nstep,
                 invmass=self.cov, sparse_mass=self.sparse_cov,
                 dUdq0=dUdq0)

        # get final energies
        K_end = self.K(p)
        U_end = self._U        # this comes from the final call to dUdq(q)
        H_end = K_end + U_end

        # assess whether this is divergent, otherwise continue as normal
        if self.is_divergent(H_start, H_end):
            Nchain = len(self.Uchain)

            # record info if desired
            if self.record_divergences:
                self._divergences.append((Nchain, q, p))

            # pick a random spot in the chain and restart from there
            if Nchain > 0:
                i = np.random.randint(0, Nchain)
                self._U = self.Uchain[i]
                self.x = ParamDict({k: torch.as_tensor(self.chain[k][i], device=self.x[k].device) for k in self.x})

            return False, torch.tensor(0.)

        # evaluate metropolis acceptance
        prob = min(torch.tensor(1.), torch.exp(H_start - H_end))
        accept = np.random.rand() < prob

        if accept:
            self.x = q
            self.p = p
        else:
            self._U = U_start
            self._gradU = dUdq0

        return accept, prob

    def dual_averaging(self, Nadapt, target=0.8, gamma=0.05, t0=10.0, kappa=0.75):
        """
        Dual averaging method for optimizing epsilon stepsize
        from Hoffman et al. 2014 Eqn (6). Uses current value
        self.eps as starting point. Updates self.eps inplace!

        Parameters
        ----------
        Nadapt : int
            Number of steps to perform eps adaptation
        target : float, optional
            Target acceptance probability [0, 1]
        gamma : float, optional
            A stepsize scheduling parameter, see Hoffman+14
        t0 : float, optional
            A stepsize scheduling parameter, see Hoffman+14
        kappa : float, optional
            A stepsize scheduling parameter, see Hoffman+14
        """
        # initialize variables
        log_eps_bar = torch.log(torch.ones(1))
        h_bar = 0.0

        if isinstance(self.eps, ParamDict):
            mu = (10 * self.eps).operator(torch.log)
        else:
            mu = torch.log(10 * self.eps)

        # iterate over adaptation steps
        for i in range(1, Nadapt+1):
            # run a leapfrog trajectory
            accept, prob = self.step()
            # compute dual averaging quantities
            eta = 1.0 / (i + t0)
            h_bar = (1 - eta) * h_bar + eta * (target - prob)
            log_eps = (mu - h_bar * torch.sqrt(torch.tensor(i)) / gamma)
            x_eta = i**(-kappa)
            log_eps_bar = x_eta * log_eps + (1 - x_eta) * log_eps_bar
            if isinstance(log_eps, ParamDict):
                self.eps = log_eps.operator(torch.exp)
            else:
                self.eps = torch.exp(torch.as_tensor(log_eps))
 
    def estimate_cov(self, Nback=None, sparse_cov=True, robust=False):
        """
        Try to estimate the covariance of self.x given 
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

        self.set_cov(cov=Cov, sparse_cov=sparse_cov)

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
                          attrs=['fn_evals', '_H', '_divergences'],
                          description=description)


class RecycledHMC(HMC):
    """
    Static trajectory, recycled HMC from Nishimura+2020
    """
    def __init__(self, potential_fn, x0, eps, cov=None, sparse_cov=True, Nstep=10,
                 dHmax=1000, record_divergences=False):
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
            Covariance matrix to inform momentum sampling.
            You can also pass the Hessian via self.set_cov().
        sparse_cov : bool, optional
            If True, cov represents the diagonal
            of the cov matrix. Otherwise, it is
            a 2D tensor for each param.ravel()
            in x0.
        Nstep : int, optional
            Number of leapfrog updates per step
        dHmax : float, optional
            Maximum allowable change in Hamiltonian for a
            step, in which case the trajectory is deemed divergent.
            In this case, the trajectory restarts from a random position
            in the chain and resamples the momentum.
        record_divergences : bool, optional
            If True, record metadata about divergences as they
            appear in self._divergences
        """
        super().__init__(potential_fn, x0, eps, cov=cov, sparse_cov=sparse_cov, Nstep=Nstep,
                         dHmax=dHmax, record_divergences=record_divergences)

    def step(self, sample_p=True):
        """
        Make a recycled HMC trajectory

        Parameters
        ----------
        sample_p : bool, optional
            If True (default) randomly re-sample momentum
            variables given mass matrix. If False, use
            existing self.p to begin (not standard, and
            only used for tracking an HMC trajectory)

        Returns
        -------
        bool
            If the sampler accepted the step
        float
            The computed acceptance probability
        """
        # copy temporary position tensor
        q = self.x.copy()

        # sample momentum vector and get starting energies
        if sample_p:
            p = self.draw_momentum()
        else:
            p = self.p.clone()
        K_start = self.K(p)
        if self._U is None:
            self.dUdx(q)
        U_start = self._U
        H_start = K_start + U_start
        gradU_start = self._gradU

        # run single-step leapfrog for Nstep iterations
        qs, ps, Us, Ks, Hs, gradUs = [], [], [], [], [], []
        for i in range(self.Nstep):
            # run single step leapfrog
            leapfrog(q, p, self.dUdx, self.eps, 1,
                     invmass=self.cov, sparse_mass=self.sparse_cov,
                     dUdq0=self._gradU)

            K = self.K(p)

            # assess whether we've hit a divergence
            if self.is_divergent(H_start, self._U + K):
                Nchain = len(self.Uchain)

                # record info if desired
                if self.record_divergences:
                    self._divergences.append((Nchain, q, p))

                # pick a random spot in the chain and restart from there
                if Nchain > 0:
                    i = np.random.randint(0, Nchain)
                    self._U = self.Uchain[i]
                    self.x = ParamDict({k: torch.as_tensor(self.chain[k][i], device=self.x[k].device) for k in self.x})

                return False, torch.tensor(0.)

            # otherwise append info to lists
            qs.append(q.clone())
            ps.append(p.clone())
            Us.append(self._U)
            gradUs.append(self._gradU)
            Ks.append(K)
            Hs.append(Us[-1] + Ks[-1])

        # now iterate through the trajectory and add samples to chain
        self._U, self._gradU = U_start, gradU_start
        for i in range(self.Nstep):
            # evaluate metropolis acceptance
            prob = min(torch.tensor(1.), torch.exp(H_start - Hs[i]))
            accept = np.random.rand() < prob

            if accept:
                self.append_chain(qs[i], Us[i])
                self.x = qs[i]
                self.p = ps[i]
                self._U = Us[i]
                self._gradU = gradUs[i]

            self._acceptances.append(accept)

        return accept, prob

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
            # chain appending is handled in step()
            self.step()
 
            # write chain if needed
            if Ncheck is not None:
                if i > 0 and i % Ncheck == 0:
                    assert outfile is not None
                    self.write_chain(outfile, overwrite=True, description=description)
        self.accept_ratio = sum(self._acceptances) / len(self._acceptances)


class TreeInfo:
    """
    q_left, p_left, grad_left are position, momentum, and dUdq variables at left edge of tree
    q_right, p_right, grad_right are position, mom. and dUdq variables at right edge of tree
    q_prop, p_prop, q_prop_U, q_prop_H are the current proposed (i.e. active) state of the tree
    and its associated potential energy and hamiltonian (i.e. total energy)
    weight is the log(tree_weight), where tree_weight = sum_i exp(-H_i) for all nodes in the tree
    turning and diverging are booleans denoting whether the tree has turned or diverged
    states is a list of (q, p, U) tuples for all states in the tree thus far. optional, for debugging.
    """
    def __init__(self, q_left, p_left, grad_left, q_right, p_right, grad_right,
                 q_prop, p_prop, q_prop_U, q_prop_H, weight, turning, diverging,
                 states=None):
        self.q_left = q_left
        self.p_left = p_left
        self.grad_left = grad_left
        self.q_right = q_right
        self.p_right = p_right
        self.grad_right = grad_right
        self.q_prop = q_prop
        self.p_prop = p_prop
        self.q_prop_U = q_prop_U
        self.q_prop_H = q_prop_H
        self.weight = weight
        self.turning = turning
        self.diverging = diverging
        self.states = states


def _logaddexp(x, y):
    minval, maxval = (x, y) if x < y else (y, x)
    return (minval - maxval).exp().log1p() + maxval


class NUTS(HMC):
    """ 
    No U-Turn Sampler variant of HMC.
    """
    def __init__(self, potential_fn, x0, eps, cov=None, sparse_cov=True,
                 dHmax=1000, record_divergences=False, max_tree_depth=6,
                 biased=True, track_states=False):
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
        max_tree_depth
        dHmax : float, optional
            Maximum allowable change in Hamiltonian for a
            step, in which case the trajectory is deemed divergent.
            In this case, the trajectory restarts from a random position
            in the chain and resamples the momentum.
        record_divergences : bool, optional
            If True, record metadata about divergences as they
            appear in self._divergences
        max_tree_depth : int, optional
            Maximum depth of NUTS binary tree.
        biased : bool, optional
            If True, use biased progressive sampling in subtree construction.
            Biased gives slight preference to the new tree over the old tree.
        track_states : bool, optional
            If True, track all states visited in a single NUTS trajectory
        """
        super().__init__(potential_fn, x0, eps, cov=cov, sparse_cov=sparse_cov,
                         dHmax=dHmax, record_divergences=record_divergences)
        self.max_tree_depth = max_tree_depth
        self.biased = biased
        self.track_states = track_states

    def is_turning(self, tree):
        """
        NUTS U-Turn criterion.

        Parameters
        ----------
        tree : TreeInfo object

        Returns
        -------
        bool
        """
        ## TODO: use Betancourt+2017 generalized momentum-based uturn
        return hoffman_uturn(tree.q_left, tree.q_right, tree.p_left, tree.p_right)

    def merge_trees(self, old_tree, new_tree, new_tree_right):
        """
        Combine two NUTS trees according to the (biased) progressive sampling
        approach of Betancourt+17 (Conceptual HMC) Sec A.3.2

        Parameters
        ----------
        old_tree : TreeInfo object
            The old tree
        new_tree : TreeInfo object
            The new tree
        new_tree_right : bool
            If True, treat new_tree as the right tree and old_tree as the left tree.
            Otherwise, new_tree as the left tree and old_tree as right tree.

        Returns
        -------
        TreeInfo object
            The merged tree
        """
        if self.biased:
            new_prob = min(1, torch.exp(new_tree.weight - old_tree.weight))
        else:
            new_prob = torch.exp(new_tree.weight - _logaddexp(old_tree.weight, new_tree.weight))

        # Bernoulli draw
        if np.random.rand() < new_prob:
            # accepted the new tree as current active state
            merged_tree = copy.deepcopy(new_tree)
            if new_tree_right:
                merged_tree.q_left = old_tree.q_left
                merged_tree.p_left = old_tree.p_left
                merged_tree.grad_left = old_tree.grad_left
                if merged_tree.states is not None:
                    merged_tree.states = old_tree.states + merged_tree.states
            else:
                merged_tree.q_right = old_tree.q_right
                merged_tree.p_right = old_tree.p_right
                merged_tree.grad_right = old_tree.grad_right
                if merged_tree.states is not None:
                    merged_tree.states = merged_tree.states + old_tree.states

        else:
            # use old tree as current active state
            merged_tree = copy.deepcopy(old_tree)
            if new_tree_right:
                merged_tree.q_right = new_tree.q_right
                merged_tree.p_right = new_tree.p_right
                merged_tree.grad_right = new_tree.grad_right
                if merged_tree.states is not None:
                    merged_tree.states = merged_tree.states + new_tree.states
            else:
                merged_tree.q_left = new_tree.q_left
                merged_tree.p_left = new_tree.p_left
                merged_tree.grad_left = new_tree.grad_left
                if merged_tree.states is not None:
                    merged_tree.states = new_tree.states + merged_tree.states

        # get new tree weights and metadata
        merged_tree.weight = _logaddexp(old_tree.weight, new_tree.weight)
        merged_tree.turning = old_tree.turning or new_tree.turning
        merged_tree.diverging = old_tree.diverging or new_tree.diverging

        return merged_tree

    def _build_basetree(self, q, p, direction, H_start, dUdq0=None):
        """
        Build base (depth=0) tree of a NUTS trajectory.
        Note this is not the (base) base tree (i.e. the origin)
        but the start of a new subtree. The (base) base tree is handled
        in step().
        """
        # run leapfrog for 1 step and collect state values
        states = []
        q_new, p_new = leapfrog(q.clone(), p.clone(), self.dUdx, direction * self.eps, 1,
                                invmass=self.cov, sparse_mass=self.sparse_cov,
                                dUdq0=dUdq0, states=states)

        # get state metadata
        dUdq_start = dUdq0 if dUdq0 is not None else states[0][3]
        dUdq_new = self._gradU
        U_new = self._U
        H_new = U_new + self.K(p_new)
        diverging = self.is_divergent(H_start, H_new)
        weight = _logaddexp(-H_start, -H_new)

        q_left     = q_new
        p_left     = p_new
        grad_left  = dUdq_new
        q_right    = q_new
        p_right    = p_new
        grad_right = dUdq_new
        q_prop     = q_new
        p_prop     = p_new
        q_prop_U   = U_new
        q_prop_H   = H_new
        states     = [(q_new, p_new, U_new)] if self.track_states else None

        return TreeInfo(q_left, p_left, grad_left, q_right, p_right, grad_right,
                        q_prop, p_prop, q_prop_U, q_prop_H, weight, False, diverging, states)

    def build_tree(self, q, p, direction, tree_depth, H_start, dUdq0=None, base_tree=None):
        """
        Build a subtree of the NUTS binary tree trajectory recursively.
        Modeled loosely after the pyro implementation (docs.pyro.ai).

        The tree state sampling uses the (biased) progressive sampling
        approach of Betancourt+17 (Conceptual Intro to HMC) Sec A.3.2,
        with multinomial state sampling.

        Parameters
        ----------
        q : ParamDict
            Starting position
        p : ParamDict
            Starting momentum
        direction : int
            1 or -1, either integrate forward (1) or backward (-1).
        tree_depth : int
            Depth of this subtree
        H_start : float
            Starting energy (Hamiltonian) for input q and p
        dUdq0 : ParamDict, optional
            Precomputed potential gradient at input q
        base_tree : TreeInfo object, optional
            This is the current base tree, only used for checking
            the Uturn criterion.

        Returns
        -------
        TreeInfo object
        """
        # if base tree, build it
        if tree_depth == 0:
            return self._build_basetree(q, p, direction, H_start, dUdq0=dUdq0)

        # build this subtree in two stages, evaluating a Uturn criterion halfway
        half_tree = self.build_tree(q, p, direction, tree_depth - 1, H_start,
                                    dUdq0=dUdq0, base_tree=base_tree)

        # check Uturn condition within half_tree
        turning = self.is_turning(half_tree)

        # check uturn conditions between base_tree and half_tree
        # TODO: perhaps only do this at the full tree level to save on computation? first profile it...
        if base_tree is not None:
            q_minus = base_tree.q_left if direction > 0 else half_tree.q_left
            p_minus = base_tree.p_left if direction > 0 else half_tree.p_left
            q_plus = half_tree.q_right if direction > 0 else base_tree.q_right
            p_plus = half_tree.p_right if direction > 0 else base_tree.p_right
            turning = hoffman_uturn(q_minus, q_plus, p_minus, p_plus)

        # check for break
        #if half_tree.diverging or half_tree.turning or turning:
        #    return half_tree

        # now build the latter half of the subtree
        q_start = half_tree.q_right if direction > 0 else half_tree.q_left
        p_start = half_tree.p_right if direction > 0 else half_tree.p_left
        grad0 = half_tree.grad_right if direction > 0 else half_tree.grad_left
        other_tree = self.build_tree(q_start, p_start, direction, tree_depth - 1, H_start,
                                     dUdq0=grad0, base_tree=base_tree)

        # merge the two half trees
        merged_tree = self.merge_trees(half_tree, other_tree, new_tree_right=direction > 0)

        # check uturn conditions between base_tree and merged_tree
        # TODO: perhaps only do this at the full tree level to save on computation? first profile it...
        turning = self.is_turning(merged_tree)
        if base_tree is not None:
            q_minus = base_tree.q_left if direction > 0 else merged_tree.q_left
            p_minus = base_tree.p_left if direction > 0 else merged_tree.p_left
            q_plus = merged_tree.q_right if direction > 0 else base_tree.q_right
            p_plus = merged_tree.p_right if direction > 0 else base_tree.p_right
            turning = hoffman_uturn(q_minus, q_plus, p_minus, p_plus)
        merged_tree.turning = merged_tree.turning or turning

        return merged_tree

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

        Returns
        -------
        bool
            If the sampler accepted the step
        float
            The computed acceptance probability
        """
        # copy temporary position tensor
        q = self.x.clone()

        # sample momentum vector and get starting energies
        if sample_p:
            p = self.draw_momentum()
        else:
            p = self.p.clone()
        K_start = self.K(p)
        if self._U is None:
            self.dUdx(q)
        U_start = self._U
        dUdq0 = self._gradU
        H_start = K_start + U_start

        # setup the trivial base NUTS tree of just the starting position
        states = [(q, p, U_start)] if self.track_states else None
        base_tree = TreeInfo(q, p, dUdq0, q, p, dUdq0, q, p, U_start, H_start, -H_start, False, False, states=states)

        # now perform multiplicative subtree iterations
        tree_depth = 0
        q_left = q_right = q
        p_left = p_right = p
        while tree_depth < self.max_tree_depth:
            # randomly sample the direction
            direction = 1.0 if np.random.rand() > 0.5 else -1.0

            # build the tree for this depth
            q_start = q_right if direction > 0 else q_left
            p_start = p_right if direction > 0 else p_left
            new_tree = self.build_tree(q_start, p_start, direction, tree_depth, H_start, dUdq0=dUdq0,
                                       base_tree=base_tree)

            # check for doubling or diverging
            if new_tree.diverging or new_tree.turning:
                break

            # otherwise, merge this with the base tree
            base_tree = self.merge_trees(base_tree, new_tree, direction > 0)

            # update new left and right nodes
            q_left, p_left = base_tree.q_left, base_tree.p_left
            q_right, p_right = base_tree.q_right, base_tree.p_right

            tree_depth += 1

        # deal with divergences
        if new_tree.diverging:
            Nchain = len(self.Uchain)

            # record info if desired
            if self.record_divergences:
                self._divergences.append(
                    (Nchain,
                     new_tree.q_right if direction > 0 else new_tree.q_left,
                     new_tree.p_right if direction > 0 else new_tree.p_left)
                    )

            # continue with base_tree, unless this happened close to origin, restart from a new place in the chain 
            if Nchain > 0 and tree_depth < 2:
                i = np.random.randint(0, Nchain)
                self._U = self.Uchain[i]
                self._gradU = None
                self.x = ParamDict({k: torch.as_tensor(self.chain[k][i], device=self.x[k].device) for k in self.x})

                return False, torch.tensor(0.)

        # evaluate metropolis acceptance
        prob = min(torch.tensor(1.), torch.exp(H_start - base_tree.q_prop_H).cpu())
        accept = np.random.rand() < prob

        # attach this tree for inspection
        self._base_tree = base_tree

        if accept:
            self.x = base_tree.q_prop
            self.p = base_tree.p_prop
            self._U = base_tree.q_prop_U
            self._gradU = None
        else:
            self._U = U_start
            self._gradU = dUdq0

        return accept, prob


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
            to update. i.e. 'main_params' means
            self.prob.main_params
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
            values. The keys of x should attached
            to self.prob. I.e. self.prob.main_params
            should have a key of 'main_params'

        Returns
        -------
        U : tensor
            Potential value
        gradU : ParamDict
            Potential gradient at q
        """
        # update params
        if x is not None:
            if isinstance(x, ParamDict):
                self.prob.update(x)
            else:
                self.prob[self.param_name] = torch.as_tensor(x)

        # zero gradients
        self.prob.zero_grad()

        # evaluate model and perform backprop
        U = self.prob.closure()

        # collect gradients
        gradU = ParamDict({k: self.prob[k].grad.clone() for k in self.prob.named_params})

        return U, gradU

    def __call__(self, x=None, **kwargs):
        return self.forward(x=x, **kwargs)


def hoffman_uturn(q_minus, q_plus, p_minus, p_plus):
    """
    NUTS turning criterion from Hoffman+2011

    Parameters
    ----------
    q_minus : ParamDict
        Position at left edge of branch
    q_plus : ParamDict
        Position at right edge of branch
    p_minus : ParamDict
        Momentum at left edge of branch
    p_plus : ParamDict
        Momentum at right edge of branch
    """
    if isinstance(q_minus, torch.Tensor):
        # wrap inputs in dummy dicts
        q_minus = {'key': q_minus}
        q_plus = {'key': q_plus}
        p_minus = {'key': p_minus}
        p_plus = {'key': p_plus}

    minus_angle = 0
    plus_angle = 0
    for key in q_minus:
        minus_angle += (q_plus[key] - q_minus[key]).ravel() @ p_minus[key].ravel()
        plus_angle += (q_plus[key] - q_minus[key]).ravel() @ p_plus[key].ravel()

    return (minus_angle < 0) or (plus_angle < 0)


def leapfrog(q, p, dUdq, eps, N, invmass=1, sparse_mass=True, dUdq0=None,
             states=None):
    """
    Perform N leapfrog steps for position and momentum
    states inplace.

    Parameters
    ----------
    q : tensor or ParamDict
        Position tensor which requires grad.
    p : tensor or ParamDict
        Momentum tensor, must be on same device
        as q.
    dUdq : callable
        Potential energy gradient at q: dU/dq(q),
        with signature dUdq(q, **kwargs)
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
    dUdq0 : tensor or ParamDict, optional
        Precomputed potential energy gradient at input q.
    states : list, optional
        If provided, will append (q, p, U, gradU) values for each
        leapfrog state to this list. Note: for potential (U) to be
        appended the dUdq callable
        must have a Ucache=[] kwarg that appends U.
        See HMC.dUdx() for example.

    Returns
    -------
    q : tensor or ParamDict
        Updated position
    p : tensor or ParamDict
        Updated momentum
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

    # get potential gradient at input q
    Ucache = []
    if dUdq0 is None:
        dUdq0 = dUdq(q, Ucache=Ucache)

    # append initial state to list
    U = None if len(Ucache) == 0 else Ucache[-1]
    if states is not None:
        states.append((q.clone(), p.clone(), U, dUdq0))

    # initial momentum half step
    p -= dUdq0 * (eps / 2)

    def pos_step(q, p, invmass, eps, sparse_mass):
        if isinstance(q, ParamDict):
            for k in q:
                imass = None if invmass is None else invmass[k]
                pos_step(q[k], p[k], imass, eps[k], sparse_mass[k])
        else:
            if invmass is None:
                q += eps * p
            elif sparse_mass:
                q += eps * (invmass * p)
            else:
                q += eps * (invmass @ p.ravel()).reshape(p.shape)

    # iterate over steps
    for i in range(N):
        # position full step update
        with torch.no_grad():
            pos_step(q, p, invmass, eps, sparse_mass)

        # momentum full step update if this isn't the last iteration
        if i != (N - 1):
            Ucache = []
            gradU = dUdq(q, Ucache=Ucache)
            gradU_eps = gradU * eps
            p -= gradU_eps

            # append q, and p with a half-step update
            if states is not None:
                U = None if len(Ucache) == 0 else Ucache[-1]
                states.append((q.clone(), p + gradU_eps / 2, U, gradU))

    # momentum half step
    Ucache = []
    gradU = dUdq(q, Ucache=Ucache)
    p -= gradU * (eps / 2)

    if states is not None:
        U = None if len(Ucache) == 0 else Ucache[-1]
        states.append((q.clone(), p.clone(), U, gradU))
 
    return q, p

