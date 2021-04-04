"""
Module for torch sky models and relevant functions
"""
import torch
import numpy as np
from scipy.special import sph_harm

from . import utils


class CompositeModel(torch.nn.Module):
    """
    Composite sky model
    """
    def __init__(self, models):
        """
        Composite sky model

        Parameters
        ----------
        models : list
            List of instantiated sky model objects
            to evaluate and sum
        """
        self.models = models

    def forward(self, params=None):
        """
        Forward model

        Parameters
        ----------
        params : list of tensor tuples, optional
            If provided, use these parameter vectors
            corresponding to each model in self.models
            intead of the parameters attached to the
            models.

        Returns
        -------
        tensor
            Summed output of self.models
        """
        if params is None:
            params = [None for m in self.models]

        # iterate over models
        output = []
        for param, model in zip(params, self.models):
            if param is None:
                param = model.param
            output.append(model(param))

        return torch.sum(output, axis=0)


class PointSourceModel(torch.nn.Module):
    """
    Point source sky model with fixed
    source locations but variable flux density.
    Relates source flux parameterization
    to per-frequency, per-stokes, per-source
    flux density vector.

    Returns point source flux density and their sky
    locations in equatorial coordinates.
    """
    def __init__(self, params, angs, Nfreqs, R=None, parameter=True):
        """
        Fixed-location point source model with
        parameterized flux density.

        Parameters
        ----------
        params : tensor
            Point source flux density amplitudes of shape
            (Npol, Npol, Nsource, ...). Npol is the number
            of feed polarizations. The first two
            axes are the coherency matrix B:
            .. math::
                B = \left(
                    \begin{array}{cc}I + U & U + iV \\
                    U - iV & I - U \end{array}
                \right)
        angs : tensor
            Point source unit vectors on the sky in equatorial
            coordinates of shape (Nsource, 2), where the
            last two axes are RA and Dec.
        Nfreqs : int
            Number of frequency bins in output tensor.
        R : callable, optional
            An arbitrary response function for the
            point source model, mapping self.params
            to a sky source tensor of shape
            (Npol, Npol, Nsources, Nfreqs)
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.
        """
        super().__init__()
        self.params = params
        self.angs = angs
        self.Npol = len(params)
        self.Nfreqs = Nfreqs
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R

    def forward(self, params=None):
        """
        Forward pass the sky parameters

        Parameters
        ----------
        params : tensor, optional
            Sky model parameter to use
            instead of self.params

        Returns
        -------
        sky : tensor
            Sky brightness at discrete locations
        angs : tensor
            Sky source locations
        """
        # setup predicted visibility
        sky = torch.zeros(self.params.shape[:3] + (self.Nfreqs,),
                          dtype=selef.params.dtype)

        # apply fed params or attr params
        if params is None:
            _params = self.params
        else:
            _params = params

        # pass through response
        return self.R(_params), self.angs


class DiffuseModel(torch.nn.Module):
    """
    Diffuse sky model at fixed locations on the sky
    but variable flux density. Can be parameterized as
    a collection of point sources on the sky, or in a
    sparse basis (e.g. spherical harmonics) that is then
    mapped to sky coordinates.

    Returns flux density of discrete cells and their
    locations on the sky.
    """
    def __init__(self, params, angs, Nfreqs, R=None, parameter=True):
        """
        Diffuse sky model with parameterzed flux density and
        frequency dependence.

        Parameters
        ----------
        params : tensor
            Diffuse model parameterization of shape
            (Npol, Npol, ...).
            Example includes spherical harmonic for
            single frequency channel models, or
            spherical Fourier-bessel for multi-channel
            bases, or simply the flux density of
            individual sky cells across frequency.
            For sparse parameterization, the output of 
            R(params) should be a tensor of shape
            (Npol, Npol, Ncells, Nfreqs), where Ncells are
            a collection of 2D cells on the sky.
            For a single feed polarization, Npol = 1, otherwise
            Npol = 2, with the first two axes being
            the coherency matrix B:
            .. math::
                B = \left(
                    \begin{array}{cc}I + U & U + iV \\
                    U - iV & I - U \end{array}
                \right)
        angs : tensor
            Unit vectors on the sky in equatorial
            coordinates of each sky cell of shape (Nsource, 2),
            where the last two axes are RA and Dec.
        Nfreqs : int
            Number of frequency bins in output tensor.
        R : callable, optional
            An arbitrary response function for the
            diffuse source model, mapping self.params
            to a collection of sky cells of shape
            (Npol, Npol, Ncells, Nfreqs)
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.
        """
        super().__init__()
        self.params = params
        self.angs = angs
        self.Npol = len(params)
        self.Nfreqs = Nfreqs
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R

    def forward(self, params=None):
        """
        Forward pass the sky parameters

        Parameters
        ----------
        params : tensor, optional
            Sky model parameter to use
            instead of self.params

        Returns
        -------
        sky : tensor
            Sky flux density at discrete locations
        angs : tensor
            Sky source locations
        """
        # setup predicted visibility
        sky = torch.zeros(self.params.shape[:3] + (self.Nfreqs,),
                          dtype=selef.params.dtype)

        # apply fed params or attr params
        if params is None:
            _params = self.params
        else:
            _params = params

        # pass through response
        return self.R(_params), self.angs


def gen_lm(lmax):
    """
    Generate array of l and m parameters

    Parameters
    ----------
    lmax : int
        Maximum l parameter

    Returns
    -------
    array
        array of shape (Nalm, 2) holding
        the (l, m) parameters.
    """
    lms = []
    for i in range(lmax):
        for j in range(-i, i + 1):
            lms.append([i, j])
    return np.array(lms)


def Ylm(angs, lmax, dtype=torch.cfloat):
    """
    Generate an array of spherical harmonics
    basis vectors.

    Parameters
    ----------
    angs : tensor
        Sky locations at which to evaluate spherical
        harmonics, of shape (Ncells, 2), where
        the last axis is RA and Dec in radians
    lmax : int
        Maximum l parameter
    mmax : int
        Maximum m parameter. Must be
        equal to or less than lmax.
    dtype : torch datatype
        Datatype of harmonic vectors

    Returns
    -------
    tensor
        Holds spherical harmonic Ylm, of shape
        (Nalm, Ncells)
    array
        array of shape (Nalm, 2) holding
        the (l, m) parameters.
    """
    # get l and m params
    lms = gen_lm(lmax)
    Nalms = len(lms)
    Nsources = len(angs)

    # iterate and generate Ylms
    Y = torch.zeros((Nsources, Nalms), dtype=dtype)

    for i, lm in enumerate(lms):
        Y[:, i] = torch.tensor(sph_harm(lm[1], lm[0], angs[:, 0], angs[:, 1]), dtype=dtype)

    return Y


def sky2alm(sky, angs, lmax=100):
    """"
    Expand sky into spherical harmonic
    coefficients, a_lm.

    Parameters
    ----------
    sky : tensor
        Sky map of shape (Npols, Npols, Nsources, Nfreqs)
        in units of flux density. For diffuse models,
        replace Nsources with Ncells, and multiply
        the sky temperature by the cell solid angle.
    angs : tensor
        Source locations in equatorial coordinates
        of the sky map.

    Returns
    -------

    """
    alms




