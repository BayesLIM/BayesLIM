"""
Module for torch instrument models and relevant functions
"""
import torch
import numpy as np
from astropy import coordinates as coord
from scipy import special
import healpy

from . import utils


class BeamModel(torch.nn.Module):
    """
    Antenna primary beam model, relating the
    directional and frequency response of the sky
    to the "perceived" sky for a baseline between
    antennas j and k

    .. math::
        I_{jk}(\hat{s}, \nu) = A_j I A_k^\ast

    - This can also be thought of as a
    direction-dependent Jones term.
    - The amplitude of the beam at boresight is assumed
    to be normalized to unity.
    - A single beam can be used for all antennsa, or
    an antenna-dependent model can be specified.
    """
    def __init__(self, beam, R=None, parameter=True):
        """
        Antenna primary beam model
 
        Parameters
        ----------
        beam : tensor
            Per-antenna beam model parameterization of shape
            (Npol, Npol, Nantenna, ...)
            where the additional axes include arbitrary
            sky and frequency parameterizations.
            Initial visibility tensor in 2-real form,
            of shape (Nvis, ..., 2). Ordering should
            match ordering of V_m input to self.forward.
        R : callable, optional
            An arbitrary response function for the
            visibility model, mapping the self.v parameters
            to the space of V_m (input to self.forward).
            Default (None) is unit response.
            Note this must use torch functions.
        parameter : bool, optional
            If True, treat v as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        """
        super().__init__()
        self.v = v
        if parameter:
            self.v = torch.nn.Parameter(self.v)
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R

    def forward(self, V_m, v=None, undo=False):
        """
        Forward pass V_m through visibility
        model term.

        Parameters
        ----------
        V_m : tensor
            Starting model visibilities in 2-real form
            of shape (Nvis, ..., 2). In the general case,
            this should be a unit matrix so that the
            predicted visibilities are simply the redundant
            model. However, if you have a model of per-baseline
            non-redundancies, these could be included by putting
            them into V_m.
        v : tensor, optional
            If not None, use this as the visibility model
            instead of self.v. Default is None.
        undo : bool, optional
            If True, push V_m backwards through the model.

        Returns
        -------
        V_p : tensor
            The predicted visibilities, having pushed V_m through
            the visibility model.
        """
        # setup predicted visibility
        V_p = torch.zeros_like(V_m)

        # apply fed r or attr r
        if v is None:
            _v = self.v
        else:
            _v = v

        # iterate through vis and apply model
        for i in range(len(V_p)):
            if not undo:
                V_p[i] = V_m[i] + _v[i]
            else:
                V_p[i] = V_m[i] - _v[i]

        return V_p


class FringeModel(torch.nn.Module):
    """
    Point source sky model (s) relating the starting
    model visibility (m) to the data visibility (d)
    for antennas j and k.

    .. math::
        V^{d}_{jk} = V^{v}_{jk} + V^{m}_{jk} 
    """
    def __init__(self, v, R=None, sum=True, parameter=True):
        """
        Visibility model

        Parameters
        ----------
        V : tensor
            Initial visibility tensor in 2-real form,
            of shape (Nvis, ..., 2). Ordering should
            match ordering of V_m input to self.forward.
        R : callable, optional
            An arbitrary response function for the
            visibility model, mapping the self.v parameters
            to the space of V_m (input to self.forward).
            Default (None) is unit response.
            Note this must use torch functions.
        parameter : bool, optional
            If True, treat v as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        """
        super().__init__()
        self.v = v
        if parameter:
            self.v = torch.nn.Parameter(self.v)
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R

    def forward(self, V_m, v=None, undo=False):
        """
        Forward pass V_m through visibility
        model term.

        Parameters
        ----------
        V_m : tensor
            Starting model visibilities in 2-real form
            of shape (Nvis, ..., 2). In the general case,
            this should be a unit matrix so that the
            predicted visibilities are simply the redundant
            model. However, if you have a model of per-baseline
            non-redundancies, these could be included by putting
            them into V_m.
        v : tensor, optional
            If not None, use this as the visibility model
            instead of self.v. Default is None.
        undo : bool, optional
            If True, push V_m backwards through the model.

        Returns
        -------
        V_p : tensor
            The predicted visibilities, having pushed V_m through
            the visibility model.
        """
        # setup predicted visibility
        V_p = torch.zeros_like(V_m)

        # apply fed r or attr r
        if v is None:
            _v = self.v
        else:
            _v = v

        # iterate through vis and apply model
        for i in range(len(V_p)):
            if not undo:
                V_p[i] = V_m[i] + _v[i]
            else:
                V_p[i] = V_m[i] - _v[i]

        return V_p




def voigt_beam(nside, sigma, gamma):
    """
    A power beam with a Voigt profile

    Parameters
    ----------
    nside : int
        HEALpix nside parameter
    sigma ; float
        Standard deviation of Gaussian component [rad]
    gamma : float
        Half-width at half-max of Cauchy component [rad]

    Returns
    -------
    beam
        HEALpix map (ring ordered) of Voigt beam
    theta, phi
        co-latitude and longitude of HEALpix map [rad]
    """
    theta, phi = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
    beam = special.voigt_profile(theta, sigma, gamma)
    beam /= beam.max()

    return beam, theta, phi


def _value_fun(start, stop, hp_map):
    value = sum(hp_map._data[start:stop])
    if hp_map._density:
        value /= stop-start
    return value


def _split_fun(start, stop):
    max_value = max(hp_map)
    return _value_fun(start, stop, hp_map) > max_value


def adaptive_healpix_mesh(hp_map, split_fun=_split_fun):
    """
    Convert a single resolution healpix map to a
    multi-order coverage (MOC) map.

    Parameters
    ----------
    hp_map : mhealpy.HealpixBase subclass
        single resolution map to convert to multi-resolution
        based on relative pixel values and split_fun.
    split_fun : callable
        Function that determines if a healpix pixel is split into
        multiple pixels. See mhealpy.adaptive_moc_mesh().

    Returns
    -------
    hp_map_moc : HealpixMap object
        Downsampled healpix map
    theta, phi : array_like
        Co-latitude and longitude of downsampled map [rad]
    """
    # convert to nested if ring
    if hp_map.is_ring:
        ring2nest = healpy.ring2nest(hp_map.nside,
                                     np.arange(healpy.nside2npix(hp_map.nside)))
        hp_map._data = hp_map._data[np.argsort(ring2nest)]
        hp_map._scheme = 'NESTED'

    # downsample healpix map grid
    hp_map_moc = hp_map.adaptive_moc_mesh(hp_map.nside, split_fun,
                                          dtype=hp_map.dtype)

    # fill data array
    rangesets = hp_map_moc.pix_rangesets(hp_map_moc.nside)
    for pix,(start, stop) in enumerate(rangesets):
        hp_map_moc[pix] = _value_fun(start, stop, hp_map)

    # get theta, phi arrays
    theta, phi = hp_map_moc.pix2ang(np.arange(hp_map_moc.npix))

    return hp_map_moc, theta, phi


