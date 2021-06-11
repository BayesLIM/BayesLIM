"""
Module for primary beam modeling
"""
import torch
import numpy as np
import warnings

from . import utils


D2R = utils.D2R


class PixelBeam(torch.nn.Module):
    """
    Handles antenna primary beam models,
    which relate the directional and frequency
    response of the sky to the "perceived" sky for
    a baseline between antennas p and q

    .. math::

        I_{pq}(\hat{s}, \nu) = A_p I A_q^\ast

    Note that this can be thought of as a direction-
    dependent Jones term which takes the form

    .. math::

        J_p = \\left[\\begin{array}{cc}J_{ee} & J_{en}\\\\
                    J_{ne} & J_{nn}\\end{array}\\right]

    where e and n index the East and North feed polarizations.
    The amplitude of the beam should be normalized to unity
    at boresight. Also, a single beam can be used for all
    antennas, or one can fit for per-antenna models.
    """
    def __init__(self, params, freqs, response=None, response_args=(), parameter=True, polmode='1pol',
                 powerbeam=True, fov=180):
        """
        A generic pixelized beam model evaluated on the sky

        Parameters
        ----------
        params : tensor or list of tensor
            Initial beam parameterization, matched to the adopted
            response function R.
            By default, params should be a tensor of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix), where Npix are the sky pixels
            where the beam is defined. For sparser parameterizations,
            one can replace Npix with Ncoeff where Ncoeff is mapped
            to Npix via R(params), or feed a list of tensors (e.g. GaussResponse).
        freqs : tensor
            Observational frequency bins [Hz]
        response : callable, optional
            Beam response function, mapping input params
            to output beam. It should return another callable
            that takes (zen [deg], az [deg], freqs [Hz])
            as arguments and returns the beam values of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix).
        parameter : bool, optional
            If True, fit for params (default), otherwise
            keep it fixed to its input value
        polmode : str, ['1pol', '2pol', '4pol'], optional
            Polarization mode. params must conform to polmode.
            1pol : single linear polarization (default)
            2pol : two linear polarizations (diag of Jones)
            4pol : four linear and cross pol (2x2 Jones)
        powerbeam : bool, optional
            If True, take the antenna beam to be a real-valued, baseline
            "power" beam, or psky = beam * sky. Only valid for 1pol or 2pol.
        fov : float, optional
            Total angular extent of the field-of-view in degrees, centered
            at the pointing center (alitude). Parts of the sky outside of fov
            are truncated from the sky integral.
            E.g. fov = 180 (default) means we view the entire sky above the horizon,
            while fov = 90 means we view the sky withih 45 deg of zenith.
            Default is full sky above the horizon.
        """
        super().__init__()
        self.params = params
        if isinstance(params, (list, tuple)):
            _params = []
            for i in range(len(params)):
                p = params[i]
                if parameter:
                    p = torch.nn.Parameter(p)
                setattr(self, 'param{}'.format(i), p)
                _params.append(p)
            if parameter:
                self.params = _params

        else:
            if parameter:
                self.params = torch.nn.Parameter(self.params)

        if response is None:
            # assumes Npix axis of params is healpix
            self.R = PixelResponse(self.params, 'healpix')
        else:
            self.R = response(self.params, *response_args)

        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.polmode = polmode
        self.powerbeam = powerbeam
        self.fov = fov
        if self.powerbeam:
            assert self.polmode in ['1pol', '2pol']
        self.Npol = 1 if polmode == '1pol' else 2

    def gen_beam(self, zen, az):
        """
        Generate a beam model given frequency and angles
        and the field-of-view (self.fov).

        Parameters
        ----------
        zen, az : array
            zenith angle (co-latitude) and azimuth angle [deg]

        Returns
        -------
        beam : tensor
            A tensor beam model of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix)
        cut : array
            Indexing of Npix axis given fov cut
        """
        # enact fov cut
        cut = zen < self.fov / 2
        zen, az = zen[cut], az[cut]

        # get beam
        beam = self.R(zen, az, self.freqs)

        if self.powerbeam:
            ## TODO: replace abs with non-neg prior on beam?
            if torch.is_complex(beam):
                beam = torch.real(beam)
            beam = torch.abs(beam)

        return beam, cut

    def apply_beam(self, beam1, sky, beam2=None):
        """
        Apply a baseline beam matrix to a representation
        of the sky.

        Parameters
        ----------
        beam1 : tensor
            Holds the beam response for one of the antennas in a
            baseline, with shape (Npol, Npol, Nfreqs, Nsources).
        sky : tensor
            sky representation (Npol, Npol, Nfreqs, Nsources)
        beam2 : tensor, optional
            The beam response for the second antenna in the baseline.
            If None, use beam1 for both ant1 and ant2.

        Returns
        -------
        psky : tensor
            perceived sky, having mutiplied beam with sky, of shape
            (Npol, Npol, Nfreqs, Npix)
        """
        if beam2 is None:
            beam2 = beam1

        if self.polmode in ['1pol', '2pol']:
            if self.powerbeam:
                # assume beam is baseline beam with identical antenna beams: only use beam1
                psky = utils.diag_matmul(beam1, sky)
            else:
                # assume antenna beams
                psky = utils.diag_matmul(utils.diag_matmul(beam1, sky), beam2.conj())
        else:
            psky = torch.einsum("ab...,bc...,dc...->ad...", beam1,
                                sky, beam2.conj())

        return psky

    def forward(self, sky_comp, telescope, obs_jd, antpairs):
        """
        Forward pass a single sky model through the beam
        at a single observation time.

        Parameters
        ----------
        sky_comp : dict
            Output of SkyBase subclass, holding
            'sky' : tensor, representation of the sky
            'kind' : str, kind of sky model, e.g. 'point', 'alm'
            and other keyword arguments specific to the model
            and expected by the beam response function.
        telescope : TelescopeModel object
            A model of the telescope location
        obs_jd : float
            Observation time in Julian Date (e.g. 2458101.23456)
        antpairs : list of 2-tuple
            A list of all unique antenna-pairs of the beam's
            "model" axis, with each 2-tuple indexing the
            unique model axis of the beam.
            For a beam with a single antenna model, or Nmodel=1,
            then antpairs = [(0, 0)].
            For a beam with 3 antenna models and a baseline
            list of [(ant1, ant2), (ant1, ant3), (ant2, ant3)],
            then antpairs = [(0, 1), (0, 2), (1, 2)]. Note that
            the following ArrayModel object should have a mapping
            of antpairs to the physical baseline list.

        Returns
        -------
        psky_comp : dict
            Same input dictionary but with psky as 'sky' of shape
            (Npol, Npol, Nantpair, Nfreqs, Nsources), where
            roughly psky = beam1 * sky * beam2. The FoV cut has
            been applied to psky as well as the 'angs' key
        """
        kind = sky_comp['kind']
        if kind in ['point', 'pixel']:
            # get coords
            alt, az = telescope.eq2top(obs_jd, sky_comp['angs'][0], sky_comp['angs'][1],
                                       sky=kind, store=False)
            zen = 90 - alt

            # evaluate beam
            beam, cut = self.gen_beam(zen, az)
            sky = sky_comp['sky'][..., cut]
            zen, alt, az = zen[cut], alt[cut], az[cut]

            # iterate over baselines
            shape = sky.shape
            psky = torch.zeros(shape[:2] + (len(antpairs),) + shape[2:], dtype=sky.dtype)
            for k, (ant1, ant2) in enumerate(antpairs):
                # get beam of each antenna
                beam1 = beam[:, :, ant1]
                beam2 = beam[:, :, ant2]

                # apply beam to sky
                psky[:, :, k] = self.apply_beam(beam1, sky, beam2=beam2)

            sky_comp['sky'] = psky
            sky_comp['angs'] = sky_comp['angs'][:, cut]
            sky_comp['altaz'] = torch.tensor([alt, az])

        else:
            raise NotImplementedError

        return sky_comp


class AlmBeam(torch.nn.Module):
    """
    A beam model representation in
    spherical harmonic space.
    This takes sky models of 'alm' kind.
    """
    def __init__(self, params, freqs, parameter=True, polmode='1pol',
                 powerbeam=False):
        raise NotImplementedError


class PixelResponse:
    """
    Pixelized representation for PixelBeam.

    .. code-block:: python

        R = PixelResponse(params, pixtype)
        beam = R(zen, az, freqs)

    where zen, az are the zenith and azimuth angles [deg]
    to evaluate the beam, computed using bilinear interpolation
    of the input beam map (params). The output beam has shape
    (Npol, Npol, Nmodel, Nfreqs, Npix).

    This object also has a caching system for the weights
    and indicies of a bilinear interpolation of the beam 
    given the zen and az arrays.
    """
    def __init__(self, params, pixtype, *args):
        """
        Parameters
        ----------
        params : tensor
            The pixel beam map, of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix). Note that
            params is mutable, so the output of this object
            will change if the input params is updated!
        pixtype : str
            Pixelization type. options = ['healpix', 'other']
        """
        self.params = params
        if pixtype != 'healpix':
            raise NotImplementedError("only supports healpix pixelization currently")
        self.pixtype = pixtype
        self.npix = self.params.shape[-1]
        self.interp_cache = {}

    def hash(self, zen):
        """
        Hash the first five entries of zen.

        Parameters
        ----------
        zen : ndarray
            Zenith angle (co-latitude) [arb. units]
        
        Returns
        -------
        hash object
        """
        return hash(str(zen[:5]))

    def get_interp(self, zen, az):
        """
        Get bilinear interpolation: nearest neighbor indicies
        and weights.

        Parameters
        ----------
        zen, az : zenith and azimuth angles [deg]

        Returns
        -------
        interp : tuple
            4 nearest neighbor (indices, weights)
            for each entry in zen, az, indexing self.params
        """
        # get hash
        h = self.hash(zen)
        if h in self.interp_cache:
            # get interpolation if present
            interp = self.interp_cache[h]
        else:
            # otherwise generate it
            if self.pixtype == 'healpix':
                nside = utils.healpy.npix2nside(self.npix)
                ## TODO: ensure az has the same starting convention as healpy phi
                inds, wgts = utils.healpy.get_interp_weights(nside, zen * D2R, az * D2R)

            else:
                raise NotImplementedError

            # store it
            interp = (inds, torch.as_tensor(wgts, dtype=self.params.dtype))
            self.interp_cache[h] = interp

        return interp

    def _interp(self, zen, az, m=None):
        # get interpolating map
        m = m if m is not None else self.params
        # get interpolation indices and weights
        inds, wgts = self.get_interp(zen, az)
        # select out 4-nearest neighbor indices for each zen, az
        # recall beam is (Npol, Npol, Nmodel, Nfreqs, Npix)
        nearest = m[:, :, :, :, inds.T]
        # multiply by weights and sum
        return torch.sum(nearest * wgts.T, axis=-1)

    def __call__(self, zen, az, *args):
        return self._interp(zen, az)


class GaussResponse:
    """
    A Gaussian beam representation for PixelBeam.

    .. code-block:: python

        R = GaussResponse(params)
        beam = R(zen, az, freqs)

    Recall azimuth is defined as the angle East of North
    """
    def __init__(self, params):
        """
        .. math::

            b = \exp(-0.5((l / sig_ew)**2 + (m / sig_ns)**2))

        Parameters
        ----------
        params : list of 2 tensors
            Each of shape (Npol, Npol, Nmodel, Nfreqs). The tensors are
            the Gaussian sigma in EW and NS sky directions, respectively,
            with units of the dimensionless image-plane l & m (azimuth sines & cosines).
        """
        self.params = params

    def __call__(self, zen, az, freqs):
        # get azimuth dependent sigma
        zen_rad, az_rad = zen * D2R, az * D2R
        l = torch.as_tensor(np.sin(zen_rad) * np.sin(az_rad))
        m = torch.as_tensor(np.sin(zen_rad) * np.cos(az_rad))
        beam = torch.exp(-0.5 * ((l / self.params[0][..., None])**2 + (m / self.params[1][..., None])**2))
        return beam
  

class YlmResponse(PixelResponse):
    """
    A spherical harmonic representation for PixelBeam,
    mapping a_lm to pixel space. Adopts a polynomial
    basis across frequency in units of MHz.

    .. code-block:: python

        R = YlmResponse(params, l, m, **kwargs)
        beam = R(zen, az, freqs)

    """ 
    def __init__(self, params, l, m, nside, mode='generate', interp_angs=None,
                 powerbeam=True, dtype=torch.complex64):
        """
        Note that for 'interpolate' mode, you must first call the object with a healpix map
        of zen, az (i.e. theta, phi) to "set" the beam, which is then interpolated with later
        calls of (zen, az) that may or not be of healpix ordering.

        Parameters
        ----------
        params : tensor
            Holds a_lm coefficients of shape (Npol, Npol, Nmodel, Ndeg, Ncoeff)
            Ncoeff is the number of lm modes.
            Ndeg is the number of polynomial degree terms wrt freqs.
            Nmodel is the number of unique antenna models.
        l, m : ndarrays
            The l and m modes of params.
        nside : int
            Nside integer of original healpix map. Note that this may be equal
            to Ncoeff, but it may not, as you can truncate the l and m modes
            as desired.
        mode : str, options=['generate', 'interpolate']
            generate - generate exact Y_lm given zen, az for each call
            interpolate - interpolate existing beam onto zen, az
        interp_angs : 2-tuple
            This is the initial (zen, az) [deg] to evaluate the Y_lm(zen, az) * a_lm
            transformation, which is then set on the object and interpolated for future
            calls. Must be a healpix map. Only needed if mode is 'interpolate'
        powerbeam : bool, optional
            If True, beam is a baseline beam, purely real and non-negative. Else,
            beam is complex antenna farfield beam.

        Notes
        -----
        Y_cache : a cache for Y_lm matrices (Npix, Ncoeff)
        ang_cache : a cache for (zen, az) arrays [deg]
        """
        super(YlmResponse, self).__init__(params, 'healpix')
        self.npix = utils.healpy.nside2npix(nside)
        self.l, self.m = l, m
        self.neg_m = np.any(m < 0)
        self.dtype = dtype
        self.powerbeam = powerbeam
        self.Ylm_cache = {}
        self.ang_cache = {}
        self.mode = mode
        self.interp_angs = interp_angs
        self.beam_cache = None

    def get_Ylm(self, zen, az):
        """
        Query cache for Y_lm matrix, otherwise generate it.

        Parameters
        ----------
        zen, az : ndarrays
            Zenith angle (co-latitude) and
            azimuth (longitude) [deg] (i.e. theta, phi)

        Returns
        -------
        Y : tensor
            Spherical harmonic tensor of shape
            (Nangle, Ncoeff)
        """
        # get hash
        h = self.hash(zen)
        if h in self.Ylm_cache:
            Ylm = self.Ylm_cache[h]
        else:
            # generate it, may take a while
            # generate exact Y_lm
            Ylm = utils.gen_sph2pix(zen * D2R, az * D2R, self.l, self.m,
                                    real_field=self.powerbeam, dtype=self.dtype)
            if self.powerbeam and not self.neg_m:
                Ylm[:, self.m > 0] *= 2
            # store it
            self.Ylm_cache[h] = Ylm
            self.ang_cache[h] = zen, az

        return Ylm

    def set_beam(self, beam, zen, az, freqs):
        self.beam_cache = beam, zen, az, freqs

    def clear_beam(self):
        self.beam_cache = None

    def forward(self, zen, az, freqs):
        """
        Perform the mapping from a_lm to pixel
        space, including the polynomial frequency response
        for multi-channel params.

        Parameters
        ----------
        zen, az : ndarrays
            zenith and azimuth angles [deg]
        freqs : ndarray
            frequency bins [Hz]

        Returns
        -------
        beam : tensor
            pixelized beam on the sky
            of shape (Npol, Npol, Nmodel, Nfreqs, Npix)
        """
        # get polynomial A matrix wrt freq
        Ndeg = self.params.shape[3]
        A = utils.gen_poly_A(freqs, Ndeg, dtype=self.params.dtype)

        # first do fast dot product along Ndeg axis
        p = (self.params.transpose(-1, -2) @ A.T).transpose(-1, -2)

        # generate Y matrix
        Ylm = self.get_Ylm(zen, az)

        # next do slower dot product over Ncoeff
        beam = p @ Ylm.transpose(-1, -2)

        if self.powerbeam:
            if torch.is_complex(beam):
                beam = torch.real(beam)
            beam = torch.abs(beam)

        return beam

    def __call__(self, zen, az, freqs):
        # for generate mode, forward model the beam exactly at zen, az
        if self.mode == 'generate':
            beam = self.forward(zen, az, freqs)

        # otherwise interpolate the pre=forwarded beam at zen, az
        elif self.mode == 'interpolate':
            if self.beam_cache is None:
                # beam must first be forwarded at self.interp_angs
                int_zen, int_az = self.interp_angs
                beam = self.forward(int_zen, int_az, freqs)
                # now cache it for future calls
                self.set_beam(beam, int_zen, int_az, freqs)

            # interpolate the beam at the desired locations
            beam = self._interp(zen, az, self.beam_cache[0])

        return beam


def airy_disk(zen, az, Dns, freqs, Dew=None):
    """
    Generate a (asymmetric) airy disk function

    Parameters
    ----------
    zen, az : ndarray
        Zenith (co-latitude) and azimuth angles [rad]
    Dns : float
        Effective diameter of aperture along the NS direction
    freqs : ndarray
        Frequency bins [Hz]
    Dew : float, optional
        Effective diameter of aperture along the EW direction

    Returns
    -------
    ndarray
        Airy disk response at zen, az angles
    """
    # set supra horizon to horizon value
    zen = zen.copy()
    zen[zen > np.pi / 2] = np.pi / 2

    # get sky angle dependent diameter
    if Dew is None:
        diameter = Dns
    else:
        ecc = np.abs(np.sin(az))**2            
        diameter = Dns + ecc * (Dew - Dns)
        diameter = diameter.reshape(-1, 1)
        
    # get xvals and evaluate airy disk
    from scipy import special
    xvals = diameter * np.sin(zen.reshape(-1, 1)) * np.pi * freqs.reshape(1, -1) / 2.99e8
    zeros = np.isclose(xvals, 0.0)
    beam = (2.0 * np.true_divide(special.j1(xvals), xvals, where=~zeros)) ** 2.0
    beam[zeros] = 1.0

    return beam
