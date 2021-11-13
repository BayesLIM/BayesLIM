"""
Module for primary beam modeling
"""
import torch
from torch.nn import Parameter
import numpy as np
import warnings
from scipy import interpolate, special as scispecial
import copy

from . import utils, linalg


D2R = utils.D2R


class PixelBeam(utils.Module):
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
    def __init__(self, params, freqs, ant2beam=None, response=None,
                 response_args=(), response_kwargs={},
                 parameter=True, polmode='1pol', pol=None,
                 powerbeam=True, fov=180, name=None):
        """
        A generic pixelized beam model evaluated on the sky

        Parameters
        ----------
        params : tensor
            Initial beam parameterization, matched to the adopted
            response function R.
            By default, params should be a tensor of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix), where Npix are the sky pixels
            where the beam is defined. For sparser parameterizations,
            one can replace Npix with Ncoeff where Ncoeff is mapped
            to Npix via R(params), or feed a list of tensors (e.g. GaussResponse).
        freqs : tensor
            Observational frequency bins [Hz]
        ant2beam : dict
            Dict of integers that map a antenna number to a particular
            index in the beam model output from beam.
            E.g. {10: 0, 11: 0, 12: 0} for 3-antennas [10, 11, 12] with
            1 shared beam model or {10: 0, 11: 1, 12: 2} for 3-antennas
            [10, 11, 12] with different 3 beam models.
            Default is all ants map to index 0.
        response : PixelResponse class, optional
            A PixelResponse class to instantiate with params, which
            maps the input params to the output beam.
            The instantiated object has a __call__ signature
            that takes (zen [deg], az [deg], freqs [Hz])
            as arguments and returns the beam values of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix). Note that unlike
            sky_model, params are stored on both the Model and
            the Response function with the same pointer.
        response_args : tuple, optional
            arguments for instantiating response object
            after passing params as first argument
        response_kwargs : tuple, optional
            Keyword arguments for instantiating response object
        parameter : bool, optional
            If True, fit for params (default), otherwise
            keep it fixed to its input value
        polmode : str, optional
            Polarization mode. params must conform to polmode.
            1pol : single linear polarization (default) Npol=1
            2pol : two linear polarizations (diag of Jones) Npol=2
            4pol : four linear and cross pol (2x2 Jones) Npol=2
        pol : str, optional
            This specifies the dipole polarization for 1pol mode.
            Only used for 1pol mode. Options = ['ee', 'nn']
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
        name : str, optional
            Name for this model, stored as self.name.
        """
        super().__init__(name=name)
        self.params = params
        self.device = self.params.device
        if parameter:
            self.params = torch.nn.Parameter(self.params)

        if response is None:
            # assumes Npix axis of params is healpix
            self.R = PixelResponse(freqs, 'healpix',
                                   params.shape[-1])
        else:
            self.R = response(*response_args, **response_kwargs)

        self.powerbeam = powerbeam
        if hasattr(self.R, 'powerbeam'):
            assert self.powerbeam == self.R.powerbeam
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.polmode = polmode
        self.fov = fov
        if self.powerbeam:
            assert self.polmode in ['1pol', '2pol']
        self.Npol = 1 if polmode == '1pol' else 2
        self.pol = pol
        if self.polmode == '1pol':
            assert self.pol is not None, "must specify pol for 1pol mode"
            assert self.pol.lower() in ['ee', 'nn'], "must be 'ee' or 'nn'"
        if ant2beam is None:
            assert params.shape[2] == 1, "only 1 model for default ant2beam"
            self.ant2beam = utils.SimpleIndex()

        # construct _args for str repr
        self._args = dict(powerbeam=powerbeam, fov=fov, polmode=polmode)
        self._args[self.R.__class__.__name__] = getattr(self.R, '_args', None)

    def push(self, device):
        """
        Wrapper around pytorch.to(device) that pushes
        params and response object to device.

        Parameters
        ----------
        device : str
            Device to push to
        """
        self.params = utils.push(self.params, device)
        self.R.push(device)
        self.device = device

    def gen_beam(self, zen, az, prior_cache=None):
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
        zen, az : tensor
            truncated zen and az tensors
        """
        # enact fov cut
        if self.fov < 360:
            cut = zen < self.fov / 2
        else:
            cut = slice(None)
        zen, az = zen[cut], az[cut]

        # get beam
        beam = self.R(self.params, zen, az, self.freqs)

        # evaluate prior
        self.eval_prior(prior_cache)

        return beam, cut, zen, az

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
        # move objects to device
        beam1 = beam1.to(self.device)
        sky = sky.to(self.device)

        if beam2 is None:
            beam2 = beam1
        else:
            beam2 = beam2.to(self.device)

        if self.polmode in ['1pol', '2pol']:
            if self.powerbeam:
                # assume beam is baseline beam with identical antenna beams: only use beam1
                psky = linalg.diag_matmul(beam1, sky)
            else:
                # assume antenna beams
                psky = linalg.diag_matmul(linalg.diag_matmul(beam1, sky), beam2.conj())
        else:
            psky = torch.einsum("ab...,bc...,dc...->ad...", beam1,
                                sky, beam2.conj())

        return psky

    def forward(self, sky_comp, telescope, time, modelpairs, prior_cache=None, **kwargs):
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
        time : float
            Observation time in Julian Date (e.g. 2458101.23456)
        modelpairs : list of 2-tuple
            A list of all unique antenna-pairs of the beam's
            "model" axis, with each 2-tuple indexing the
            unique model axis of the beam.
            For a beam with a single antenna model, or Nmodel=1,
            then modelpairs = [(0, 0)].
            For a beam with 3 antenna models and a baseline
            list of [(ant1, ant2), (ant1, ant3), (ant2, ant3)],
            then modelpairs = [(0, 1), (0, 2), (1, 2)]. Note that
            the following ArrayModel object should have a mapping
            of modelpairs to the physical baseline list.
        prior_cache : dict, optional
            Cache for storing computed priors as self.name

        Returns
        -------
        psky_comp : dict
            Same input dictionary but with psky as 'sky' of shape
            (Npol, Npol, Nmodelpair, Nfreqs, Nsources), where
            roughly psky = beam1 * sky * beam2. The FoV cut has
            been applied to psky as well as the 'angs' key
        """
        sky_comp = dict(sky_comp.items())
        kind = sky_comp['kind']
        if kind in ['point', 'pixel']:
            # get coords
            alt, az = telescope.eq2top(time, sky_comp['angs'][0], sky_comp['angs'][1],
                                       sky=kind, store=False)
            zen = utils.colat2lat(alt, deg=True)

            # evaluate beam
            beam, cut, zen, az = self.gen_beam(zen, az, prior_cache=prior_cache)
            sky = sky_comp['sky'][..., cut]
            alt = alt[cut]

            # iterate over baselines
            shape = sky.shape
            psky = torch.zeros(shape[:2] + (len(modelpairs),) + shape[2:], dtype=sky.dtype, device=self.device)
            for k, (ant1, ant2) in enumerate(modelpairs):
                # get beam of each antenna
                beam1 = beam[:, :, ant1]
                beam2 = beam[:, :, ant2]

                # apply beam to sky
                psky[:, :, k] = self.apply_beam(beam1, sky, beam2=beam2)

            sky_comp['sky'] = psky
            sky_comp['angs'] = sky_comp['angs'][0][cut], sky_comp['angs'][1][cut]
            sky_comp['altaz'] = torch.vstack([alt, az])

        else:
            raise NotImplementedError

        # evaluate prior
        self.eval_prior(prior_cache)

        return sky_comp

    def eval_prior(self, prior_cache, inp_params=None, out_params=None):
        """
        Parameters
        ----------
        prior_cache : dict
            Dictionary to hold computed prior, assigned as self.name
        inp_params, out_params : tensor, optional
            self.params and self.R(self.params), respectively
        """
        if prior_cache is not None and self.name not in prior_cache:
            # configure inp_params if needed
            if self.priors_inp_params is not None and inp_params is None: 
                inp_params = self.params
            # configure out_params if needed
            if self.priors_out_params is not None and out_params is None:
                out_params = None
                # we can evaluate prior on PixelResponse beam
                if hasattr(self.R, 'beam_cache') and self.R.beam_cache is not None:
                    out_params = self.beam_cache

            self._eval_prior(prior_cache, inp_params, out_params)

    def freq_interp(self, freqs, kind='linear'):
        """
        Interpolate params onto new set of frequencies
        if freq_mode is channel. If freq_mode is
        poly, powerlaw or other, just update response frequencies

        Parameters
        ----------
        freqs : tensor
            Updated frequency array to interpolate to [Hz]
        kind : str, optional
            Kind of interpolation if freq_mode is channel
            see scipy.interp1d for options
        """
        if self.R.freq_mode == 'channel':
            # interpolate params across frequency
            freq_ax = 3 if not hasattr(self.R, 'freq_ax') else self.R.freq_ax
            interp = interpolate.interp1d(utils.tensor2numpy(self.freqs),
                                          utils.tensor2numpy(self.params),
                                          axis=freq_ax, kind=kind, fill_value='extrapolate')
            params = torch.as_tensor(interp(utils.tensor2numpy(freqs)), device=self.device,
                                     dtype=self.params.dtype)
            if self.params.requires_grad:
                self.params = torch.nn.Parameter(params)
            else:
                self.params = params
            self.R.params = self.params
            self.freqs = freqs

        self.R.freqs = freqs
        self.R._setup()


class PixelResponse(utils.PixInterp):
    """
    Pixelized representation for PixelBeam.

    .. code-block:: python

        R = PixelResponse(freqs, pixtype, npix, **kwargs)
        beam = R(params, zen, az, freqs)

    where zen, az are the zenith and azimuth angles [deg]
    to evaluate the beam, computed using nearest or bilinear
    interpolation of the input beam map (params). The output
    beam has shape (Npol, Npol, Nmodel, Nfreqs, Npix).

    This object also has a caching system for the weights
    and indicies of a bilinear interpolation of the beam 
    given the zen and az arrays.

    Warning: if mode = 'interpolate' and parameter = True,
    you need to clear_beam() after every backwards call,
    otherwise the graph of the cached beam is freed
    and you get a RunTimeError.
    """
    def __init__(self, freqs, pixtype, npix, interp_mode='bilinear',
                 theta=None, phi=None, freq_mode='channel',
                 device=None, log=False, f0=None, Ndeg=None, poly_dtype=None,
                 poly_kwargs={}, powerbeam=True):
        """
        Parameters
        ----------
        freqs : tensor
            frequency array of params [Hz]
        pixtype : str
            Pixelization type. options = ['healpix', 'other']
        npix : int
            Number of sky pixels in the beam
        interp_mode : str, optional
            Spatial interpolation method. ['nearest', 'bilinear']
        theta, phi : array_like, optional
            Co-latitude and azimuth arrays [deg] of
            input params if pixtype is 'other'
        freq_mode : str, optional
            Frequency parameterization model.
            channel - each freq channel is an independent parameter
            poly - low-order polynomial basis
        device : str, optional
            Device to put intermediary products on
        log : bool, optional
            If True, assume params is logged and take
            exp before returning.
        f0 : float, optional
            Fiducial frequency [Hz] for freq_mode = 'poly'
        Ndeg : int, optional
            Number of poly degrees for freq_mode = 'poly'
        poly_dtype : torch dtype, optional
            Cast poly A matrix to this dtype, freq_mode = 'poly'
        poly_kwargs : dict, optional
            Optional kwargs to pass to utils.gen_poly_A
        powerbeam : bool, optional
            If True treat beam as non-negative and real-valued.
        """
        super().__init__(pixtype, npix, interp_mode=interp_mode,
                         device=device, theta=theta, phi=phi)
        self.powerbeam = powerbeam
        self.freqs = freqs
        self.device = device
        self.log = log
        self.freq_mode = freq_mode
        self.f0 = f0
        self.Ndeg = Ndeg
        self.poly_dtype = poly_dtype
        self.freq_ax = 3
        self.poly_kwargs = poly_kwargs
        self.clear_beam()

        self._setup()

        # construct _args for str repr
        self._args = dict(interp_mode=interp_mode, freq_mode=freq_mode)

    def _setup(self):
        if self.freq_mode == 'channel':
            pass
        elif self.freq_mode == 'poly':
            # get polynomial A matrix wrt freq
            if self.f0 is None:
                self.f0 = self.freqs.mean()
            self.dfreqs = (self.freqs - self.f0) / 1e6  # MHz
            self.A = utils.gen_poly_A(self.dfreqs, self.Ndeg, device=self.device,
                                      **self.poly_kwargs).to(self.poly_dtype)

    def push(self, device):
        """push attrs to device"""
        # call PixInterp push for cache and self.device
        super().push(device)
        # other attrs
        self.freqs = self.freqs.to(device)
        if self.freq_mode == 'poly':
            self.dfreqs = self.dfreqs.to(device)
            self.A = self.A.to(device)

    def __call__(self, params, zen, az, *args):
        # get pre-forwarded beam if it exists
        if self.beam_cache is None:
            # pass through frequency response
            if self.freq_mode == 'poly':
                params = (params.transpose(-1, -2) @ self.A.T).transpose(-1, -2)

            # now cache it for future calls
            self.beam_cache = params

        # interpolate at sky values
        b = self.interp(self.beam_cache, zen, az)

        if self.powerbeam:
            ## TODO: replace abs with non-neg prior on beam?
            if torch.is_complex(b):
                b = torch.real(b)
            b = torch.abs(b)

        if self.log:
            b = torch.exp(b)

        return b

    def clear_beam(self):
        self.beam_cache = None


class GaussResponse:
    """
    A Gaussian beam representation for PixelBeam.

    .. code-block:: python

        R = GaussResponse()
        beam = R(params, zen, az, freqs)

    Recall azimuth is defined as the angle East of North.

    The input params should have shape (Npol, Npol, Nmodel, Nfreqs, 2).
    The tensors are the Gaussian sigma in EW and NS sky directions,
    respectively (last axis), with units of the dimensionless image-plane
    l & m (azimuth sines & cosines).
    The output beam has shape (Npol, Npol, Nmodel, Nfreqs, Npix)
    """
    def __init__(self):
        """
        .. math::

            b = \exp(-0.5((l / sig_ew)**2 + (m / sig_ns)**2))

        Parameters
        ----------
        freqs : tensor
            frequency array of params [Hz]
        """
        self.freq_mode = 'channel'
        self.freq_ax = 3

    def _setup(self):
        pass

    def __call__(self, params, zen, az, freqs):
        # get azimuth dependent sigma
        zen_rad, az_rad = zen * D2R, az * D2R
        srad = np.sin(zen_rad)
        srad[zen_rad > np.pi/2] = 1.0  # ensure sine_zen doesn't wrap around back to zero below horizon
        l = torch.as_tensor(srad * np.sin(az_rad), device=params.device)
        m = torch.as_tensor(srad * np.cos(az_rad), device=params.device)
        beam = torch.exp(-0.5 * ((l / params[..., 0:1])**2 + (m / params[..., 1:2])**2))
        return beam
  
    def push(self, device):
        pass


class AiryResponse:
    """
    An Airy Disk representation for PixelBeam.
    -- Note this is not differentiable!

    .. code-block:: python

        R = AiryResponse(**kwargs)
        beam = R(params, zen, az, freqs)

    Recall azimuth is defined as the angle East of North.

    params has shape (Npol, Npol, Nmodel, 1, 2). The tensors are
    the aperture diameter [meters] in the EW and NS aperture directions,
    respectively (last axis). The second-to-last axis is an empty slot
    for frequency broadcasting. In case one wants a single aperture diameter
    for both EW and NS directions, this is shape (Npol, Npol, Nmodel, 1, 1)

    The output beam has shape (Npol, Npol, Nmodel, Nfreqs, Npix).
    """
    def __init__(self, freq_ratio=1.0):
        """
        .. math::

            b = \left[\frac{2J_1(2\pi\nu a\sin\theta/c)}{2\pi\nu a\sin\theta/c}\right]^2

        Parameters
        ----------
        freq_ratio : float, optional
            Multiplicative scalar acting on freqs before airy disk is
            evaluated. Makes the beam mimic a higher or lower frequency beam.
        """
        self.freq_ratio = 1.0
        self.freq_mode = 'other'
        self.freq_ax = None

    def _setup(self):
        pass

    def __call__(self, params, zen, az, freqs):
        """
        Parameters
        ----------
        params : tensor
            parameter tensor of shape (Npol, Npol, Nmodel, 2)
        zen, az : array or tensor
            zenith and azimuth arrays [deg]
        freqs : array or tensor
            Frequency array [Hz]
        """
        # get azimuth dependent sigma
        Dew = self.params[..., 0:1]
        Dns = self.params[..., 1:2] if params.shape[-1] > 1 else None
        beam = airy_disk(zen * D2R, az * D2R, Dew, freqs, Dns, self.freq_ratio)
        return beam
  
    def push(self, device):
        pass


class YlmResponse(PixelResponse):
    """
    A spherical harmonic representation for PixelBeam,
    mapping a_lm to pixel space. Adopts a polynomial
    basis across frequency in units of MHz.

    .. code-block:: python

        R = YlmResponse(l, m, **kwargs)
        beam = R(params, zen, az, freqs)

    Warning: if mode = 'interpolate' and parameter = True,
    you need to clear_beam() after every backwards call,
    otherwise the graph of the cached beam is freed
    and you get a RunTimeError.

    params holds a_lm coefficients of shape
    (Npol, Npol, Nmodel, Ndeg, Ncoeff). Ncoeff is the number
    of lm modes. Ndeg is the number of polynomial degree terms
    wrt freqs (or Nfreqs). Nmodel is the number of unique antenna models.

    The output beam has shape (Npol, Npol, Nmodel, Nfreqs, Npix)
    """
    def __init__(self, l, m, freqs, mode='interpolate', device=None,
                 interp_mode='bilinear', theta=None, phi=None, npix=None,
                 powerbeam=True, log=False, freq_mode='channel', f0=None,
                 Ndeg=None, poly_dtype=None, poly_kwargs={},
                 Ylm_kwargs={}):
        """
        Note that for 'interpolate' mode, you must first call the object with a healpix map
        of zen, az (i.e. theta, phi) to "set" the beam, which is then interpolated with later
        calls of (zen, az) that may or not be of healpix ordering.

        Parameters
        ----------
        l, m : ndarrays
            The l and m modes of params.
        freqs : tensor
            frequency array [Hz]
        mode : str, options=['generate', 'interpolate']
            generate - generate exact Y_lm for each zen, az call. Slow and not recommended.
            interpolate - interpolate existing beam onto zen, az. See warning
            in docstring above.
        interp_mode : str, optional
            If mode is interpolate, this is the kind (see utils.PixelInterp)
        theta, phi : array_like, optional
            This is the initial (zen, az) [deg] to evaluate the Y_lm(zen, az) * a_lm
            transformation, which is then set on the object and interpolated for future
            calls. Only needed if mode is 'interpolate'
        npix : int, optional
            Number of pixels of output map. Currently only healpix supported.
            If a partial healpix map is used, this is the full map size given nside.
        powerbeam : bool, optional
            If True, beam is a baseline beam, purely real and non-negative. Else,
            beam is complex antenna farfield beam.
        log : bool, optional
            If True assume params is logged and take exp(params) before returning.
        freq_mode : str, optional
            Frequency parameterization ['channel', 'poly']
        f0 : float, optional
            fiducial frequency [Hz], for 'poly' freq_mode
        Ndeg : int, optional
            Number of poly terms, for 'poly' freq_mode
        poly_dtype : torch dtype, optional
            Cast poly A matrix to this dtype, freq_mode = 'poly'
        poly_kwargs : dict, optional
            Kwargs for generating poly modes, for 'poly' freq_mode
        Ylm_kwargs : dict, optional
            Kwargs for generating Ylm modes

        Notes
        -----
        Y_cache : a cache for Y_lm matrices (Npix, Ncoeff)
        ang_cache : a cache for (zen, az) arrays [deg]
        """
        ## TODO: enable pix_type other than healpix
        super(YlmResponse, self).__init__(freqs, 'healpix', npix,
                                          interp_mode=interp_mode,
                                          freq_mode=freq_mode, f0=f0, Ndeg=Ndeg,
                                          poly_dtype=poly_dtype, poly_kwargs=poly_kwargs,
                                          theta=theta, phi=phi)
        self.l, self.m = l, m
        self.mult = torch.ones(len(m), dtype=utils._cfloat(), device=device)
        if np.all(m >= 0):
            self.mult[m > 0] = 2.0
        self.powerbeam = powerbeam
        self.Ylm_cache = {}
        self.ang_cache = {}
        self.mode = mode
        self.beam_cache = None
        self.freq_ax = 3
        self.Ylm_kwargs = Ylm_kwargs
        self.device = device
        self.log = log

        # construct _args for str repr
        self._args = dict(mode=mode, interp_mode=interp_mode, freq_mode=freq_mode)

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
        h = utils.ang_hash(zen)
        if h in self.Ylm_cache:
            Ylm = self.Ylm_cache[h]
        else:
            # generate it, may take a while
            # generate exact Y_lm
            Ylm = utils.gen_sph2pix(zen * D2R, az * D2R, l=self.l, m=self.m, device=self.device,
                                    real_field=self.powerbeam, **self.Ylm_kwargs)
            # store it
            self.Ylm_cache[h] = Ylm
            self.ang_cache[h] = zen, az

        return Ylm

    def set_cache(self, Ylm, angs):
        """
        Insert a Ylm tensor into Ylm_cache, hashed on the
        zenith array. We use the forward transform convention

        .. math::

            T = \sum_{lm} a_{lm} Y_{lm}

        Parameters
        ----------
        Ylm : tensor
            Ylm forward model matrix of shape (Nmodes, Npix)
        angs : tuple
            sky angles of Ylm pixels of shape (2, Npix)
            holding (zenith, azimuth) in [deg]
        """
        assert len(self.l) == len(Ylm)
        zen, az = angs
        h = utils.ang_hash(zen)
        self.Ylm_cache[h] = Ylm
        self.ang_cache[h] = (zen, az)

    def forward(self, params, zen, az, freqs):
        """
        Perform the mapping from a_lm to pixel
        space, in addition to possible transformation
        over frequency.

        Parameters
        ----------
        params : tensor
            Ylm coefficients of shape (Npol, Npol, Nmodel, Ndeg, Ncoeff)
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
        # detect if params needs to be casted into complex
        if not torch.is_complex(params):
            params = utils.viewcomp(params)

        if self.freq_mode == 'channel':
            p = params
        elif self.freq_mode == 'poly':
            # first do fast dot product along frequency axis
            p = (params.transpose(-1, -2) @ self.A.T).transpose(-1, -2)

        # generate Y matrix
        Ylm = self.get_Ylm(zen, az)

        # next do slower dot product over Ncoeff
        beam = (p * self.mult) @ Ylm

        if self.powerbeam:
            if torch.is_complex(beam):
                beam = torch.real(beam)
            beam = torch.abs(beam)

        if self.log:
            beam = torch.exp(beam)

        return beam

    def __call__(self, params, zen, az, freqs):
        # for generate mode, forward model the beam exactly at zen, az
        if self.mode == 'generate':
            beam = self.forward(params, zen, az, freqs)

        # otherwise interpolate the pre-forwarded beam at zen, az
        elif self.mode == 'interpolate':
            if self.beam_cache is None:
                # beam must first be forwarded at theta and phi
                beam = self.forward(params, self.theta, self.phi, freqs)
                # now cache it for future calls
                self.beam_cache = beam

            # interpolate the beam at the desired sky locations
            beam = self.interp(self.beam_cache, zen, az)

        return beam

    def push(self, device):
        """push attrs to device"""
        self.device = device
        super().push(device)
        self.mult = self.mult.to(device)
        for k, Ylm in self.Ylm_cache.items():
            self.Ylm_cache[k] = Ylm.to(device)
        if self.beam_cache is not None:
            self.beam_cache = utils.push(self.beam_cache, device)


class AlmBeam(utils.Module):
    """
    A beam model representation in
    spherical harmonic space.
    This takes sky models of 'alm' kind.
    """
    def __init__(self, freqs, parameter=True, polmode='1pol',
                 powerbeam=False):
        raise NotImplementedError


def airy_disk(zen, az, Dew, freqs, Dns=None, freq_ratio=1.0):
    """
    Generate a (asymmetric) airy disk function.
    Note: this is not differentiable!

    .. math::

        b = \left[\frac{2J_1(x)}{x}\right]^2

    Parameters
    ----------
    zen, az : ndarray
        Zenith (co-latitude) and azimuth angles [rad]
    Dew : float or array
        Effective diameter of aperture along the EW direction
    freqs : ndarray
        Frequency bins [Hz]
    Dns : float or array, optional
        Effective diameter of aperture along the NS direction
    freq_ratio : float, optional
        Optional scalar to multiply frequencies by before
        evaluating airy disk. Can make the beam look like a
        lower or higher frequency beam
        (i.e. have a smaller or wider main lobe)

    Returns
    -------
    ndarray
        Airy disk response at zen, az angles of
        shape (..., Nfreqs, Npix)
    """
    # determine if ndarray or tensor
    mod = np if isinstance(zen, np.ndarray) else torch
    # move to numpy
    zen = copy.deepcopy(zen)

    # set supra horizon to horizon value
    zen[zen > np.pi / 2] = np.pi / 2

    # get sky angle dependent diameter
    if Dns is None:
        diameter = Dew
    else:
        ecc = mod.abs(mod.sin(az))**2            
        diameter = Dns + ecc * (Dew - Dns)

    # get xvals
    xvals = diameter * mod.sin(zen) * np.pi * freqs.reshape(-1, 1) * freq_ratio / 2.9979e8
    # add a small value to handle x=0: introduces error on level of 1e-10
    xvals += 1e-10
    beam = (2.0 * scispecial.j1(xvals) / xvals)**2

    return beam

