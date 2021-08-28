"""
Module for primary beam modeling
"""
import torch
from torch.nn import Parameter
import numpy as np
import warnings
from scipy import interpolate, special as scispecial
import copy

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
    def __init__(self, params, freqs, response=None,
                 response_args=(), response_kwargs={},
                 parameter=True, polmode='1pol',
                 powerbeam=True, fov=180):
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
        response : response object, optional
            Beam response class to instantiate, which
            maps the input params to the output beam.
            The instantiated object has a __call__ signature
            that takes (zen [deg], az [deg], freqs [Hz])
            as arguments and returns the beam values of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix). Note that unlike
            sky_model, params are stored on both the Model and
            the Response function with the same pointer.
        response_args : tuple, optional
            arguments for instantiating response object
        response_kwargs : tuple, optional
            Keyword arguments for instantiating response object
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
        self.device = self.params.device
        if parameter:
            self.params = torch.nn.Parameter(self.params)

        if response is None:
            # assumes Npix axis of params is healpix
            self.R = PixelResponse(self.params, freqs, 'healpix',
                                   params.shape[-1])
        else:
            self.R = response(self.params, *response_args, **response_kwargs)

        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.polmode = polmode
        self.powerbeam = powerbeam
        self.fov = fov
        if self.powerbeam:
            assert self.polmode in ['1pol', '2pol']
        self.Npol = 1 if polmode == '1pol' else 2
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
        beam = self.R(zen, az, self.freqs)

        if self.powerbeam:
            ## TODO: replace abs with non-neg prior on beam?
            if torch.is_complex(beam):
                beam = torch.real(beam)
            beam = torch.abs(beam)

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

    def forward(self, sky_comp, telescope, obs_jd, modelpairs):
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
            alt, az = telescope.eq2top(obs_jd, sky_comp['angs'][0], sky_comp['angs'][1],
                                       sky=kind, store=False)
            zen = utils.colat2lat(alt, deg=True)

            # evaluate beam
            beam, cut, zen, az = self.gen_beam(zen, az)
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
            sky_comp['angs'] = sky_comp['angs'][:, cut]
            sky_comp['altaz'] = torch.vstack([alt, az])

        else:
            raise NotImplementedError

        return sky_comp

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

        R = PixelResponse(params, pixtype)
        beam = R(zen, az, freqs)

    where zen, az are the zenith and azimuth angles [deg]
    to evaluate the beam, computed using nearest or bilinear
    interpolation of the input beam map (params). The output
    beam has shape (Npol, Npol, Nmodel, Nfreqs, Npix).

    This object also has a caching system for the weights
    and indicies of a bilinear interpolation of the beam 
    given the zen and az arrays.
    """
    def __init__(self, params, freqs, pixtype, npix, interp_mode='bilinear',
                 freq_mode='channel', f0=None, poly_kwargs={}):
        """
        Parameters
        ----------
        params : tensor
            The pixel beam map, of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix). Nfreqs
            can be interchanged with Ncoeff where Ncoeff
            is the number of coefficients used to model
            frequency response (see freq_mode)
        freqs : tensor
            frequency array of params [Hz]
        pixtype : str
            Pixelization type. options = ['healpix', 'other']
        npix : int
            Number of sky pixels in the beam
        interp_mode : str, optional
            Spatial interpolation method. ['nearest', 'bilinear']
        freq_mode : str, optional
            Frequency parameterization model.
            channel - each freq channel is an independent parameter
            poly - low-order polynomial basis
        f0 : float, optional
            Fiducial frequency for poly freq_mode
        poly_kwargs : dict, optional
            Optional kwargs to pass to utils.gen_poly_A
        """
        device = params.device
        super().__init__(pixtype, npix, interp_mode=interp_mode,
                         device=device)
        self.freqs = freqs
        self.params = params
        self.device = device
        self.freq_mode = freq_mode
        self.f0 = f0
        self.freq_ax = 3
        self.poly_kwargs = poly_kwargs

        self._setup()

        # construct _args for str repr
        self._args = dict(interp_mode=interp_mode, freq_mode=freq_mode)

    def _setup(self):
        if self.freq_mode == 'channel':
            assert self.params.shape[3] == len(self.freqs)
        elif self.freq_mode == 'poly':
            # get polynomial A matrix wrt freq
            if self.f0 is None:
                self.f0 = self.freqs.mean()
            self.dfreqs = (self.freqs - self.f0) / 1e6  # MHz
            Ndeg = self.params.shape[3]
            self.A = utils.gen_poly_A(self.dfreqs, Ndeg, device=self.device, **self.poly_kwargs)

    def push(self, device):
        """push params and other attrs to device"""
        self.params = utils.push(self.params, device)
        self.device = self.params.device
        self.freqs = self.freqs.to(device)
        for k, interp in self.interp_cache.items():
            self.interp_cache[k] = (interp[0], interp[1].to(device))

    def __call__(self, zen, az, *args):
        # interpolate or generate sky values
        b = self.interp(self.params, zen, az)

        # evaluate frequency values
        if self.freq_mode == 'poly':
            b = (self.params.transpose(-1, -2) @ self.A.T).transpose(-1, -2)

        return b


class GaussResponse:
    """
    A Gaussian beam representation for PixelBeam.

    .. code-block:: python

        R = GaussResponse(params)
        beam = R(zen, az, freqs)

    Recall azimuth is defined as the angle East of North.
    The output beam has shape (Npol, Npol, Nmodel, Nfreqs, Npix)
    """
    def __init__(self, params, freqs):
        """
        .. math::

            b = \exp(-0.5((l / sig_ew)**2 + (m / sig_ns)**2))

        Parameters
        ----------
        params : tensor
            Tensor of shape (2, Npol, Npol, Nmodel, Nfreqs). The tensors are
            the Gaussian sigma in EW and NS sky directions, respectively (0th axis),
            with units of the dimensionless image-plane l & m (azimuth sines & cosines).
        freqs : tensor
            frequency array of params [Hz]
        """
        self.params = params
        self.freqs = freqs
        self.device = self.params.device
        self.freq_mode = 'channel'
        self.freq_ax = 4
        assert self.params.shape[4] == len(self.freqs)

    def _setup(self):
        pass

    def __call__(self, zen, az, freqs):
        # get azimuth dependent sigma
        zen_rad, az_rad = zen * D2R, az * D2R
        srad = np.sin(zen_rad)
        srad[zen_rad > np.pi/2] = 1.0  # ensure sine_zen doesn't wrap around back to zero below horizon
        l = torch.as_tensor(srad * np.sin(az_rad), device=self.device)
        m = torch.as_tensor(srad * np.cos(az_rad), device=self.device)
        beam = torch.exp(-0.5 * ((l / self.params[0][..., None])**2 + (m / self.params[1][..., None])**2))
        return beam
  
    def push(self, device):
        """push params and other attrs to device"""
        self.params = utils.push(self.params, device)
        self.device = self.params.device


class AiryResponse:
    """
    An Airy Disk representation for PixelBeam.
    -- Note this is not differentiable!

    .. code-block:: python

        R = AiryResponse(params)
        beam = R(zen, az, freqs)

    Recall azimuth is defined as the angle East of North.
    The output beam has shape (Npol, Npol, Nmodel, Nfreqs, Npix).
    """
    def __init__(self, params, freq_ratio=1.0):
        """
        .. math::

            b = \left[\frac{2J_1(2\pi\nu a\sin\theta/c)}{2\pi\nu a\sin\theta/c}\right]^2

        Parameters
        ----------
        params : tensor
            Tensor of shape (2, Npol, Npol, Nmodel). The tensors are
            the aperture diameter [meters] in the EW and NS aperture directions,
            respectively (0th axis). In case one wants a single aperture diameter
            for both EW and NS directions, this is shape (1, Npol, Npol, Nmodel)
        freqs : tensor
            frequency array of params [Hz]
        freq_ratio : float, optional
            Multiplicative scalar acting on freqs before airy disk is
            evaluated. Makes the beam mimic a higher or lower frequency beam.
        """
        assert not params.requires_grad, "AiryResponse not differentiable"
        self.params = params
        self.freq_ratio = 1.0
        self.device = self.params.device
        self.freq_mode = 'other'
        self.freq_ax = None

    def _setup(self):
        pass

    def __call__(self, zen, az, freqs):
        """
        Parameters
        ----------
        zen, az : array or tensor
            zenith and azimuth arrays [deg]
        freqs : array or tensor
            Frequency array [Hz]
        """
        # get azimuth dependent sigma
        Dew = self.params[0][..., None, None]
        Dns = self.params[1][..., None, None] if len(self.params) > 1 else None
        beam = airy_disk(zen * D2R, az * D2R, Dew, freqs, Dns, self.freq_ratio)
        return torch.as_tensor(beam, device=self.device)
  
    def push(self, device):
        """push params and other attrs to device"""
        self.params = utils.push(self.params, device)
        self.device = self.params.device


class YlmResponse(PixelResponse):
    """
    A spherical harmonic representation for PixelBeam,
    mapping a_lm to pixel space. Adopts a polynomial
    basis across frequency in units of MHz.

    .. code-block:: python

        R = YlmResponse(params, l, m, **kwargs)
        beam = R(zen, az, freqs)

    Warning: if mode = 'interpolate' and parameter = True,
    you need to clear_beam() after every backwards call,
    otherwise the graph of the cached beam is freed
    and you get a RunTimeError.
    The output beam has shape (Npol, Npol, Nmodel, Nfreqs, Npix)
    """ 
    def __init__(self, params, l, m, freqs, mode='generate',
                 interp_mode='bilinear', interp_angs=None,
                 powerbeam=True, freq_mode='channel', f0=None,
                 Ylm_kwargs={}):
        """
        Note that for 'interpolate' mode, you must first call the object with a healpix map
        of zen, az (i.e. theta, phi) to "set" the beam, which is then interpolated with later
        calls of (zen, az) that may or not be of healpix ordering.

        Warning: if mode = 'interpolate' and parameter = True,
        you need to clear_beam() after every backwards call,
        otherwise the graph of the cached beam is freed
        and you get a RunTimeError.

        Parameters
        ----------
        params : tensor
            Holds a_lm coefficients of shape (Npol, Npol, Nmodel, Ndeg, Ncoeff)
            Ncoeff is the number of lm modes.
            Ndeg is the number of polynomial degree terms wrt freqs.
            Nmodel is the number of unique antenna models.
        l, m : ndarrays
            The l and m modes of params.
        freqs : tensor
            frequency array of params [Hz]
        mode : str, options=['generate', 'interpolate']
            generate - generate exact Y_lm given zen, az for each call
            interpolate - interpolate existing beam onto zen, az. See warning
            in docstring above.
        interp_mode : str, optional
            If mode is interpolate, this is the kind (see utils.PixelInterp)
        interp_angs : 2-tuple
            This is the initial (zen, az) [deg] to evaluate the Y_lm(zen, az) * a_lm
            transformation, which is then set on the object and interpolated for future
            calls. Must be a healpix map. Only needed if mode is 'interpolate'
        powerbeam : bool, optional
            If True, beam is a baseline beam, purely real and non-negative. Else,
            beam is complex antenna farfield beam.
        freq_mode : str, optional
            Frequency parameterization ['channel', 'poly']
        f0 : float, optional
            fiducial frequency [Hz] for poly freq_mode

        Notes
        -----
        Y_cache : a cache for Y_lm matrices (Npix, Ncoeff)
        ang_cache : a cache for (zen, az) arrays [deg]
        """
        self.npix = interp_angs[0].shape[-1] if interp_angs is not None else None
        super(YlmResponse, self).__init__(params, freqs, 'healpix', self.npix,
                                          interp_mode=interp_mode,
                                          freq_mode=freq_mode, f0=f0)
        self.l, self.m = l, m
        self.neg_m = np.any(m < 0)
        self.powerbeam = powerbeam
        self.Ylm_cache = {}
        self.ang_cache = {}
        self.mode = mode
        self.interp_angs = interp_angs
        self.beam_cache = None
        self.freq_ax = 3
        self.Ylm_kwargs = Ylm_kwargs

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

    def load_cache(self, fname):
        """
        Load an .npz file with Ylm and ang tensors
        and insert into the cache

        Parameters
        ----------
        fname : str
            Filepath to .npz file with Ylm and angs keys.
            Ylm is tensor output from utils.gen_sph2pix and
            angs is (zen, az) tensors [deg]
        """
        with np.load(fname) as f:
            Ylm = f['Ylm'].item()
            zen, az = f['angs'].item()
        # compute hash
        h = utils.ang_hash(zen)
        self.Ylm_cache[h] = Ylm
        self.ang_cache[h] = zen, az

    def set_beam(self, beam, zen, az, freqs):
        self.beam_cache = dict(beam=beam, zen=zen, az=az, freqs=freqs)

    def clear_beam(self):
        self.beam_cache = None

    def forward(self, zen, az, freqs):
        """
        Perform the mapping from a_lm to pixel
        space, in addition to possible transformation
        over frequency.

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
        if self.freq_mode == 'channel':
            p = self.params
        elif self.freq_mode == 'poly':
            # first do fast dot product along frequency axis
            p = (self.params.transpose(-1, -2) @ self.A.T).transpose(-1, -2)

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

            # interpolate the beam at the desired sky locations
            beam = self.interp(self.beam_cache['beam'], zen, az)

        return beam

    def push(self, device):
        """push params and other attrs to device"""
        self.params = utils.push(self.params, device)
        self.device = self.params.device
        for k, Ylm in self.Ylm_cache.items():
            self.Ylm_cache[k] = Ylm.to(device)
        if self.beam_cache is not None:
            self.beam_cache['beam'] = utils.push(self.beam_cache['beam'], device)


class AlmBeam(torch.nn.Module):
    """
    A beam model representation in
    spherical harmonic space.
    This takes sky models of 'alm' kind.
    """
    def __init__(self, params, freqs, parameter=True, polmode='1pol',
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
