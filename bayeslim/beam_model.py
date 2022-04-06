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
    Handles antenna primary beam models for a
    discrete pixelized or point source sky model.
    This relates the directional and frequency
    response of the sky to the "perceived" sky for
    a baseline between antennas p and q

    .. math::

        B_{pq}(\hat{s}, \nu) = A_p B A_q^\ast

    Note that this can be thought of as a direction-
    dependent Jones term which takes the form

    .. math::

        J_a = \left[\begin{array}{cc}J_e^\alpha & J_e^\delta\\\\
                    J_n^\alpha & J_n^\delta\end{array}\right]

    where e and n index the East and North feed polarizations.
    The amplitude of the beam should be normalized to unity
    at boresight. Also, a single beam can be used for all
    antennas, or one can fit for per-antenna models.

    There are three distinct operating modes, which are determined
    based on the shape of params and the shape of the input sky_comp.
        '1pol' : One auto-feed polarization
            powerbeam = True  : (Npol=1, Nvec=1, Nmodel=1, ...)
            powerbeam = False : (Npol=1, Nvec=2, ...)
        '2pol' : Two auto-feed polarization
            powerbeam = True  : (Npol=2, Nvec=1, Nmodel=1, ...)
        '4pol' : All four auto and cross pols
            powerbeam = False : (Npol=2 Nvec=2, ...)
    """
    def __init__(self, params, freqs, ant2beam=None, response=None,
                 response_args=(), response_kwargs={},
                 parameter=True, pol=None, powerbeam=True,
                 fov=180, name=None):
        """
        A generic beam model evaluated on the pixelized sky

        Parameters
        ----------
        params : tensor
            Initial beam parameterization, matched to the adopted
            response function R. By default, params should be a tensor
            of shape (Npol, Nvec, Nmodel, Nfreqs, Npix), where Npix are
            the sky pixels where the beam is defined, Nmodel are the number
            of unique antenna beam models, Npol are the number of 
            feed polarizations and Nvec is the number of electric
            field vectors (Nvec = 1 for Stokes I, 2 for full-Stokes).
            If params is complex, it should have an additional dim of
            (..., 2) via utils.viewreal(), in which case, set its response
            function comp_params=True.
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
            (Npol, Nvec, Nmodel, Nfreqs, Npix). Note that unlike
            sky_model, params are stored on both the Model and
            the Response function with the same pointer! Hence the
            different intialization API.
        response_args : tuple, optional
            arguments for instantiating response object
            after passing params as first argument
        response_kwargs : tuple, optional
            Keyword arguments for instantiating response object
        parameter : bool, optional
            If True, fit for params (default), otherwise
            keep it fixed to its input value
        pol : str, optional
            If Npol = 1, this is its polarization, either ['e', 'n'].
            If Npol = 2, params ordering must be ['e', 'n'], so this
            attribute is ignored.
        powerbeam : bool, optional
            If True, take the antenna beam to be a real-valued, baseline
            "power" beam, or psky = beam * sky. Only valid under strict
            assumption of identical antenna beams and a Stokes I sky model.
            i.e. Nmodel = Nvec = Nstokes = 1.
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
                                   nside=utils.healpy.npix2nside(params.shape[-1]))
        else:
            self.R = response(*response_args, **response_kwargs)

        self.powerbeam = powerbeam
        if hasattr(self.R, 'powerbeam'):
            assert self.powerbeam == self.R.powerbeam
        self.Npol = params.shape[0]
        self.Nvec = params.shape[1]
        self.Nmodel = params.shape[2]
        if self.powerbeam:
            assert self.Nmodel == self.Nvec == 1
        else:
            assert self.Nvec == 2, "Jones matrix must be Npol x 2"
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.fov = fov
        self.pol = pol
        if ant2beam is None:
            assert params.shape[2] == 1, "only 1 model for default ant2beam"
            self.ant2beam = utils.SimpleIndex()

        # construct _args for str repr
        self._args = dict(powerbeam=powerbeam, fov=fov, Npol=self.Npol, Nmodel=self.Nmodel)
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
            (Npol, Nvec, Nmodel, Nfreqs, Npix)
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
            Holds the beam response for the first antenna in a
            baseline, with shape (Npol, Nvec, Nfreqs, Nsources).
        sky : tensor
            sky coherency matrix (Nvec, Nvec, Nfreqs, Nsources)
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

        # multiply in the beam(s) depending on polmode
        if self.Npol == 1:
            # only one feed polarization: 1-pol mode
            if self.Nvec == 1:
                # only stokes I
                assert sky.shape[:2] == (1, 1)
                # direct multiply
                if self.powerbeam:
                    psky = beam1 * sky
                else:
                    psky = beam1 * sky * beam2.conj()
            else:
                # full stokes
                assert sky.shape[:2] == (2, 2)
                psky = torch.einsum("ab...,bc...,dc...->ad...", beam1, sky, beam2.conj())
        else:
            # two feed polarizations
            if self.powerbeam:
                # this is 2-pol mode, a simplified version of the full 4-pol mode
                assert self.Nvec == 1
                assert sky.shape[:2] == (1, 1)
                psky = torch.zeros(2, 2, *sky.shape[2:], dtype=sky.dtype, device=self.device)
                psky[0, 0] = beam1[0, 0] * sky[0, 0]
                psky[1, 1] = beam1[1, 0] * sky[0, 0]

            else:
                # this is 4-pol mode
                assert not self.powerbeam
                assert sky.shape[:2] == (2, 2)
                psky = torch.einsum("ab...,bc...,dc...->ad...", beam1, sky, beam2.conj())

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
                                       store=False)
            zen = utils.colat2lat(alt, deg=True)

            # evaluate beam
            beam, cut, zen, az = self.gen_beam(zen, az, prior_cache=prior_cache)
            sky = sky_comp['sky'][..., cut]
            alt = alt[cut]

            # iterate over baselines
            shape = sky.shape
            psky = torch.zeros((self.Npol, self.Npol) + (len(modelpairs),) + shape[2:],
                               dtype=sky.dtype, device=self.device)
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
                    out_params = self.R.beam_cache

            self._eval_prior(prior_cache, inp_params, out_params)

    def freq_interp(self, freqs, kind='linear'):
        """
        Interpolate params onto new set of frequencies
        if freq_mode is channel. If freq_mode is
        linear, powerlaw or other, just update response frequencies

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

        R = PixelResponse(freqs, pixtype, **kwargs)
        beam = R(params, zen, az, freqs)

    where zen, az are the zenith and azimuth angles [deg]
    to evaluate the beam, computed using nearest or bilinear
    interpolation of the input beam map (params). The output
    beam has shape (Npol, Nvec, Nmodel, Nfreqs, Npix).

    This object also has a caching system for the weights
    and indicies of a bilinear interpolation of the beam 
    given the zen and az arrays.

    Warning: if mode = 'interpolate' and parameter = True,
    you need to clear_beam() after every backwards call,
    otherwise the graph of the cached beam is freed
    and you get a RunTimeError.

    Note that this also supports the multiplication of a
    DD polarization rotation matrix, which transforms
    the polarized response of the beam from TOPO spherical
    coordinates to Equatorial coordaintes, i.e. from

    .. math::

        J_{\phi\theta} = \left[\begin{array}{cc}
        J_e^\phi & J_e^\theta \\\\
        J_n^\phi & J_n^\theta \end{array}\right]

    to

    .. math::

        J_{\alpha\delta} = \left[\begin{array}{cc}
        J_e^\alpha & J_e^\delta \\\\
        J_n^\alpha & J_n^\delta \end{array}\right]

    via

    .. math::

        J_{\alpha\delta} = J_{\phi\theta} R_\chi

    where the rotation matrix is DD and is derived in
    BayesLIM Memo 1.
    """
    def __init__(self, freqs, pixtype, comp_params=False, interp_mode='nearest',
                 Nnn=4, theta=None, phi=None, freq_mode='channel', nside=None,
                 device=None, log=False, f0=None, Ndeg=None,
                 freq_kwargs={}, powerbeam=True, Rchi=None):
        """
        Parameters
        ----------
        freqs : tensor
            frequency array of params [Hz]
        pixtype : str
            Pixelization type. options = ['healpix', 'other']
        comp_params : bool, optional
            If True, cast params to complex via utils.viewcomp
        interp_mode : str, optional
            Spatial interpolation method. ['nearest']
        Nnn : int, optional
            Nearest neighbors for interpolation.
        theta, phi : array_like, optional
            Co-latitude and azimuth arrays [deg] of
            input params if pixtype is 'other'
        nside : int, optional
            nside of healpix map if pixtype is healpix
        freq_mode : str, optional
            Frequency parameterization model.
            channel - each freq channel is an independent parameter
            linear - linear mapping
        device : str, optional
            Device to put intermediary products on
        log : bool, optional
            If True, assume params is logged and take
            exp before returning.
        freq_kwargs : dict, optional
            Optional kwargs to pass to utils.gen_linear_A
        powerbeam : bool, optional
            If True treat beam as non-negative and real-valued.
        Rchi : tensor, optional
            Polarization rotation matrix, rotating polarized Jones
            matrix from J phi theta to J alpha delta (see Memo 1),
            should be shape (2, 2, Npix) where Npix is the spatial
            size of the pixelized beam cache, i.e. len(theta)
        """
        super().__init__(pixtype, interp_mode=interp_mode, Nnn=Nnn, nside=nside,
                         device=device, theta=theta, phi=phi)
        self.powerbeam = powerbeam
        self.freqs = freqs
        self.comp_params = comp_params
        assert isinstance(comp_params, bool)
        self.device = device
        self.log = log
        self.freq_mode = freq_mode
        self.freq_ax = 3
        self.freq_kwargs = freq_kwargs
        self.Rchi = Rchi
        self.clear_beam()

        self._setup()

        # construct _args for str repr
        self._args = dict(interp_mode=interp_mode, Nnn=Nnn, freq_mode=freq_mode)

    def _setup(self):
        if self.freq_mode == 'channel':
            pass
        elif self.freq_mode == 'linear':
            # get A matrix wrt freq
            freq_kwargs = copy.deepcopy(self.freq_kwargs)
            freq_kwargs['x'] = self.freqs
            if 'x0' not in freq_kwargs:
                freq_kwargs['x0'] = freq_kwargs.get("f0", None)
            assert 'linear_mode' in self.freq_kwargs, "must specify linear_mode"
            linear_mode = freq_kwargs.pop('linear_mode')
            dtype = utils._cfloat() if self.comp_params else utils._float()
            self.A = utils.gen_linear_A(linear_mode, device=self.device, dtype=dtype,
                                        **freq_kwargs)

    def push(self, device):
        """push attrs to device"""
        # call PixInterp push for cache and self.device
        super().push(device)
        # other attrs
        self.freqs = self.freqs.to(device)
        if self.freq_mode == 'linear':
            self.A = self.A.to(device)

    def __call__(self, params, zen, az, *args):
        # cast to complex if needed
        if self.comp_params:
            params = utils.viewcomp(params)

        # set beam cache if it doesn't exist
        if self.beam_cache is None:
            # pass to device
            if utils.device(params.device) != utils.device(self.device):
                params = params.to(self.device)

            # pass through frequency response
            if self.freq_mode == 'linear':
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

        # apply polarization rotation if desired
        b = self.apply_Rchi(b)

        return b

    def clear_beam(self):
        self.beam_cache = None

    def apply_Rchi(self, beam):
        """
        Apply DD polarization rotation matrix to
        pixel beam model. beam must have
        Nvec == 2.

        Parameters
        ----------
        beam : tensor
            Pixelized beam tensor of shape
            (Npol, Nvec, Nmodel, Nfreqs, Npix)

        Returns
        -------
        tensor
        """
        if self.Rchi is None:
            return beam
        assert self.Rchi.shape[-1] == beam.shape[-1]
        assert self.beam.shape[1] == 2, "Nvec must be 2"
        return torch.einsum("ij...l,jkl->ik...l", beam, self.Rchi)


class GaussResponse:
    """
    A Gaussian beam representation for PixelBeam.

    .. code-block:: python

        R = GaussResponse()
        beam = R(params, zen, az, freqs)

    Recall azimuth is defined as the angle East of North.

    The input params should have shape (Npol, Nvec, Nmodel, Nfreqs, 2).
    The tensors are the Gaussian sigma in EW and NS sky directions,
    respectively (last axis), with units of the dimensionless image-plane
    l & m (azimuth sines & cosines).
    The output beam has shape (Npol, Nvec, Nmodel, Nfreqs, Npix)
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

    params has shape (Npol, Nvec, Nmodel, 1, 2). The tensors are
    the aperture diameter [meters] in the EW and NS aperture directions,
    respectively (last axis). The second-to-last axis is an empty slot
    for frequency broadcasting. In case one wants a single aperture diameter
    for both EW and NS directions, this is shape (Npol, Nvec, Nmodel, 1, 1)

    The output beam has shape (Npol, Nvec, Nmodel, Nfreqs, Npix).
    """
    def __init__(self, freq_ratio=1.0, powerbeam=True):
        """
        .. math::

            b = \left[\frac{2J_1(2\pi\nu a\sin\theta/c)}{2\pi\nu a\sin\theta/c}\right]^2

        Parameters
        ----------
        freq_ratio : float, optional
            Multiplicative scalar acting on freqs before airy disk is
            evaluated. Makes the beam mimic a higher or lower frequency beam.
        powerbeam : bool, optional
            If True, treat this as a squared, "baseline beam"
            otherwise treat this as a per-antenna beam (unsquared)
        """
        self.freq_ratio = freq_ratio
        self.freq_mode = 'other'
        self.freq_ax = None
        self.powerbeam = powerbeam

    def _setup(self):
        pass

    def __call__(self, params, zen, az, freqs):
        """
        Parameters
        ----------
        params : tensor
            parameter tensor of shape (Npol, Nvec, Nmodel, 2)
        zen, az : array or tensor
            zenith and azimuth arrays [deg]
        freqs : array or tensor
            Frequency array [Hz]
        """
        # get azimuth dependent sigma
        Dew = params[..., 0:1]
        Dns = params[..., 1:2] if params.shape[-1] > 1 else None
        beam = airy_disk(zen * D2R, az * D2R, Dew, freqs, Dns, self.freq_ratio,
                         square=self.powerbeam)
        return beam
  
    def push(self, device):
        pass


class YlmResponse(PixelResponse):
    """
    A spherical harmonic representation for PixelBeam,
    mapping a_lm to pixel space. Adopts a linear
    mapping across frequency in units of MHz.

    .. code-block:: python

        R = YlmResponse(l, m, **kwargs)
        beam = R(params, zen, az, freqs)

    Warning: if mode = 'interpolate' and parameter = True,
    you need to clear_beam() after every backwards call,
    otherwise the graph of the cached beam is freed
    and you get a RunTimeError.

    params holds a_lm coefficients of shape
    (Npol, Nvec, Nmodel, Ndeg, Ncoeff). Ncoeff is the number
    of lm modes. Ndeg is the number of linear mapping terms
    wrt freqs (or Nfreqs). Nmodel is the number of unique antenna models.

    The output beam has shape (Npol, Nvec, Nmodel, Nfreqs, Npix)
    """
    def __init__(self, l, m, freqs, pixtype='healpix', comp_params=False,
                 mode='interpolate', device=None, interp_mode='nearest',
                 Nnn=4, theta=None, phi=None, nside=None,
                 powerbeam=True, log=False, freq_mode='channel',
                 freq_kwargs={}, Ylm_kwargs={}, Rchi=None):
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
        pixtype : str, optional
            Beam pixelization type, ['healpix', 'other']
        comp_params : bool, optional
            Cast params to compelx if True.
        mode : str, options=['generate', 'interpolate']
            generate - generate exact Y_lm for each zen, az call. Slow and not recommended.
            interpolate - interpolate existing beam onto zen, az. See warning
            in docstring above.
        interp_mode : str, optional
            If mode is interpolate, this is the kind (see utils.PixInterp)
        Nnn : int, optional
            Number of nearest neighbors in interpolation (see utils.PixInterp)
        theta, phi : array_like, optional
            This is the initial (zen, az) [deg] to evaluate the Y_lm(zen, az) * a_lm
            transformation, which is then set on the object and interpolated for future
            calls. Only needed if mode is 'interpolate'
        nside : int, optional
            nside of healpix map if pixtype is healpix
        powerbeam : bool, optional
            If True, beam is a baseline beam, purely real and non-negative. Else,
            beam is complex antenna farfield beam.
        log : bool, optional
            If True assume params is logged and take exp(params) before returning.
        freq_mode : str, optional
            Frequency parameterization ['channel', 'linear']
        freq_kwargs : dict, optional
            Kwargs for generating linear modes, see utils.gen_linear_A()
        Ylm_kwargs : dict, optional
            Kwargs for generating Ylm modes
        Rchi : tensor, optional
            Polarization rotation matrix, rotating polarized Jones
            matrix from J phi theta to J alpha delta (see Memo 1),
            should be shape (2, 2, Npix) where Npix is the spatial
            size of the pixelized beam cache, i.e. len(theta)

        Notes
        -----
        Y_cache : a cache for Y_lm matrices (Npix, Ncoeff)
        ang_cache : a cache for (zen, az) arrays [deg]
        """
        ## TODO: enable pix_type other than healpix
        super(YlmResponse, self).__init__(freqs, pixtype, nside=nside,
                                          interp_mode=interp_mode, Nnn=Nnn,
                                          freq_mode=freq_mode,
                                          comp_params=comp_params, freq_kwargs=freq_kwargs,
                                          theta=theta, phi=phi, Rchi=Rchi)
        self.l, self.m = l, m
        dtype = utils._cfloat() if comp_params else utils._float()
        self.mult = torch.ones(len(m), dtype=dtype, device=device)
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
        self._args = dict(mode=mode, interp_mode=interp_mode, Nnn=Nnn, freq_mode=freq_mode)

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
            Ylm coefficients of shape (Npol, Nvec, Nmodel, Ndeg, Ncoeff)
        zen, az : ndarrays
            zenith and azimuth angles [deg]
        freqs : ndarray
            frequency bins [Hz]

        Returns
        -------
        beam : tensor
            pixelized beam on the sky
            of shape (Npol, Nvec, Nmodel, Nfreqs, Npix)
        """
        # pass to device
        if utils.device(params.device) != utils.device(self.device):
            params = params.to(self.device)

        if self.freq_mode == 'channel':
            p = params
        elif self.freq_mode == 'linear':
            # first do fast dot product along frequency axis
            p = (params.transpose(-1, -2) @ self.A.T).transpose(-1, -2)

        # generate Y matrix
        Ylm = self.get_Ylm(zen, az)

        # next do slower dot product over Ncoeff
        beam = (p * self.mult) @ Ylm

        if self.powerbeam:
            if torch.is_complex(beam):
                beam = torch.real(beam)
            #beam = torch.abs(beam)

        if self.log:
            beam = torch.exp(beam)

        if self.mode != 'generate':
            # apply polarization rotation to beam_cache
            beam = self.apply_Rchi(beam)

        return beam

    def __call__(self, params, zen, az, freqs):
        # cast to complex if needed
        if self.comp_params and torch.is_complex(params) is False:
            params = utils.viewcomp(params)

        # for generate mode, forward model the beam exactly at zen, az
        if self.mode == 'generate':
            beam = self.forward(params, zen, az, freqs)

        # otherwise interpolate the pre-forwarded beam in beam_cache at zen, az
        elif self.mode == 'interpolate':
            if self.beam_cache is None:
                # beam must first be forwarded at theta and phi
                beam = self.forward(params, self.theta, self.phi, freqs)
                # now cache it for future calls
                self.beam_cache = beam

            # interpolate the beam at the desired sky locations
            beam = self.interp(self.beam_cache, zen, az)

        return beam


def __call__(skyelf, params, zen, az, *args):
    # cast to complex if needed
    if self.comp_params:
        params = utils.viewcomp(params)

    # set beam cache if it doesn't exist
    if self.beam_cache is None:
        # pass to device
        if utils.device(params.device) != utils.device(self.device):
            params = params.to(self.device)

        # pass through frequency response
        if self.freq_mode == 'linear':
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


def airy_disk(zen, az, Dew, freqs, Dns=None, freq_ratio=1.0, square=True):
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
        Effective diameter [m] of aperture along the EW direction
    freqs : ndarray
        Frequency bins [Hz]
    Dns : float or array, optional
        Effective diameter [m] of aperture along the NS direction
    freq_ratio : float, optional
        Optional scalar to multiply frequencies by before
        evaluating airy disk. Can make the beam look like a
        lower or higher frequency beam
        (i.e. have a smaller or wider main lobe)
    square : bool, optional
        If True, square the output, otherwise don't
        square it.

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
    xvals = diameter * mod.sin(zen) * np.pi * freqs.reshape(-1, 1) * freq_ratio / 2.99792458e8
    # add a small value to handle x=0: introduces error on level of 1e-10
    xvals += 1e-10
    beam = 2.0 * scispecial.j1(xvals) / xvals
    if square:
        beam = beam**2

    return beam


# define the transformation from eq to xyz
def R_eq_to_xyz(alpha, delta):
    """
    Expresses Equatorial alpha and delta unit vectors,
    each at location (alpha, delta), in terms of xyz
    unit vectors. Dotted into a alpha, delta unit
    vectors [1, 0], [0, 1], this is a rotation matrix.

    Parameters
    ----------
    alpha : ndarray
        Right ascension [rad]
    delta : ndarray
        Declination [rad]
        Note this is an altitude, not polar angle!

    Returns
    -------
    Req2xyz : ndarray
        Of shape (3, 2)
    """
    ## Depending on whether you use declination as an altitude (standard) or polar unit vector, you can negate
    ## the second column, however, this leads to R_chi having a det of 1 or -1 (rotation / reflection).
    ## not sure which is most appropriate, to study...
    return np.array([[-np.sin(alpha), np.cos(alpha)*np.sin(delta)],#, np.cos(alpha)*np.cos(delta)],
                     [np.cos(alpha), np.sin(alpha)*np.sin(delta)],# np.sin(alpha)*np.cos(delta)],
                     [np.zeros_like(alpha), -np.cos(delta)],# np.sin(delta)]
                     ])

def R_beta(beta):
    """
    Rotation matrix from xyz to XYZ by angle beta
    about hat(y) (rotate in the x-z plane)

    Parameters
    ----------
    beta : float
        Angle [rad]

    Returns
    -------
    R_beta : ndarray
        3x3 rotation matrix
    """
    # define the transformation from xyz to XYZ
    return np.array([[np.cos(beta), 0, np.sin(beta)],
                     [0, 1, 0],
                     [-np.sin(beta), 0, np.cos(beta)]])

def R_XYZ_to_top(phi, theta):
    """
    Project XYZ unit vectors onto topocentric
    unit vectors, phi, theta.

    Parameters
    ----------
    phi : ndarray
        Azimuth angle [rad]
    theta : ndarray
        Polar (zenith) angle [rad]

    Returns
    -------
    R_xyz2top : ndarray
        2x3 projection matrix, projecting from
        XYZ to (phi,theta)
    """
    return np.array([[-np.sin(phi), np.cos(phi), np.zeros_like(phi)],
                     [np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)]])

def R_chi(alpha, delta, beta):
    """
    Compute rotation matrix for rotating equatorial basis vectors
    [e_a, e_d] to spherical basis vectors [e_phi, e_theta]
    (defined in the xyz frame)
    
    Parameters
    ----------
    alpha : ndarray
        right ascension [rad]
    delta : ndarray
        declination [rad]
    beta : float
        angle of observer latitude from north pole [rad]

    Returns
    -------
    Rchi : tensor
        (2, 2, Npix) rotation matrix for each sky
        pixel
    """
    # get the pointing vector in xyz coords
    s_hat_xyz = np.array([np.cos(delta)*np.cos(alpha), np.cos(delta)*np.sin(alpha), np.sin(delta)])

    # (3, 2, Nsamples)
    R_eq = R_eq_to_xyz(alpha, delta)

    # (3, 3)
    R_b = R_beta(beta)
    s_hat_XYZ = R_b @ s_hat_xyz

    theta = np.arctan2(np.sqrt(s_hat_XYZ[0]**2+s_hat_XYZ[1]**2), s_hat_XYZ[2])
    phi = np.arctan2(s_hat_XYZ[1], s_hat_XYZ[0])

    # (2, 3, Nsamples)
    R_top = R_XYZ_to_top(phi, theta)

    # get total transformation matrix
    R_chi = np.einsum('ijk,jl,lmk->imk', R_top, R_b, R_eq)

    return R_chi
