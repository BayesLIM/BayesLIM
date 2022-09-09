"""
Module for primary beam modeling
"""
import torch
from torch.nn import Parameter
import numpy as np
import warnings
from scipy import interpolate
import copy

from . import utils, linalg, special


D2R = utils.D2R


class PixelBeam(utils.Module):
    """
    Handles antenna primary beam models for a
    discrete pixelized or point source sky model.
    Note: this class does not require the beam to
    be pixelated, but the sky to be pixelated.
    Analytic beam response functions can be used.

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
    def __init__(self, params, freqs, R=None, ant2beam=None,
                 parameter=True, pol=None, powerbeam=True,
                 fov=180, name=None, p0=None, offset=None):
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
            index in the Nmodel dimension from beam.
            E.g. {10: 0, 11: 0, 12: 0} for 3-antennas [10, 11, 12] with
            1 shared beam model or {10: 0, 11: 1, 12: 2} for 3-antennas
            [10, 11, 12] with different 3 beam models.
            Default is all ants map to index 0.
        R : PixelResponse or other, optional
            A response function mapping the beam params
            to the requested beam values at zenith and azimuth
            angles and at each frequency.
            Must have a signature of (params, zen, az, freqs)
            and returns a beam of shape
            (Npol, Nvec, Nmodel, Nfreqs, Npix).
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
        p0 : tensor, optional
            A fixed "prior starting model" for params, which is summed with
            params before passing through the response function.
            This recasts params as a delta perturbation from p0.
            Must have same shape as params.
        offset : tuple, optional
            A small-angle pointing offset in (theta_x, theta_y)
            where theta_x is a rotation about x-hat vector [rad]
        """
        super().__init__(name=name)
        self.params = params
        self.p0 = p0
        self.device = self.params.device
        if parameter:
            self.params = torch.nn.Parameter(self.params)

        if R is None:
            # assumes uniform beam response
            self.R = UniformResponse()
        else:
            self.R = R

        self.powerbeam = powerbeam
        if hasattr(self.R, 'powerbeam'):
            assert self.powerbeam == self.R.powerbeam
        self.Npol = params.shape[0]
        self.Nvec = params.shape[1]
        self.Nmodel = params.shape[2]
        if self.powerbeam:
            assert self.Nmodel == self.Nvec == 1
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.fov = fov
        self.pol = pol
        if ant2beam is None:
            assert params.shape[2] == 1, "only 1 model for default ant2beam"
            self.ant2beam = utils.SimpleIndex()

        # set pointing offset
        offset = (0, 0) if offset is None else offset
        self.set_pointing_offset(*offset)

        # caching
        self.clear_cache()

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
        self.freqs = self.freqs.to(device)
        if self.p0 is not None:
            self.p0 = self.p0.to(device)
        # push prior functions
        if self.priors_inp_params is not None:
            for pr in self.priors_inp_params:
                pr.push(device)
        if self.priors_out_params is not None:
            for pr in self.priors_out_params:
                pr.push(device)

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
            FOV integer index along Npix axis
        zen, az : tensor
            truncated zen and az tensors
        """
        # enact fov cut
        if self.fov < 360:
            cut = torch.where(zen < self.fov / 2)[0]
        else:
            cut = slice(None)
        zen, az = zen[cut], az[cut]

        # add prior model for params
        if self.p0 is None:
            p = self.params
        else:
            p = self.params + self.p0

        # introduce pointing offset if needed
        theta_x = self.theta_x if hasattr(self, 'theta_x') else 0
        theta_y = self.theta_y if hasattr(self, 'theta_y') else 0
        if theta_x > 0 or theta_y > 0:
            new_zen, new_az = pointing_offset(utils.tensor2numpy(zen)*np.pi/180,
                                              utils.tensor2numpy(az)*np.pi/180,
                                              theta_x, theta_y)
            if isinstance(zen, torch.Tensor):
                new_zen = torch.as_tensor(new_zen) * 180 / np.pi
                new_az = torch.as_tensor(new_az) * 180 / np.pi
        else:
            new_zen, new_az = zen, az

        # evaluate the beam!
        beam = self.R(p, new_zen, new_az, self.freqs)

        # register gradient hook if desired (only works for interpolate mode)
        if hasattr(self, '_hook_response_grad') and self._hook_response_grad:
            if hasattr(self.R, 'beam_cache') and self.R.beam_cache is not None:
                if self.R.beam_cache.requires_grad:
                    self.R.beam_cache.register_hook(self.response_grad_hook)

        # evaluate prior
        self.eval_prior(prior_cache)

        return beam, cut, zen, az

    def apply_beam(self, beam, bls, sky):
        """
        Apply a beam matrix to a pixel representation
        of the sky.

        Parameters
        ----------
        beam : tensor
            Holds the full beam output from gen_beam()
            of shape (Npol, Nvec, Nmodel, Nfreqs, Nsources)
        bls : 2-tuple or list of tuples
            Specifies which baselines (antenna-integer pairs)
            to use in applying beam to sky.
            self.ant2beam maps antenna integers to beam model indices.
        sky : tensor
            sky coherency matrix (Nvec, Nvec, Nfreqs, Nsources)

        Returns
        -------
        psky : tensor
            perceived sky, having mutiplied beam with sky, of shape
            (Npol, Npol, Nbls, Nfreqs, Npix)
        """
        # get modelpairs from baseline(s)
        if not isinstance(bls, list):
            bls = [bls]
        bl2mp = {_bl: (self.ant2beam[_bl[0]], self.ant2beam[_bl[1]]) for _bl in bls}
        modelpairs = sorted(set(bl2mp.values()))
        Nmp = len(modelpairs)

        # move objects to device
        if utils.device(beam.device) != utils.device(self.device):
            beam = beam.to(self.device)
        if utils.device(sky.device) != utils.device(self.device):
            sky = sky.to(self.device)

        # expand beam to (Npol, Nvec, Nmodelpair, Nfreqs, Nsources)
        if Nmp == 1:
            # simple case that is just a slice
            p1 = modelpairs[0][0]
            beam1 = beam[:, :, p1:p1+1]
            if not self.powerbeam:
                p2 = modelpairs[0][1]
                beam2 = beam[:, :, p2:p2+1]
        else:
            # for multiple pairs need to index along Nmodel dimension
            ant1_idx = torch.as_tensor([mp[0] for mp in modelpairs], device=self.device)
            beam1 = torch.index_select(beam, 2, ant1_idx)
            if not self.powerbeam:
                ant2_idx = torch.as_tensor([mp[1] for mp in modelpairs], device=self.device)
                beam2 = torch.index_select(beam, 2, ant2_idx)

        # give sky an Nmodelpair dimension if needed
        if sky.ndim == 4:
            sky = sky[:, :, None]

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

        # expand Nmodelpair dimension to Nbls length
        if Nmp > 1:
            mp_idx = torch.as_tensor([modelpairs.index(bl2mp[_bl]) for _bl in bls])
            psky = torch.index_select(psky, 2, mp_idx)
        else:
            psky = psky.expand(psky.shape[:2] + (len(bls),) + psky.shape[3:])

        return psky

    def forward(self, sky_comp, telescope, time, bls, prior_cache=None, **kwargs):
        """
        Forward pass a single sky model through the beam
        at a single observation time.

        Parameters
        ----------
        sky_comp : MapData object
            Output of a SkyBase subclass
        telescope : TelescopeModel object
            A model of the telescope location
        time : float
            Observation time in Julian Date (e.g. 2458101.23456)
        bls : 2-tuple or list of 2-tuples
            A list of baselines (antenna integer pairs) to use in
            applying beam to sky. e.g. [(0, 1), (0, 10)].
            Note that the self.ant2beam dictionary controls
            which antenna integer maps to which beam model
            along the Nmodel dimension of self.params
        prior_cache : dict, optional
            Cache for storing computed priors as self.name

        Returns
        -------
        psky : dict
            Dictionary holding perceived sky data as 'sky' of shape
            (Npol, Npol, Nbls, Nfreqs, Nsources), where
            roughly psky = beam1 * sky * beam2. The FoV cut has
            been applied to psky as well as the angs vectors.
            'angs' holds ra,dec and 'altaz' holds alt,az [deg]
        """
        # get coords
        alt, az = telescope.eq2top(time, sky_comp.angs[0], sky_comp.angs[1],
                                   store=False)
        zen = utils.colat2lat(alt, deg=True)

        # evaluate beam
        beam, cut, zen, az = self.gen_beam(zen, az, prior_cache=prior_cache)
        sky = cut_sky_fov(sky_comp.data, cut)
        alt = alt[cut]

        # apply beam to sky to get perceived sky
        psky = self.apply_beam(beam, bls, sky)

        out_comp = {}
        out_comp['sky'] = psky
        out_comp['angs'] = sky_comp.angs[:, cut]
        out_comp['altaz'] = torch.vstack([alt, az])

        return out_comp

    def eval_prior(self, prior_cache, inp_params=None, out_params=None):
        """
        Evaluate prior on params (not params + p0) and R(params + p0).
        This overloads Module.eval_prior given non-standard API of
        PixelBeam. See utils.Module.eval_prior for more details.

        Parameters
        ----------
        prior_cache : dict
            Dictionary to hold computed prior, assigned as self.name
        inp_params, out_params : tensor, optional
            self.params and self.R(self.params+self.p0), respectively
        """
        # append to cache
        if prior_cache is not None and self.name not in prior_cache:
            # start starting log prior value
            prior_value = torch.as_tensor(0.0)

            # look for prior on inp_params
            if self.priors_inp_params is not None:
                # try to get inp_params if not provided
                if inp_params is None:
                    if hasattr(self, 'params'):
                        inp_params = self.params
                # iterate over priors
                for prior in self.priors_inp_params:
                    if prior is not None:
                        prior_value = prior_value + prior(inp_params)

            # look for prior on out_params
            if self.priors_out_params is not None:
                if out_params is None:
                    # we can evaluate prior on PixelResponse beam if mode is interpolate
                    if hasattr(self.R, 'beam_cache') and self.R.beam_cache is not None:
                        p = self.params if self.p0 is None else self.params + self.p0
                        self.R.set_beam_cache(p)
                        out_params = self.R.beam_cache
                # iterate over priors
                for prior in self.priors_out_params:
                    if prior is not None:
                        prior_value = prior_value + prior(out_params)

            prior_cache[self.name] = prior_value

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
        # only interpolate if new freqs don't match current freqs to 1 Hz
        if len(freqs) != len(self.freqs) or not np.isclose(self.freqs, freqs, atol=1.0).all():
            freqs = torch.as_tensor(freqs)
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

    def set_pointing_offset(self, theta_x=0, theta_y=0):
        """
        Set a small-angle pointing offset in the beam
        (non-differentiable). Note that the Euler rotations
        commute under small angle approximation.
        This offset is introduced when calling self.gen_beam(...)

        Note: if you are using PixInterp subclass for the
        PixelResponse class, you will want to model
        the beam out to a zenith angle that is buffered
        beyond fov/2 by theta_x & theta_y, such that the interpolation
        routines don't suffer from extrapolation at the fov boundaries.

        Parameters
        ----------
        theta_x : float, optional
            Angle to rotate beam about x-hat vector [rad]
        theta_y : float, optional
            Angle to rotate beam about y-hat vector [rad]
        """
        self.theta_x = theta_x
        self.theta_y = theta_y

    def set_sky_cut(self, zen, cut, device=None):
        """
        Insert a sky cut index array into the cache

        Parameters
        ----------
        zen : tensor
            zenith angle tensor for a sky model
        cut : tensor
            indexing tensor of that sky model given FOV
        device : str, optional
            Device to push cut
        """
        h = utils.arr_hash(zen)
        if h not in self.cache:
            self.cache[h] = cut.to(device)

    def query_cache(self, zen):
        """
        Query sky cut cache
        """
        h = utils.arr_hash(zen)
        if h in self.cache:
            return self.cache[h]

    def clear_cache(self):
        """clear the sky cut cache"""
        self.cache = {}

    def hook_response_grad(self, value=True):
        """
        Store gradient of response output as self.response_grad
        """
        self._hook_response_grad = value


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
    you need to clear_beam_cache() after every backwards call,
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
                 theta=None, phi=None, theta_grid=None, phi_grid=None,
                 freq_mode='channel', nside=None, device=None, log=False, freq_kwargs=None,
                 powerbeam=True, Rchi=None, interp_gpu=False, edge_alpha=None, edge_fov=180):
        """
        Parameters
        ----------
        freqs : tensor
            frequency array of params [Hz]
        pixtype : str
            Pixelization type. options = ['healpix', 'rect'].
            For healpix, pixel ordering is RING.
            For rect, pixel ordering should be
            x, y = meshgrid(phi_grid, theta_grid)
            x, y = x.ravel(), y.ravel()
        comp_params : bool, optional
            If True, cast params to complex via utils.viewcomp
        interp_mode : str, optional
            Spatial interpolation method for 'rect' pixtype.
            e.g. ['nearest', 'linear', 'quadratic', 'linear,quadratic']
            where mixed mode is for 'az,zen' respectively
        theta, phi : array_like, optional
            This is the (zen, az) [deg] of the beam
            See pixtype above for theta/phi ordering given 'rect' or 'healpix'.
            Note, this can contain additional points not specified by rect or healpix,
            (e.g. theta=0) but these additional points must come after the points
            required by the pixtype. 
        theta_grid, phi_grid : array_like, optional
            For interp_mode = 'rect', these are 1D float arrays (monotonically increasing)
            in zenith and azimuth [deg] that make-up the 2D grid to be interpolated against.
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
            Kwargs for frequency parameterization.
            'linear' : pass kwargs to utils.LinearModel
        powerbeam : bool, optional
            If True treat beam as non-negative and real-valued.
        Rchi : tensor, optional
            Polarization rotation matrix, rotating polarized Jones
            matrix from J phi theta to J alpha delta (see Memo 1),
            should be shape (2, 2, Npix) where Npix is the spatial
            size of the pixelized beam cache, i.e. len(theta)
        interp_gpu : bool, optional
            If True and pixtype is 'rect', use GPU when solving
            for pixel interpolation weights for speedup (PixInterp)
        edge_alpha : float, optional
            If not None, this is the alpha parameter of a tukey mask
            for tapering the edge of the beam out to fov / 2 zenith angle.
            Default (None) is no tapering.
        edge_fov : float, optional
            This is the fov parameter for the beam object in degrees,
            i.e. the diameter of the observable hemisphere
        """
        super().__init__(pixtype, interp_mode=interp_mode, nside=nside,
                         device=device, theta_grid=theta_grid, phi_grid=phi_grid,
                         gpu=interp_gpu)
        self.theta, self.phi = theta, phi
        self.powerbeam = powerbeam
        self.freqs = freqs
        self.comp_params = comp_params
        assert isinstance(comp_params, bool)
        self.device = device
        self.log = log
        self.freq_mode = freq_mode
        self.freq_ax = 3
        self.Rchi = Rchi
        self.clear_beam_cache()
        self.edge_alpha, self.edge_fov = edge_alpha, edge_fov

        freq_kwargs = freq_kwargs if freq_kwargs is not None else {}
        self._setup(**freq_kwargs)

        # construct _args for str repr
        self._args = dict(interp_mode=interp_mode, freq_mode=freq_mode)

    def _setup(self, **kwargs):
        if self.freq_mode == 'channel':
            pass
        elif self.freq_mode == 'linear':
            # get A matrix wrt freq
            kwgs = copy.deepcopy(kwargs)
            kwgs['x'] = self.freqs
            linear_mode = kwgs.pop('linear_mode')
            kwgs['dtype'] = utils._cfloat() if self.comp_params else utils._float()
            self.freq_LM = utils.LinearModel(linear_mode, dim=-2, device=self.device,
                                             **kwgs)

    def push(self, device):
        """push attrs to device"""
        # call PixInterp push for cache and self.device
        super().push(device)
        # other attrs
        self.freqs = self.freqs.to(device)
        if self.freq_mode == 'linear':
            self.freq_LM.push(device)

    def forward(self, params):
        """forward pixelized beam through frequency response"""
        if utils.device(params.device) != utils.device(self.device):
            params = params.to(self.device)
        # pass through frequency response
        if self.freq_mode == 'channel':
            p = params
        elif self.freq_mode == 'linear':
            p = self.freq_LM(params)

        # apply mask if necessary
        if hasattr(self, 'edge_alpha') and self.edge_alpha is not None:
            p *= beam_edge_taper(zen,
                                 alpha=self.edge_alpha,
                                 fov=self.edge_fov,
                                 device=p.device)

        return p

    def __call__(self, params, zen, az, *args):
        # cast to complex if needed
        if self.comp_params:
            params = utils.viewcomp(params)

        # set beam cache if it doesn't exist
        if self.beam_cache is None:
            self.set_beam_cache(params)

        # interpolate at sky values
        b = self.interp(self.beam_cache, zen, az)

        if torch.is_complex(b):
            b = torch.real(b)

#        if self.powerbeam:
#            ## TODO: replace abs with non-neg prior on beam?
#            b = torch.abs(b)

        if self.log:
            b = torch.exp(b)

        # apply polarization rotation if desired
        b = self.apply_Rchi(b)

        return b

    def clear_beam_cache(self):
        self.beam_cache = None

    def set_beam_cache(self, params):
        """
        Forward params through frequency response
        and set beam_cache
        """
        # forward params and set beam cache
        self.beam_cache = self.forward(params)

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
        # need to fix order of operations
        # either apply to beam_cache then interpolate
        # or interpolate then apply Rchi functional on theta, phi
        if self.Rchi is None:
            return beam
        raise NotImplementedError
        assert self.Rchi.shape[-1] == beam.shape[-1]
        assert beam.shape[1] == 2, "Nvec must be 2"
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
    def __init__(self, powerbeam=True):
        """
        .. math::

            b = \exp(-0.5((l / sig_ew)**2 + (m / sig_ns)**2))

        Parameters
        ----------
        freqs : tensor
            frequency array of params [Hz]
        powerbeam : bool, optional
            If True, treat this as a squared, "baseline beam"
            otherwise treat this as a per-antenna beam (unsquared)
        """
        self.freq_mode = 'channel'
        self.freq_ax = 3
        self.powerbeam = powerbeam

    def _setup(self):
        pass

    def __call__(self, params, zen, az, freqs):
        # get azimuth dependent sigma: azimuth is East of North (clockwise)
        zen_rad, az_rad = zen * D2R, az * D2R
        srad = np.sin(zen_rad)
        srad[zen_rad > np.pi/2] = 1.0  # ensure sine_zen doesn't wrap around back to zero below horizon
        l = torch.as_tensor(srad * np.sin(az_rad), device=params.device)
        m = torch.as_tensor(srad * np.cos(az_rad), device=params.device)
        beam = torch.exp(-0.5 * ((l / params[..., 0:1])**2 + (m / params[..., 1:2])**2))
        if not self.powerbeam:
            beam = torch.sqrt(beam)
        return beam
  
    def push(self, device):
        pass


class AiryResponse:
    """
    An Airy Disk representation for PixelBeam.

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
    def __init__(self, freq_ratio=1.0, powerbeam=True, brute_force=True, Ntau=100,
                 edge_alpha=None, edge_fov=180):
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
        brute_force : bool, optional
            If True (default) makes this differentiable.
            See airy_disk() for details.
        Ntau : int, optional
            Integral pixelization, see airy_disk() for details
        edge_alpha : float, optional
            If not None, this is the alpha parameter of a tukey mask
            for tapering the edge of the beam out to fov / 2 zenith angle.
            Default (None) is no tapering.
        edge_fov : float, optional
            This is the fov parameter for the beam object in degrees,
            i.e. the diameter of the observable hemisphere
        """
        self.freq_ratio = freq_ratio
        self.freq_mode = 'other'
        self.freq_ax = None
        self.powerbeam = powerbeam
        self.brute_force = brute_force
        self.Ntau = Ntau
        self.edge_alpha, self.edge_fov = edge_alpha, edge_fov

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
        if isinstance(az, torch.Tensor):
            if utils.device(params.device) != utils.device(az.device):
                az = az.to(params.device)
        if isinstance(zen, torch.Tensor):
            if utils.device(params.device) != utils.device(zen.device):
                zen = zen.to(params.device)
        beam = airy_disk(zen * D2R, az * D2R, Dew, freqs, Dns, self.freq_ratio,
                         square=self.powerbeam, brute_force=self.brute_force,
                         Ntau=self.Ntau)
        beam = torch.as_tensor(beam, device=params.device)

        # apply mask if necessary
        if hasattr(self, 'edge_alpha') and self.edge_alpha is not None:
            beam *= beam_edge_taper(zen,
                                    alpha=self.edge_alpha,
                                    fov=self.edge_fov,
                                    device=beam.device)

        return beam
  
    def push(self, device):
        pass


class UniformResponse:
    """
    A uniform beam response
    """
    def __init__(self, edge_alpha=None, edge_fov=180, device=None):
        self.edge_alpha, self.edge_fov = edge_alpha, edge_fov
        self.device = device

    def _setup(self):
        pass

    def __call__(self, params, zen, az, freqs):
        out = torch.ones(params.shape[:3] + (len(freqs), len(zen)),
                          dtype=utils._float(), device=self.device)

        # apply mask if necessary
        if hasattr(self, 'edge_alpha') and self.edge_alpha is not None:
            out *= beam_edge_taper(zen,
                                   alpha=self.edge_alpha,
                                   fov=self.edge_fov,
                                   device=out.device)
        return out

    def push(self, device):
        self.device = device


class YlmResponse(PixelResponse):
    """
    A spherical harmonic representation for PixelBeam,
    mapping params to pixel space. Adopts a linear
    mapping across frequency in units of MHz.
    For further compression, can adopt a polynomial mapping
    across degree l for fixed order m (lm_poly_setup).

    params should hold the a_lm coefficients of shape
    (Npol, Nvec, Nmodel, Ndeg, Ncoeff). Ncoeff is the number
    of lm modes. Ndeg is the number of linear mapping terms
    wrt freqs (or Nfreqs). Nmodel is the number of unique antenna models.
    The output beam has shape (Npol, Nvec, Nmodel, Nfreqs, Npix)

    .. code-block:: python

        R = YlmResponse(l, m, **kwargs)
        beam = R(params, zen, az, freqs)

    Warning: if mode = 'interpolate' and parameter = True,
    you need to clear_beam_cache() after every back-propagation,
    otherwise the graph of the cached beam is stale
    and you get a RunTimeError. Note this is performed
    automatically when using the rime_model.RIME object.
    """
    def __init__(self, l, m, freqs, pixtype='healpix', comp_params=False,
                 mode='interpolate', device=None, interp_mode='nearest',
                 theta=None, phi=None, theta_grid=None, phi_grid=None,
                 nside=None, powerbeam=True, log=False, freq_mode='channel',
                 freq_kwargs=None, Ylm_kwargs=None, Rchi=None, interp_gpu=False,
                 edge_alpha=None, edge_fov=180):
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
            Beam pixelization type, ['healpix', 'rect']. Only needed
            for 'interpolate' mode.
            For healpix, pixel ordering is RING. For rect,
            pixel ordering should be
            x, y = meshgrid(phi_grid, theta_grid)
            x, y = x.ravel(), y.ravel()
        comp_params : bool, optional
            Cast params to compelx if True.
        mode : str, options=['generate', 'interpolate']
            generate - generate exact Y_lm for each zen, az call. Slow and not recommended.
            interpolate - interpolate existing beam onto zen, az. See warning
            in docstring above.
        interp_mode : str, optional
            If mode is interpolate, this is the kind (see utils.PixInterp)
        theta, phi : array_like, optional
            Only needed if mode is 'interpolate'. This is the initial (zen, az) [deg]
            to evaluate the Y_lm(zen, az) * a_lm transformation, which is then set on
            the object and interpolated for future calls. 
            See pixtype above for theta/phi ordering given 'rect' or 'healpix'.
            Note, this can contain additional points not specified by rect or healpix,
            (e.g. theta=0) but these additional points must come after the points
            required by the pixtype. 
        theta_grid, phi_grid : array_like, optional
            For interp_mode = 'rect', these are 1D float arrays (monotonically increasing)
            in zenith and azimuth [deg] that make-up the 2D grid to be interpolated against.
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
        interp_gpu : bool, optional
            If True and pixtype is 'rect', use GPU when solving
            for pixel interpolation weights for speedup (PixInterp)
        edge_alpha : float, optional
            If not None, this is the alpha parameter of a tukey mask
            for tapering the edge of the beam out to fov / 2 zenith angle.
            Default (None) is no tapering.
        edge_fov : float, optional
            This is the fov parameter for the beam object in degrees,
            i.e. the diameter of the observable hemisphere

        Notes
        -----
        Y_cache : a cache for Y_lm matrices (Npix, Ncoeff)
        ang_cache : a cache for (zen, az) arrays [deg]
        """
        ## TODO: enable pix_type other than healpix
        super(YlmResponse, self).__init__(freqs, pixtype, nside=nside,
                                          interp_mode=interp_mode, theta=theta, phi=phi,
                                          freq_mode=freq_mode, comp_params=comp_params,
                                          freq_kwargs=freq_kwargs, Rchi=Rchi,
                                          theta_grid=theta_grid, phi_grid=phi_grid,
                                          interp_gpu=interp_gpu)
        self.l, self.m = l, m
        dtype = utils._cfloat() if comp_params else utils._float()
        self.powerbeam = powerbeam
        self.Ylm_cache = {}
        self.ang_cache = {}
        self.mode = mode
        self.beam_cache = None
        self.freq_ax = 3
        self.Ylm_kwargs = Ylm_kwargs if Ylm_kwargs is not None else {}
        self.device = device
        self.log = log
        self.edge_alpha, self.edge_fov = edge_alpha, edge_fov

        # default is no mapping across l
        self.lm_poly_setup()

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
        alm_mult : tensor
            Multiplication tensor to alm,
            if not stored return None
        """
        # get hash
        h = utils.arr_hash(zen)
        if h in self.Ylm_cache:
            Ylm = self.Ylm_cache[h]
            if isinstance(Ylm, (list, tuple)):
                Ylm, alm_mult = Ylm
            else:
                alm_mult = None
        else:
            # generate exact Y_lm, may take a while
            Ylm, norm, alm_mult = utils.gen_sph2pix(zen * D2R, az * D2R,
                                                    self.l, self.m,
                                                    device=self.device,
                                                    **self.Ylm_kwargs)
            # store it
            self.Ylm_cache[h] = (Ylm, alm_mult)
            self.ang_cache[h] = (zen, az)

        return Ylm, alm_mult

    def set_cache(self, Ylm, angs, alm_mult=None):
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
        alm_mult : tensor, optional
            multiply this (Nmodes,) tensor into alm tensor
            before forward pass
        """
        assert len(self.l) == len(Ylm)
        zen, az = angs
        h = utils.arr_hash(zen)
        self.Ylm_cache[h] = (Ylm, alm_mult)
        self.ang_cache[h] = (zen, az)

    def forward(self, params, zen, az, *args):
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

        Returns
        -------
        beam : tensor
            pixelized beam on the sky
            of shape (Npol, Nvec, Nmodel, Nfreqs, Npix)
        """
        # pass to device
        if utils.device(params.device) != utils.device(self.device):
            params = params.to(self.device)

        # first handle frequency axis
        if self.freq_mode == 'channel':
            p = params
        elif self.freq_mode == 'linear':
            # first do fast dot product along frequency axis
            p = self.freq_LM(params)

        # transform into a_lm space if using poly lm compression
        p = self.lm_poly_forward(p)

        # next handle lm axis
        Ylm, alm_mult = self.get_Ylm(zen, az)

        # multiply alm_mult
        if alm_mult is not None:
            p = p * alm_mult

        # next do slower dot product over Ncoeff
        beam = p @ Ylm

        if torch.is_complex(beam):
            beam = torch.real(beam)

        # apply edge mask if necessary
        if hasattr(self, 'edge_alpha') and self.edge_alpha is not None:
            beam *= beam_edge_taper(zen,
                                    alpha=self.edge_alpha,
                                    fov=self.edge_fov,
                                    device=beam.device)

        if self.log:
            beam = torch.exp(beam)

        if self.mode != 'generate':
            # apply polarization rotation to beam_cache
            beam = self.apply_Rchi(beam)

        return beam

    def __call__(self, params, zen, az, *args):
        # cast to complex if needed
        if self.comp_params and torch.is_complex(params) is False:
            params = utils.viewcomp(params)

        # for generate mode, forward model the beam exactly at zen, az
        if self.mode == 'generate':
            beam = self.forward(params, zen, az)

        # otherwise interpolate the pre-forwarded beam in beam_cache at zen, az
        elif self.mode == 'interpolate':
            if self.beam_cache is None:
                # forward beam at pre-designated points before interpolating
                self.set_beam_cache(params)

            # interpolate the beam at the desired sky locations
            beam = self.interp(self.beam_cache, zen, az)

        return beam

    def set_beam_cache(self, params):
        """
        Forward beam at pre-designated theta, phi
        values and store as self.beam_cache.
        Used for mode = 'interpolate'
        """
        # forward params at theta/phi and set beam cache
        self.beam_cache = self.forward(params, self.theta, self.phi)

    def push(self, device):
        """push attrs to device"""
        self.device = device
        super().push(device)
        for k, Ylm in self.Ylm_cache.items():
            if isinstance(Ylm, (tuple, list)):
                self.Ylm_cache[k] = (Ylm[0].to(device), Ylm[1].to(device))
            else:
                self.Ylm_cache[k] = Ylm.to(device)
        if self.beam_cache is not None:
            self.beam_cache = utils.push(self.beam_cache, device)
        if self._lm_poly:
            for key, (lm_inds, p_inds, A) in self.lm_poly_A.items():
                self.lm_poly_A[k] = (lm_inds, p_inds, A.to(device))

    # lm_poly_fit is experimental...
    def lm_poly_setup(self, lm_poly_kwargs=None):
        """
        Setup polynomial compression along degree l if desired using
        utils.gen_poly_A().
        This means the last dimension of the input params tensor
        are the weights to self.lm_poly_A matrices ordered
        according to increasing unique m value.
        Default (None) is not compression.

        Parameters
        ----------
        lm_poly_kwargs : dict, optional
            Kwargs for utils.gen_poly_A(), compressing along l
            for each fixed integer m. Default is no compression.
            Can also be a kwarg dictionary for each unique m integer.
            If Ndeg is fed as None for a particular m mode dictionary,
            then don't perform any compression for this m mode.
        """
        self._lm_poly_kwargs = lm_poly_kwargs if lm_poly_kwargs is not None else {}
        self._lm_poly = True if lm_poly_kwargs not in [None, {}] else False

        if self._lm_poly:
            # assert m is only integers
            munique = np.unique(self.m)
            assert np.isclose(munique, munique.astype(int)).all()

            # generate poly mappings for each integer m
            self.lm_poly_A = {}
            i = 0
            for m in munique:
                # get indices in self.m
                lm_inds = np.where(self.m == m)[0]

                # get kwarg dictionary for this m
                if m in lm_poly_kwargs:
                    kwargs = lm_poly_kwargs[m].copy()
                else:
                    kwargs = lm_poly_kwargs.copy()

                # get Ndeg: if fed as None then no compression for this m
                assert 'Ndeg' in kwargs
                Ndeg = kwargs.pop('Ndeg')
                compress = True
                if Ndeg is None:
                    compress = False
                    Ndeg = len(lm_inds)

                # get indices in eventual params tensor
                p_inds = i + np.arange(Ndeg)
                i += Ndeg

                if compress:
                    # generate A matrix
                    A = utils.gen_poly_A(self.l[lm_inds], Ndeg, **kwargs)
                    dtype = utils._float() if not self.comp_params else utils._cfloat()
                    A = A.to(dtype)
                else:
                    A = None

                self.lm_poly_A[m] = (lm_inds, p_inds, A)

            self.lm_poly_Ndeg = i

    def lm_poly_fit(self, params, fit_kwargs=None):
        """
        Take a normal a_lm params tensor of shape
        (..., Nalm) and use least squares to fit
        poly modes across l for each fixed m mode
        given the A matrices in self.lm_poly_A,
        see self.lm_poly_setup().
        kwargs are fed to linalg.least_squares().
        kwargs can also hold a kwarg dict for each
        key of self.lm_poly_A.
        Returns a tensor of shape (..., Npoly).
        """
        assert self._lm_poly
        out = torch.zeros(params.shape[:-1] + (self.lm_poly_Ndeg,),
                          dtype=params.dtype, device=params.device)

        # iterate over m mode A matrices
        for key, (lm_inds, p_inds, A) in self.lm_poly_A.items():
            # get kwargs
            fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
            if key in fit_kwargs:
                kwargs = fit_kwargs[key]
            else:
                kwargs = fit_kwargs

            # run fit
            if A is not None:
                xhat, _ = linalg.least_squares(A, params[..., lm_inds], dim=-1, **kwargs)
            else:
                xhat = params[..., lm_inds]

            # insert into out
            out[..., p_inds] = xhat

        return out

    def lm_poly_forward(self, params):
        """
        Forward pass a params tensor of shape
        (..., Npoly) to be (..., Nalm)
        using self.lm_poly_A. Assumes
        the last dimension of params is ordered
        according to increasing unique m value.
        """
        if not self._lm_poly:
            return params

        # generate empty output tensor
        out = torch.zeros(params.shape[:-1] + (len(self.l),),
                          dtype=params.dtype, device=params.device)
        
        # iterate over each m mode A matrix
        for key, (lm_inds, p_inds, A) in self.lm_poly_A.items():
            if A is not None:
                out[..., lm_inds] = params[..., p_inds] @ A.T
            else:
                out[..., lm_inds] = params[..., p_inds]

        return out


class AlmBeam(utils.Module):
    """
    A beam model representation in
    spherical harmonic space.
    """
    def __init__(self, freqs, parameter=True, polmode='1pol',
                 powerbeam=False):
        raise NotImplementedError


def airy_disk(zen, az, Dew, freqs, Dns=None, freq_ratio=1.0,
              square=True, Ntau=100, brute_force=True):
    """
    Generate a (asymmetric) airy disk function.

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
    Ntau : int, optional
        Bessel integral pixelization density
    brute_force : bool, optional
        If True (default) numerically integrate Bessel integral
        making this differentiable, otherwise use scipy routine.

    Returns
    -------
    ndarray
        Airy disk response at zen, az angles of
        shape (..., Nfreqs, Npix)
    """
    # determine if ndarray or tensor
    mod = np if isinstance(zen, np.ndarray) else torch

    # copy
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
    xvals = xvals.clip(1e-10)

    # evaluate beam
    beam = 2.0 * special.j1(xvals, Ntau=Ntau, brute_force=brute_force) / xvals
    if square:
        beam = beam**2

    return beam


def R_eq_to_xyz(alpha, delta):
    """
    Expresses equatorial alpha and delta unit vectors,
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


def rotation(beta, axis):
    """
    Rotation matrix in xyz by angle beta
    about hat(y) (rotate in the x-z plane)
    or hat(x) (rotate in the y-z plane)

    Parameters
    ----------
    beta : float
        Angle [radians]
    axis : str
        Axis about which to rotate

    Returns
    -------
    R : ndarray
        3x3 rotation matrix
    """
    if axis.lower() == 'x':
        R = np.array([[1.0, 0, 0],
                     [0, np.cos(beta), -np.sin(beta)],
                     [0, np.sin(beta), np.cos(beta)]])

    elif axis.lower() == 'y':
        R = np.array([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1.0, 0],
                      [-np.sin(beta), 0, np.cos(beta)]])

    else:
        raise ValueError

    return R


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
    return rotation(beta, 'y')


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


def pointing_offset(theta, phi, theta_x=0, theta_y=0):
    """
    Introduce a small-angle pointing offset
    onto zenith and azimuth arrays

    Parameters
    ----------
    theta : ndarray
        Zenith angle [radians]
    phi : ndarray
        Azimuth angle [radians]
    theta_x : float, optional
        Small rotation about the x-hat vector [rad]
    theta_y : float, optional
        Small rotation about the y-hat vector [rad]

    Returns
    -------
    new_theta, new_phi
    """
    # get XYZ coordinates on unit-sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    r = np.array([x, y, z])

    # perform rotations
    if theta_x > 0:
        r = rotation(theta_x, 'x') @ r
    if theta_y > 0:
        r = rotation(theta_y, 'y') @ r

    # convert back to spherical coordinates
    new_theta = np.arccos(r[2])
    xzero = np.isclose(r[0], 0)
    yzero = np.isclose(r[1], 0)
    xneg = r[0] < 0
    ypos = r[1] > 0
    new_phi = np.zeros_like(new_theta)
    new_phi[~xzero] = np.arctan(r[1][~xzero] / r[0][~xzero])
    new_phi[xneg & ypos] += np.pi
    new_phi[xneg & ~ypos] -= np.pi
    new_phi[(xzero & yzero)] = 0.0
    new_phi[(xzero & ypos)] = np.pi/2
    new_phi[(xzero & ~ypos)] = -np.pi/2
    new_phi = new_phi % (2 * np.pi)

    return new_theta, new_phi


def cut_sky_fov(sky, cut):
    """
    Given a sky tensor (..., Npixels) and a FOV cut
    indexing array (Nfov_pixels,), apply the indexing to sky
    and return the cut_sky. Optimized for backprop through
    integer indexing.
    """
    if isinstance(cut, slice):
        cut_sky = sky[..., cut]
    else:
        if isinstance(cut, np.ndarray):
            cut = torch.as_tensor(cut)
        if utils.device(cut.device) != utils.device(sky.device):
            cut = cut.to(sky.device)
        # for integer index, this is faster than sky[...,cut] on GPU
        cut_sky = sky.index_select(-1, cut)

    return cut_sky


def beam_edge_taper(zen, alpha=0.1, fov=180, device=None):
    """
    Create a Tukey window tapering for a beam response

    Parameters
    ----------
    zen : ndarray or tensor
        Zenith angles [deg]
    alpha : float, optional
        Alpha parameter for the Tukey mask extending
        from -fov/2 to fov/2
    fov : float, optional
        fov parameter [deg] of the beam
    device : str, optional
        Device to create mask

    Returns
    -------
    tensor
    """
    zen = utils.tensor2numpy(zen, clone=False)
    # theta mask: 5000 yields ang. resolution < nside 512
    th_arr = np.linspace(-fov/2, fov/2, 5000, endpoint=True)
    mask = utils.windows.tukey(5000, alpha=alpha)
    intp = utils.interp1d(th_arr, mask, fill_value=0, bounds_error=False, kind='linear')
    out = intp(zen)

    return torch.as_tensor(out, device=device)


