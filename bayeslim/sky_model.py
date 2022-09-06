"""
Module for torch sky models and relevant functions
"""
import torch
import numpy as np
from scipy import special, interpolate
import copy

from . import utils, cosmology
from .utils import _float, _cfloat


class SkyBase(utils.Module):
    """
    Base class for various sky model representations
    """
    def __init__(self, params, kind, freqs, R=None, name=None,
                 parameter=True, p0=None):
        """
        Base class for a torch sky model representation.

        Parameters
        ----------
        params : tensor
            A sky model parameterization as a tensor to
            be pushed through the response function R().
            In general this should be (Nstokes, 1, Nfreqs, Nsources)
        kind : str
            Kind of sky model. options = ['point', 'pixel', 'alm']
            for point source model, pixelized model, and spherical
            harmonic model.
        freqs : tensor
            Frequency array of sky model [Hz]
        R : callable, optional
            An arbitrary response function for the
            point source model, mapping self.params
            to a sky source tensor of shape
            (Nstokes, 1, Nfreqs, Nsources)
        name : str, optional
            Name for this object, stored as self.name
        parameter : bool
            If True, treat params as variables to be fitted,
            otherwise hold them fixed as their input value
        p0 : tensor, optional
            Fixed starting params tensor (default is zero),
            which is summed with params before entering
            response function. Redefines params as a deviation
            from p0. Must have same shape as params.
        """
        super().__init__(name=name)
        self.params = params
        self.device = self.params.device
        self.p0 = p0
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        self.kind = kind
        if R is None:
            R = DefaultResponse()
        self.R = R
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        if self.R.freq_mode == 'channel':
            assert len(self.freqs) == self.params.shape[2]

        # construct _args for str repr
        self._args = dict(name=name)
        self._args[self.R.__class__.__name__] = getattr(self.R, '_args', None)

    def push(self, device, attrs=[]):
        """
        Wrapper around nn.Module.to(device) method
        but always pushes self.params whether its a 
        parameter or not.

        Parameters
        ----------
        device : str
            Device to push to, e.g. 'cpu', 'cuda:0'
        attrs : list of str
            List of additional attributes to push
        """
        # push basic attrs
        self.params = utils.push(self.params, device)
        self.device = device
        self.freqs = self.freqs.to(device)
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device))
        # push response
        self.R.push(device)
        # push starting p0
        if self.p0 is not None:
            self.p0 = self.p0.to(device)
        # push prior functions
        if self.priors_inp_params is not None:
            for pr in self.priors_inp_params:
                pr.push(device)
        if self.priors_out_params is not None:
            for pr in self.priors_out_params:
                pr.push(device)

    def freq_interp(self, freqs, kind='linear'):
        """
        Interpolate params onto new set of frequencies
        if response freq_mode is channel. If freq_mode is
        linear or powerlaw, just update response frequencies

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
                interp = interpolate.interp1d(utils.tensor2numpy(self.freqs),
                                              utils.tensor2numpy(self.params),
                                              axis=2, kind=kind, fill_value='extrapolate')
                params = torch.as_tensor(interp(utils.tensor2numpy(freqs)), device=self.device,
                                         dtype=self.params.dtype)
                if self.params.requires_grad:
                    self.params = torch.nn.Parameter(params)
                else:
                    self.params = params

            self.freqs = freqs.to(self.device)
            self.R.freqs = freqs.to(self.device)
            self.R._setup()

    def hook_response_grad(self, value=True):
        """
        Store gradient of response output as self.response_grad
        """
        self._hook_response_grad = value


class DefaultResponse:
    """
    Default response function for SkyBase  
    """
    def __init__(self):
        self.freq_mode = 'channel'

    def _setup(self):
        pass

    def __call__(self, params):
        return params

    def push(self, device):
        pass


class PointSky(SkyBase):
    """
    Point source sky model with fixed
    source locations but variable flux density.
    Relates source flux parameterization
    to per-frequency, per-stokes, per-source
    flux density vector.

    Returns point source flux density and their sky
    locations in equatorial coordinates.
    """
    def __init__(self, params, angs, freqs, R=None, name=None,
                 parameter=True, p0=None):
        """
        Fixed-location point source model with
        parameterized flux density.

        Parameters
        ----------
        params : tensor
            Point source flux parameterization adopted by R().
            In general, this is of shape (Nstokes, 1, Ncoeff, Nsources),
            where Ncoeff is the chosen parameterization across frequency.
            For no parameterization (default) this should be a tensor
            of shape (Nstokes, 1, Nfreqs, Nsources).
            Nstokes is the number of Stokes parameters ordered as
            I, Q, U, V. The sky block should be followed by a 
            Stokes2Coherency block. However, for just Stokes I
            and in 1pol mode, this can be (1, 1, Nfreqs, Nsources)
            with no Stokes2Coherency block.
        angs : tensor
            Point source unit vectors on the sky in equatorial
            coordinates of shape (2, Nsources), where the
            last two axes are RA and Dec [deg].
        freqs : tensor
            Frequency array of sky model [Hz].
        R : callable, optional
            An arbitrary response function for the
            point source model, mapping self.params
            to a sky source tensor of shape
            (Ntokes, 1, Nfreqs, Nsources)
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.
        p0 : tensor, optional
            Fixed starting params tensor (default is zero),
            which is summed with params before entering
            response function. Redefines params as a deviation
            from p0. Must have same shape as params.

        Examples
        --------
        Here is an example for a simple point source model
        with a frequency power law parameterization.
        Note that the frequency array must be defined
        in the global namespace.

        .. code-block:: python

            Nfreqs = 16
            freqs = np.linspace(100e6, 120e6, Nfreqs)  # Hz
            phi = np.random.rand(100) * 180            # dec
            theta = np.random.rand(100) * 360          # ra
            angs = torch.tensor([theta, phi])
            amps = scipy.stats.norm.rvs(20, 1, 100)
            amps = torch.tensor(amps.reshape(1, 100, 1))
            alpha = torch.tensor([-2.2])
            def R(params, freqs=freqs):
                S = params[0][..., None]
                spix = params[1]
                return S * (freqs / freqs[0])**spix
            P = bayeslim.sky.PointSky([amps, alpha],
                                      angs, Nfreqs, R=R)
        """
        super().__init__(params, 'point', freqs, R=R, name=name,
                         parameter=parameter, p0=p0)
        self.angs = angs

    def forward(self, params=None, prior_cache=None, **kwargs):
        """
        Forward pass the sky parameters

        Parameters
        ----------
        params : list of tensors, optional
            Set of parameters to use instead of self.params.
        prior_cache : dict, optional
            Cache for storing computed priors as self.name

        Returns
        -------
        dictionary
            kind : str
                Kind of sky model ['point', 'pixel', 'alm']
            sky : tensor
                Source brightness at discrete locations
                (Nstokes, 1, Nfreqs, Nsources)
            angs : tensor
                Sky source locations (RA, Dec) [deg]
                (2, Nsources)
        """
        # fed params or attr params
        if params is None:
            params = self.params

        if self.p0 is not None:
            p = params + self.p0
        else:
            p = params

        # pass through response
        sky = self.R(p)

        # register gradient hook if desired
        if self._hook_response_grad:
            if sky.requires_grad:
                sky.register_hook(self.response_grad_hook)

        # evaluate prior on self.params (not params + p0)
        self.eval_prior(prior_cache, inp_params=self.params, out_params=sky)

        # pass through response
        name = getattr(self, 'name', None)

        return dict(kind=self.kind, sky=sky, angs=self.angs, name=name)


class PointSkyResponse:
    """
    Frequency parameterization of point sources at
    fixed locations but variable flux wrt frequency
    options include
        - channel : vary all frequency channels
        - linear : linear mapping across frequency
        - powerlaw : amplitude and exponent, centered at f0.
    """
    def __init__(self, freqs, freq_mode='linear', log=False, device=None, **freq_kwargs):
        """
        Choose a frequency parameterization for PointSky

        Parameters
        ----------
        freqs : tensor
            Frequency array [Hz]
        freq_mode : str, optional
            options = ['channel', 'linear', 'powerlaw']
            Frequency parameterization mode. Choose between
            channel - each frequency is a parameter
            linear - linear (e.g. polynomial) basis
            powerlaw - amplitude and powerlaw basis
        log : bool, optional
            Treat params as log(amplitude).
            For channel and linear freq modes, this means
            we take exp(R(params)).
            For powerlaw mode we only take exp(params[...,0,:])
        device : str, optional
            Device of point source params
        freq_kwargs : dict, optional
            Kwargs for different freq_modes, see Notes below

        Notes
        -----
        freq_mode == 'linear'
            See utils.gen_linear_A() for necessary kwargs
        freq_mode == 'powerlaw':
            f0 : anchor frequency [Hz]
        """
        self.log = log
        self.freqs = freqs
        self.freq_mode = freq_mode
        self.device = device
        self._setup(**freq_kwargs)

        # construct _args for str repr
        self._args = dict(freq_mode=self.freq_mode)

    def _setup(self, **kwargs):
        # setup
        if self.freq_mode == 'linear':
            assert 'linear_mode' in kwargs
            kwgs = copy.deepcopy(kwargs)
            linear_mode = kwgs.pop('linear_mode')
            kwgs['x'] = self.freqs
            self.freq_LM = utils.LinearModel(linear_mode, dim=-2,
                                             device=self.device, **kwgs)

        elif self.freq_mode == 'powerlaw':
            self.f0 = kwargs['f0']

    def __call__(self, params):
        # pass to device
        if utils.device(params.device) != utils.device(self.device):
            params = params.to(self.device)

        if self.freq_mode == 'channel':
            pass

        elif self.freq_mode == 'linear':
            params = self.freq_LM(params)

        elif self.freq_mode == 'powerlaw':
            amp = params[..., 0:1, :]
            if self.log:
                amp = torch.exp(amp)
            params = amp * (self.freqs[:, None] / self.f0)**params[..., 1:2, :]

        if self.log and self.freq_mode in ['channel', 'linear']:
            params = torch.exp(params)

        return params

    def push(self, device):
        self.device = device
        self.freqs = self.freqs.to(device)
        if self.freq_mode == 'linear':
            self.freq_LM.push(device)


class PixelSky(SkyBase):
    """
    Pixelized model (e.g. healpix or other representation)
    of the sky specific intensity distribution
    (aka brightness or temperature) at fixed locations in
    Equatorial coordinates but with variable amplitude.

    While the input sky model (params) should be in units of
    specific intensity (Kelvin or Jy / str), the output
    of the forward model is in flux density [Jy]
    (i.e. we multiply by each cell's solid angle).
    """
    def __init__(self, params, angs, freqs, px_area, R=None, name=None,
                 parameter=True, p0=None):
        """
        Pixelized model of the sky brightness distribution.
        This can be parameterized in any generic way via params,
        but the output of R(params) must be
        a representation of the sky brightness at fixed
        cells, which are converted to flux density
        by multiplying by each cell's solid angle.

        Parameters
        ----------
        params : tensor
            Sky model flux parameterization of shape
            (Nstokes, 1, Nfreq_coeff, Nsky_coeff), where Nsky_coeff is
            the free parameters describing angular fluctations, and Nfreq_coeff
            is the number of free parameters describing frequency fluctuations,
            both of which should be expected by the response function R().
            By default, this is just Nfreqs and Npix, respectively.
            Nstokes is the number of Stokes polarizations ordered as I, Q, U, V.
            Unless operating in 1pol mode with only Stokes I, this should
            be followed by a Stokes2Coherency block.
        angs : tensor
            Point source unit vectors on the sky in equatorial
            coordinates of shape (2, Nsources), where the
            last two axes are RA and Dec [deg].
        freqs : tensor
            Frequency array of sky model [Hz].
        px_area : tensor or float
            Contains the solid angle of each pixel [str]. This is multiplied
            into the final sky model, and thus needs to be a scalar or
            a tensor of shape (Npix,) to allow for broadcasting
            rules to apply.
        R : callable, optional
            An arbitrary response function for the sky model, mapping
            self.params to a sky pixel tensor of shape
            (Nstokes, 1, Nfreqs, Npix)
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.
        p0 : tensor, optional
            Fixed starting params tensor (default is zero),
            which is summed with params before entering
            response function. Redefines params as a deviation
            from p0. Must have same shape as params.
        """
        super().__init__(params, 'pixel', freqs, R=R, name=name,
                         parameter=parameter, p0=p0)
        self.angs = angs
        self.px_area = px_area

    def forward(self, params=None, prior_cache=None, **kwargs):
        """
        Forward pass the sky parameters.

        Parameters
        ----------
        params : list of tensors, optional
            Set of parameters to use instead of self.params.
        prior_cache : dict, optional
            Cache for storing compute priors as self.name

        Returns
        -------
        dictionary
            kind : str
                Kind of sky model ['point', 'pixel', 'alm']
            amps : tensor
                Pixel flux density at fixed locations on the sky
                (Nstokes, 1, Nfreqs, Npix)
            angs : tensor
                Sky source locations (RA, Dec) [deg]
                (2, Npix)
        """
        # apply fed params or attr params
        if params is None:
            params = self.params

        if self.p0 is not None:
            p = params + self.p0
        else:
            p = params

        # pass through response
        sky = self.R(p)

        # register gradient hook if desired
        if self._hook_response_grad:
            if sky.requires_grad:
                sky.register_hook(self.response_grad_hook)

        # evaluate prior on self.params (not params + p0)
        self.eval_prior(prior_cache, inp_params=self.params, out_params=sky)

        name = getattr(self, 'name', None)

        return dict(kind=self.kind, sky=sky * self.px_area, angs=self.angs, name=name)


class PixelSkyResponse:
    """
    Spatial and frequency parameterization for PixelSky.
    Takes a params tensor (Nstokes, 1, Nfreq_coeff, Npix_coeff)
    and returns a sky tensor (Nstokes, 1, Nfreq, Npix)

    options for spatial parameterization include
        - 'pixel' : sky pixel
        - 'linear' : any generic linear model
        - 'alm' : spherical harmonic model

    options for frequency parameterization include
        - 'channel' : frequency channels
        - 'linear' : linear mapping
        - 'powerlaw' : power law model
        - 'bessel' : spherical bessel g_l (for spatial mode 'alm')
    """
    def __init__(self, freqs, comp_params=False, spatial_mode='pixel',
                 freq_mode='channel', device=None, transform_order=0,
                 cosmo=None, spatial_kwargs={}, freq_kwargs={}, log=False):
        """
        Parameters
        ----------
        freqs : tensor
            Frequency bins [Hz]
        comp_params : bool, optional
            If True, params should be transformed to complex
            before passing through self
        spatial_mode : str, optional
            Choose the spatial parameterization (default is pixel)
            options = ['pixel', 'linear', 'alm']
        freq_mode : str, optional
            Choose the freq parameterization (default is channel)
            options = ['channel', 'linear', 'powerlaw', 'bessel']
        device : str, optional
            Device to put model on
        transform_order : int, optional
            0 - spatial then frequency transform (default)
            1 - frequency then spatial transform
        cosmo : Cosmology object
        spatial_kwargs : dict, optional
            Kwargs used to generate spatial transform matrix.
            These are required kwargs for each option.
            'pixel' : None
            'alm' :
                l, m : array, holding l, m vaues
                theta, phi : array, holds co-latitude and azimuth
                    angles [deg] of pixel model, used for generating
                    Ylm in alm mode
                alm_mult : array, (Ncoeff,)
                Ylm : tensor, holds (Ncoeff, Npix) Ylm transform
                    matrix. If None, will compute it given l, m
            'linear' : 
                kwargs to LinearModel()

        freq_kwargs : dict, optional
            Kwargs used to generate freq transform matrix
            for 'linear' freq_mode, see utils.gen_linear_A()
            gln : dict, dictionary of gln modes for bessel mode
            kbins : dict, dictionary of kln values for bessel mode
            f0 : float, fiducial frequency [Hz], used for poly
            Ndeg : int, number of degrees, used for poly
            kbins : ndarray, wavevector bins [Mpc-1], used for bessel
            radial_method : str, radial convention, used for bessel
            kmax : float, maximum k to compute
            decimate : bool, if True, decimate every other kbin
        log : bool, optional
            If True, assumed params is log sky and take
            exp of params just before return
        """
        self.freqs = freqs
        self.comp_params = comp_params
        self.Nfreqs = len(freqs)
        self.spatial_mode = spatial_mode
        self.freq_mode = freq_mode
        self.device = device
        self.transform_order = transform_order
        if cosmo is None:
            cosmo = cosmology.Cosmology()
        self.cosmo = cosmo
        self.log = log

        self._spatial_setup(spatial_kwargs=spatial_kwargs)
        self._freq_setup(freq_kwargs=freq_kwargs)

        # construct _args for str repr
        self._args = dict(freq_mode=self.freq_mode, spatial_mode=self.spatial_mode)

    def _freq_setup(self, freq_kwargs={}):
        # freq setup
        if self.freq_mode == 'channel':
            pass

        elif self.freq_mode == 'powerlaw':
            self.f0 = freq_kwargs['f0']

        elif self.freq_mode == 'linear':
            fkwgs = copy.deepcopy(freq_kwargs)
            assert 'linear_mode' in fkwgs, "must specify linear_mode"
            linear_mode = fkwgs.pop('linear_mode')
            fkwgs['x'] = self.freqs
            fkwgs['dtype'] = utils._cfloat() if self.comp_params else utils._float()
            self.freq_LM = utils.LinearModel(linear_mode, dim=-2, device=self.device, **fkwgs)

        elif self.freq_mode == 'bessel':
            assert self.spatial_transform == 'alm'
            # compute comoving line of sight distances
            self.z = self.cosmo.f2z(utils.tensor2numpy(self.freqs))
            self.r = self.cosmo.comoving_distance(self.z).value
            self.dr = self.r.max() - self.r.min()
            if 'gln' in freq_kwargs and 'kbins' in freq_kwargs:
                self.gln = freq_kwargs['gln']
                self.kbins = freq_kwargs['kbins']

            else:
                gln, kbins = utils.gen_bessel2freq(freq_kwargs['l'],
                                                  utils.tensor2numpy(self.freqs), self.cosmo,
                                                  kmax=freq_kwargs.get('kmax'),
                                                  decimate=freq_kwargs.get('decimate', True),
                                                  dk_factor=freq_kwargs.get('dk_factor', 1e-1),
                                                  device=self.device,
                                                  method=freq_kwargs.get('radial_method', 'shell'),
                                                  Nproc=freq_kwargs.get('Nproc', None),
                                                  Ntask=freq_kwargs.get('Ntask', 10),
                                                  renorm=freq_kwargs.get('renorm', False))
                self.gln = gln
                self.kbins = kbins

    def _spatial_setup(self, spatial_kwargs={}):        
        # spatial setup
        if self.spatial_mode == 'alm':
            if 'Ylm' in spatial_kwargs:
                # assign Ylm to self if present
                self.Ylm = spatial_kwargs['Ylm']
                self.alm_mult = spatial_kwargs['alm_mult']
            elif not hasattr(self, 'Ylm'):
                # if Ylm is not already defined and not in dict, create it
                # warning: this can be *VERY* slow for large ang_pix and l,m arrays
                self.Ylm, _, self.alm_mult = utils.gen_sph2pix(spatial_kwargs['theta'] * D2R,
                                             spatial_kwargs['phi'] * D2R,
                                             spatial_kwargs['l'],
                                             spatial_kwargs['m'],
                                             device=self.device)
            self.l, self.m = spatial_kwargs['l'], spatial_kwargs['m']
            self.theta, self.phi = spatial_kwargs['theta'], spatial_kwargs['phi']

        elif self.spatial_mode == 'linear':
            skwgs = copy.deepcopy(spatial_kwargs)
            assert 'linear_mode' in skwgs, "must specify linear_mode"
            linear_mode = skwgs.pop('linear_mode')
            skwgs['dtype'] = utils._cfloat() if self.comp_params else utils._float()
            self.spat_LM = utils.LinearModel(linear_mode, dim=-1, device=self.device, **skwgs)

    def spatial_transform(self, params):
        """
        Forward model the sky params tensor
        through a spatial transform.

        Parameters
        ----------
        params : tensor
            Sky model parameters (Nstokes, 1, Ndeg, Ncoeff)
            where Ndeg may equal Nfreqs, and Ncoeff
            are the coefficients for the sky representations.

        Returns
        -------
        tensor
            Sky model of shape (Nstokes, 1, Ndeg, Npix)
        """
        # detect if params needs to be casted into complex
        if self.comp_params:
            if not torch.is_complex(params):
                params = utils.viewcomp(params)

        if self.spatial_mode == 'pixel':
            return params

        elif self.spatial_mode == 'alm':
            return (params * self.alm_mult) @ self.Ylm

        elif self.spatial_mode == 'linear':
            return self.spat_LM(params)

    def freq_transform(self, params):
        """
        Forward model the sky params tensor
        through a frequency transform.

        Parameters
        ----------
        params : tensor
            Sky model parameters (Nstokes, 1, Ndeg, Ncoeff)
            where Ncoeff may equal Npix, and Ndeg
            are the coefficients for the frequency representations.
    
        Returns
        -------
        tensor
            Sky model of shape (Nstokes, 1, Nfreqs, Ncoeff)
        """
        # detect if params needs to be casted into complex
        if self.comp_params:
            if not torch.is_complex(params):
                params = utils.viewcomp(params)

        if self.freq_mode == 'channel':
            return params

        elif self.freq_mode == 'linear':
            return self.freq_LM(params)

        elif self.freq_mode == 'powerlaw':
            return params[..., 0:1, :] * (self.freqs[:, None] / self.f0)**params[..., 1:2, :]

        elif self.freq_mode == 'bessel':
            assert self.transform_order == 1, "only support freq-spatial transform order for bessel mode"
            out = torch.zeros(params.shape[:-2] + (self.Nfreqs,) + params.shape[-1:],
                              device=params.device, dtype=params.dtype)
            for l in np.unique(self.freq_kwargs['l']):
                inds = self.freq_kwargs['l'] == l
                out[..., inds] += (params[:, :, inds].transpose(-1, -2) @ self.gln[l]).transpose(-1, -2)
            return out

    def __call__(self, params):
        if utils.device(params.device) != utils.device(self.device):
            params = params.to(self.device)
        if self.transform_order == 0:
            params = self.spatial_transform(params)
            params = self.freq_transform(params)
        else:
            params = self.freq_transform(params)
            params = self.spatial_transform(params)

        if torch.is_complex(params):
            params = params.real

        if hasattr(self, 'log') and self.log:
            params = torch.exp(params)

        return params

    def push(self, device):
        if self.spatial_mode == 'alm':
            self.Ylm = self.Ylm.to(device)
            self.alm_mult = self.alm_mult.to(device)
        elif self.spatial_mode == 'linear':
            self.spat_LM.push(device)

        if self.freq_mode == 'linear':
            self.freq_LM.push(device)
        elif self.freq_mode == 'bessel':
            for k in self.gln:
                self.gln[k] = self.gln[k].to(device)

        self.freqs = self.freqs.to(device)
        self.device = device


class SphHarmSky(SkyBase):
    """
    Spherical harmonic expansion of a sky temperature field
    at pointing direction s and frequency f

    .. math::

        T(s, f) = \sum_{lm} = Y_{lm}(s) a_{lm}(f)

    where Y is a spherical harmonic of order l and m
    and t is its   coefficient.
    """
    def __init__(self, params, lms, freqs, R=None, parameter=True):
        """
        Spherical harmonic representation of the sky brightness.
        Can also accomodate a spherical Fourier Bessel model.

        Parameters
        ----------
        params : list of tensors
            Spherical harmonic parameterization of the sky.
            The first element of params must be a tensor holding
            the a_lm coefficients of shape
            (Nstokes, 1, Nfreqs, Ncoeff). Nfreqs may also be
            replaced by Nk for a spherical Fourier Bessel model.
            Additional tensors can also parameterize frequency axis.
        lms : array
            Array holding spherical harmonic orders (l, m) of shape
            (2, Ncoeff).
        freqs : tensor
            Frequency array of sky model [Hz].
        R : callable, optional
            An arbitrary response function for the
            spherical harmonic model, mapping input self.params
            to an output a_lm basis of shape
            (Nstokes, 1, Nfreqs, Ncoeff).
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.
        """
        raise NotImplementedError


class CompositeModel(utils.Module):
    """
    Multiple sky models, possibly on different devices.
    To keep graph memory as small as possible, place
    sky models that don't have parameters first in "models"
    """
    def __init__(self, models, sum_output=False, device=None):
        """
        Multiple sky models to be evaluated
        and returned in a list

        Parameters
        ----------
        models : OrderedDict
            Dictionary of SkyBase objects to evaluate
        sum_output : bool, optional
            If True, sum output sky model from
            each model before returning. This only
            works if each input model is of the
            same kind, and if they have the same
            shape.
        device : str, optional
            Device to move all outputs to before summing
            if sum_output
        """
        super().__init__()
        self.models = list(models.keys())
        for k in models:
            setattr(self, k, models[k])
        self.sum_output = sum_output
        self.device = device

    def forward(self, *args, prior_cache=None):
        """
        Forward pass sky models and append in a list
        or sum the sky maps and return a sky_component
        dictionary
        """
        # forward each sky model
        sky_components = []
        for mod in self.models:
            skycomp = self[mod].forward(prior_cache=prior_cache)
            sky_components.append(skycomp)

        if self.sum_output:
            # assert only one kind of sky models
            assert len(set([comp['kind'] for comp in sky_components])) == 1
            # use first sky_component as placeholder
            output = sky_components[0]
            sky = torch.zeros_like(output['sky'], device=self.device)
            for comp in sky_components:
                sky += comp['sky'].to(self.device)
            output['sky'] = sky
            # make sure other keys are on the same device
            for k in output:
                if isinstance(output[k], torch.Tensor):
                    if output[k].device.type != self.device:
                        output[k] = output[k].to(self.device)
        else:
            output = sky_components

        return output

    def freq_interp(self, freqs, kind='linear'):
        """
        Interpolate params onto new set of frequencies
        if response freq_mode is channel. If freq_mode is
        linear or powerlaw, just update response frequencies

        Parameters
        ----------
        freqs : tensor
            Updated frequency array to interpolate to [Hz]
        kind : str, optional
            Kind of interpolation if freq_mode is channel
            see scipy.interp1d for options
        """
        for model in self.models:
            model.freq_interp(freqs, kind)

    def push(self, device):
        for model in self.models:
            self[model].push(device)
        self.device = device


def read_catalogue(catfile, freqs=None, device=None,
                   parameter=False, freq_interp='linear'):
    """
    Read a point source catalogue YAML file.
    See bayeslim.data.DATA_PATH for examples.

    Parameters
    ----------
    catfile : str
        Path to a YAML point source catalogue file
    freqs : tensor, optional
        Frequencies to evaluate model [Hz].
        If catalogue is 'channel' along the frequency
        axis, then it is interpolated with freq_interp
        onto freqs bins. Required for powerlaw model.
    device : str, optional
        Device to transport model to
    parameter : bool, optional
        Make params a Parameter object
    freq_interp : str, optional
        Kind of frequency interpolation if necessary

    Returns
    -------
    tensor
        PointSky object
    list
        Source names
    """
    import yaml
    with open(catfile) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    # ensure frequencies are float
    if 'freqs' in d:
        d['freqs'] = torch.as_tensor(np.array(d['freqs'], dtype=float), dtype=_float())

    # load point positions
    sources = d['sources']
    angs = torch.tensor([np.array(sources['ra']),
                         np.array(sources['dec'])], dtype=_float())

    if d['freq_mode'] == 'channel':
        # collect Stokes I fluxes at specified frequencies
        S = np.array([sources['freq{}'.format(i)] for i in range(len(d['freqs']))])

        # interpolate onto freqs
        if freqs is not None:
            interp = interpolate.interp1d(d['freqs'], S, kind=freq_interp, axis=0,
                                          fill_value='extrapolate')
            params = torch.tensor(interp(freqs), dtype=_float())[None, None, :, None]
        else:
            assert 'freqs' in d, "must pass freqs if not in catalogue file"
            freqs = d['freqs']
            params = torch.ones(len(freqs), dtype=_float())[None, None, :, None]

    elif d['freq_mode'] == 'powerlaw':
        assert freqs is not None
        # ensure frequencies are float
        d['mode_kwargs']['f0'] = float(d['mode_kwargs']['f0'])

        # collect parameters
        params = torch.tensor([np.array(sources['amp']), np.array(sources['alpha'])])[None, None, :, :]

    else:
        raise NotImplementedError

    R = PointSkyResponse(freqs, freq_mode=d['freq_mode'], device=device, **d['mode_kwargs'])
    sky = PointSky(params, angs, freqs, R=R, parameter=parameter)

    if 'polarizaton' in d:
        # still under development
        raise NotImplementedError
        Nsources = params.shape[-1]
        sparams = torch.tensor(np.array(d['Qfrac'], d['Ufrac'], d['Vfrac']).reshape(3, 1, Nsources))
        stokes = FullStokesModel(frac_pol=True)
        sky = utils.Sequential(dict(sky=sky, stokes=stokes))

    return sky, sources['name']


def write_catalogue(catfile, sky, names, overwrite=False):
    """
    Write a point source catalogue to YAML file

    Parameters
    ----------
    catfile : str
        Name of output catalogue yml file
    sky : PointSky object
    names : list
        List of str holding source names
    """
    import os
    import yaml
    if not os.path.exists(catfile) or overwrite:
        d = {}
        assert sky.params.shape[0] == 1, "Writing Stokes QUV not currently implemented"

        # insert header metadata
        d['freq_mode'] = sky.R.freq_mode
        d['mode_kwargs'] = {}

        if sky.R.freq_mode == 'channel':
            d['freqs'] = utils.tensor2numpy(sky.freqs).tolist()
            d['mode_kwargs']['f0'] = None

        elif sky.R.freq_mode == 'powerlaw':
            d['mode_kwargs']['f0'] = sky.R.f0

        elif sky.R.freq_mode == 'linear':
            raise NotImplementedError
        
        else:
            raise ValueError("{} not recognized".format(sky.R.freq_mode))

        # insert source data
        d['sources'] = {}
        d['sources']['name'] = list(names)
        d['sources']['ra'], d['sources']['dec'] = utils.tensor2numpy(sky.angs).tolist()

        if sky.R.freq_mode == 'channel':
            for i in range(len(sky.freqs)):
                d['sources']['freq{:d}'.format(i)] = utils.tensor2numpy(sky.params[0, 0, i]).tolist()

        elif sky.R.freq_mode == 'powerlaw':
            d['sources']['amp'] = utils.tensor2numpy(sky.params[0, 0, 0]).tolist()
            d['sources']['alpha'] = utils.tensor2numpy(sky.params[0, 0, 1]).tolist()

        with open(catfile, 'w') as f:
            yaml.dump(d, f)


def Jy2K(freqs, steradians):
    """
    Convert from Jansky (flux density) to
    Kelvin (specific intensity).
    Returns [K] / [Jy] factor w.r.t. frequency
    given solid angle of pixel.

    Parameters
    ----------
    freqs : ndarray
        Frequencies [Hz]
    steradians : float
        Solid angle [str] to use in conversion.
    """
    c_cmps = 2.99792458e10  # cm/s
    k_boltz = 1.38064852e-16  # erg/K
    lam = c_cmps / freqs    # cm
    return 1e-23 * lam ** 2 / (2 * k_boltz * steradians)


class Stokes2Coherency(utils.Module):
    """
    Convert Stokes parameters to coherency matrix
    for xy cartesian (i.e. linear) feed basis.
    Input to forward call should be of shape (4, 1, ...)
    with the zeroth axis holding the Stokes parameters
    in the order of [I, frac_Q, frac_U, frac_V], where
    frac_Q is defined as

    .. math::

        Q = I * f_Q

    and is not the normal fractional pol of (Q/I)^2.
    Optionally, if S is of shape (2, 2, ...) then it is assumed
    that Stokes parameters are ordered
        | I      frac_Q |
        | frac_U frac_V |

    Returns the coherency matrix of the form

    .. math::

        B = \left(
            \begin{array}{cc}I + Q & U - iV \\
            U + iV & I - Q \end{array}
        \right)

    Note: the Stokes QUV parameters should not
    sum in quadrature to more than I, i.e.

    .. math::

        (Q^2 + U^2 + V^2) / I^2 \le 1

    which should be enforced by setting a
    prior on this quantity.
    """
    def __init__(self, params=None, parameter=False):
        """
        Parameters
        ----------
        params : tensor or utils.Module object, optional
            If provided, this is the fractional polarization
            tensor shape (3, 1, ...) where the zeroth dimension
            holds fractional polarization for Q, U, V in
            that order. This assumes that the input to the
            forward call is a Stokes-I only sky model.
            Alternatively, this can be a sky_model.SkyBase
            object representing a forward model of the
            fractional polarization sky, whose output['sky']
            tensor takes the form of the params tensor above.
            If params is not provided, assume that the input
            sky model contains Stokes I[frac_Q,frac_U,frac_V].
        parameter : bool, optional
            Make the fractional pol tensor a parameter.
        """
        super().__init__()
        self.params = params
        if parameter:
            if isinstance(self.params, torch.Tensor):
                self.params = torch.nn.Parameter(self.params)

    def forward(self, sky_comp, prior_cache=None):
        """
        Forward sky model into coherency matrix form

        Parameters
        ----------
        sky_comp : tensor or dict
            Of shape (4, 1, ...) or (2, 2, ...)
            holding [IQUV] or [[I, Q], [U, V]] respectively.
            Can also be a sky_component dictionary, which will
            index sky_comp['sky'] as the sky tensor

        Returns
        -------
        tensor or dict
        """
        # check if params is actually a sky_component
        if isinstance(sky_comp, dict):
            sky_comp['sky'] = self.forward(sky_comp['sky'], prior_cache=prior_cache)
            return sky_comp

        # assume sky_comp is a (Nstokes, 1, ...) tensor from here on out
        device = sky_comp.device
        noV = True
        if len(sky_comp) == 1:
            # only Stokes I was fed
            I = sky_comp[0, 0]
            if self.params is not None:
                if not isinstance(self.params, torch.Tensor):
                    # forward model if params is a sky model
                    params = self.params()['sky']
                else:
                    # otherwise use self.params
                    params = self.params
                # setup output coherency matrix
                B = torch.zeros(2, 2, *sky_comp.shape[2:], dtype=_float(), device=device)
                frac_Q = params[0, 0]
                Q = I * frac_Q
                if len(params) > 1:
                    frac_U = params[1, 0]
                    U = I * frac_U
                else:
                    frac_U = 0
                    U = 0
                if len(params) > 2:
                    frac_V = params[2, 0]
                    V = I * frac_V
                    B = B.to(_cfloat())
                    noV = False
                else:
                    frac_V = 0
                    V = 0

            else:
                # no fractional polarization just use Stokes I
                B = torch.zeros(1, 1, *sky_comp.shape[2:], dtype=_float(), device=device)
                frac_Q, frac_U, frac_V = 0, 0, 0
                Q, U, V = 0, 0, 0
        else:
            # assume some or all of Q, U, V was fed along with I
            B = torch.zeros(2, 2, *sky_comp.shape[2:], dtype=_float(), device=device)
            I = sky_comp[0, 0]
            if sky_comp.shape[:2] == (2, 2):
                # index sky_comp as (2, 2, ...)
                frac_Q, frac_U, frac_V = sky_comp[0, 1], sky_comp[1, 0], sky_comp[1, 1]
                B = B.to(_cfloat())
                noV = False
            else:
                # index sky_comp as (N, 1, ...)
                frac_Q = sky_comp[1, 0]
                if len(sky_comp) > 2:
                    frac_U = sky_comp[2, 0]
                else:
                    frac_U = 0
                if len(sky_comp) > 3:
                    frac_V = sky_comp[3, 0]
                    B = B.to(_cfloat())
                    noV = False
                else:
                    frac_V = 0
            Q, U, V = I * frac_Q, I * frac_U, I * frac_V

        # populate coherency matrix
        B[0, 0] = I + Q
        if len(B) > 1:
            B[0, 1] = U
            B[1, 0] = U
            B[1, 1] = I - Q
            if not noV:
                B[0, 1] -= 1j * V
                B[1, 0] += 1j * V

        # evaluate prior on fractional polarization
        # this should be less than or equal to 1.0
        frac_pol = frac_Q**2 + frac_U**2 + frac_V**2
        self.eval_prior(prior_cache, inp_params=frac_pol)

        return B

    def eval_prior(self, prior_cache, inp_params=None, out_params=None):
        """
        Prior evalution function specific to Stokes2Coherency class.
        See utils.Module for details

        Parameters
        ----------
        prior_cache : dict
            Dictionary to hold computed prior, assigned as self.name
        inp_params, out_params : tensor, optional
            self.params and self.R(self.params), respectively
        """
        # append to cache
        if prior_cache is not None and self.name not in prior_cache:
            # start starting log prior value
            prior_value = torch.as_tensor(0.0)

            # try to get inp_params
            if inp_params is None:
                if hasattr(self, 'params'):
                    inp_params = self.params

            # look for prior on inp_params
            if self.priors_inp_params is not None and inp_params is not None:
                for prior in self.priors_inp_params:
                    if prior is not None:
                        prior_value = prior_value + prior(inp_params)

            # no prior on out_params for now...

            # append prior value
            prior_cache[self.name] = prior_value


def pixelsky_Ylm_cut(obj, lmin=None, lmax=None, mmin=None, mmax=None, other=None):
    """
    Cut the lm modes of a PixelSky object with a alm spatial response function.
    Operates inplace

    Parameters
    ----------
    obj : PixelSky object
    lmin : float, optional
    lmax : float, optional
    mmin : float, optional
    mmax : float, optional
    other : array, optional
        A custom boolean array
        indexing lm axis.
    """
    s = np.ones(len(obj.R.l), dtype=bool)
    if lmin is not None:
        s = s & (obj.R.l >= lmin)
    if lmax is not None:
        s = s & (obj.R.l <= lmax)
    if mmin is not None:
        s = s & (obj.R.m >= mmin)
    if mmax is not None:
        s = s & (obj.R.m <= mmax)
    if other is not None:
        s = s & (other)

    obj.R.Ylm = obj.R.Ylm[s]
    with torch.no_grad():
        if obj.p0 is not None:
            obj.p0 = obj.p0[..., s, :]
        obj['params'] = obj.params[..., s, :]
    obj.R.alm_mult = obj.R.alm_mult[s]
    obj.R.l, obj.R.m = obj.R.l[s], obj.R.m[s]
