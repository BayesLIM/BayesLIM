"""
Module for torch sky models and relevant functions
"""
import torch
import numpy as np
from scipy import special, interpolate
import copy

from . import utils, cosmology, dataset, sph_harm, linear_model
from .utils import _float, _cfloat


class SkyBase(utils.Module):
    """
    Base class for various pixelized sky model representations
    """
    def __init__(self, params, R=None, name=None,
                 parameter=True, p0=None):
        """
        Base class for a pixel sky model representation.

        Parameters
        ----------
        params : tensor
            A sky model parameterization as a tensor to
            be pushed through the response function R().
            In general this should be (Nstokes, 1, Nfreqs, Nsources)
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
        if R is None:
            R = DefaultResponse()
        self.R = R

        # construct _args for str repr
        self._args = dict(name=name)
        self._args[self.R.__class__.__name__] = getattr(self.R, '_args', None)

    def _push(self, device, attrs=[]):
        """
        Wrapper around nn.Module.to(device) method
        but always pushes self.params whether its a 
        parameter or not.

        Parameters
        ----------
        device : str
            Device to push to, e.g. 'cpu', 'cuda:0'
            Can also be a dtype.
        attrs : list of str
            List of additional attributes to push
        """
        dtype = isinstance(device, torch.dtype)
        # push basic attrs
        if not dtype: self.device = device
        self.params = utils.push(self.params, device)
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device))
        # push response
        self.R.push(device)
        # push starting p0
        if self.p0 is not None:
            self.p0 = utils.push(self.p0, device)
        if isinstance(self.angs, torch.Tensor):
            self.angs = utils.push(self.angs, device)
        else:
            self.angs = (utils.push(self.angs[0], device),
                         utils.push(self.angs[1], device)
                        )
        # push prior functions
        if self.priors_inp_params is not None:
            for pr in self.priors_inp_params:
                if pr is not None:
                    pr.push(device)
        if self.priors_out_params is not None:
            for pr in self.priors_out_params:
                if pr is not None:
                    pr.push(device)

    def freq_interp(self, freqs, kind='linear'):
        """
        Interpolate params onto new set of frequencies
        if response freq_mode is channel. If freq_mode is
        powerlaw, just update response frequencies.

        Parameters
        ----------
        freqs : tensor
            Updated frequency array to interpolate to [Hz]
        kind : str, optional
            Kind of interpolation if freq_mode is channel
            see scipy.interp1d for options
        """
        # only interpolate if new freqs don't match current freqs to 1 Hz
        if len(freqs) != len(self.R.freqs) or not np.isclose(self.R.freqs, freqs, atol=1.0).all():
            freqs = torch.as_tensor(freqs)
            if self.R.freq_mode == 'channel':
                # interpolate params across frequency
                interp = interpolate.interp1d(utils.tensor2numpy(self.R.freqs),
                                              utils.tensor2numpy(self.params),
                                              axis=2, kind=kind, fill_value='extrapolate')
                params = torch.as_tensor(interp(utils.tensor2numpy(freqs)), device=self.device,
                                         dtype=self.params.dtype)
                if self.params.requires_grad:
                    self.params = torch.nn.Parameter(params)
                else:
                    self.params = params

            self.R.freqs = freqs.to(self.device)
            self.R._setup()


class DefaultResponse:
    """
    Default response function for SkyBase  
    """
    def __init__(self, freqs=None):
        self.freqs = freqs
        self.freq_mode = 'channel'

    def set_freq_index(self, idx=None):
        pass

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
    def __init__(self, params, angs, R=None, name=None,
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
            first axis holds RA and Dec [deg], respectively.
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
            P = PointSky([amps, alpha], angs, R=R)
        """
        super().__init__(params, R=R, name=name,
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
        MapData object
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

        # register gradient hooks if desired
        if hasattr(self, '_hook_registry') and self._hook_registry is not None:
            if sky.requires_grad:
                for r in self._hook_registry:
                    sky.register_hook(r)

        # evaluate prior on self.params (not params + p0)
        self.eval_prior(prior_cache, inp_params=self.params, out_params=sky)

        # pass through response
        name = getattr(self, 'name', None)

        skycomp = dataset.MapData()
        skycomp.setup_meta(name=name)
        if isinstance(self.angs, (tuple, list)):
            angs = torch.vstack(self.angs)
        else:
            angs = torch.as_tensor(self.angs)

        freqs = self.R.freqs
        if hasattr(self.R, '_freq_idx') and self.R._freq_idx is not None:
            freqs = freqs[self.R._freq_idx]
        skycomp.setup_data(freqs=freqs, data=sky, angs=angs)

        return skycomp

    def push(self, device, **kwargs):
        """
        Wrapper around SkyBase._push()
        """
        self._push(device, **kwargs)


class PointSkyResponse:
    """
    Frequency parameterization of point sources at
    fixed locations but variable flux wrt frequency.
    The params tensor of shape (Nstokes, 1, Ncoeff, Nsources).
    frequency parameterizations include
        - channel : vary all frequency channels
        - linear : linear mapping across frequency
        - powerlaw : amplitude and exponent, centered at f0.
    """
    def __init__(self, freqs, freq_mode='linear', log=False, device=None, LM=None, **freq_kwargs):
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
        LM : LinearModel object, optional
            Pass the input params through this LinearModel
            object before passing through the response function.
        freq_kwargs : dict, optional
            Kwargs for different freq_modes, see Notes below

        Notes
        -----
        freq_mode == 'linear'
            See linear_model.gen_linear_A() for necessary kwargs
        freq_mode == 'powerlaw':
            f0 : anchor frequency [Hz]
        """
        self.log = log
        self.freqs = freqs
        self.freq_mode = freq_mode
        self.device = device
        self.LM = LM
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
            self.freq_LM = linear_model.LinearModel(linear_mode, dim=-2,
                                                    device=self.device, **kwgs)

        elif self.freq_mode == 'powerlaw':
            if 'f0' in kwargs:
                self.f0 = kwargs['f0']

    def __call__(self, params):
        # pass to device
        if not utils.check_devices(params.device, self.device):
            params = params.to(self.device)

        if self.LM is not None:
            params = self.LM(params)

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

        if hasattr(self, '_freq_idx') and self._freq_idx is not None:
            params = params[..., self._freq_idx, :]

        return params

    def set_freq_index(self, idx=None):
        """
        Set indexing of frequency axis after pushing through
        the response

        Parameters
        ----------
        idx : list or slice object, optional
            Indexing along frequency axis
        """
        self._freq_idx = idx

    def push(self, device):
        dtype = isinstance(device, torch.dtype)
        if not dtype: self.device = device
        self.freqs = self.freqs.to(device)
        if self.LM is not None: self.LM.push(device)
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
    def __init__(self, params, angs, px_area, R=None, name=None,
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
        super().__init__(params, R=R, name=name,
                         parameter=parameter, p0=p0)
        self.angs = angs
        self.px_area = torch.as_tensor(px_area)

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
        MapData object
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

        # register gradient hooks if desired
        if hasattr(self, '_hook_registry') and self._hook_registry is not None:
            if sky.requires_grad:
                for r in self._hook_registry:
                    sky.register_hook(r)

        # evaluate prior on self.params (not params + p0)
        self.eval_prior(prior_cache, inp_params=self.params, out_params=sky)

        name = getattr(self, 'name', None)

        skycomp = dataset.MapData()
        skycomp.setup_meta(name=name)
        if isinstance(self.angs, (tuple, list)):
            angs = torch.vstack(self.angs)
        else:
            angs = torch.as_tensor(self.angs)

        freqs = self.R.freqs
        if hasattr(self, '_freq_idx') and self.R._freq_idx is not None:
            freqs = freqs.self.R._freq_idx
        skycomp.setup_data(freqs=freqs, data=sky * self.px_area, angs=angs)

        return skycomp

    def push(self, device, **kwargs):
        """
        Wrapper around SkyBase._push()
        """
        self._push(device, **kwargs)
        self.px_area = utils.push(self.px_area, device)


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
                 cosmo=None, spatial_kwargs={}, freq_kwargs={}, log=False,
                 real_output=True, abs_output=True, LM=None, sky0=None):
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
                'Alm' : AlmModel object, holding all metadata
                or
                See kwargs to AlmModel.__init__() and setup_Ylm():
                l, m : array, holding l, m vaues
                theta, phi : array, holds co-latitude and azimuth
                    angles [deg] of pixel model, used for generating
                    Ylm in alm mode. See AlmModel for details.
                alm_mult : array, (Ncoeff,)
                Ylm : tensor, holds (Ncoeff, Npix) Ylm transform
                    matrix. If None, will compute it given l, m
            'linear' : 
                kwargs to LinearModel()
        freq_kwargs : dict, optional
            Kwargs used to generate freq transform matrix
            for 'linear' freq_mode, see linear_model.gen_linear_A()
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
        real_output : bool, optional
            If True, ensure that the output of self is a real-valued tensor.
        abs_output : bool, optional
            If True, take abs of final output in self() call.
        LM : LinearModel object, optional
            Pass the input params through this LinearModel
            object before passing through the response function.
        sky0 : tensor, optional
            Starting sky model to add with self(params) after
            forward modeling. This redefines the forward model
            as a perturbation about sky0. Must have the same
            shape as the forward model.
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
        self.LM = LM
        self.real_output = real_output
        self.sky0 = sky0

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
            self.freq_LM = linear_model.LinearModel(linear_mode, dim=-2, device=self.device, **fkwgs)

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
                gln, kbins = sph_harm.gen_bessel2freq(freq_kwargs['l'],
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
        sk = spatial_kwargs
        if self.spatial_mode == 'alm':
            if 'Alm' in spatial_kwargs:
                # check for pre-built Alm
                self.Alm = spatial_kwargs['Alm']
            else:
                # build it now
                self.Alm = sph_harm.AlmModel(sk['l'], sk['m'],
                                             default_kw=sk.get('default_kwargs', None))
                # this attaches self.Alm.Ylm and self.Alm.alm_mult
                self.Alm.setup_Ylm(sk['theta'], sk['phi'],
                                   Ylm=sk.get('Ylm', None), alm_mult=sk.get('alm_mult', None),
                                   cache=False)

        elif self.spatial_mode == 'linear':
            skwgs = copy.deepcopy(spatial_kwargs)
            assert 'linear_mode' in skwgs, "must specify linear_mode"
            linear_mode = skwgs.pop('linear_mode')
            skwgs['dtype'] = utils._cfloat() if self.comp_params else utils._float()
            self.spat_LM = linear_model.LinearModel(linear_mode, dim=-1, device=self.device, **skwgs)

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
        if self.comp_params and not torch.is_complex(params):
            params = utils.viewcomp(params)

        if self.spatial_mode == 'pixel':
            return params

        elif self.spatial_mode == 'alm':
            return self.Alm(params)

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
        if not utils.check_devices(params.device, self.device):
            params = params.to(self.device)
        if self.LM is not None:
            params = self.LM(params)
        if self.transform_order == 0:
            params = self.spatial_transform(params)
            params = self.freq_transform(params)
        else:
            params = self.freq_transform(params)
            params = self.spatial_transform(params)

        if self.real_output:
            params = params.real

        if hasattr(self, 'log') and self.log:
            params = torch.exp(params)

        if hasattr(self, '_freq_idx') and self._freq_idx is not None:
            params = params[..., self._freq_idx, :]

        if hasattr(self, 'sky0') and self.sky0 is not None:
            params = params + self.sky0

        if hasattr(self, 'abs_output') and self.abs_output:
            params = params.abs()

        return params

    def set_freq_index(self, idx=None):
        """
        Set indexing of frequency axis after pushing through
        the response

        Parameters
        ----------
        idx : list or slice object, optional
            Indexing along frequency axis
        """
        self._freq_idx = idx

    def push(self, device):
        dtype = isinstance(device, torch.dtype)
        if self.spatial_mode == 'alm':
            self.Alm.push(device)
        elif self.spatial_mode == 'linear':
            self.spat_LM.push(device)

        if self.freq_mode == 'linear':
            self.freq_LM.push(device)
        elif self.freq_mode == 'bessel':
            for k in self.gln:
                self.gln[k] = self.gln[k].to(device)

        self.freqs = self.freqs.to(device)
        if self.LM is not None: self.LM.push(device)
        if not dtype: self.device = device
        if self.sky0 is not None:
            self.sky0 = utils.push(self.sky0, device)


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
    def __init__(self, models, sum_output=False, device=None,
                 index=None, name=None):
        """
        Multiple sky models to be evaluated
        and returned in a list

        Parameters
        ----------
        models : dict
            Dictionary of SkyBase objects to evaluate
            and stack together.
            E.g. {'diff': PointSky, 'eor': PixelSky}
        sum_output : bool, optional
            If True, sum output sky model from
            each model before returning. This only
            works if each successive sky object in models
            can be summed with its predecessor, optionally
            with an indexing along the Npix axis.
            All sky models are summed into the zeroth model.
        device : str, optional
            Device to move all outputs to before summing
            if sum_output
        index : dict, optional
            This holds indexing arrays used when sum_output=True
            for combining the SkyBase objects in models when
            they are of different shapes. The keys are the same
            as keys in models (the first entry in models doesn't need
            any indexing, and should have the highest spatial resolution
            of all input sky models), and the values are tuples holding
            (predecessor_indexing, this_indexing), where the predecessor_indexing
            indexes the previous sky model data array along its Npix axis
            and this_indexing reshapes the current sky model along its Npix axis.
            E.g. #1 models = {'diff': PixelSky(nside=64), 'eor': PixelSky(nside=64)}
            Both are healpix maps of the same NSIDE but 'eor' has truncated
            spatial extent.
                index = {'eor': ([0, 1, 3, ...], None)}
            In this case we do diff().data[..., [0, 1, 3, ...]] += eor().data[..., :],
            where diff is indexed and eor needs no reshaping. Another way this could
            be achieved is to reshape eor such that missing pixels are created as zero,
            which can then just be directly summed with diff() without indexing it.
            E.g. #2 models = {'diff1': PixelSky(nside=64), 'diff2': PixelSky(nside=32)}
            Here the second model is of lower resolution, so its values need to be copied
            onto the higher resolution map and then directly summed w/ its predecessor
                index = {'diff2': (None, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, ....])}
        name : str, optional
            A name for this object
        """
        super().__init__(name=name)
        self.eval_models = list(models.keys())
        for k in models:
            setattr(self, k, models[k])
        self.sum_output = sum_output
        self.device = device
        self.index = index

    @property
    def models(self):
        return [child[0] for child in self.named_children()]

    def set_eval_models(self, models=None):
        """
        Set the sky models to evaluate upon forward call.
        Default is to use all named_children()

        Parameters
        ----------
        models : list, optional
            List of str names for sky models to evaluate
            upon successive forward() calls. Sets self.eval_models
        """
        if models is None:
            self.eval_models = self.models
        else:
            if isinstance(models, str):
                models = [models]
            self.eval_models = models

    def forward(self, *args, prior_cache=None):
        """
        Forward pass sky models and append in a list
        or sum the sky maps and return a sky_component
        MapData object (or a list of these)
        """
        # forward each sky model
        sky_components = []
        for mod in self.eval_models:
            skycomp = self[mod].forward(prior_cache=prior_cache)
            sky_components.append(skycomp)

        if self.sum_output:
            # use first sky_component as placeholder
            output = sky_components[0]
            sky = output.data

            # iterate over sky components and sum with sky
            for i, (comp, mod) in enumerate(zip(sky_components[1:], self.eval_models[1:])):
                # check if we need to index sky or comp
                if self.index is not None and mod in self.index:
                    data = comp.data
                    idx = self.index[mod]
                    assert isinstance(idx, (tuple, list))
                    # check if indexing is provided for comp
                    if idx[1] is not None:
                        data = comp.data.index_select(-1, idx[1])
                    # check if indexing is provided for sky
                    if idx[0] is not None:
                        sky = sky.index_add(-1, idx[0], data)
                    else:
                        sky += data.to(self.device)
                else:
                    sky += comp.data.to(self.device)

            # assign sky to output
            output.data = sky

            # make sure other keys are on the same device
#            for k in output:
#                if isinstance(output[k], torch.Tensor):
#                    if output[k].device.type != self.device:
#                        output[k] = output[k].to(self.device)
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
            self[model].freq_interp(freqs, kind)

    def push(self, device):
        dtype = isinstance(device, torch.dtype)
        for model in self.models:
            self[model].push(device)
        if not dtype:
            self.device = device
            if self.index is not None:
                for k, v in self.index.items():
                    self.index[k] = (utils.push(v[0], device),
                                     utils.push(v[1], device))


def ang_index(theta, phi, theta_min=None, theta_max=None, phi_min=None, phi_max=None):
    """
    Given two theta (co-lat) and phi (azimuth) arrays and possible cuts in theta and phi,
    return an integer indexing tensor on the theta, phi arrays

    Parameters
    ----------
    theta : array
        Colatitude in rad or degree
    phi : array
        Azimuth in rad or degree
    theta_min, theta_max : float
        Constraints on theta
    phi_min, phi_max : float
        Constraints on phi

    Returns
    -------
    tensor
    """
    idx = torch.ones(len(theta), dtype=bool)
    if phi_min:
        idx = idx & (phi >= phi_min)
    if phi_max:
        idx = idx & (phi <= phi_max)
    if theta_min:
        idx = idx & (theta >= theta_min)
    if theta_max:
        idx = idx & (theta <= theta_max)

    return torch.where(idx)[0]


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
    sky = PointSky(params, angs, R=R, parameter=parameter)

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
        sky_comp : tensor or MapData
            Of shape (4, 1, ...) or (2, 2, ...)
            holding [IQUV] or [[I, Q], [U, V]] respectively.
            Can also be a sky_component dictionary, which will
            index sky_comp['sky'] as the sky tensor

        Returns
        -------
        tensor or MapData
        """
        # check if params is actually a sky_component
        if isinstance(sky_comp, dataset.MapData):
            sky_comp.data = self.forward(sky_comp.data, prior_cache=prior_cache)
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
    assert hasattr(obj.R, 'Alm')

    # get indexing tensor
    s = obj.R.Alm.select(lmin=lmin, lmax=lmax, mmin=mmin, mmax=mmax, other=other)

    # now index params
    with torch.no_grad():
        if obj.p0 is not None:
            obj.p0 = obj.p0[..., s, :]
        obj.params.data = obj.params.data[..., s, :]


def eqarea_grid(resol):
    """
    Create an equal-area grid in phi, theta (longitude, colatitude)
    given a specified d_phi resolution at the equator. The pixel area
    of all cells in resol^2

    Parameters
    ----------
    resol : float
        Resolution of grid in degrees
        (i.e. sidelength of cell at equator)

    Returns
    -------
    theta : ndarray
        Theta samples (colat) in radians
    phi : ndarray
        Phi samples (lat) in radians
    """
    n = int(2*np.pi / (resol * np.pi / 180))
    phi, dphi = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)

    t = np.arange(0, 1, dphi)
    t = np.concatenate([t[::-1], -t[1:]])
    theta = np.arccos(t)

    return theta, phi


def index_sky_pixels(angs_large, angs_small):
    """
    Given a large area sampling of the sky, and a smaller
    but overlapping sampling of the sky, compute
    the indices that map angs_small into angs_large.
    All angles in angs_small must have a counterpart
    in angs_large, but not vice versa.

    Parameters
    ----------
    angs_large : tensor
        The phi and theta tensors in degrees (2, Npixels_large) for
        a large area sampling of the sky.
    angs_small : tensor
        The phi and theta tensors in degrees (2, Npixels_small) for
        a smaller, subset sampling of the sky. 
    atol : float, optional
        Absolute tolerance for matching phi and theta arrays.

    Returns
    -------
    idx : tensor
        The indexing tensor that satisfies:
        angs_large[:, idx] = angs_small
    """
    idx = []
    for ph, th in zip(*angs_small):
        # ph % 360 % 360 ensures -1e-15 -> 0 instead of 360
        idx.append(np.argmin(np.linalg.norm(angs_large.T - torch.tensor([ph % 360 % 360, th]), axis=1)))

    return torch.as_tensor(idx)

