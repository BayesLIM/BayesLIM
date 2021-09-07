"""
Module for torch sky models and relevant functions
"""
import torch
import numpy as np
from scipy import special, interpolate

from . import utils, cosmology
from .utils import _float, _cfloat


class SkyBase(torch.nn.Module):
    """
    Base class for various sky model representations
    """
    def __init__(self, params, kind, freqs, R=None, name=None, parameter=True):
        """
        Base class for a torch sky model representation.

        Parameters
        ----------
        params : tensor
            A sky model parameterization as a tensor to
            be pushed through the response function R().
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
            (Npol, Npol, Nfreqs, Nsources)
        name : str, optional
            Name for this object
        parameter : bool
            If True, treat params as variables to be fitted,
            otherwise hold them fixed as their input value
        """
        super().__init__()
        self.params = params
        self.device = self.params.device
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        self.kind = kind
        if R is None:
            R = DefaultResponse()
        self.R = R
        self.name = name
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
        self.params = utils.push(self.params, device)
        self.device = self.params.device
        if hasattr(self, 'angs'):
            self.angs = self.angs.to(device)
        for attr in attrs:
            setattr(self, attr, getattr(self, attr).to(device))
        self.R.push(device)

    def freq_interp(self, freqs, kind='linear'):
        """
        Interpolate params onto new set of frequencies
        if response freq_mode is channel. If freq_mode is
        poly or powerlaw, just update response frequencies

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
            interp = interpolate.interp1d(utils.tensor2numpy(self.freqs),
                                          utils.tensor2numpy(self.params),
                                          axis=2, kind=kind, fill_value='extrapolate')
            params = torch.as_tensor(interp(utils.tensor2numpy(freqs)), device=self.device,
                                     dtype=self.params.dtype)
            if self.params.requires_grad:
                self.params = torch.nn.Parameter(params)
            else:
                self.params = params
            self.freqs = freqs

        self.R.freqs = freqs
        self.R._setup()


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
    def __init__(self, params, angs, freqs, R=None, name=None, parameter=True):
        """
        Fixed-location point source model with
        parameterized flux density.

        Parameters
        ----------
        params : tensor
            Point source flux parameterization adopted by R().
            In general, this is of shape (Npol, Npol, Ncoeff, Nsources),
            where Ncoeff is the chosen parameterization across frequency.
            For no parameterization (default) this should be a tensor
            of shape (Npol, Npol, Nfreqs, Nsources).
            Npol is the number of feed polarizations, and
            the first two axes are the coherency matrix B:

            .. math::

                B = \left(
                    \begin{array}{cc}I + Q & U + iV \\
                    U - iV & I - Q \end{array}
                \right)

            See bayeslim.sky_model.stokes2linear() for details.
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
            (Npol, Npol, Nfreqs, Nsources)
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.

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
        super().__init__(params, 'point', freqs, R=R, name=name, parameter=parameter)
        self.angs = angs

    def forward(self, params=None):
        """
        Forward pass the sky parameters

        Parameters
        ----------
        params : list of tensors, optional
            Set of parameters to use instead of self.params.

        Returns
        -------
        dictionary
            kind : str
                Kind of sky model ['point', 'pixel', 'alm']
            sky : tensor
                Source brightness at discrete locations
                (Npol, Npol, Nfreqs, Nsources)
            angs : tensor
                Sky source locations (RA, Dec) [deg]
                (2, Nsources)
        """
        # fed params or attr params
        if params is None:
            params = self.params

        # pass through response
        return dict(kind=self.kind, sky=self.R(params), angs=self.angs)


class PointSkyResponse:
    """
    Frequency parameterization of point sources at
    fixed locations but variable flux wrt frequency
    options include
        - channel : vary all frequency channels
        - poly : low-order polynomial across freqs, centered at f0.
        - powerlaw : amplitude and freq exponent, centered at f0.
    """
    def __init__(self, freqs, freq_mode='poly', f0=None,
                 device=None, Ndeg=None):
        """
        Choose a frequency parameterization for PointSky

        Parameters
        ----------
        freqs : tensor
            Frequency array [Hz]
        freq_mode : str, optional
            options = ['channel', 'poly', 'powerlaw']
            Frequency parameterization mode. Choose between
            channel - each frequency is a parameter
            poly - polynomial basis of Ndeg
            powerlaw - amplitude and powerlaw basis anchored at f0
        f0 : float, optional
            Fiducial frequency [Hz]. Used for poly and powerlaw.
        device : str, optional
            Device of point source params
        Ndeg : int, optional
            Polynomial degrees if freq_mode is 'poly'

        Notes
        -----
        The ordering of the coeff axis in params should be
            poly - follows that of utils.gen_poly_A()
            powerlaw - ordered as (amplitude, exponent)
        """
        self.freqs = freqs
        self.f0 = f0
        self.freq_mode = freq_mode
        self.Ndeg = Ndeg
        self.device = device
        self._setup()

        # construct _args for str repr
        self._args = dict(freq_mode=self.freq_mode)

    def _setup(self):
        # setup
        if self.freq_mode == 'poly':
            self.dfreqs = (self.freqs - self.f0) / 1e6  # MHz
            self.A = utils.gen_poly_A(self.dfreqs, self.Ndeg, device=self.device)

    def __call__(self, params):
        if self.freq_mode == 'channel':
            return params
        elif self.freq_mode == 'poly':
            return self.A @ params
        elif self.freq_mode == 'powerlaw':
            return params[..., 0, :] * (self.freqs[None, None, :, None] / self.f0)**params[..., 1, :]

    def push(self, device):
        self.A = self.A.to(device)
        self.freqs = self.freqs.to(device)
        self.device = device

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
    def __init__(self, params, angs, freqs, px_area, R=None, name=None, parameter=True):
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
            (Npol, Npol, Nfreq_coeff, Nsky_coeff), where Nsky_coeff is
            the free parameters describing angular fluctations, and Nfreq_coeff
            is the number of free parameters describing frequency fluctuations,
            both of which should be expected by the response function R().
            By default, this is just Nfreqs and Npix, respectively.
            Npol is the number of feed polarizations.
            The first two axes are the coherency matrix B:

            .. math::

                B = \left(
                    \begin{array}{cc}I + Q & U + iV \\
                    U - iV & I - Q \end{array}
                \right)

            See bayeslim.sky_model.stokes2linear() for details.
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
            (Npol, Npol, Nfreqs, Npix)
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.
        """
        super().__init__(params, 'pixel', freqs, R=R, name=name, parameter=parameter)
        self.angs = angs
        self.px_area = px_area

    def forward(self, params=None):
        """
        Forward pass the sky parameters.

        Parameters
        ----------
        params : list of tensors, optional
            Set of parameters to use instead of self.params.

        Returns
        -------
        dictionary
            kind : str
                Kind of sky model ['point', 'pixel', 'alm']
            amps : tensor
                Pixel flux density at fixed locations on the sky
                (Npol, Npol, Nfreqs, Npix)
            angs : tensor
                Sky source locations (RA, Dec) [deg]
                (2, Npix)
        """
        # apply fed params or attr params
        if params is None:
            params = self.params

        # pass through response
        sky = self.R(params) * self.px_area
        return dict(kind=self.kind, sky=sky, angs=self.angs)


class PixelSkyResponse:
    """
    Spatial and frequency parameterization for PixelSky

    options for spatial parameterization include
        - 'pixel' : sky pixel
        - 'alm' : spherical harmonic

    options for frequency parameterization include
        - 'channel' : frequency channels
        - 'poly' : low-order polynomials
        - 'powerlaw' : power law model
        - 'bessel' : spherical bessel j_l (for spatial mode 'alm')
            For this mode, the all elements in params must be
            from a single l mode
    """
    def __init__(self, freqs, spatial_mode='pixel', freq_mode='channel',
                 device=None, transform_order=0, cosmo=None,
                 spatial_kwargs={}, freq_kwargs={}):
        """
        Parameters
        ----------
        freqs : tensor
            Frequency bins [Hz]
        spatial_mode : str, optional
            Choose the spatial parameterization (default is pixel)
            options = ['pixel', 'alm']
        freq_mode : str, optional
            Choose the freq parameterization (default is channel)
            options = ['channel', 'poly', 'bessel']
        device : str, optional
            Device to put model on
        transform_order : int, optional
            0 - spatial then frequency transform (default)
            1 - frequency then spatial transform
        cosmo : Cosmology object
        spatial_kwargs : dict, optional
            Kwargs used to generate spatial transform matrix
            lms : array, holding l, m vaues of shape (2, Nlm)
            theta, phi : array, holds co-latitude and azimuth
                angles [deg] of pixel model, used for generating
                Ylm in alm mode
            Ylm : tensor, holds (Ncoeff, Npix) Ylm transform
                matrix. If None, will compute it given l, m
        freq_kwargs : dict, optional
            Kwargs used to generate freq transform matrix
            f0 : float, fiducial frequency [Hz], used for poly
            Ndeg : int, number of degrees, used for poly
            kbins : ndarray, wavevector bins [Mpc-1], used for bessel
            radial_method : str, radial convention, used for bessel
            kmax : float, maximum k to compute
            decimate : bool, if True, decimate every other kbin
            kbin_file : str, filepath to csv of k-bins, used for bessel
        """
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.spatial_mode = spatial_mode
        self.freq_mode = freq_mode
        self.device = device
        self.transform_order = transform_order
        self.spatial_kwargs = spatial_kwargs
        self.freq_kwargs = freq_kwargs
        if cosmo is None:
            cosmo = cosmology.Cosmology()
        self.cosmo = cosmo

        # assertions
        if self.freq_mode == 'bessel':
            assert self.spatial_mode == 'alm'

        self._setup()

        # construct _args for str repr
        self._args = dict(freq_mode=self.freq_mode, spatial_mode=self.spatial_mode)

    def _setup(self):
        # freq setup
        self.A, self.jl = None, None
        if self.freq_mode == 'poly':
            self.dfreqs = (self.freqs - self.freq_kwargs['f0']) / 1e6  # MHz
            self.A = utils.gen_poly_A(self.dfreqs, self.freq_kwargs['Ndeg'],
                                      basis=getattr(self.freq_kwargs, 'basis', 'direct'),
                                      whiten=getattr(self.freq_kwargs, 'whiten', None),
                                      device=self.device)
        elif self.freq_mode == 'bessel':
            assert self.spatial_transform == 'alm'
            # compute comoving line of sight distances
            self.z = self.cosmo.f2z(utils.tensor2numpy(self.freqs))
            self.r = self.cosmo.comoving_distance(self.z).value
            self.dr = self.r.max() - self.r.min()
            if 'jl' in self.freq_kwargs and 'kbins' in self.freq_kwargs:
                self.jl = self.freq_kwargs['jl']
                self.kbins = self.freq_kwargs['kbins']
            else:
                jl, kbins = utils.gen_bessel2freq(self.spatial_kwargs['l'],
                                                  utils.tensor2numpy(self.freqs), self.cosmo,
                                                  kmax=getattr(freq_kwargs, 'kmax'),
                                                  decimate=getattr(freq_kwargs, 'decimate', True),
                                                  dk_factor=getattr(freq_kwargs, 'dk_factor', 1e-1),
                                                  device=self.device,
                                                  method=getattr(freq_kwargs, 'radial_method', 'default'),
                                                  Nproc=getattr(freq_kwargs, 'Nproc', None),
                                                  Ntask=getattr(freq_kwargs, 'Ntask', 10),
                                                  renorm=getattr(freq_kwargs, 'renorm', False))
                self.jl = jl
                self.kbins = kbins

        # spatial setup
        self.Ylm = None
        if self.spatial_mode == 'alm':
            if 'Ylm' in self.spatial_kwargs:
                self.Ylm = self.spatial_kwargs['Ylm']
            else:
                self.Ylm = utils.gen_sph2pix(self.spatial_kwargs['theta'] * D2R,
                                             self.spatial_kwargs['phi'] * D2R,
                                             self.spatial_kwargs['l'],
                                             self.spatial_kwargs['m'],
                                             device=self.device, real_field=True)

    def spatial_transform(self, params):
        """
        Forward model the sky params tensor
        through a spatial transform.

        Parameters
        ----------
        params : tensor
            Sky model parameters (Npol, Npol, Ndeg, Ncoeff)
            where Ndeg may equal Nfreqs, and Ncoeff
            are the coefficients for the sky representations.

        Returns
        -------
        tensor
            Sky model of shape (Npol, Npol, Ndeg, Npix)
        """
        if self.spatial_mode == 'pixel':
            return params
        elif self.spatial_mode == 'alm':
            return params @ self.Ylm

    def freq_transform(self, params):
        """
        Forward model the sky params tensor
        through a frequency transform.

        Parameters
        ----------
        params : tensor
            Sky model parameters (Npol, Npol, Ndeg, Ncoeff)
            where Ncoeff may equal Npix, and Ndeg
            are the coefficients for the frequency representations.
    
        Returns
        -------
        tensor
            Sky model of shape (Npol, Npol, Nfreqs, Ncoeff)
        """
        if self.freq_mode == 'channel':
            return params
        elif self.freq_mode == 'poly':
            return (params.transpose(-1, -2) @ self.A.T).transpose(-1, -2)
        elif self.freq_mode == 'powerlaw':
            return params[..., 0, :] * (self.freqs / self.freq_kwargs['f0'])**params[..., 1, :]
        elif self.freq_mode == 'bessel':
            out = torch.zeros(params.shape[:-2] + (self.Nfreqs,) + params.shape[-1:],
                              device=params.device, dtype=params.dtype)
            for l in np.unique(self.freq_kwargs['l']):
                inds = self.freq_kwargs['l'] == l
            out += (params[:, :, inds].transpose(-1, -2) @ self.jl[l]).transpose(-1, -2)
            return out

    def __call__(self, params):
        if params.device != self.device:
            params = utils.push(params, self.device)
        if self.transform_order == 0:
            params = self.spatial_transform(params)
            params = self.freq_transform(params)
        else:
            params = self.freq_transform(params)
            params = self.spatial_transform(params)

        return params

    def push(self, device):
        if self.spatial_mode == 'alm':
            self.Ylm = self.Ylm.to(device)

        if self.freq_mode == 'poly':
            self.A = self.A.to(device)
        elif self.freq_mode == 'bessel':
            for k in self.jl:
                self.jl[k] = self.jl[k].to(device)

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
            (Npol, Npol, Nfreqs, Ncoeff). Nfreqs may also be
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
            (Npol, Npol, Nfreqs, Ncoeff).
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.
        """
        raise NotImplementedError


class CompositeModel(torch.nn.Module):
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

    def forward(self, *args):
        """
        Forward pass sky models and append in a list
        or sum the sky maps and return a sky_component
        dictionary
        """
        # forward the models
        sky_components = [getattr(self, mod).forward() for mod in self.models]
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
        poly or powerlaw, just update response frequencies

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
            model.push(device)
        self.device = device


def parse_catalogue(catfile, freqs, device=None,
                    parameter=False, freq_interp='linear'):
    """
    Read a point source catalogue YAML file.
    See bayeslim.data.DATA_PATH for examples.

    Parameters
    ----------
    catfile : str
        Path to a YAML point source catalogue file
    freqs : tensor
        Frequencies to evaluate model [Hz].
        If catalogue is 'channel' along the frequency
        axis, then it is interpolated with freq_interp
        onto freqs bins.

    Returns
    -------
    tensor
        PointSky object
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

    if d['mode'] == 'channel':
        # collect Stokes I fluxes at specified frequencies
        S = np.array([sources['freq{}'.format(i)] for i in range(len(d['freqs']))])

        # interpolate onto freqs
        interp = interpolate.interp1d(d['freqs'], S, kind=freq_interp, axis=0,
                                      fill_value='extrapolate')
        params = torch.tensor(interp(freqs), dtype=_float())[None, None, :, :]

    elif d['mode'] == 'powerlaw':
        # ensure frequencies are float
        d['mode_kwargs']['f0'] = float(d['mode_kwargs']['f0'])

        # collect parameters
        params = torch.tensor([np.array(sources['amp']), np.array(sources['alpha'])])[None, None, :, :]

    else:
        raise NotImplementedError

    R = PointSkyResponse(freqs, freq_mode=d['mode'], device=device, **d['mode_kwargs'])
    sky = PointSky(params, angs, freqs, R=R, parameter=parameter)

    if 'polarizaton' in d:
        # still under development
        Nsources = params.shape[-1]
        sparams = torch.tensor(np.array(d['Qfrac'], d['Ufrac'], d['Vfrac']).reshape(3, 1, Nsources))
        stokes = PolStokesModel(sparams, parameter=parameter)
        sky = torch.nn.Sequential(sky, stokes)

    return sky


def stokes2linear(S):
    """
    Convert Stokes parameters to coherency matrix
    for xyz cartesian (aka linear) feed basis.
    This can be included at the beginning of
    the response matrix (R) of any of the sky model
    objects in order to properly account for Stokes
    Q, U, V parameters in a sky model.

    .. math::

        S = \left(
            \begin{array}{c}I\\ Q\\ U\\ V\end{array}
        \right)

    Parameters
    ----------
    S : tensor
        Holds the Stokes parameter of a generalized
        sky model parameterization, of shape (4, ...)
        with the zeroth axis holding the Stokes parameters
        in the order of [I, Q, U, V].

    Returns
    -------
    B : tensor
        Coherency matrix of electric field in xyz cartesian
        basis of shape (2, 2, ...) with the form

        .. math::

            B = \left(
                \begin{array}{cc}I + Q & U + iV \\
                U - iV & I - Q \end{array}
            \right)
    """
    device = S.device
    if len(S) == 1:
        # assume Stokes I was fed
        B = torch.zeros(1, 1, *S.shape[1:], dtype=_float(), device=device)
        B[0, 0] = S[0]
    elif len(S) == 3:
        # assume fractional Q, U, V was fed.
        B = torch.zeros(2, 2, *S.shape[1:], dtype=_float(), device=device)
        B[0, 0] = 1 + S[0]
        B[0, 1] = S[1] + 1j * S[2]
        B[1, 0] = S[1] - 1j * S[2]
        B[1, 1] = 1 - S[0]
    elif len(S) == 4:
        # assume 4-Stokes was fed
        B = torch.zeros(2, 2, *S.shape[1:], dtype=_float(), device=device)
        B[0, 0] = S[0] + S[1]
        B[0, 1] = S[2] + 1j * S[3]
        B[1, 0] = S[2] - 1j * S[3]
        B[1, 1] = S[0] - S[1]

    return B


class PolStokesModel(torch.nn.Module):
    """
    A model for Stokes Q, U, V parameters of a SkyBase object

    The output of a Stokes I-only sky model can be pushed
    through this to become a polarized Stokes sky model, with
    parameters dictating the relationship between
    Stokes I and Stokes Q, U, V.

    .. math::

            \left(
                \begin{array}{cc}I + Q & U + iV \\
                U - iV & I - Q \end{array}
            \right) = 
            \left(
                \begin{array}{cc}1 + f_Q & f_U + i*f_V \\
                f_U - i*f_V & 1 - f_Q \end{array}
            \right)\cdot I

    Note: the fractional stokes parameters should not
    sum in quadrature to more than 1, i.e.

    .. math::

        f_Q^2 + f_U^2 + f_V^2 \le 1

    which should be enforced by setting a prior
    on their quadrature sum.
    """
    def __init__(self, params, parameter=True):
        """
        Parameters
        ----------
        params : tensor
            The fractional relationship between
            Stokes I and Stokes Q, U, V, i.e.
            f_Q, f_U, f_V, of shape
            (3, Nfreqs, Nsources), where both
            Nfreqs and Nsources can be 1 to
            reduce degrees of freedom and then
            be broadcasted on the sky model.
        parameter : bool, optional
            Treat params as tunable parameters,
            or leave them fixed as is.
        """
        super().__init__()
        self.params = params
        self.device = self.params.device
        if parameter:
            self.params = torch.nn.Parameter(self.params)

    def push(self, device):
        """
        Parameters
        ----------
        device : str
            Device to push to, e.g. 'cpu', 'cuda:0'
        """
        self.params = utils.push(self.params, device)
        self.device = self.params.device

    def forward(self, sky_comp):
        """
        Forward model polarization state onto Stokes I basis

        Parameters
        ----------
        sky_comp : dict or tensor
            A Stokes-I sky component dictionary output from
            a SkyBase subclass, or a sky parameter tensor
            of shape (1, 1, Nfreqs, Nsources). To reduce
            dimensionality, one can set Nfreqs or Nsources = 1
            and will broadcast the parameter across the sky model.

        Returns
        -------
        dict or tensor
            Polarized coherency matrix of shape (2, 2, Nfreqs, Nsources)
        """
        if isinstance(sky_comp, dict):
            sky_comp['sky'] = self.forward(sky_comp['sky'])
            return sky_comp

        # get fractional polarized coherency matrix
        B = stokes2linear(self.params)
        return B * sky_comp

