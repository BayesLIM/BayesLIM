"""
Module for torch sky models and relevant functions
"""
import torch
import numpy as np
from scipy.special import sph_harm

from . import utils


class SkyBase(torch.nn.Module):
    """
    Base class for various sky model representations
    """
    def __init__(self, params, kind, freqs, R=None, parameter=True):
        """
        Base class for a torch sky model representation.

        Parameters
        ----------
        params : tensor or list of tensors
            A sky model parameterization as a tensor or list
            of tensors to be pushed through the response R().
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
        parameter : bool
            If True, treat params as variables to be fitted,
            otherwise hold them fixed as their input value
        """
        super().__init__()
        self.params = params
        oneparam = True
        if isinstance(self.params, (list, tuple)):
            oneparam = False
            _params = []
            for i, p in enumerate(self.params):
                if parameter:
                    p = torch.nn.Parameter(p)
                name = "param{}".format(i)
                setattr(self, name, p)
                _params.append(p)
            if parameter:
                self.params = _params
        else:
            if parameter:
                self.params = torch.nn.Parameter(self.params)
        self.kind = kind
        if R is None:
            if oneparam: R = lambda x: x
            else: R = lambda x: x[0]
        self.R = R
        self.freqs = freqs
        self.Nfreqs = len(freqs)


class PointSourceModel(SkyBase):
    """
    Point source sky model with fixed
    source locations but variable flux density.
    Relates source flux parameterization
    to per-frequency, per-stokes, per-source
    flux density vector.

    Returns point source flux density and their sky
    locations in equatorial coordinates.
    """
    def __init__(self, params, angs, freqs, R=None, parameter=True):
        """
        Fixed-location point source model with
        parameterized flux density.

        Parameters
        ----------
        params : list of tensors
            A list of point source parameters. Bare minimum, the
            first element should be a tensor of shape
            (Npol, Npol, Nfreqs, Nsources), also referred to as "sky",
            containing the flux density of each source.
            Npol is the number of feed polarizations.
            The first two axes are the coherency matrix B:

            .. math::

                B = \left(
                    \begin{array}{cc}I + Q & U + iV \\
                    U - iV & I - Q \end{array}
                \right)

            See bayescal.sky.stokes2linear() for details.
            Additionally, params may contain other tensor parameters
            e.g. a frequency power law, which should be expected
            by the R function.
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
            P = bayescal.sky.PointSourceModel([amps, alpha],
                                              angs, Nfreqs, R=R)

        """
        super().__init__(params, 'point', freqs, R=R, parameter=parameter)
        self.angs = angs
        self.Npol = len(self.param0)
        if R is None:
            # dummy params eval
            R = lambda x: x[0]
        self.R = R

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
                (Npol, Npol, Nsources, Nfreqs)
            angs : tensor
                Sky source locations (RA, Dec) [deg]
                (2, Nsources)
        """
        # fed params or attr params
        if params is None:
            _params = self.params
        else:
            _params = params

        # pass through response
        return dict(kind=self.kind, sky=self.R(_params), angs=self.angs)


class PixelModel(SkyBase):
    """
    Pixelized model (e.g. Healpix) of the sky
    specific intensity (aka brightness or temperature)
    at fixed locations in Equatorial coordinates
    but with variable amplitude.

    While the input sky model (params) should be in units of
    specific intensity (Kelvin or Jy / str), the output
    of the forward model is in flux density [Jy]
    (i.e. we multiply by each cell's solid angle).
    """
    def __init__(self, params, angs, freqs, areas, R=None, parameter=True):
        """
        Pixelized model of the sky brightness distribution.
        This can be parameterized in any generic way via params,
        but the output of R(params) must be
        a representation of the sky brightness at fixed
        cells, which are converted to flux density
        by multiplying by each cell's solid angle.

        Parameters
        ----------
        params : list of tensors
            A list of source parameters. Bare minimum, the
            first element should be a tensor of shape
            (Npol, Npol, Nfreqs, Npix), where Npix is the number
            of free parameters. This could be individual pixels, but
            it could also be some sparse parameterization that is eventually
            mapped to pixel space after passing through R().
            Npol is the number of feed polarizations.
            The first two axes are the coherency matrix B:

            .. math::

                B = \left(
                    \begin{array}{cc}I + Q & U + iV \\
                    U - iV & I - Q \end{array}
                \right)

            See bayescal.sky.stokes2linear() for details.
            Additionally, params may contain other tensor parameters
            e.g. a frequency power law, which should be expected
            by the R function.
        angs : tensor
            Point source unit vectors on the sky in equatorial
            coordinates of shape (2, Nsources), where the
            last two axes are RA and Dec.
        freqs : tensor
            Frequency array of sky model [Hz].
        areas : tensor
            Contains the solid angle of each pixel. This is multiplied
            into the final sky modle, and thus needs to be of shape
            (1, 1, 1, Npix) to allow for broadcasting rules to apply.
        R : callable, optional
            An arbitrary response function for the sky model, mapping
            self.params to a sky pixel tensor of shape
            (Npol, Npol, Nfreqs, Npix)
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.
        """
        super().__init__(params, 'pixel', freqs, R=R, parameter=parameter)
        self.angs = angs
        self.areas = areas
        self.Npol = len(self.param0)

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
            _params = self.params
        else:
            _params = params

        # pass through response
        sky = self.R(_params) * self.areas
        return dict(kind=self.kind, sky=sky, angs=self.angs)


class SphHarmModel(SkyBase):
    """
    Spherical harmonic expansion of a sky temperature field
    at pointing direction s and frequency f

    .. math::

        T(s, f) = \sum_{lm} = Y_{lm}(s) a_{lm}(f)

    where Y is a spherical harmonic of order l and m
    and a is the coefficient.
    """
    def __init__(self, params, lms, freqs, R=None, parameter=True):
        """
        Spherical harmonic representation of the sky brightness.
        Note that this model does not by default transform the
        representation to the spatial domain on the sky.
        By default, this model operates in harmonic space all the
        way down to the visibility level, however, there are methods
        to transform to the sky spatial domain if desired.

        Parameters
        ----------
        params : list of tensors
            Spherical harmonic parameterization of the sky.
            The first element of params must be a tensor holding
            the a_lm coefficients of shape (Npol, Npol, Ncoeff, ...).
            Additional tensors may expand the parameterization.
            Example includes spherical harmonic for a single or
            multiple frequency channels, or a spherical
            Fourier-bessel for multiple channels.
        lms : array
            Array holding spherical harmonic orders (l, m) of shape
            (Ncoeff, 2).
        freqs : tensor
            Frequency array of sky model [Hz].
        R : callable, optional
            An arbitrary response function for the
            spherical harmonic model, mapping input self.params
            to an output a_lm basis of shape
            (Npol, Npol, Ncoeff, Nfreqs).
        parameter : bool, optional
            If True, treat params as parameters to be fitted,
            otherwise treat as fixed to its input value.
        """
        super().__init__(params, 'alm', freqs, R=R, parameter=parameter)
        self.lms = lms
        self.Npol = len(params)

    def forward(self, params=None):
        """
        Forward pass the sky parameters

        Parameters
        ----------
        params : list of tensors, optional
            Set of parameters to use instead of self.params.

        Returns
        -------
        kind : str
            Kind of sky model ['point', 'pixel', 'alm']
        sky : tensor
            Sky flux density at discrete locations
        """
        # apply fed params or attr params
        if params is None:
            _params = self.params
        else:
            _params = params

        # pass through response
        return dict(kind=self.kind, sky=self.R(_params), lms=self.lms)

    def alm2sky(self):
        pass




class CompositeModel(torch.nn.Module):
    """
    Multiple sky models
    """
    def __init__(self, models):
        """
        Multiple sky models to be evaluated
        and returned in a list

        Parameters
        ----------
        models : list
            List of sky model objects
        """
        self.models = models

    def forward(self, models=None):
        """
        Forward pass sky models and append in a list

        Parameters
        ----------
        models : list
            List of sky models to use instead of self.models

        Returns
        -------
        list
            List of each sky model output
        """
        _models = self.models
        if models is not None:
            _models = models

        return [mod.forward() for mod in _models]


def stokes2linear(stokes, dtype=torch.cfloat):
    """
    Convert Stokes parameters to coherency matrix
    for xyz cartesian (aka linear) feed basis.
    This can be included at the beginning of
    the response matrix (R) of any of the sky model
    objects in order to properly account for Stokes
    Q, U, V parameters in your sky model.

    Parameters
    ----------
    stokes : tensor
        Holds the Stokes parameter of your generalized
        sky model parameterization, of shape (4, ...)
        with the zeroth axis holding the Stokes parameters
        in the order of [I, Q, U, V].
    dtype : torch dtype object
        dtype of output coherency matrix, default is cfloat

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
    B = torch.zeros(2, 2, stokes.shape[1:], dtype=dtype)
    B[0, 0] = stokes[0] + stokes[1]
    B[0, 1] = stokes[2] + 1j * stokes[3]
    B[1, 0] = stokes[2] - 1j * stokes[3]
    B[1, 1] = stokes[0] - stokes[1]

    return B


def gen_lm(lmax, real=True):
    """
    Generate array of l and m parameters

    Parameters
    ----------
    lmax : int
        Maximum l parameter
    real : bool, optional
        If True, treat sky as real-valued (default)
        so truncate negative m values.

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
    Nsources = angs.shape[1]

    # iterate and generate Ylms
    Y = torch.zeros((Nsources, Nalms), dtype=dtype)

    for i, lm in enumerate(lms):
        Y[:, i] = torch.tensor(sph_harm(lm[1], lm[0], angs[0], angs[1]), dtype=dtype)

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
    pass


def alm2sky(alm, angs):
    """

    """
    pass


def alm_convolve(alm, conv):
    """
    """
    pass




