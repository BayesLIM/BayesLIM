"""
Utility module
"""
import numpy as np
import torch
from scipy.special import voigt_profile
from scipy.signal import windows
from scipy.interpolate import interp1d
import copy
import warnings
import os
from astropy import constants
import h5py

from . import special, version

# try to import healpy
try:
    import healpy
    import_healpy = True
except ImportError:
    import_healpy = False
if not import_healpy:
    try:
        # note this will have more limited capability
        # than healpy, but can do what we need
        from astropy_healpix import healpy
        import_healpy = True
    except ImportError:
        warnings.warn("could not import healpy")

# try to import nullcontext (for Py>=3.7)
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result

# try to import sklearn
try:
    from sklearn.neighbors import BallTree
    import_sklearn = True
except ImportError:
    import_sklearn = False
    warnings.warn("could not import sklearn")


D2R = np.pi / 180
viewreal = torch.view_as_real
viewcomp = torch.view_as_complex


def _float(numpy=False):
    """Manipulate with torch.set_default_dtype()"""
    float_type = torch.get_default_dtype()
    if not numpy:
        return float_type
    else:
        if float_type == torch.float16:
            return np.float16
        elif float_type == torch.float32:
            return np.float32
        elif float_type == torch.float64:
            return np.float64


def _cfloat(numpy=False):
    """Manipulate with torch.set_default_dtype()"""
    float_type = torch.get_default_dtype()
    if not numpy:
        if float_type == torch.float64:
            return torch.complex128
        elif float_type == torch.float32:
            return torch.complex64
        elif float_type == torch.float16:
            return torch.complex32
    else:
        if float_type == torch.float64:
            return np.complex128
        elif float_type == torch.float32:
            return np.complex64
        elif float_type == torch.float16:
            return np.complex32


######################################
######### Sky Modeling Tools #########
######################################

def colat2lat(theta, deg=True):
    """
    Convert colatitude to latitude and vice versa

    Parameters
    ----------
    theta : ndarray
        Colatitude
    deg : bool, optional
        If True, theta is in deg, otherwise in rad

    Returns
    -------
    ndarray
        Converted angles
    """
    if deg:
        return 90 - theta
    else:
        return np.pi / 2 - theta


def stripe_tukey_mask(theta, theta_min, theta_max,
                      phi, phi_min, phi_max,
                      theta_alpha=0.5, phi_alpha=0.5):
    """
    Generate a tukey apodization mask for a spherical stripe

    Parameters
    ----------
    theta : array_like
        co-latitude (polar) [rad]
    theta_min, theta_max : float
        minimum and maximum extent in theta of the stripe [rad]
    phi : array_like
        azimuth [rad]
    phi_min, phi_max : float
        minimum and maximum extent in phi of the stripe [rad]
    theta_alpha, phi_alpha : float, optional
        alpha parameter of the tukey window for the theta
        and phi axes respectively

    Returns
    -------
    ndarray
        Apodization mask wrt theta, phi spanning [0, 1]
    """
    # theta mask: 5000 yields ang. resolution < nside 512
    th_arr = np.linspace(theta_min, theta_max, 5000, endpoint=True)
    mask = windows.tukey(5000, alpha=theta_alpha)
    theta_mask = interp1d(th_arr, mask, fill_value=0, bounds_error=False, kind='linear')(theta)
    # phi mask
    ph_arr = np.linspace(phi_min, phi_max, 5000, endpoint=True)
    mask = windows.tukey(5000, alpha=phi_alpha)
    phi_mask = interp1d(ph_arr, mask, fill_value=0, bounds_error=False, kind='linear')(phi)

    return theta_mask * phi_mask


def gen_linear_A(linear_mode, A=None, x=None, d0=None, logx=False,
                 whiten=True, x0=None, dx=None, Ndeg=None, basis='direct',
                 device=None, dtype=None, **kwargs):
    """
    Generate a linear mapping design matrix A

    Parameters
    ----------
    linear_mode : str
        One of ['poly', 'custom']
    A : tensor, optional
        (mode='custom') Linear mapping of shape (Nsamples, Nfeatures)
    x : tensor, optional
        (mode='poly') sample values
    d0 : float, optional
        (mode='poly') divide x by d0 before any other operation
        if provided
    logx : bool, optional
        If True, take logarithm of x before generating
        A matrix (mode='poly')
    whiten : bool, optional
        (mode='poly') whiten samples
    x0 : float, optional
        (mode='poly') center x by x0
    dx : float, optional
        (mode='poly') scale x by 1/dx
    Ndeg : int, optional
        (mode='poly') Number of poly degrees
    basis : str, optional
        (mode='poly') poly basis
    device : str, optional
        Device to push A to
    dtype : type, optional
        data type to cast A to. Default is float

    Returns
    -------
    A : tensor
        Design matrix of shape (Nsamples, Nfeatures)
    """
    dtype = dtype if dtype is not None else _float()
    if linear_mode == 'poly':
        A = gen_poly_A(x, Ndeg, basis=basis, d0=d0, logx=logx, whiten=whiten, x0=x0, dx=dx)
    elif linear_mode == 'custom':
        A = torch.as_tensor(A)

    return A.to(dtype).to(device)


def gen_poly_A(x, Ndeg, device=None, basis='direct', d0=None,
               logx=False, whiten=True, x0=None, dx=None):
    """
    Generate design matrix (A) for polynomial of Ndeg across x,
    with coefficient ordering

    .. math::

        y = Ax = a_0 * x^0 + a_1 * x^1 + a_2 * x^2 + \ldots

    Parameters
    ----------
    x : ndarray
    Ndeg : int
        Polynomial degree
    device : str, optional
        device to send A matrix to before return
    basis : str, optional
        Polynomial basis to use. See emupy.linear.setup_polynomial
        ['direct', 'legendre', 'chebyshevt', 'chebyshevu']
        direct (default) is a standard polynomial (x^0 + x^1 + ...)
    d0 : float, optional
        Divide x by x0 before any other operation if provided
    logx : bool, optional
        If True, take log of x before generating A matrix or whitening.
    whiten : bool, optional
        If True, center (i.e. subtract mean) and scale (i.e. range of -1, 1) x.
        Useful when using orthogonal polynomial bases
    x0 : float, optional
        If whiten, use this centering instead of x.mean()
    dx : float, optional
        If whiten, use this scaling instead of (x-x0).max()

    Returns
    -------
    A : tensor
        Polynomial design matrix (Nx, Ndeg)
    """
    x, _, _ = prep_xarr(x, d0=d0, logx=logx, whiten=whiten, x0=x0, dx=dx)

    # setup the polynomial
    from emupy.linear import setup_polynomial
    A = setup_polynomial(x[:, None], Ndeg - 1, basis=basis)[0]
    A = torch.as_tensor(A, dtype=_float(), device=device)

    return A


def prep_xarr(x, d0=None, logx=False,  whiten=False, x0=None, dx=None):
    """
    Prepare input x tensor for anchoring, log, and/or whitening

    Parameters
    ----------
    x : tensor
        Independent x tensor of shape (Nsample,).
        monotonically increasing but not necc. uniform
    d0 : float, optional
        Divide x by d0 before any other operation if provided
    logx : bool, optional
        Take log of x before whitening
    whiten : bool, optional
        Center and scale x to be from [-1, 1].
        Default w/ uniform sampling in x returns
        range of [-1+dx/2, 1-dx/2]
    x0 : float, optional
        Center x by this value
    dx : float, optional
        Scale the centered x by this value

    Returns
    -------
    x : New x tensor
    x0 : x0 value
    dx : dx value
    """
    # anchor by d0 if desired
    if d0:
        x = x / d0

    # logx if desired
    if logx:
        x = torch.log(x)

    # whiten the input if desired
    # solve for x0 and dx if passed as None
    if whiten:
        x, x0, dx = whiten_xarr(x, x0, dx)

    return x, x0, dx


def whiten_xarr(x, x0=None, dx=None):
    """
    Whiten a monotonically increasing
    vector x to have a range of [-1, 1]
    for optimal polynomial orthogonality.
    For uniformly increasing x, the whitened
    range is [-1+dx/2, 1-dx/2] if dx is None.

    Parameters
    ----------
    x : tensor
        Monotonically increasing, but not
        necessarily uniform.
    x0 : float, optional
        mean to use in centering x
    dx : float, optional
        scale to use in whitening x

    Returns
    -------
    xw : tensor
        Whitened x tensor
    x0 : float
        mean used to center x
    dx : float
        scale used to whiten x
    """
    if x0 is None:
        x0 = x.mean()
    x = x - x0

    if dx is None:
        dx = x.max() + (x[-1] - x[0]) / (len(x) - 1) / 2
    x = x / dx

    return x, x0, dx


class LinearModel:
    """
    A linear model of

        y = Ax
    """
    def __init__(self, linear_mode, dim=0, coeff=None,
                 out_dtype=None, meta=None, **kwargs):
        """
        Parameters
        ----------
        linear_model : str
            The kind of model to generate: ['custom', 'poly'].
            See utils.gen_linear_A.
        dim : int, optional
            The dimension of the input params tensor to sum over.
        coeff : tensor, optional
            A tensor of params.shape to multiply with params before
            forward transform (e.g. alm_mult for sph. harm.)
        out_dtype : dtype, optional
            Cast output of forward as this dtype if desired
        meta : dict, optional
            Additional metadata to attach to self as self.meta
        kwargs : dict
            keyword arguments for utils.gen_linear_A()
        """
        self.linear_mode = linear_mode
        self.dim = dim
        self.coeff = coeff
        self.out_dtype = out_dtype
        self.meta = meta if meta is not None else {}

        if self.linear_mode in ['poly']:
            if kwargs.get('whiten', False):
                # if using whiten, get x0, dx parameters now
                # and store in kwargs for later
                _, x0, dx = prep_xarr(kwargs.get('x'),
                                      d0=kwargs.get('d0', None),
                                      logx=kwargs.get('logx', False),
                                      whiten=kwargs.get('whiten', False),
                                      x0=kwargs.get('x0', None),
                                      dx=kwargs.get('dx', None))

                if not kwargs.get('x0', None):
                    kwargs['x0'] = x0
                if not kwargs.get('dx', None):
                    kwargs['dx'] = dx

        self.kwargs = kwargs
        self.A = gen_linear_A(linear_mode, **kwargs)
        self.device = self.A.device

    def forward(self, params, A=None, coeff=None):
        """
        Forward pass parameter tensor through design matrix

        Parameters
        ----------
        params : tensor
            Parameter tensor to forward model
        A : tensor, optional
            Use this (Nsamples, Nfeatures) design
            matrix instead of self.A when taking
            forward model
        coeff : tensor, optional
            Use this tensor with params.shape instead
            of self.coeff and multiply by params before
            forward transform (e.g. alm_mult for sph. harm.)

        Returns
        -------
        tensor
        """
        A = A if A is not None else self.A
        coeff = coeff if coeff is not None else self.coeff
        if coeff is not None:
            params = params * coeff
        ndim = params.ndim
        if self.dim == 0:
            # trivial matmul
            out = A @ params
        elif self.dim == ndim or self.dim == -1:
            # trivial transpose
            out = params @ A.T
        else:
            # would involve a transpose,
            # dot, then re-transpose,
            # so let's just use einsum
            t1 = 'ab'
            assert ndim <= 8
            t2 = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'][:ndim]
            t2[self.dim] = 'b'
            t2 = ''.join(t2)
            out = t2.replace('b', 'a')
            out = torch.einsum("{},{}->{}".format(t1, t2, out), A, params)

        if self.out_dtype is not None:
            out = out.to(self.out_dtype)

        return out

    def __call__(self, params, A=None):
        return self.forward(params, A=A)

    def least_squares(self, y, **kwargs):
        """
        Estimate a params tensor from the data vector

        Parameters
        ----------
        y : tensor
            Data vector to use in estimating a
            new params tensor
        kwargs : dict
            keyword arguments for linalg.least_squares()

        Returns
        -------
        tensor
        """
        from bayeslim.linalg import least_squares
        if y.dtype != self.A.dtype:
            y = y.to(self.A.dtype)
        return least_squares(self.A, y, dim=self.dim, **kwargs)

    def generate_A(self, x, **interp1d_kwargs):
        """
        Generate a new A matrix at new x values.
        If linear_mode is 'custom', then we interpolate
        the existing A, otherwise we generate
        a new A using the existing setup parameters.

        Parameters
        ----------
        x : tensor
            New x values to generate A
        kwargs : dict
            Kwargs for scipy interp1d(), used
            if linear_mode is custom.

        Returns
        -------
        tensor
        """
        if self.linear_mode == 'custom':
            # perform interpolation of existing A
            from scipy.interpolate import interp1d
            A = interp1d(self.kwargs['x'], self.A.cpu().numpy(),
                         axis=0, **interp1d_kwargs)(x)
            A = torch.as_tensor(A).to(self.device)
        else:
            kwargs = copy.deepcopy(self.kwargs)
            kwargs['x'] = x
            A = gen_linear_A(self.linear_mode, **kwargs)
            A = A.to(self.device)

        return A

    def push(self, device):
        """
        Push items to new device
        """
        dtype = isinstance(device, torch.dtype)
        self.A = push(self.A, device)
        if self.coeff is not None:
            self.coeff = push(self.coeff, device)
        if not dtype: self.device = device


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
    beam = voigt_profile(theta, sigma, gamma)
    beam /= beam.max()

    return beam, theta, phi


def _value_fun(start, stop, hp_map):
    value = sum(hp_map._data[start:stop])
    if hp_map._density:
        value /= stop - start
    return value


def adaptive_healpix_mesh(hp_map, split_fun=None):
    """
    Convert a single resolution healpix map to a
    multi-order coverage (MOC) map based on
    mhealpy's pixel value algorithm (density = False)

    Parameters
    ----------
    hp_map : mhealpy.HealpixBase subclass
        single resolution map to convert to multi-resolution
        based on relative pixel values and split_fun.
        Note that this should have density = False.
    split_fun : callable
        Function that determines if a healpix pixel is split into
        multiple pixels. See mhealpy.adaptive_moc_mesh().
        Default is mhealpy default function.

    Returns
    -------
    grid : HealpixMap object
        Downsampled healpix grid. Note that, due to how
        mhealpy.get_interp_val works, this will have density = True.
    theta, phi : array_like
        Co-latitude and longitude of downsampled map [rad]

    Notes
    -----
    See multires_map for downsampling a sky map onto
    output grid.
    """
    # set split_fun
    if split_fun is None:
        def split_fun(start, stop):
            max_value = max(hp_map)
            return _value_fun(start, stop, hp_map) > max_value

    # convert to nested if ring
    if hp_map.is_ring:
        ring2nest = healpy.ring2nest(hp_map.nside,
                                     np.arange(healpy.nside2npix(hp_map.nside)))
        hp_map._data = hp_map._data[np.argsort(ring2nest)]
        hp_map._scheme = 'NESTED'

    # downsample healpix map grid
    grid = hp_map.adaptive_moc_mesh(hp_map.nside, split_fun,
                                    dtype=hp_map.dtype)
    grid._density = True

    # fill data array
    rangesets = grid.pix_rangesets(grid.nside)
    for pix,(start, stop) in enumerate(rangesets):
        grid[pix] = _value_fun(start, stop, hp_map)

    # get theta, phi arrays
    theta, phi = grid.pix2ang(np.arange(grid.npix))

    return grid, theta, phi 


def multires_map(hp_map, grid, weights=None, dtype=None):
    """
    Given a multi-resolution grid, downsample
    a singe-res healpix map to multi-res grid.

    Parameters
    ----------
    hp_map : array_like or mhealpy.HealpixMap object
        A single-res healpix map to downsample (NESTED)
        If array_like, the last axis must be sky pixel axis
    grid : mhealpy.HealpixMap object
        Multi-resolution HealpixMap object containing
        grid to downsample to.
    weights : array_like or mhealpy.HealpixMap object, optional
        Optional weights to use when averaging
        child pixels of hp_map within a parent
        pixel in grid. Must be same nside as hp_map.
    dtype : object
        Data type of output map. Default is grid.dtype.

    Returns
    -------
    hp_map_mr
        Multiresolution healpix object of hp_map.
    """
    if isinstance(grid, mhealpy.HealpixBase):
        hp_map_mr = copy.deepcopy(grid)
        hp_map_mr._data = hp_map_mr._data.astype(dtype)
        nside = hp_map.nside
    else:
        hp_map_mr = np.zeros(hp_map.shape[:-1] + grid.data.shape,
                             dtype=dtype)
        nside = healpy.npix2nside(hp_map.shape[-1])

    # average hp_map
    for i, rs in enumerate(grid.pix_rangesets(nside)):
        # get weights
        w = np.ones(rs[1] - rs[0])
        if weights is not None:
            w = weights[..., rs[0]:rs[1]]
        # take average of child pixels
        hp_map_mr[..., i] = np.sum(hp_map[..., rs[0]:rs[1]] * w, axis=-1) / np.sum(w, axis=-1).clip(1e-40, np.inf)

    return hp_map_mr


def _recursive_pixelization(bsky, prev_ind, prev_nside, max_nside, theta, phi, nsides, total_nsides,
                           sigma=None, target_nside=None):
    """
    A dynamic pixelization scheme. See dynamic_pixelization() for operation.

    Parameters
    ----------
    bsky : array_like
        beam weighted healpix sky in NEST order (at high nside resolution)
    prev_ind : int
        HEALpix index of the leaf we are currently subdividing
    prev_nside : int
        HEALpix nside of the leaf we are currently subdividing
    max_nside : int
        Maximum nside of dynamic pixelization.
    theta, phi, nsides, total_nsides: list
        Empty lists to append to
    sigma : float, optional
        Sigma threshold for beam weighted pixelization
    target_nside : int
        The nside assigned to the leaf we are currently subdividign.
    """
    # get new nside
    this_nside = prev_nside * 2
    # determine if prev_nside is enough
    if (prev_nside >= max_nside) or (target_nside is not None and prev_nside >= target_nside):
        angs = healpy.pix2ang(prev_nside, prev_ind, nest=True)
        theta.append(angs[0])
        phi.append(angs[1])
        nsides.append(prev_nside)
        total_nsides.extend([prev_nside] * int(4**(np.log(max_nside / this_nside) / np.log(2) + 1)))
        return
    # get the four indices of this leaf in this_nside nest ordering
    start_ind = 4 * prev_ind
    inds = range(start_ind, start_ind + 4)
    # get the bsky interpolated values
    angs = healpy.pix2ang(this_nside, inds, nest=True)
    # figure out if we need to subdivide or not
    if sigma is not None:
        vals = healpy.get_interp_val(bsky, *angs, nest=True)
        stop_divide = np.std(vals) < sigma
    if this_nside >= max_nside:
        stop_divide = True
    if target_nside is not None:
        stop_divide = this_nside >= target_nside
    if stop_divide:
        theta.extend(angs[0])
        phi.extend(angs[1])
        nsides.extend([this_nside] * 4)
        total_nsides.extend([this_nside] * int(4**(np.log(max_nside / this_nside) / np.log(2) + 1)))
    # otherwise, iterate over each leaf and subdivide again
    else:
        for ind in inds:
            _recursive_pixelization(bsky, ind, this_nside, max_nside, theta, phi, nsides, total_nsides,
                                    sigma=sigma, target_nside=target_nside)


def nside_binning(zen, ra, zen_sigma=5, zen_gamma=15, ra_sigma=5, ra_gamma=15,
                  ra_min_max=None, min_nside=32, max_nside=256):
    """
    Compute nside binning using a voigt profile given
    a map of sky angles. Note for the ra axis: be mindful
    of how the ra_min_max cuts depend on the wrapping of
    the input ra array.

    Parameters
    ----------
    zen : array_like
        Zenith sky coordinate along the declination axis [deg].
    ra : array_like
        Right ascension coordinate [deg]. 
    zen_sigma, zen_gamma : float
        Sigma and gamma parameters of voigt profile of the
        zenith angle along the declination axis [deg]
    ra_sigma, ra_gamma : float
        Sigma and gamma parameters of voigt profile
        along right ascension [deg]
    ra_min_max : 2-tuple
        Minimum and maximum ra [deg] cut to keep a flat, max
        nside response of the binning. Outside of these cuts
        the voigt profile parameters lower the nside resolution.
    min_nside : int
        Minimum nside resolution. Must be a power of 2.
    max_nside : int
        Maximum nside resolution. Must be a power of 2.

    Returns
    -------
    curve : array_like
        Voigt profile curve used to set the nside binning
    nside_bins : array_like
        The nside of each pixel on the sky
    """
    # get zen component of voigt profile
    curve = voigt_profile(zen, zen_sigma, zen_gamma)
    curve -= curve.min()
    curve /= curve.max()

    # get ra component of voigt profile
    if ra_min_max is not None:
        # enact a nside res decay for ra less than min ra
        assert ra_min_max[0] > ra.min()
        ra_low = ra < ra_min_max[0]
        ra_low_curve = voigt_profile(ra[ra_low] - ra_min_max[0], ra_sigma, ra_gamma)
        ra_low_curve -= ra_low_curve.min()
        ra_low_curve /= ra_low_curve.max()
        curve[ra_low] *= ra_low_curve
        # enact a nside res decay for ra greater than max ra
        ra_hi = ra > ra_min_max[1]
        assert ra_min_max[1] < ra.max()
        ra_hi_curve = voigt_profile(ra[ra_hi] - ra_min_max[1], ra_sigma, ra_gamma)
        ra_hi_curve -= ra_hi_curve.min()
        ra_hi_curve /= ra_hi_curve.max()
        curve[ra_hi] *= ra_hi_curve

    # normalize curve to max and min_nside
    curve *= (max_nside - min_nside)
    curve += min_nside

    # bin the inputs
    bins = np.array([2 ** i for i in range(int(np.log(min_nside)/np.log(2)), int(np.log(max_nside)/np.log(2)) + 1)])
    inds = np.array([np.argmin(np.abs(bins - c)) for c in curve])
    nside_bins = np.array([bins[i] for i in inds])

    return curve, nside_bins


def dynamic_pixelization(base_nside, max_nside, sigma=None, bsky=None, target_nsides=None):
    """
    Two dynamic HEALpix pixelization schemes.
    1. Based on Zheng+2016 MITEOR Map Making (sigma)
    2. Manual pixelization (set by target_nsides)

    Parameters
    ----------
    base_nside : int
        The starting, minimum nside of the map. Must be power of 2.
    max_nside : int
        The upper limit on nside resolution. Must be a power of 2.
    sigma : float, optional
        If using algorithm (1), this is the standard deviation
        threshold of the bsky map, above which the healpix pixel
        is subdivded, below which the pixelization stops.
    bsky : array_like, optional
        If using algorithm (1), this is the beam weighted sky (NEST)
        to compute the standard deviations. This should be fed
        at an nside resolution higher than max_nside.
    target_nsides : array_like, optional
        If using algorithm (2), this should be an array of integers
        that has a length nside2npix(base_nside). Each element
        sets the nside resolution of that healpix pixel.
        See nside_binning() for examples.

    Returns
    -------
    theta, phi : array_like
        Co-latitude and longitude [radians] of dynamic pixels
    nsides : mhealpy HealpixBase object
        nside resolution of each pixel in theta, phi. This also
        holds the pixrangesets used in multires_map for downsampling
        a single-resolution healpix map to the dynamic res map.
    total_nsides : array_like
        An array that has the full shape of nside2npix(max_nside),
        with each element containing the nside resolution of the
        dynamic pixelization map at that location. This is used
        to plot the nside resolution of the map in healpix format.
    """
    import mhealpy
    theta, phi, nsides, total_nsides = [], [], [], []
    for i in range(healpy.nside2npix(base_nside)):
        target = target_nsides[i] if target_nsides is not None else None
        _recursive_pixelization(bsky, i, base_nside, max_nside, theta, phi, nsides, total_nsides,
                                sigma=sigma, target_nside=target)
    theta, phi, total_nsides = np.array(theta), np.array(phi), np.array(total_nsides)
    # turn nsides into mhealpy HealpixMap object
    ipix = [healpy.ang2pix(ns, th, ph, nest=True) for ns, th, ph in zip(nsides, theta, phi)]
    uniq = [4 * ns**2 + ip for ns, ip in zip(nsides, ipix)]
    nsides = mhealpy.HealpixMap(nsides, uniq=uniq, scheme='nested', dtype=np.int16)

    return theta, phi, nsides, total_nsides


def split_healpix_grid(theta, phi, nside, phi_min=None, phi_max=None, theta_min=None, theta_max=None):
    """
    Split a healpix map into four distinct
    components:
    1. southern cap
    2. central grid 1
    3. central grid 2
    4. northern cap

    If theta and phi have already been down-selected, they must
    still retain full 2pi range along phi (i.e. can only pre-
    down select on theta)

    Parameters
    ----------
    theta : array
        Theta values of healpix map [rad]
    phi : array
        Phi values of healpix map [rad]
    nside : int
        NSIDE of the healpix map
    phi_theta_min_max : float, optional
        Min and max ranges of phi or theta in radians

    Returns
    -------
    southern_idx : Indices for the southern cap
    central1_idx : Indices for the central 1 grid
    central2_idx : Indices for the central 2 grid
    northern_idx : Indices for the northern cap
    """
    # the declination boundary between central and caps
    magic_dec = 41.84 * np.pi / 180

    # get theta, phi
    dec = np.pi / 2 - theta
    idx = np.arange(len(theta))

    # setup theta/phi selections
    def theta_phi_select(theta, phi):
        f = np.ones(len(theta), dtype=bool)
        if phi_min:
            f = f & (phi >= phi_min)
        if phi_max:
            f = f & (phi <= phi_max)
        if theta_min:
            f = f & (theta >= theta_min)
        if theta_max:
            f = f & (theta <= theta_max)
        return f

    f = theta_phi_select(theta, phi)

    # southern cap
    s = (dec < -magic_dec) & f
    southern = np.where(s & f)[0]

    # northern cap
    s = (dec > magic_dec) & f
    northern = np.where(s & f)[0]

    # central grids
    s = (dec > -magic_dec) & (dec < magic_dec)

    # first grid
    th = theta[s].reshape(-1, nside*4)[::2].ravel()
    ph = phi[s].reshape(-1, nside*4)[::2].ravel()
    _idx = idx[s].reshape(-1, nside*4)[::2].ravel()
    f = theta_phi_select(th, ph)
    central1 = _idx[f]

    # first grid
    th = theta[s].reshape(-1, nside*4)[1::2].ravel()
    ph = phi[s].reshape(-1, nside*4)[1::2].ravel()
    _idx = idx[s].reshape(-1, nside*4)[1::2].ravel()
    f = theta_phi_select(th, ph)
    central2 = _idx[f]

    return southern, central1, central2, northern


class PixInterp:
    """
    Sky pixel spatial interpolation object
    """
    def __init__(self, pixtype, nside=None, interp_mode='nearest',
                 theta_grid=None, phi_grid=None, device=None,
                 gpu=False, downcast=False, interp_cache_depth=None):
        """
        Interpolation is a weighted average of nearest neighbors.
        If pixtype is 'healpix', this is bilinear interpolation.
        If pixtype is 'rect' use interp_mode to set interpolation

        Parameters
        ----------
        pixtype : str
            Pixelization type. options are ['healpix', 'rect']
            For healpix, pixel ordering is RING.
            For rect, pixel ordering should be
            x, y = meshgrid(phi_grid, theta_grid)
            x, y = x.ravel(), y.ravel()
        nside : int, optional
            nside of healpix map if pixtype is'healpix'
        interp_mode : str, optional
            Spatial interpolation method.
            'nearest' : nearest neighbor interpolation
            'linear' : bilinear interpolation, needs rect grid
            'quadratic' : biquadratic interpolation, needs rect grid 
            'cubic' : bicubic interpolation, needs rect grid
            'linear,quadratic' : linear along az, quadratic along zen
            For pixtype of 'healpix', we always use healpy bilinear interp.
        theta_grid, phi_grid : array_like
            1D zen and azimuth arrays [deg] if pixtype is 'rect'
            defining the grid to be interpolated against. These
            should mark the pixel centers.
        device : str, optional
            Device to place object on
        gpu : bool, optional
            If True and pixtype is 'rect', perform grid sorting
            on GPU for speedups
        downcast : bool, optional
            If True and using gpu, downcast grids to float32
            before sending to GPU for further speedup (with
            slight loss of accuracy)
        interp_cache_depth : int, optional
            The number of caches in the interp_cache allowed.
            Follows first-in, first-out rule. Default is no limit.
        """
        self.pixtype = pixtype
        self.nside = nside
        self.interp_cache = {}
        self.interp_mode = interp_mode
        self.theta_grid = theta_grid
        self.phi_grid = phi_grid
        self.device = device
        self.gpu = gpu
        self.downcast = downcast
        self.interp_cache_depth = interp_cache_depth

    def clear_cache(self, depth=None):
        """Clears interpolation cache.
        If depth is specified, then use FIFO to
        clear until depth is reached."""
        if depth is None:
            self.interp_cache = {}
        else:
            clear_cache_depth(self.interp_cache, depth)

    def get_interp(self, zen, az):
        """
        Get interpolation metadata

        Parameters
        ----------
        zen, az : zenith (co-lat) and azimuth angles [deg]

        Returns
        -------
        interp : tuple
            nearest neighbor (indices, weights)
            for each entry in zen, az for interpolation
        """
        # get hash
        h = (arr_hash(zen), arr_hash(az))
        if h in self.interp_cache:
            # retrieve interpolation if cached
            interp = self.interp_cache[h]
        else:
            # otherwise generate weights and indices
            if self.pixtype == 'healpix':
                # use healpy bilinear interp regardless of interp_mode
                inds, wgts = healpy.get_interp_weights(self.nside,
                                                       tensor2numpy(zen) * D2R,
                                                       tensor2numpy(az) * D2R)
                inds = inds.T
                wgts = wgts.T

            # this is bipolynomial interpolation
            elif self.pixtype == 'rect':
                # get poly degree for az and zen
                if ',' not in self.interp_mode:
                    degree =  [self.interp_mode, self.interp_mode]
                else:
                    degree = [s.strip() for s in self.interp_mode.split(',')]

                # map string to interpolation degree
                s2d = {'nearest': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3}
                degree = [s2d[d] for d in degree]

                # get grid
                xgrid = self.phi_grid
                ygrid = self.theta_grid
                dx = np.median(np.diff(xgrid))
                dy = np.median(np.diff(ygrid))
                # TODO: zen and az can be on gpu, which can speed up bipoly_grid_index
                xnew, ynew = tensor2numpy(az), tensor2numpy(zen)

                # get map indices
                inds, xyrel = bipoly_grid_index(xgrid, ygrid, xnew, ynew,
                                                degree[0]+1, degree[1]+1,
                                                wrapx=True, ravel=True,
                                                gpu=self.gpu, downcast=self.downcast)

                # get weights
                Ainv, Anew = setup_bipoly_interp(degree, dx, dy, xyrel[0], xyrel[1])
                wgts = Anew @ Ainv

            # store it in the cache
            if self.interp_cache_depth is None or self.interp_cache_depth > 0:
                interp = (torch.as_tensor(inds, device=self.device),
                          torch.as_tensor(wgts, dtype=_float(), device=self.device))
                self.interp_cache[h] = interp

                # clear cache if needed
                if self.interp_cache_depth is not None:
                    self.clear_cache(depth=self.interp_cache_depth)

        return interp

    def interp(self, m, zen, az):
        """
        Interpolate a map m at zen, az points

        Parameters
        ----------
        m : array_like or tensor
            Map to interpolate of shape (..., Npix).
            If Healpix map must be ring ordered.
        zen, az : array_like or tensor
            Zenith angle (co-latitude) and azimuth [deg]
            points at which to interpolate map
        """
        # get interpolation indices and weights
        inds, wgts = self.get_interp(zen, az)

        # get nearest neighbors to use for interpolation
        # inds has shape (Nnearest, Nangles)
        nearest = m[..., inds]

        # note that simple tests show that
        # above fancy indexing is faster than
        # an expand->gather indexing on CPU
        # but is similar in speed on GPU

        # multiply by weights and sum
        out = torch.sum(nearest * wgts, axis=-1)

        ## LEGACY
#        if self.interp_mode == 'nearest':
#            # use nearest neighbor
#            if self.pixtype == 'healpix':
#                return m[..., inds]
#            elif self.pixtype == 'rect':
#                return m[..., inds[0], inds[1]]
#        elif self.interp_mode == 'bilinear':
#            # select out nearest neighbor indices for each zen, az
#            if self.pixtype == 'healpix':
#                nearest = m[..., inds.T]
#            else:
#                nearest = m[..., inds[0].T, inds[1].T]
#            # multiply by weights and sum
#            return torch.sum(nearest * wgts.T, axis=-1)
#        else:
#            raise ValueError("didnt recognize interp_mode")
        
        return out

    def push(self, device):
        """
        Push cache onto a new device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype: self.device = device
        for k in self.interp_cache:
            cache = self.interp_cache[k]
            self.interp_cache[k] = (cache[0].to(device),
                                    push(cache[1], device))


def clear_cache_depth(cache, depth):
    """
    Use FIFO to clear a provided cache (must be ordered dictionary)
    down to depth. Operates inplace.

    Parameters
    ----------
    cache : dict
        An ordered dict to clear
    depth : int
        Number of entries to keep
    """
    if depth is None:
        return
    cache_len = len(cache)
    if cache_len > depth:
        keys = list(cache.keys())
        for k in keys[:cache_len - depth]:
            del cache[k]


def freq_interp(params, param_freqs, freqs, kind, axis,
                fill_value='extrapolate'):
    """
    Interpolate the params tensor onto a new frequency basis

    Parameters
    ----------
    params : tensor
        A tensor to interpolate along axis
    param_freqs : tensor
        The frequencies [Hz] of params
    freqs : tensor
        Frequencies to interpolate onto [Hz]
    kind : str
        Kind of interpolation, 'nearest', linear', 'quadratic', ...
    axis : int
        Axis of params to interpolate
    fill_value : str, optional
        If nan, raise error if freqs is out of param_freqs
        otherwise extrapolate

    Returns
    -------
    tensor
        Interpolated params
    """
    with torch.no_grad():
        # determine if interpolation is necessary
        match = [np.isclose(f, param_freqs, atol=1) for f in freqs]
        if np.all(match):
            indices = [np.argmin(param_freqs - f) for f in freqs]
            index = [slice(None) for i in range(params.ndim)]
            index[axis] - indices
            return params[index]

        from scipy.interpolate import interp1d
        param = tensor2numpy(params, clone=True)
        param_freqs = tensor2numpy(param_freqs)
        freqs = tensor2numpy(freqs)

        interp = interp1d(param_freqs, param, kind=kind, axis=axis,
                          fill_value=fill_value)
        interp_param = interp(freqs)

        return torch.tensor(interp_param, device=params.device, dtype=params.dtype)


def bipoly_grid_index(xgrid, ygrid, xnew, ynew, Nx, Ny,
                      wrapx=False, ravel=True, gpu=False, downcast=False):
    """
    For uniform grid in x and y, pick out N nearest grid indices
    in x and y given a sampling of new xy values.
    
    Parameters
    ----------
    xgrid : array
        1D float array of grid x values
    ygrid : array
        1D float array of grid y values
    xnew : array
        New x samples to get nearest neighbors
    ynew : array
        New y samples to get nearest neighbors
    Nx : int
        Number of NN in x to index
    Ny : int
        Number of NN in y to index
    wrapx : bool, optional
        If True, wrap x axis such that samples near xgrid
        boundaries wrap around their indexing
    ravel : bool, optional
        If True, flatten inds to index the raveled
        grid (as oppossed to the 2D grid), assuming
        the grid is ordered as
            X, Y = np.meshgrid(xgrid, ygrid)
            grid = X.ravel(), Y.ravel()
        i.e. [(x1, y1), (x2, y1), ..., (x1, y2), (x2, y2), ...]
    gpu : bool or str, optional
        If True, pass input arrays to GPU to speed up argsort.
        Default is to send to 'cuda:0', but gpu can be passed
        as a str specifying the exact GPU to use, e.g. 'cuda:1'
    downcast : bool, optional
        If also using gpu, cast input arrays down to torch.float32,
        which speeds CPU->GPU transfer and argsort.

    Returns
    -------
    inds : array
        Indexes the nearest neighbors of xnew and ynew at
        xgrid and ygrid points. If not ravel,
        this is a 2-tuple holding (Xnn, Ynn), where
        Xnn is of shape (Nnew, Nx) and similar for Ynn.
        If ravel, this is a (Nnew, Nx*Ny) array meant
        to index the raveled grid.
    (xrel, yrel) : tuple
        xnew and ynew but cast into dimensionless units
        relative to the start of inds and dx,dy spacing
    """
    # set gpu to false if no cuda
    gpu = gpu if torch.cuda.is_available() else False

    # parse gpu kwarg
    if gpu and isinstance(gpu, bool):
        gpu = 'cuda:0'

    # get dx, dy
    dx, dy = np.median(np.diff(xgrid)), np.median(np.diff(ygrid))
    
    # wrap xgrid
    N = len(xgrid)
    if wrapx:
        xgrid = np.concatenate([xgrid[-Nx:]-N*dx, xgrid, xgrid[:Nx]+N*dx])

    # get xgrid and ygrid indices for each xynew
    if gpu:
        # send arrays to the GPU
        dtype = torch.float32 if downcast else None
        _xgrid = torch.as_tensor(xgrid, dtype=dtype).to(gpu)
        _ygrid = torch.as_tensor(ygrid, dtype=dtype).to(gpu)
        _xnew = torch.as_tensor(xnew, dtype=dtype).to(gpu)
        _ynew = torch.as_tensor(ynew, dtype=dtype).to(gpu)

        # get indices
        xnn = torch.sort(torch.argsort(torch.abs(_xgrid - _xnew[:, None]), dim=-1)[:, :Nx], dim=-1).values
        ynn = torch.sort(torch.argsort(torch.abs(_ygrid - _ynew[:, None]), dim=-1)[:, :Ny], dim=-1).values
        xnn, ynn = xnn.cpu().numpy(), ynn.cpu().numpy()

    else:
        # if using numpy, argpartition is ~4x faster than argsort for large arrays
        xnn = np.sort(np.argpartition(-np.abs(xgrid - xnew[:, None]), -Nx, axis=-1)[:, -Nx:], axis=-1)
        ynn = np.sort(np.argpartition(-np.abs(ygrid - ynew[:, None]), -Ny, axis=-1)[:, -Ny:], axis=-1)

    # get xnew ynew in coords relative to xnn[:, 0] and ynn[:, 0]
    xrel = (xnew - xgrid[xnn[:, 0]]) / dx
    yrel = (ynew - ygrid[ynn[:, 0]]) / dy

    # unwrap
    if wrapx:
        xnn -= Nx
        xnn = xnn % N
    
    # ravel
    if ravel:
        inds = (xnn[:, None, :] + N * ynn[:, :, None]).reshape(len(ynew), -1)
    else:
        inds = (xnn, ynn)
        
    return inds, (xrel, yrel)


def setup_bipoly_interp(degree, dx, dy, xnew, ynew):
    """
    Setup bi-polynomial interpolation weight matrix
    on a uniform grid.
    
    For bi-linear interpolation we use 2x2 grid points.
    Assume interpolation is given by evaluating a bi-linear poly
        
        f(x, y) = a_{00} + a_{10}x + a_{01}y + a_{11}xy
        
    then the 4 points on the 2x2 grid can be expressed as
        
        | 1 x_1 y_1 x_1y_1 | | a_{00} |   | f(x_1,y_1) |
        | 1 x_2 y_1 x_2y_1 | | a_{10} | = | f(x_2,y_1) |
        | 1 x_1 y_2 x_1y_2 | | a_{01} |   | f(x_1,y_2) |
        | 1 x_2 y_2 x_2y_2 | | a_{11} |   | f(x_2,y_2) |
        
    or
    
        Ax = y
        
    The weights "x_hat" can be found via least squares inversion,
    
        x_hat = (A^T A)^-1 A^T y
        
    Note that this function returns the (A^T A)^-1 A^T portion of x_hat,
    which must be dotted into f(x,y) vector with the ordering given above.
    
    Interpolation of any new point within the 2x2 grid can be computed as
    
        f_xy = A_xy x_hat
        
    Note that this generalizes to bi-quadratic and bi-cubic, as
    well as mixed polynomials (e.g. bi-linear-quadratic).

    Parameters
    ----------
    degree : int or list of int
        Degree of polynomial interpolation
        (e.g. 0 for nearest, 1 for bilinear, 2 for biquadratic).
        Can also be a len-2 list specifying polynomial degree along (x, y)
        respectively.
    dx : float
        Spacing along the x direction
    dy : float
        Spacing along the y direction
    xnew : tensor
        New x positions relative to dx grid
    ynew : tensor
        New y positions relative to dy grid

    Returns
    -------
    AtAinvAt : tensor
        (A^T A)^-1 A^T portion of the x_hat vector
    Anew : tensor
        A matrix at xnew and ynew points
    """
    from emupy.linear import setup_polynomial
    if not isinstance(degree, (list, tuple)):
        degree = [degree, degree]
    assert len(degree) == 2
    # setup grid given degree
    Npoints = [degree[0]+1, degree[1]+1]
    x, y = np.meshgrid(np.arange(Npoints[0]) * dx,  np.arange(Npoints[1]) * dy)
    X = np.array([x.ravel(), y.ravel()]).T

    # get design matrix
    A, _ = setup_polynomial(X, degree, feature_degree=True, basis='direct')
    # get inverse
    AtAinvAt = np.linalg.pinv(A.T @ A, hermitian=True) @ A.T
    
    # get new design matrix
    X = np.array([xnew * dx, ynew * dy]).T
    Anew, _ = setup_polynomial(X, degree, feature_degree=True, basis='direct')
    
    return AtAinvAt, Anew


###########################
######### Modules #########
###########################

class Module(torch.nn.Module):
    """
    A shallow wrapper around torch.nn.Module, with added
    utility features. Subclasses should overload
    self.forward, which will propagate to self.__call__
    """
    def __init__(self, name=None):
        super().__init__()
        # add version
        self.__version__ = version.__version__
        self.set_priors()
        self._name = name
        self.clear_response_grad()

    @property
    def name(self):
        return self._name if self._name is not None else self.__class__.__name__

    @property
    def named_params(self):
        return [k[0] for k in self.named_parameters()]

    def forward(self, inp=None, prior_cache=None, **kwargs):
        """
        The forward operator. Should have a kwarg for
        starting input to the model (inp) and a cache
        dictionary for holding the result of eval_prior

        Parameters
        ----------
        inp : object, optional
            Starting input for model
        prior_cache : dict, optional
            Cache for storing computed prior
        """
        raise NotImplementedError

    def __getitem__(self, name):
        return get_model_attr(self, name)

    def __setitem__(self, name, value):
        with torch.no_grad():
            set_model_attr(self, name, value)

    def __delitem__(self, name):
        del_model_attr(self, name)

    def update(self, pdict, clobber_param=False):
        """
        Update model attributes from pdict

        Parameters
        ----------
        pdict : ParamDict
            dictionary of values to assign
            to model
        clobber_param : bool, optional
            If True and key from pdict is an existing
            Parameter on self, del the param then assign
            it from pdict (this removes Parameter object
            but keeps memory address from pdict).
            If False (default), insert value from pdict
            into existing Parameter object, changing
            its memory address.
        """
        for key, val in pdict.items():
            # uses set_model_attr for no_grad context
            set_model_attr(self, key, val, clobber_param=clobber_param)

    def unset_param(self, name):
        """
        Unset a Parameter tensor "name"
        as a non-Parameter
        """
        if isinstance(name, list):
            for n in name:
                self.unset_param(n)
            return
        param = self[name].detach()
        del self[name]
        self[name] = param

    def set_param(self, name):
        """
        Set tensor "name" as a Parameter
        """
        if isinstance(name, list):
            for n in name:
                self.set_param(n)
            return
        param = self[name]
        if not isinstance(param, torch.nn.Parameter):
            self[name] = torch.nn.Parameter(param)

    def set_priors(self, priors_inp_params=None, priors_out_params=None):
        """
        Set log prior(s) on this module's input params tensor
        and/or on the output params tensor (i.e. after mapping
        through a response function)

        Parameter
        ---------
        priors_inp_params : optim.LogPrior object or list
            Takes params as input and returns a scalar
            log prior. Can feed list of priors as well.
        priors_out_params : optim.LogPrior object or list
            Takes the tensor output after pushing
            params through its response function
            as input and returns a scalar log prior.
            Can feed list of priors as well.
        """
        if (priors_inp_params is not None and 
            not isinstance(priors_inp_params, (list, tuple))):
            priors_inp_params = [priors_inp_params]
        self.priors_inp_params = priors_inp_params

        if (priors_out_params is not None and
            not isinstance(priors_out_params, (list, tuple))):
            priors_out_params = [priors_out_params]
        self.priors_out_params = priors_out_params

    def eval_prior(self, prior_cache, inp_params=None, out_params=None):
        """
        Evaluate prior and insert into prior_cache.
        Will use self.name (default) or __class__.__name__ as key.
        If the key already exists, the prior will not be computed or stored.
        This override is needed when minibatching.

        If inp_params is None, will try self.params.
        If out_params is None, will try self.R(self.params + self.p0).
        For special classes with non-standard API (like PixelBeam) this
        function needs to be overloaded.

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

            # try to get inp_params
            if inp_params is None:
                if hasattr(self, 'params'):
                    inp_params = self.params

            # look for prior on inp_params
            if self.priors_inp_params is not None and inp_params is not None:
                for prior in self.priors_inp_params:
                    if prior is not None:
                        prior_value = prior_value + prior(inp_params)

            # try to get out_params
            if out_params is None:
                if hasattr(self, 'params'):
                    p = self.params
                if hasattr(self, 'p0') and self.p0 is not None:
                    p = p + self.p0
                if hasattr(self, 'R'):
                    out_params = self.R(p)

            # look for prior on out_params
            if self.priors_out_params is not None and out_params is not None:
                for prior in self.priors_out_params:
                    if prior is not None:
                        prior_value = prior_value + prior(out_params)

            prior_cache[self.name] = prior_value

    def clear_response_grad(self):
        self.response_grad = None
        for mod in self.modules():
            if hasattr(mod, 'clear_response_grad'):
                mod.response_grad = None

    def response_grad_hook(self, grad):
        if not hasattr(self, 'response_grad') or self.response_grad is None:
            self.response_grad = grad.clone()
        else:
            self.response_grad += grad

    def clear_graph_tensors(self):
        """
        If any graph tensors are attached to self, use
        this function to clear them, i.e. del them
        or set them to None
        """
        pass


class Sequential(Module):
    """
    A minimal mirror of torch.nn.Sequential with added features.
    Inherits from bayeslim.utils.Module

    .. code-block:: python

        S = Sequential(OrderedDict(model1=model1, model2=model2))

    where evaluation order is S(params) -> model2( model1( params ) )

    The forward call takes a parameter dictionary that updates
    the model before evaluation, which must conform to 
    torch.nn.Module.named_parameters() styel
    """
    def __init__(self, models):
        """
        Parameters
        ----------
        models : dict
            Models to evaluate in sequential order.
        """
        super().__init__()
        # get ordered list of model names
        self._models = list(models)
        # assign models as sub modules
        for name, model in models.items():
            self.add_module(name, model)

    def forward(self, inp=None, pdict=None, prior_cache=None, **kwargs):
        """
        Evaluate model in sequential order,
        optionally updating all parameters beforehand

        Parameters
        ----------
        inp : object, optional
            Starting input
        pdict : ParamDict, optional
            Parameter dictionary with keys
            conforming to nn.Module.get_parameter
            syntax, and values as tensors
        prior_cache : dict, optional
            Cache for storing computed priors
        """
        # update parameters of module and sub-modules
        if pdict is not None:
            self.update(pdict)

        for name in self._models:
            inp = self.get_submodule(name)(inp, prior_cache=prior_cache, **kwargs)

        return inp

    @property
    def Nbatch(self):
        """get total number of batches in model"""
        if hasattr(self.get_submodule(self._models[0]), 'Nbatch'):
            return self.get_submodule(self._models[0]).Nbatch
        else:
            return 1

    @property
    def batch_idx(self):
        """return current batch index"""
        if hasattr(self.get_submodule(self._models[0]), 'batch_idx'):
            return self.get_submodule(self._models[0]).batch_idx
        else:
            return 0

    def set_batch_idx(self, idx):
        """Set the current batch index"""
        if hasattr(self.get_submodule(self._models[0]), 'set_batch_idx'):
            self.get_submodule(self._models[0]).set_batch_idx(idx)
        elif idx > 0:
            raise ValueError("No method set_batch_idx and requested idx > 0")

    def push(self, device):
        """
        push all models to device
        """
        for name in self._models:
            model = self.get_submodule(name)
            if hasattr(model, 'push'):
                model.push(device)


def has_model_attr(model, name):
    """
    Return True if model has name
    """
    if isinstance(name, str):
        name = name.split('.')
    if len(name) == 1:
        return hasattr(model, name[0])
    else:
        if hasattr(model, name[0]):
            return has_model_attr(get_model_attr(model, name[0]), '.'.join(name[1:]))
        else:
            return False


def get_model_attr(model, name, pop=0):
    """
    Get attr model.name

    Parameters
    ----------
    pop : int, optional
        period-delimited chunks of 'name'
        to pop from the end before getting.
        E.g. if name = 'rime.sky.params'
        and pop = 0, returns self.rime.sky.params
        or if pop = 1 returns self.rime.sky
    """
    if isinstance(name, str):
        name = name.split('.')
    if pop > 0:
        name = name[:-pop]
    attr = getattr(model, name[0])
    if len(name) == 1:
        return attr
    else:
        return get_model_attr(attr, '.'.join(name[1:]))


def set_model_attr(model, name, value, clobber_param=False,
                   no_grad=True, idx=None):
    """
    Set value to model as model.name

    If name is a torch.nn.Parameter, cast
    value as Parameter before setting.

    Parameters
    ----------
    model : utils.Module object
    name : str
        Name of attribute to set
    value : tensor
        New tensor values to set
    clobber_param : bool, optional
        If True and name already exists as a Parameter,
        detach & clone it then assign value as name
        (this removes existing name from graph).
        If False (default) and name already exists,
        try to insert value into existing model.name. 
    no_grad : bool, optional
        If True, enter a torch.no_grad() context,
        otherwise enter a nullcontext.
    idx : tuple, optional
        If model.name already exists, insert value as
        model.name[idx] = value. Note that if clobber_param = False
        and model.name is a Paramter, this will only work with
        simple indexing schemes. Note that value should already
        have a shape that matches model.name[idx]
    """
    context = torch.no_grad if no_grad else nullcontext
    with context():
        if isinstance(name, str):
            name = name.split('.')
        if len(name) == 1:
            # this is the last entry in name = "first.second.third"
            # so assign value as name[0]
            name = name[0]
            device = None
            parameter = False
            if clobber_param:
                # if clobbering name, del then reattach w/o requires_grad
                if hasattr(model, name):
                    p = getattr(model, name).detach().clone()
                    delattr(model, name)
                    setattr(model, name, p)
            elif hasattr(model, name):
                # if not clobber and this attr exists
                # get device and parameter status from attr
                device = getattr(model, name).device
                parameter = isinstance(getattr(model, name), torch.nn.Parameter)
                value = value.to(device)
                if parameter and not value.requires_grad:
                    # if model.name is a Parameter and value is not a leaf
                    # or view of a leaf, make it a Parameter too
                    value = torch.nn.Parameter(value)
            # set the attribute!
            if idx is None:
                setattr(model, name, value)
            else:
                # this only works if model.name already exists
                # and 1. requires_grad=False or 2. clobber_param=True
                model[name][idx] = value

        else:
            # recurse through the '.' names until you get to the end
            set_model_attr(get_model_attr(model, '.'.join(name[:-1])),
                           name[-1], value, clobber_param=clobber_param,
                           idx=idx, no_grad=no_grad)


def del_model_attr(model, name):
    """
    Delete model.name
    """
    if isinstance(name, str):
        name = name.split('.')
    if len(name) > 1:
        model = get_model_attr(model, name, pop=1)
        name = name[-1:]
    delattr(model, name[0])


class Difference(Module):
    """
    A difference block. Can act
    on an input tensor, or on an input
    VisData, CalData, MapData object
    """
    def __init__(self, idx1, idx2):
        """
        Parameters
        ----------
        idx1 : tuple of ints or dict
            If fed as a tuple, treat
            the block input as a tensor
            and index it with idx1.
            If this is a dict, treat
            input as a Dataset object
            and use its select(**idx1)
        idx2 : tuple of ints or dict
            Same format as idx1. The
            difference is
            params[idx1] - params[idx2]
        """
        super().__init__()
        self.idx1 = idx1
        self.idx2 = idx2
        
    def __call__(self, params, **kwargs):
        return self.forward(params)
    
    def forward(self, params, **kwargs):
        if isinstance(self.idx1, dict):
            # treat params as a VisData or CalData
            params1 = params.copy(detach=False)
            params1.select(**self.idx1)
            params2 = params.copy(detach=False)
            params2.select(**self.idx2)
            params1.data -= params2.data
            res = params1

        else:
            # treat params as a tensor
            res = params[self.idx1] - params[self.idx2]

        return res


#################################
######### Miscellaneous #########
#################################


def Jy_to_KStr(freqs):
    """
    Compute conversion from Jy to Kelvin-Str [K Str / Jy]

    Parameters
    ----------
    freqs : tensor
        Frequencies [Hz]

    Returns
    -------
    tensor
    """
    return 1e-26 * (constants.c.value / freqs)**2 / (2 * constants.k_B.value)


def white_noise(*args):
    """
    Generate complex white Gaussian noise
    with unit variance

    Parameters
    ----------
    shape : shape of tensor

    Returns
    -------
    tensor
    """
    n = torch.randn(*args) + 1j * torch.randn(*args)
    return n / np.sqrt(2)


def arr_hash(arr):
    """
    Hash an array or list by using its
    first value, last value and length as a
    unique identifier of the array.
    Note that if arr is a tensor, the device and
    require_grad values will affect the hash!
    Also note, normally array hash is not allowed.

    Parameters
    ----------
    arr : ndarray or tensor or list

    Returns
    -------
    hash object
    """
    if hasattr(arr, '_arr_hash'):
        return arr._arr_hash
    if isinstance(arr, torch.Tensor):
        key = (arr[0].cpu().tolist(), arr[-1].cpu().tolist(), len(arr))
    else:
        key = (arr[0], arr[-1], len(arr))
    h = hash(key)
    if isinstance(arr, (np.ndarray, torch.Tensor)):
        arr._arr_hash = h
    return h


def push(tensor, device, parameter=False):
    """
    Push a tensor to a new device or dtype. If the tensor
    is a parameter, it instantiates the parameter
    class on device.

    Parameters
    ----------
    tensor : tensor
        A pytorch tensor, optionally a pytorch Parameter
    device : str
        The device to push it to, either str or device object.
        Can also be a dtype, in which case tensor will
        be cast as dtype. If complex, tensor stay complex.
        e.g. complex128 -> complex64 for dtype of float32.
    parameter : bool, optional
        Make new tensor a parameter. This is done
        by default if input is a Parameter

    Returns
    -------
    tensor
        The tensor on device (or as new dtype)
    """
    dtype = isinstance(device, torch.dtype)
    if dtype and torch.is_complex(tensor) and not device.is_complex:
        if device == torch.float16: device = torch.complex32
        elif device == torch.float32: device = torch.complex64
        elif device == torch.float64: device = torch.complex128
        else: raise ValueError("tensor is complex but output dtype is not float or complex")

    if parameter or isinstance(tensor, torch.nn.Parameter):
        return torch.nn.Parameter(tensor.to(device))
    else:
        return tensor.to(device)


def tensor2numpy(tensor, clone=True):
    """
    Convert a tensor (on any device)
    to a numpy ndarray on the cpu

    Parameters
    ----------
    tensor : tensor
        A pytorch tensor on any device
    clone : bool, optional
        If True, clone tensor, i.e. output
        has different memory address.

    Returns
    -------
    ndarray
        The tensor as an ndarray on cpu
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach()
        if device(tensor.device) != device('cpu'):
            tensor = tensor.cpu()
        if clone:
            tensor = tensor.clone()
        return tensor.numpy()
    return tensor


def device(d):
    """
    parse a device into a standard pytorch format

    Parameters
    ----------
    d : str or torch.device object
    """
    if d is None:
        d = 'cpu'
    return torch.device(d)


def fit_zero(x, y):
    """fit a quadratic and solve for roots"""
    a, b, c = np.polyfit(x, y, 2)
    d = b**2 - 4*a*c
    x1 = (-b + np.sqrt(d)) / (2 * a)
    x2 = (-b - np.sqrt(d)) / (2 * a)
    sol = x1 if np.abs(x1 - x[0]) < np.abs(x2 - x[0]) else x2
    return sol


def get_zeros(x, y):
    """iterate over y and get zeros"""
    # get roots
    roots = []
    for i in range(len(y)):
        if i == 0:
            prev = np.sign(y[i])
            continue
        # get current sign
        curr = np.sign(y[i])
        # check for zero crossing: abs(y) condition avoids y jitters around zero
        if (curr != prev) and (np.abs(y[i]) > 1e-40) and (curr != 0.0) and np.isfinite(prev):
            # check for initial divergence from zero, which is not a real root
            if prev == 0.0:
                # set prev to curr sign such that future crossings are counted
                prev = curr
                continue

            # this is zero crossing: get 3 nn points and fit quadratic for root
            start = max([i-3, 0])
            nn = np.argsort(np.abs(y)[start:i+3])[:3] + start
            roots.append(fit_zero(x[nn], y[nn]))
            prev = curr

    return roots


def _make_hex(N, D=15):
    x, y, ants = [], [], []
    ant = 0
    k = 0
    start = 0
    for i in range(2*N - 1):
        for j in range(N + k):
            x.append(j + start)
            y.append(i * np.sin(np.pi/3))
            ants.append(ant)
            ant += 1
        if i < N-1:
            k += 1
            start -= .5
        else:
            k -= 1
            start += .5
    x = np.array(x) - np.mean(x)
    y = np.array(y) - np.mean(y)
    return ants, np.vstack([x, y, np.zeros_like(x)]).T * D


class SimpleIndex:
    """
    Returns value for any getitem call
    """
    def __init__(self, value=0):
        self.value = value

    def __getitem__(self, k):
        return self.value


def split_into_groups(arr, Nelem=10):
    """
    Split a list or array of elements into
    a nested list of groups each being Nelem long

    Parametrs
    ---------
    arr : array or list
        List of elements
    Nelem : int, optional
        Number of elements in each sublist

    Returns
    -------
    array or list
    """
    N = len(arr)
    sublist = [arr[i*Nelem:(i+1)*Nelem] for i in range(N//Nelem + 1)]
    if len(sublist[-1]) == 0:
        sublist.pop(-1)

    return sublist


def smi(name, fname='nvidia_mem.txt', verbose=True):
    os.system("nvidia-smi -f {}".format(fname))
    with open(fname) as f: lines = f.readlines()
    date = lines[0].strip()
    output = []
    usage = {'date': date}
    for i, line in enumerate(lines):
        if name in line:
            gpu = line[4]
            mems = lines[i+1].split('|')[2].strip().split('/')
            usage[gpu] = "{:>5} / {:>5} MiB".format(mems[0].strip()[:-3], mems[1].strip()[:-3])
            alloc = "{:>6} MiB".format("{:,}".format(int(mems[0].strip()[:-3])))
            total = "{:>6} MiB".format("{:,}".format(int(mems[1].strip()[:-3])))
            mem = "{} / {}".format(alloc, total)
            output.append("| GPU {} | {} |".format(gpu, mem))
    if verbose:
        print('\n'.join([date] + ['o' + '-'*33 + 'o'] + output + ['o' + '-'*33 + 'o']))

    return usage


def flatten(arr, Nelem=None):
    """
    Flatten nested list or array

    Parameters
    ----------
    arr : list or ndarray
        Nested set of lists
    Nitems : int, optional
        Number of elements in the subarray
        to keep. default is all elements.

    Returns
    -------
    list
        Flatten list
    """
    flat = []
    if Nelem is None:
        s = slice(None)
    else:
        s = slice(0, Nelem)
    for subarr in arr:
        for item in subarr[s]:
            flat.append(item)
    return flat


def _list2slice(inds):
    """convert list indexing to slice if possible"""
    if isinstance(inds, list):
        diff = list(set(np.diff(inds)))
        if len(diff) == 1:
            if (inds[1] - inds[0]) > 0:
                # only return as slice if inds is increasing
                return slice(inds[0], inds[-1]+diff[0], diff[0])
    return inds

