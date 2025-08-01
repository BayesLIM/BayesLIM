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

# 2-real to complex, vice versa
viewreal = torch.view_as_real
viewcomp = torch.view_as_complex

# degree to radian
D2R = np.pi / 180

# log-base-e to log-base-10
log2ten = np.log(np.e) / np.log(10)


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


def _cfloat(float_type=None, numpy=False):
    """Manipulate with torch.set_default_dtype()"""
    float_type = float_type if float_type is not None else torch.get_default_dtype()
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


_float_resol = {
    torch.float16: 0,
    torch.complex32: 1,
    torch.float32 : 2,
    torch.complex64: 3,
    torch.float64: 4,
    torch.complex128: 5,
}

_cfloat2float = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

_float2cfloat = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}

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


def half_gaussian_taper(x, xcenter, sigma, below=True):
    """
    Create a single-sided Gaussian tapering function

    Parameters
    ----------
    x : tensor
        x-values (e.g. theta sky coordinates)
    xcenter : float
        Center of Gaussian in x-values
    sigma : float
        Sigma of Gaussian
    below : bool, optional
        If True, apply Gaussian for x < xcenter
        otherwise for x > xcenter

    Returns
    -------
    tensor
    """
    win = torch.ones_like(x)
    if below:
        s = x < xcenter
    else:
        s = x > xcenter

    win[s] = torch.exp(-.5 * (x[s] - xcenter)**2 / sigma**2)

    return win


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
                 interp_cache_depth=None):
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
        theta_grid, phi_grid : tensor
            1D zen and azimuth arrays [deg] if pixtype is 'rect'
            defining the grid to be interpolated against. These
            should mark the pixel centers.
        device : str, optional
            Device to place object on
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
        zen, az : tensors, zenith (co-lat) and azimuth angles [deg]

        Returns
        -------
        interp : tuple
            nearest neighbor (indices, weights)
            for each entry in zen, az for interpolation
        """
        # get hash
        h = arr_hash(zen)
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
                inds = torch.as_tensor(inds.T)
                wgts = torch.as_tensor(wgts.T)

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
                dx = xgrid[1] - xgrid[0]
                dy = ygrid[1] - ygrid[0]
                xnew, ynew = az, zen

                # get map indices
                inds, xyrel = bipoly_grid_index(xgrid, ygrid, xnew, ynew,
                                                degree[0]+1, degree[1]+1,
                                                wrapx=True, ravel=True)

                # get weights
                Ainv, Anew = setup_bipoly_interp(degree, dx, dy, xyrel[0], xyrel[1],
                                                 device=inds.device)
                wgts = Anew @ Ainv

            interp = (inds, wgts)

            # store it in the cache
            if self.interp_cache_depth is None or self.interp_cache_depth > 0:
                if not check_devices(inds.device, self.device):
                    inds = inds.to(self.device)
                    wgts = wgts.to(self.device)
                self.interp_cache[h] = (inds, wgts)

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
        if m.requires_grad:
            # in autodiff mode this leads to faster backprop
            nearest = m.index_select(-1, inds.view(-1)).view(m.shape[:-1] + inds.shape)
        else:
            # in inference mode this is faster
            nearest = m[..., inds]

        # multiply by weights and sum
        out = torch.einsum('...i,...i->...', nearest, wgts)

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
        if not dtype:
            self.device = device
        for k in self.interp_cache:
            cache = self.interp_cache[k]
            self.theta_grid = push(self.theta_grid, device)
            self.phi_grid = push(self.phi_grid, device)
            if dtype:
                cache = (cache[0], push(cache[1], device))
            else:
                cache = (push(cache[0], device), push(cache[1], device))
            self.interp_cache[k] = cache


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
                      wrapx=False, ravel=True):
    """
    For uniform grid in x and y, pick out N nearest neighbor (NN) indices
    in x and y given a sampling of new xy values.
    
    Parameters
    ----------
    xgrid : tensor
        1D float array of grid x values (monotonic. increasing)
    ygrid : tensor
        1D float array of grid y values (monotonic. increasing)
    xnew : tensor
        New x samples to get nearest neighbors
    ynew : tensor
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

    Returns
    -------
    inds : tensor
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
    # get dx, dy
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    # wrap xgrid
    N = len(xgrid)
    if wrapx:
        xgrid = torch.cat([xgrid[-Nx:]-N*dx, xgrid, xgrid[:Nx]+N*dx])

    # get xgrid and ygrid indices for each xynew (Nsamples, Nnn)
    xnn = torch.sort(torch.argsort(torch.abs(xgrid - xnew[:, None]), dim=-1)[:, :Nx], dim=-1).values
    ynn = torch.sort(torch.argsort(torch.abs(ygrid - ynew[:, None]), dim=-1)[:, :Ny], dim=-1).values

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


def setup_bipoly_interp(degree, dx, dy, xnew, ynew, device=None):
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
    device : str, optional
        Device to create polynomial on.

    Returns
    -------
    AtAinvAt : tensor
        (A^T A)^-1 A^T portion of the x_hat vector
    Anew : tensor
        A matrix at xnew and ynew points
    """
    if not isinstance(degree, (list, tuple)):
        degree = [degree, degree]
    assert len(degree) == 2

    # setup grid given degree
    Npoints = [degree[0]+1, degree[1]+1]
    x, y = torch.meshgrid(
        torch.arange(Npoints[0], device=device) * dx,
        torch.arange(Npoints[1], device=device) * dy,
        indexing='xy')
    X = torch.stack([x.ravel(), y.ravel()]).T

    # get polynomial design matrix
    A = torch.zeros(len(X), Npoints[0] * Npoints[1], device=device)
    k = 0
    for i in range(Npoints[0]):
        for j in range(Npoints[1]):
            A[:, k] = X[:, 0]**i * X[:,1]**j
            k += 1

    # get inverse
    AtAinvAt = torch.linalg.pinv(A.T @ A, hermitian=True) @ A.T

    # get new design matrix
    X = torch.stack([xnew * dx, ynew * dy]).T
    Anew = torch.zeros(len(X), Npoints[0] * Npoints[1], device=device)
    k = 0
    for i in range(Npoints[0]):
        for j in range(Npoints[1]):
            Anew[:, k] = X[:, 0]**i * X[:,1]**j
            k += 1

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
        parameter : bool, optional
            Set tensors from pdict as Parameters
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

    def register_response_hooks(self, registry=None):
        """
        Setup a registry of gradient hooks to apply to 
        the output of the params tensor after passing through
        the response function. Default is no hooks. Clear these hooks
        by calling this function with empty kwargs.
        Sets self._hook_registry

        Parameters
        ----------
        registry : list, optional
            List of hook() callables. grad_hook*() functions.
        """
        if registry is not None and not isinstance(registry, (list, tuple)):
            registry = [registry]
        self._hook_registry = registry

    def clear_graph_tensors(self):
        """
        If any graph tensors are attached to self, use
        this function to clear them, i.e. del them
        or set them to None, in order to send them
        to garbage collection and not lead to a
        stale graph. For example, for PixelBeam
        in interpolate mode, the forward modeled
        beam is set as self.beam_cache, which can
        be interpolated against at all unique times
        of the simulation. Before each new forward
        call (i.e. after the previous L.backward())
        this tensor must be cleared
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
    torch.nn.Module.named_parameters() stye.

    Note that for batch_idx API, this assumes a
    RIME object is the first model in self._models
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

    @batch_idx.setter
    def batch_idx(self, val):
        """Set the current batch index"""
        if hasattr(self.get_submodule(self._models[0]), 'batch_idx'):
            self.get_submodule(self._models[0]).batch_idx = val
        elif val > 0:
            raise ValueError("No batch_idx and requested idx > 0")

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
                   no_grad=True, idx=None, add=False, fill=None):
    """
    Set value to model as model.name.

    Parameters
    ----------
    model : utils.Module object
    name : str
        Name of attribute to set
    value : tensor
        New tensor values to set
    clobber_param : bool, optional
        If True and name already exists as a Parameter,
        detach it then assign value as name
        (this removes existing name from graph).
        If False (default) and name already exists as Parameter,
        try to insert value into existing model.name as a Parameter.
    no_grad : bool, optional
        If True, enter a torch.no_grad() context,
        otherwise enter a nullcontext.
    idx : tuple, optional
        If model.name already exists, insert value as
        model.name[idx] = value. Note that if clobber_param = False
        and model.name is a Parameter, this will only work with
        simple indexing schemes. Note that value should already
        have a shape that matches model.name[idx]
    add : bool, optional
        If True, add value to existing tensor values, thus preserving
        graph of value and existing graph of tensor. Otherwise
        simply replace the elements with value
    fill : float, optional
        If tensor exists, fill the non-indexed elements
        with this value. Default is to use existing elements.
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
            param = getattr(model, name) if hasattr(model, name) else None
            parameter = isinstance(param, torch.nn.Parameter)

            if param is not None:
                # if clobber_param, del and reset it (even if not a Parameter)
                if clobber_param or parameter:
                    pd = param.data
                    delattr(model, name)
                    setattr(model, name, pd)
                    param = getattr(model, name)

                # check device
                device = param.device
                if not check_devices(device, value.device):
                    value = value.to(device)

                # fill elements if needed
                if fill is not None:
                    param.data[:] = fill.to(param.data.dtype)

                # insert value
                if add:
                    # add value
                    if idx is None:
                        param += value
                    else:
                        param[idx] += value
                else:
                    # replace with value
                    if idx is None:
                        # note this actually replaces the whole
                        # tensor with value. all others assign in-place
                        setattr(model, name, value)
                    else:
                        param[idx] = value

                # set as parameter if needed
                if not clobber_param and parameter:
                    setattr(model, name, torch.nn.Parameter(getattr(model, name)))

            else:
                # model.name doesn't exist, so just set it
                setattr(model, name, value)

        else:
            # recurse through the '.' names until you get to the end
            set_model_attr(get_model_attr(model, '.'.join(name[:-1])),
                           name[-1], value, clobber_param=clobber_param,
                           idx=idx, no_grad=no_grad, add=add, fill=fill)


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


def arr_hash(arr, pntr=False):
    """
    'Hash' an array or tensor by using its
    first value, last value and length as a
    unique tuple identifier of the array.
    Note that this moves tensors to cpu, meaning
    GPU synchronization will occur. To avoid this,
    precompute the hash and store as arr._arr_hash.

    Parameters
    ----------
    arr : ndarray or tensor
    pntr : bool, optional
        If True, hash arr by its memory id. This can
        lead to erroneous results if arr is mutable.
        If False (default), hash by values in the array.

    Returns
    -------
    int or tuple
    """
    if pntr:
        return id(arr)

    if hasattr(arr, '_arr_hash'):
        return arr._arr_hash
    if isinstance(arr, torch.Tensor):
        key = (arr[0].cpu().item(), arr[-1].cpu().item(), len(arr))
        h = hash(key)
        arr._arr_hash = h
    elif isinstance(arr, np.ndarray):
        key = (arr[0], arr[-1], len(arr))
        h = hash(key)
    else:
        key = (arr[0], arr[-1], len(arr))
        h = hash(key)

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
    if tensor is None or device is None: return tensor
    if hasattr(tensor, 'push') and not isinstance(tensor, torch.Tensor):
        # this is not a tensor but an object with a push() method
        tensor.push(device)
        return tensor
    if not isinstance(tensor, torch.Tensor):
        # this isn't a tensor or an obj with push(), so just return it
        return tensor
    # this is a tensor
    dtype = isinstance(device, torch.dtype)
    if dtype:
        # only change tensor.dtype for float or complex tensors
        is_complex = tensor.is_complex()
        is_float = tensor.is_floating_point()
        if (not is_float) and (not is_complex):
            # int or bool tensor
            return tensor
        elif is_complex and (not device.is_complex):
            # convert complex to another complex
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
        if not check_devices(tensor.device, 'cpu'):
            tensor = tensor.cpu()
        if clone:
            tensor = tensor.clone()
        return tensor.numpy()
    return tensor


def parse_device(d):
    """
    parse a device into a standard pytorch format

    Parameters
    ----------
    d : str or torch.device object
    """
    if d is None:
        d = 'cpu'
    if isinstance(d, torch.device):
        if d.index is None and d.type == 'cuda':
            d = torch.device('cuda:0')
    elif d == 'cuda':
        d = 'cuda:0'

    return torch.device(d)


def check_devices(d1, d2):
    """
    Check the equivalence of two devices.
    If either are None, return True

    Parameters
    ----------
    d1 : str or torch.device
    d2 : str or torch.device

    Returns
    -------
    bool
    """
    if d1 is None or d2 is None:
        return True
    return parse_device(d1) == parse_device(d2)


def check_cuda(d1, d2):
    """
    Check if d1 and d2 are both cuda (any ID)

    Parameters
    ----------
    d1 : str or torch.device
    d2 : str or torch.device

    Returns
    -------
    bool
    """
    return parse_device(d1).type == parse_device(d2).type == 'cuda'


def grad_hook_store(store, assign):
    """
    This returns a callable hook function, which assigns
    an input grad tensor into the store dictionary with
    key assign

    Parameters
    ----------
    store : dictionary
    assign : str

    Returns
    -------
    callable
    """
    def hook(grad):
        store[assign] = grad

    return hook


def grad_hook_assign(value, index=()):
    """
    This retuns a callable hook function, which indexes
    the input grad tensor and assigns its elements as value.

    Parameters
    ----------
    value : float or tensor
    index : tuple, optional

    Returns
    -------
    callable
    """
    def hook(grad):
        new_grad = grad.clone()
        new_grad[index] = value
        return new_grad

    return hook


def grad_hook_mult(value, index=()):
    """
    This retuns a callable hook function, which indexes
    the grad tensor and multiplies by value.

    Parameters
    ----------
    value : float or tensor
    index : tuple, optional

    Returns
    -------
    callable
    """
    def hook(grad):
        new_grad = grad.clone()
        new_grad[index] *= value
        return new_grad

    return hook


def grad_hook_modify(func):
    """
    This retuns a callable hook function, takes
    grad and passes it through a generalized
    function func.

    Parameters
    ----------
    func : callable, optional

    Returns
    -------
    callable

    """
    def hook(grad):
        return func(grad)

    return hook


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


def split_into_groups(arr, Nelem=None, Ngroup=None, interleave=False):
    """
    Split a list or array of elements into
    a nested list of groups each being Nelem long

    Parametrs
    ---------
    arr : array or list
        List of elements
    Nelem : int, optional
        Max number of elements in each sub-group. Pass
        either this or Ngroup
    Ngroup : int, optional
        Max number of total sub-groups to use. Pass
        either this or Nelem
    interleave : bool, optional
        If True, split into groups by interleaving,
        otherwise split groups by chunks (default)

    Returns
    -------
    array or list
    """
    if Nelem is not None:
        assert Ngroup is None
    N = len(arr)
    if interleave:
        if Ngroup is None:
            Ngroup = int(np.ceil(N / Nelem))
        sublist = [arr[i::Ngroup] for i in range(Ngroup)]
    else:
        if Nelem is None:
            Nelem = int(np.ceil(N / Ngroup))
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
            util = lines[i+1].split('|')[3].strip().split('%')[0]
            usage[gpu] = "{:>6} / {:>6} MiB | {:>2}%".format(mems[0].strip()[:-3], mems[1].strip()[:-3], util)
            alloc = "{:>6} MiB".format("{:,}".format(int(mems[0].strip()[:-3])))
            total = "{:>6} MiB".format("{:,}".format(int(mems[1].strip()[:-3])))
            mem = "{} / {}".format(alloc, total)
            output.append("| GPU {} | {} | {:>2}% |".format(gpu, mem, util))
    if verbose:
        print('\n'.join([date] + ['o' + '-'*39 + 'o'] + output + ['o' + '-'*39 + 'o']))

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


def inflate_bls(red_bls, bl2red, all_bls=None):
    """
    Inflate unique baseline to all physical baselines

    Parameters
    ----------
    red_bls : list of ant-pair tuples
        List of unique bls of current redundant data set.
    bl2red : dict, optional
        A {bl: int} dictionary mapping a physical baseline
        to its redundant group index.
    all_bls : list of ant-pair tuples, optional
        If provided, only inflate to physical bls
        present in all_bls. If a redundant baseline
        match doesn't exist in red_bls then drop
        it from output.

    Returns
    -------
    new_bls : list
        List of physical baselines
    red_inds : list
        List of indices of bls for
        each bl in new_bls
    """
    if all_bls is None:
        all_bls = list(bl2red.keys())

    # get redundant indices of current baselines
    red_indices = set(bl2red.get(bl, None) for bl in red_bls)

    # iterate over new_bls and get corresponding redundant index
    new_bls, red_inds = [], []
    for bl in all_bls:
        red_idx = bl2red.get(bl, -1)
        if red_idx in red_indices:
            new_bls.append(bl)
            red_inds.append(red_idx)

    return new_bls, red_inds


def _list2slice(inds):
    """convert list/tuple/tensor indexing to slice if possible"""
    if isinstance(inds, range) and inds.step > 0:
        return slice(inds.start, inds.stop, inds.step)
    if isinstance(inds, (int, np.integer)):
        return slice(inds, inds+1)
    if isinstance(inds, (list, tuple, torch.Tensor, np.ndarray)):
        if len(inds) == 0:
            # only 1 element
            return inds
        elif len(inds) == 1:
            # only 1 element
            start = int(inds[0])
            return slice(start, start+1, 1)
        # non-trivial
        if isinstance(inds, torch.Tensor):
            diff = list(set(np.diff(inds.cpu())))
        else:
            diff = list(set(np.diff(inds)))
        if len(diff) == 1:
            # constant step size
            if (inds[1] - inds[0]) > 0:
                # only return as slice if inds is increasing
                return slice(int(inds[0]), int(inds[-1]+diff[0]), int(diff[0]))

    return inds


def _slice2tensor(obj, device=None):
    """Convert a slice object to a integer tensor"""
    if isinstance(obj, slice):
        start = obj.start if obj.start is not None else 0
        stop = obj.stop
        step = obj.step if obj.step is not None else 1
        obj = torch.arange(start, stop, step, device=device)

    return obj


def _idx2ten(idx, device=None):
    """Convert a 1d indexing list or ndarray to tensor
    and push to a desired device.
    if idx is a slice or int, do nothing"""
    if isinstance(idx, (list, np.ndarray, tuple)):
        # check if idx is a bool ndarray, if so convert to int
        if isinstance(idx, np.ndarray) and idx.dtype == np.dtype(bool):
            idx = torch.as_tensor(np.where(idx)[0])
        else:
            idx = torch.as_tensor(idx, dtype=torch.long)
    # check if idx is a bool tensor
    if isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
        idx = torch.where(idx)[0]
    if device is not None and isinstance(idx, torch.Tensor):
        idx = idx.to(device)

    return idx


def _cat_idx(idx, device=None):
    """Concatenate multiple 1D indexing objects (tensors, slice objects)
    packed in a list and return as a int tensor"""
    if isinstance(idx, list):
        idx = torch.cat([_slice2tensor(_idx2ten(i, device=device), device=device) for i in idx])

    return idx


def _tensor_concat(tensors, dim=0, interleave=False):
    """
    Given a list of tensors of the same ndim but
    possibly different length along dim, concatenate them
    with optional interleaving
    """
    try:
        if interleave:
            shape = list(tensors[0].shape)
            shape[dim] = sum(t.shape[dim] for t in tensors)
            N = shape[dim]
            out = torch.zeros(shape, dtype=tensors[0].dtype, device=tensors[0].device)
            indices = [torch.arange(i, N, len(tensors), device=tensors[0].device) for i in range(len(tensors))]
            for ten, idx in zip(tensors, indices):
                out.index_add_(dim, idx, ten)
        else:
            out = torch.cat(tensors, dim=dim)
    except TypeError as err:
        # this happens if one entry in tensors is not a Tensor
        # which is possible if we are concat tensors with
        # read_data = False, or lazy_load=True
        if tensors[0] is None:
            out = None
        else:
            raise err

    return out


class AntposDict:
    """
    A dictionary for antenna positions
    that functions like a normal dictionary
    but holds its values in contiguous memory
    under the hood.
    Note: this means repeateadly setting keys
    scales as N^2. Better to re-instantiate
    existing keys along with new keys at once.
    """
    def __init__(self, ants, antvecs):
        """
        Parameters
        ----------
        ants : list or tensor
            Antenna integers
        antvecs : tensor
            Antenna positions in ENU coordinates [meters]
            of shape (Nants, 3)
        """
        self.ants = list(ants)
        self._ant_idx = {a: self.ants.index(a) for a in self.ants}
        try:
            # this works if antvec is 1) ndarray
            # 2) list of ndarray or 3) tensor
            self.antvecs = torch.as_tensor(antvecs)
        except ValueError:
            # this works if antvec is list of tensor
            self.antvecs = torch.vstack(antvecs)

    def keys(self):
        return (a for a in self.ants)

    def values(self):
        return (av for av in self.antvecs)

    def items(self):
        return zip(self.ants, self.antvecs)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.antvecs[self._ant_idx[key]]
        elif isinstance(key, (list, tuple, np.ndarray, torch.Tensor)):
            if isinstance(key, torch.Tensor):
                key = key.tolist()
            return self.antvecs[[self._ant_idx[k] for k in key]]

    def __setitem__(self, key, value):
        idx = self._ant_idx[key]
        self.antvecs[idx] = value

    def __repr__(self):
        return "Antpos{{{}}}".format(self.ants)

    def __len__(self):
        return len(self.ants)

    def __contains__(self, key):
        return key in self.ants

    def __iter__(self):
        return self.keys()

    def push(self, device):
        self.antvecs = push(self.antvecs, device)


def blnum2ants(blnum, separate=False):
    """
    Convert baseline integer to tuple of antenna numbers.

    Parameters
    ----------
    blnum : integer or ndarray
        baseline integers, e.g. 102103 -> (2, 3)
    separate : bool, optional
        If True, return ant1, ant2 lists, otherwise
        return them as tuples (ant1, ant2) (default)

    Returns
    -------
    antnums : tuple
        tuple containing baseline antenna numbers. Ex. (ant1, ant2)
    """
    if isinstance(blnum, tuple):
        # assumed already antnums tuple
        return blnum
    elif isinstance(blnum, list) and isinstance(blnum[0], tuple):
        # assumed already list of antnum tuples
        if separate:
            return list(zip(*blnum))
        else:
            return blnum

    # get antennas
    if isinstance(blnum, (int, np.integer)):
        ant1 = int(np.floor(blnum / 1e3))
        ant2 = blnum - ant1 * 1000
        ant1 -= 100
        ant2 -= 100

        return (ant1, ant2)

    elif isinstance(blnum, (list, np.ndarray)):
        ant1 = (np.floor(blnum) / 1e3).astype(np.int64)
        ant2 = np.asarray(blnum) - ant1 * 1000
        ant1 -= 100
        ant2 -= 100
        ant1 = ant1.tolist()
        ant2 = ant2.tolist()

        if separate:
            return ant1, ant2
        else:
            return list(zip(ant1, ant2))

    elif isinstance(blnum, torch.Tensor):
        blnum = blnum.cpu()
        ant1 = torch.floor(blnum / 1e3).to(torch.int64)
        ant2 = blnum - ant1 * 1000
        ant1 -= 100
        ant2 -= 100
        ant1 = ant1.tolist()
        ant2 = ant2.tolist()

        if separate:
            return ant1, ant2
        else:
            return list(zip(ant1, ant2))


def ants2blnum(antnums, separate=False, tensor=False):
    """
    Convert tuple of antenna numbers to baseline integer.
    A baseline integer is the two antenna numbers + 100
    directly (i.e. string) concatenated. Ex: (1, 2) -->
    101 + 102 --> 101102.

    Parameters
    ----------
    antnums : tuple or list
        tuple containing integer antenna numbers for a baseline.
        Ex. (ant1, ant2)
    separate : bool, optional
        If True, return tuple of separated baseline numbers
        otherwise return as a single number (default)
    tensor : bool, optional
        If True, return as torch.Tensor, otherwise
        return as np.ndarray (default)

    Returns
    -------
    blnum : integer or ndarray
        baseline integer
    """
    if isinstance(antnums, tuple):
        # get antennas
        ant1 = antnums[0] + 100
        ant2 = antnums[1] + 100

        if separate:
            bl = (ant1, ant2)
        else:
            # form bl
            bl = int(ant1*1000 + ant2)

    elif isinstance(antnums, list) and isinstance(antnums[0], tuple):
        # assumed list of antnum tuples
        bl = np.asarray(antnums) + 100
        if separate:
            bl = (bl[:, 0] * 1000, bl[:, 1])
        else:
            bl = bl[:, 0] * 1000 + bl[:, 1]

    else:
        # assumed antnums already a blnum
        bl = antnums
        if separate:
            bl = (bl // 1000, bl % 1000)

    if tensor:
        bl = torch.as_tensor(bl)

    return bl


def conjbl(bl):
    """
    Conjugate a blnum or antpair tuple
    """
    if isinstance(bl, tuple):
        # antpair tuple
        return bl[::-1]
    elif isinstance(bl, list) and isinstance(bl[0], tuple):
        # list of antpair tuples
        return [conjbl(b) for b in bl]
    else:
        # single blnum or blnum ndarray
        return 1000 * (bl % 1000) + bl // 1000

