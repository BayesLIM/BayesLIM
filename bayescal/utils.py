"""
Utility module
"""
import numpy as np
import torch
import healpy, mhealpy
from scipy import special
import copy


########################################
######### Linear Algebra Tools #########
########################################

viewreal = torch.view_as_real
viewcomp = torch.view_as_complex


def cmult(a, b):
    """
    Complex multiplication of two real-valued torch
    tensors in "2-real" form, or of shape (..., 2)
    where the last axis indexes the real and imag
    component of the tensor, respectively.

    Parameters
    ----------
    a : tensor
        In 2-real form
    b : tensor
        In 2-real form

    Returns
    -------
    tensor
        Complex product of a and b in 2-real form
    """
    return viewreal(viewcomp(a) * viewcomp(b))


def cdiv(a, b):
    """
    Complex division (a / b) of two real-valued torch
    tensors in "2-real" form, or of shape (..., 2)
    where the last axis indexes the real and imag
    component of the tensor, respectively.

    Parameters
    ----------
    a : tensor
        In 2-real form
    b : tensor
        In 2-real form

    Returns
    -------
    tensor
        Complex division of a / b in 2-real form
    """
    return viewreal(viewcomp(a) / viewcomp(b))


def cconj(z):
    """
    Complex conjugate of a real-valued torch
    tensor in "2-real" form, or of shape (..., 2)
    where the last axis indexes the real and imag
    component of the tensor, respectively.

    Parameters
    ----------
    z : tensor
        In 2-real form

    Returns
    -------
    tensor
        Complex conjugate of z in 2-real form
    """
    return viewreal(viewcomp(z).conj())


def ceinsum(equation, *operands):
    """
    A shallow wrapper around torch.einsum,
    taking 2-real operands and returning
    2-real output.

    Parameters
    ----------
    equation : str
        A torch.einsum equation
    operands : tensor
        torch tensors to operate on in 2-real form

    Returns
    -------
    tensor
        Output of einsum in 2-real form
    """
    raise NotImplementedError("pytorch doesn't yet support complex autodiff for this")
    operands = (viewcomp(op) for op in operands)
    return viewreal(torch.einsum(equation, *operands))


def cinv(z):
    """
    Take the inverse of z
    across the first two axes

    Parameters
    ----------
    z : tensor
        torch tensor in 2-real form

    Returns
    -------
    tensor
        inverse of z in 2-real form
    """
    return viewreal(torch.inverse(viewcomp(z).T).T)


def diag_matmul(a, b):
    """
    Multiply two, diagonal 1x1 or 2x2 matrices manually.
    This is generally faster than matmul or einsum
    for large, high dimensional stacks of 2x2 matrices.

    !! Note: this ignores the off-diagonal for 2x2 matrices !!
    If you need off-diagonal components, you are
    better off using torch.matmul or torch.einsum directly.

    Parameters
    ----------
    a, b : tensor
        of shape (Nax, Nax, ...), where Nax = 1 or 2

    Returns
    -------
    c : tensor
        of shape (Nax, Nax, ...)
    """
    if a.shape[0] == 1:
        # 1x1: trivial
        return a * b
    elif a.shape[0] == 2:
        # 2x2
        c = torch.zeros_like(a)
        c[0, 0] = a[0, 0] * b[0, 0]
        c[1, 1] - a[1, 1] * b[1, 1]
        return c
    else:
        raise ValueError("only 1x1 or 2x2 tensors")


def angle(z):
    """
    Compute phase of the 2-real tensor z

    Parameters
    ----------
    z : tensor
        In 2-real form

    Returns
    -------
    float or ndarray
        Phase of z in radians
    """
    return torch.angle(viewcomp(z))


def abs(z):
    """
    Take the abs of a 2-real tensor z.

    Parameters
    ----------
    z : tensor
        In 2-real form

    Returns
    -------
    tensor
        The amplitude of the input 2-real tensor
        projected into the complex plane with
        zero phase

    """
    zabs = torch.clone(z)
    zabs[..., 0] = torch.linalg.norm(z, axis=-1)
    zabs[..., 1] = 0
    return zabs


def apply_phasor(z, phi):
    """
    Apply a complex phasor to z

    Parameters
    ----------
    z : tensor
        In 2-real form
    phi : float
        Phase of phasor in radians

    Returns
    -------
    tensor
        z in 2-real form with phi applied
    """
    return viewreal(viewcomp(z) * np.exp(1j * phi))


def project_out_phase(z, avg_axis=None, select=None):
    """
    Compute and project out the phase of z

    Parameters
    ----------
    z : tensor
        In 2-real form
    avg_axis : int, optional
        Average z along avg_axis before computing
        its phase. Default is None.
    select : list, optional
        Use this to index z after any averaging
        before computing the phase.
        E.g.: select = [slice(None), slice(0, 1)].
        Note that this indexing must keep z's dimensionality.
        Default is None.

    Returns
    -------
    tensor
        z in 2-real form with phase projected out
    """
    if avg_axis is not None:
        za = torch.mean(z, axis=avg_axis, keepdim=True)
    else:
        za = z
    if select is not None:
        za = z[select]
    z_phs = angle(za)

    return apply_phasor(z, -z_phs)


def ones(*args, **kwargs):
    """
    Construct a 2-real tensor of ones

    Parameters
    ----------
    shape : tuple
        Shape of tensor

    Returns
    -------
    tensor
        A 2-real tensor full of ones

    Notes
    -----
    keyword arguments passed to torch.ones
    """
    ones = torch.ones(*args, **kwargs)
    ones[..., 1] = 0
    return ones


def cmatmul(a, b):
    """
    Perform 1x1 or 2x2 matrix multiplication
    along the first two axes of a and b
    in 2-real form. Note: this is slow
    compared to torch.einsum, but doesn't need
    to cast 

    Parameters
    -----------
    a : tensor
        In 2-real form with shape of b
    b : tensor
        In 2-real form with shape of a

    Returns
    -------
    tensor
        Matrix multiplication of a and b along
        their 0th and 1st axes
    """
    # determine if 1x1 or 2x2 matmul
    assert b.shape[0] == b.shape[1] == a.shape[0] == a.shape[1]
    assert a.shape[0] in [1, 2]
    twodim = True if a.shape[0] == 2 else False

    if not twodim:
        # 1x1 matmul is trivial
        return cmult(a, b)
    else:
        # 2x2 matmul
        c = torch.zeros_like(a)

        # upper left real
        c[0, 0, ..., 0] = a[0, 0, ..., 0] * b[0, 0, ..., 0] - a[0, 0, ..., 1] * b[0, 0, ..., 1] \
                          + a[0, 1, ..., 0] * b[1, 0, ..., 0] - a[0, 1, ..., 1] * b[1, 0, ..., 1]

        # upper left imag
        c[0, 0, ..., 1] = a[0, 0, ..., 0] * b[0, 0, ..., 1] + a[0, 0, ..., 1] * b[0, 0, ..., 0] \
                          + a[0, 1, ..., 0] * b[1, 0, ..., 1] + a[0, 1, ..., 1] * b[1, 0, ..., 0]

        # upper right real
        c[0, 1, ..., 0] = a[0, 0, ..., 0] * b[0, 1, ..., 0] - a[0, 0, ..., 1] * b[0, 1, ..., 1] \
                          + a[0, 1, ..., 0] * b[1, 1, ..., 0] - a[0, 1, ..., 1] * b[1, 1, ..., 1]

        # upper right imag
        c[0, 1, ..., 1] = a[0, 0, ..., 0] * b[0, 1, ..., 1] + a[0, 0, ..., 1] * b[0, 1, ..., 0] \
                          + a[0, 1, ..., 0] * b[1, 1, ..., 1] + a[0, 1, ..., 1] * b[1, 1, ..., 0]

        # lower left real
        c[1, 0, ..., 0] = a[1, 0, ..., 0] * b[0, 0, ..., 0] - a[1, 0, ..., 1] * b[0, 0, ..., 1] \
                          + a[1, 1, ..., 0] * b[1, 0, ..., 0] - a[1, 1, ..., 1] * b[1, 0, ..., 1]

        # lower left imag
        c[1, 0, ..., 1] = a[1, 0, ..., 0] * b[0, 0, ..., 1] + a[1, 0, ..., 1] * b[0, 0, ..., 0] \
                          + a[1, 1, ..., 0] * b[1, 0, ..., 1] + a[1, 1, ..., 1] * b[1, 0, ..., 0]

        # lower right real
        c[1, 1, ..., 0] = a[1, 0, ..., 0] * b[0, 1, ..., 0] - a[1, 0, ..., 1] * b[0, 1, ..., 1] \
                          + a[1, 1, ..., 0] * b[1, 1, ..., 0] - a[1, 1, ..., 1] * b[1, 1, ..., 1]

        # lower right imag
        c[1, 1, ..., 1] = a[1, 0, ..., 0] * b[0, 1, ..., 1] + a[1, 0, ..., 1] * b[0, 1, ..., 0] \
                          + a[1, 1, ..., 0] * b[1, 1, ..., 1] + a[1, 1, ..., 1] * b[1, 1, ..., 0]


    return c


#####################################
######### Sky Mapping Tools #########
#####################################


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
    beam = special.voigt_profile(theta, sigma, gamma)
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


def nside_binning(zen_dec, ra, dec_sigma=5, dec_gamma=15, ra_sigma=5, ra_gamma=15,
                  ra_min_max=None, min_nside=32, max_nside=256):
    """
    Compute nside binning using a voigt profile given
    a map of sky angles. Note for the ra axis: be mindful
    of how the ra_min_max cuts depend on the wrapping of
    the input ra array.

    Parameters
    ----------
    zen_dec : array_like
        Zenith sky angle along the declination axis [deg].
    ra : array_like
        Right ascension coordinate [deg]. 
    dec_sigma, dec_gamma : float
        Sigma and gamma parameters of voigt profile along
        declination axis [deg]
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
    # get dec component of voigt profile
    curve = special.voigt_profile(zen_dec, dec_sigma, dec_gamma)
    curve -= curve.min()
    curve /= curve.max()

    # get ra component of voigt profile
    if ra_min_max is not None:
        # enact a nside res decay for ra less than min ra
        assert ra_min_max[0] > ra.min()
        ra_low = ra < ra_min_max[0]
        ra_low_curve = special.voigt_profile(ra[ra_low] - ra_min_max[0], ra_sigma, ra_gamma)
        ra_low_curve -= ra_low_curve.min()
        ra_low_curve /= ra_low_curve.max()
        curve[ra_low] *= ra_low_curve
        # enact a nside res decay for ra greater than max ra
        ra_hi = ra > ra_min_max[1]
        assert ra_min_max[1] < ra.max()
        ra_hi_curve = special.voigt_profile(ra[ra_hi] - ra_min_max[1], ra_sigma, ra_gamma)
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
    theta, phi, nsides, total_nsides = [], [], [], []
    for i in range(healpy.nside2npix(base_nside)):
        target = target_nsides[i] if target_nsides is not None else None
        _recursive_pixelization(bsky, i, base_nside, max_nside, theta, phi, nsides, total_nsides,
                                sigma=sigma, target_nside=target)
    theta, phi, total_nsides = np.array(theta), np.array(phi), np.array(total_nsides)
    # turn nsides into mhealpy HealpixMap object
    ipix = [healpy.ang2pix(ns, th, ph, nest=True) for ns, th, ph in zip(nsides, theta, phi)]
    uniq = [4 * ns**2 + ip for ns, ip in zip(nsides, ipix)]
    nsides = mhealpy.HealpixMap(nsides, uniq=uniq, scheme='nest', dtype=np.float32)

    return theta, phi, nsides, total_nsides
