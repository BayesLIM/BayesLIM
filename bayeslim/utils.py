"""
Utility module
"""
import numpy as np
import torch
from scipy.special import voigt_profile
from scipy.integrate import quad
from scipy.signal import windows
from scipy.interpolate import interp1d
import copy
import warnings
import os
from astropy import constants
import time as timer
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


def gen_lm(lmax, real_field=True):
    """
    Generate array of l and m parameters.
    Matches healpy.sphtfunc.Alm.getlm order.

    Parameters
    ----------
    lmax : int
        Maximum l parameter
    real_field : bool, optional
        If True, treat sky as real-valued (default)
        so truncate negative m values.

    Returns
    -------
    l, m : array_like
        array of shape (2, Ncoeff) holding
        the (l, m) parameters.
    """
    lms = []
    lowm = 0 if real_field else -lmax
    for m in range(lowm, lmax + 1):
        for l in range(0, lmax + 1):
            if np.abs(m) > l: continue
            lms.append([l, m]) 
    return np.array(lms).T


def sph_stripe_lm(phi_max, mmax, theta_min, theta_max, lmax, dl=0.1,
                  mmin=0, high_prec=True, add_sectoral=True):
    """
    Compute associated Legendre function degrees l on
    the spherical stripe or cap given boundary conditions.

    theta boundary conditions:
        theta_min == 0 and theta_max < pi or
        theta_max == 0 and theta_min > 0:
            This is a spherical cap, with boundary conditions
                P_lm(theta_max) = 0 and
                m == 0: d P_lm(theta_min) / d theta = 0
                m  > 0: P_lm(theta_min) = 0
        theta_min > 0 and theta_max < pi:
            This is a spherical stripe with BC
                P_lm(theta_min) = 0 and
                P_lm(theta_max) = 0
                for m > 0, and replace
                P_lm with d P_lm / d theta
                for m == 0.

    phi boundary conditions:
        Phi(0) = Phi(phi_max), assuming
        mask extends from phi = 0 to phi = phi_max

    Parameters
    ----------
    phi_max : float
        Maximum extent of mask in azimuth [rad]
    mmax : int
        Maximum m mode to compute
    theta_min : float
        Minimum co-latitude of stripe [rad]
    theta_max : float
        Maximum co-latitude of stripe [rad]
    lmax : int
        Maximum degree l to compute for each m
    dl : float, optional
        Sampling density in l from m to lmax
    mmin : int, optional
        Minimum m to compute, default is 0.
    high_prec : bool, optional
        If True, use precise mpmath for hypergeometric
        calls, else use faster but less accurate scipy.
        Matters mostly for non-integer degree
    add_sectoral : bool, optional
        If True, include sectoral modes (l == m)
        regardless of whether they satisfy BC

    Returns
    -------
    l, m
        Array of l and m values
    """
    # solve for m modes
    spacing = 2 * np.pi / phi_max
    assert np.isclose(spacing % 1, 0), "phi_max must evenly divide into 2pi"
    mmin = max([0, mmin])
    m = np.arange(mmin, mmax + 1.1, spacing)

    # solve for l modes
    assert theta_max < np.pi, "if theta_max must be < pi for spherical cap or stripe"
    ls = {}
    x_min, x_max = np.cos(theta_min), np.cos(theta_max)
    m = np.atleast_1d(m)
    for _m in m:
        # construct array of test l's, skip l == m
        larr = _m + np.arange(1, (lmax - _m)//dl + 1) * dl
        marr = np.ones_like(larr) * _m
        if len(larr) < 1:
            continue
        # boundary condition is derivative is zero for m == 0
        deriv = _m == 0
        if np.isclose(theta_min, 0):
            # spherical cap
            y = special.Plm(larr, marr, x_max, deriv=deriv, high_prec=high_prec, keepdims=True)

        elif np.isclose(theta_max, 0):
            # inverted spherical ap
            y = special.Plm(larr, marr, x_min, deriv=deriv, high_prec=high_prec, keepdims=True)

        else:
            # spherical stripe
            y = special.Plm(larr, marr, x_min, deriv=deriv, high_prec=high_prec, keepdims=True) \
                * special.Qlm(larr, marr, x_max, deriv=deriv, high_prec=high_prec, keepdims=True) \
                - special.Plm(larr, marr, x_max, deriv=deriv, high_prec=high_prec, keepdims=True) \
                * special.Qlm(larr, marr, x_min, deriv=deriv, high_prec=high_prec, keepdims=True)

        y = y.ravel()
        zeros = get_zeros(larr, y)
        # add sectoral
        if add_sectoral:
            zeros = [_m] + zeros
        ls[_m] = np.asarray(zeros)

    larr, marr = [], []
    for _m in ls:
        larr.extend(ls[_m])
        marr.extend([_m] * len(ls[_m]))

    return np.array(larr), np.array(marr)


def _gen_sph2pix_multiproc(job):
    (l, m), args, kwargs = job
    Y = gen_sph2pix(*args, **kwargs)
    Ydict = {(_l, _m): Y[i] for i, (_l, _m) in enumerate(zip(l, m))}
    return Ydict


def gen_sph2pix(theta, phi, method='sphere', theta_max=None, l=None, m=None,
                lmax=None, real_field=True, Nproc=None, Ntask=10, device=None,
                high_prec=True, renorm=False):
    """
    Generate spherical harmonic forward model matrix.

    Note, this can take a *long* time: dozens of minutes for a few hundred
    lm at tens of thousands of theta, phi points even with Nproc of 20.
    The code is limited by the internal high precision computation of the
    Legendre functions via mpmath, which enables stable evaluation of
    large, non-integer degree harmonics.
    It is advised to compute these once and store them.
    Also, pip installing the "gmpy" module can offer modest speed up.
    For computing integer degree spherical harmonics, high_prec is
    likely not needed.

    The orthonormalized, full-sky spherical harmonics are

    .. math::

        Y_{lm}(\theta,\phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}
                                e^{im\phi}(P_{lm} + A_{lm}Q_{lm})(\cos(\theta))

    Parameters
    ----------
    theta : array_like
        Co-latitude (i.e. zenith angle) [rad]
    phi : array_like
        Longitude (i.e. azimuth) [rad]
    method : str, optional
        Spherical harmonic mode ['sphere', 'stripe', 'cap']
        For 'sphere', l modes are integer
        For 'stripe' or 'cap', l modes are float
    theta_max : float, optional
        For method == 'stripe', this is the maximum theta
        boundary [radians] of the mask, used for boundary conditions.
    l : array_like, optional
        Integer or float array of spherical harmonic l modes
    m : array_like, optional
        Integer array of spherical harmonic m modes
    lmax : int, optional
        If l, m are None, this generates integer l and m
        arrays
    real_field : bool, optional
        If True, treat sky as real-valued
        so truncate negative m values (used for lmax).
    Nproc : int, optional
        If not None, this starts multiprocessing mode, and
        specifies the number of independent processes.
    Ntask : int, optional
        This is the number of tasks (i.e. modes) computed
        per process.
    device : str, optional
        Device to push Ylm to.
    high_prec : bool, optional
        If True, use precise mpmath for hypergeometric
        calls, else use faster but less accurate scipy.
        Matters mostly for non-integer degree
    renorm : bool, optional
        Re-normalize the spherical harmonics such that their
        norm is 1 over the cut-sky. This is done using the sampled
        theta, phi points as a numerical inner product approx.
        Note this assumes that the theta, phi are drawn from
        part of a HEALpix grid with a pixelization
        density enough to resolve the fastest spatial frequency

    Returns
    -------
    Ylm : array_like
        An (Npix x Ncoeff) matrix encoding a spherical
        harmonic transform from a_lm -> map

    Notes
    -----
    The output dtype can be set using torch.set_default_dtype
    """
    # setup l modes
    if lmax is not None:
        assert method == 'sphere'
        l, m = gen_lm(lmax, real_field=real_field)

    # run multiproc mode
    if Nproc is not None:
        # setup multiprocessing
        import multiprocessing
        Njobs = len(l) / Ntask
        if Njobs % 1 > 0:
            Njobs = np.floor(Njobs) + 1
        Njobs = int(Njobs)
        jobs = []
        for i in range(Njobs):
            _l = l[i*Ntask:(i+1)*Ntask]
            _m = m[i*Ntask:(i+1)*Ntask]
            jobs.append([(_l, _m), (theta, phi), dict(method=method, theta_max=theta_max,
                                                      l=_l, m=_m, high_prec=high_prec,
                                                      renorm=renorm)])

        # run jobs
        try:
            pool = multiprocessing.Pool(Nproc)
            output = pool.map(_gen_sph2pix_multiproc, jobs)
        finally:
            pool.close()
            pool.join()

        # combine
        Y = torch.zeros((len(l), len(theta)), dtype=_cfloat(), device=device)
        for Ydict in output:
            for k in Ydict:
                _l, _m = k
                index = np.where((l == _l) & (m == _m))[0][0]
                Y[index] = Ydict[k].to(device)

        return Y

    # run single proc mode
    if isinstance(l, (int, float)):
        l = np.array([l])
    if isinstance(m, (int, float)):
        m = np.array([m])
    l = l[:, None]
    m = m[:, None]
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    # compute assoc. legendre: orthonorm is already in Plm and Qlm
    x = np.cos(theta)
    if method == 'sphere':
        theta_max = np.pi
    x_max = np.cos(theta_max)
    H = legendre_func(x, l, m, method, x_max=x_max, high_prec=high_prec)
    Phi = np.exp(1j * m * phi)
    Y = torch.as_tensor(H * Phi, dtype=_cfloat(), device=device)

    # renormalize
    if renorm:
        # Note: theta and phi must be part of a HEALpix grid
        norm = torch.sqrt(torch.sum(torch.abs(Y)**2, axis=1))
        Y /= norm[:, None]

    return Y


def write_Ylm(fname, Ylm, angs, l, m, overwrite=False):
    """
    Write a Ylm basis to HDF5 file

    Parameters
    ----------
    fname : str
        Filepath of output hdf5 file
    Ylm : array
        Ylm matrix of shape (Ncoeff, Npix)
    angs : array
        theta, phi sky positions of Ylm in degrees
        of shape (2, Npix). This is either
        co-lat and azimuth or Dec, RA depending on
        whether this is used for beam or sky
    l, m : array
        Ylm degree l and order m of len Ncoeff
    """
    if not os.path.exists(fname) or overwrite:
        with h5py.File(fname, 'w') as f:
            f.create_dataset('Ylm', data=Ylm)
            f.create_dataset('angs', data=np.array(angs))
            f.create_dataset('l', data=l)
            f.create_dataset('m', data=m)


def load_Ylm(fname, lmin=None, lmax=None, discard=None, cast=None,
             colat_min=None, colat_max=None, device=None, read_data=True):
    """
    Load an hdf5 file with Ylm and ang arrays

    Parameters
    ----------
    fname : str
        Filepath to hdf5 file with Ylm, angs, l, and m as datasets.
    lmin : float, optional
        Truncate all Ylm modes with l < lmin
    lmax : float, optional
        Truncate all Ylm modes with l > lmax
    discard : tensor, optional
        Of shape (2, Nlm), holding [l, m] modes
        to discard from fname. Discards any Ylm modes
        that match the provided l and m.
    cast : torch.dtype
        Data type to cast Ylm into
    colat_min : float, optional
        truncate Ylm response for colat < colat_min [deg]
        assuming angs[0] is colatitude (zenith)
    colat_max : float, optional
        truncate Ylm response for colat > colat_max [deg]
        assuming angs[0] is colatitude (zenith)
    device : str, optional
        Device to place Ylm
    read_data : bool, optional
        If True, read and return Ylm
        else return None, angs, l, m 

    Returns
    -------
    Ylm, angs, l, m
    """
    import h5py
    with h5py.File(fname, 'r') as f:
        # load angles and all modes
        angs = f['angs'][:]
        l, m = f['l'][:], f['m'][:]

        # truncate modes
        keep = np.ones_like(l, dtype=bool)
        if lmin is not None:
            keep = keep & (l >= lmin)
        if lmax is not None:
            keep = keep & (l <= lmax)
        if discard is not None:
            cut_l, cut_m = discard
            for i in range(len(cut_l)):
                keep = keep & ~(np.isclose(l, cut_l[i], atol=1e-6) & np.isclose(m, cut_m[i], atol=1e-6))

        # refactor keep slicing
        keep = np.where(keep)[0]
        if len(keep) == len(l):
            keep = slice(None)
        elif len(set(np.diff(keep))) == 1:
            keep = slice(keep[0], keep[-1]+1, keep[1] - keep[0])

        l, m = l[keep], m[keep]
        if read_data:
            Ylm = f['Ylm'][keep, :]
        else:
            Ylm = None

    # truncate sky
    if colat_min is not None:
        colat, az = angs
        keep = colat >= colat_min
        colat = colat[keep]
        az = az[keep]
        angs = colat, az
        if read_data:
            Ylm = Ylm[:, keep]
    if colat_max is not None:
        colat, az = angs
        keep = colat <= colat_max
        colat = colat[keep]
        az = az[keep]
        angs = colat, az
        if read_data:
            Ylm = Ylm[:, keep]

    if read_data:
        Ylm = torch.tensor(Ylm, device=device)
        if cast is not None:
            Ylm = Ylm.to(cast)

    return Ylm, angs, l, m


def legendre_func(x, l, m, method, x_max=None, high_prec=True):
    """
    Generate (un-normalized) assoc. Legendre basis

    Parameters
    ----------
    x : array_like
        Array of x values [-1, 1]
    l : array_like
        float degree
    m : array_like
        integer order
    method : str, ['stripe', 'sphere', 'cap']
        boundary condition method
    x_max : float, optional
        If method is stripe, this the max x value
    high_prec : bool, optional
        If True, use arbitrary precision for Plm and Qlm
        otherwise use standard (faster) scipy method

    Returns
    -------
    H : array_like
        Legendre basis: P + A * Q
    """
    # compute assoc. legendre: orthonorm is already in Plm and Qlm
    P = special.Plm(l, m, x, high_prec=high_prec, keepdims=True)
    if method == 'stripe':
        # spherical stripe: uses Plm and Qlm
        assert x_max is not None
        # compute Qlms
        Q = special.Qlm(l, m, x, high_prec=high_prec, keepdims=True)
        # compute A coefficients
        A = -special.Plm(l, m, x_max, high_prec=high_prec, keepdims=True) \
            / special.Qlm(l, m, x_max, high_prec=high_prec, keepdims=True)
        # Use deriv = True for m == 0
        if 0 in m:
            mzero = np.ravel(m) == 0
            A[mzero] = -special.Plm(l[mzero], m[mzero], x_max, high_prec=high_prec, keepdims=True, deriv=True) \
                       / special.Qlm(l[mzero], m[mzero], x_max, high_prec=high_prec, keepdims=True, deriv=True)

        H = P + A * Q
    else:
        H = P

    return H


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


def _gen_bessel2freq_multiproc(job):
    args, kwargs = job
    return gen_bessel2freq(*args, **kwargs)


def gen_bessel2freq(l, freqs, cosmo, kmax, method='default', kbin_file=None,
                    dk_factor=1e-1, decimate=False, device=None,
                    Nproc=None, Ntask=10, renorm=False):
    """
    Generate spherical Bessel forward model matrices sqrt(2/pi) r^2 k g_l(kr)
    from Fourier domain (k) to LOS distance or frequency domain (r_nu)

    The inverse transformation from Fourier space (k)
    to configuration space (r) is

    .. math::

        T_{lm}(r) &= \sqrt{\frac{2}{\pi}} \int dk k g_l(k r) T_{lm}(k) \\
        T(r,\theta,\phi) &= \sqrt{\frac{2}{\pi}} \int dk k g_l(k r)
                            T_l(k,\theta,\phi)

    Parameters
    ----------
    l : array_like
        Spherical harmonic l modes for g_l(kr) terms
    freqs : array_like
        Frequency array [Hz]
    cosmo : Cosmology object
        For freq -> r [comoving Mpc] conversion
    kmax : float
        Maximum k-mode to compute [Mpc^-1]
    method : str, optional
        Method for constraining radial basis functions.
        options=['default', 'samushia', 'gebhardt']
        See sph_bessel_kln for details.
    dk_factor : float, optional
        The delta-k spacing in the k_array used for sampling
        for the roots of the boundary condition is given as
        dk = k_min * dk_factor where k_min = 2pi / (rmax-rmin)
        A smaller dk_factor leads to higher resolution in k
        when solving for roots, but is slower to compute.
    decimate : bool, optional
        Use every other g_l(x) zero as k bins (i.e. DFT convention)
    device : str, optional
        Device to push g_l(kr) to.
    Nproc : int, optional
        If not None, enable multiprocessing mode with Nproc processes
    Ntask : int, optional
        Number of modes to compute per process
    renorm : bool, optional
        If True, renormalize the g_l modes
        such that inner product of r^1 g_l(k_n r) with
        itself equals pi/2 k^-2

    Returns
    -------
    gln : dict
        A dictionary holding a series of Nk x Nfreqs
        spherical Fourier Bessel transform matrices,
        one for each unique l mode.
        Keys are l mode integers, values are matrices.
    kln : dict
        A dictionary holding a series of k modes [Mpc^-1]
        for each l mode. same keys as gln
    """
    # convert frequency to LOS distance
    r = cosmo.f2r(freqs)
    r_max, r_min = r.max(), r.min()
    ul = np.unique(l)

    # multiproc mode
    if Nproc is not None:
        assert kbin_file is None, "no multiproc necessary if passing kbin_file"
        import multiprocessing
        Njobs = len(ul) / Ntask
        if Njobs % 1 > 0:
            Njobs = np.floor(Njobs) + 1
        Njobs = int(Njobs)
        jobs = []
        for i in range(Njobs):
            _l = ul[i*Ntask:(i+1)*Ntask]
            jobs.append([(_l, freqs, cosmo, kmax), dict(method=method, dk_factor=dk_factor,
                                                        decimate=decimate,
                                                        device=device, renorm=renorm)])

        # run jobs
        try:
            pool = multiprocessing.Pool(Nproc)
            output = pool.map(_gen_bessel2freq_multiproc, jobs)
        finally:
            pool.close()
            pool.join()

        # collect output
        jl, kbins = {}, {}
        for out in output:
            jl.update(out[0])
            kbins.update(out[1])

        return jl, kbins

    # run single proc mode
    jl = {}
    kbins = {}
    for _l in ul:
        # get k bins for this l mode
        k = sph_bessel_kln(_l, r_min, r_max, kmax, dk_factor=dk_factor, decimate=decimate,
                           method=method, filepath=kbin_file)
        # add monopole term if l = 0
        if _l == 0:
            k = np.concatenate([[0], k[:-1]])
        # get basis function g_l
        j = sph_bessel_func(_l, k, r, method=method, renorm=renorm, device=device)
        # form transform matrix: sqrt(2/pi) k g_l
        rt = torch.as_tensor(r, device=device, dtype=_float())
        kt = torch.as_tensor(k, device=device, dtype=_float())
        jl[_l] = np.sqrt(2 / np.pi) * rt**2 * kt[:, None].clip(1e-3) * j
        kbins[_l] = k

    return jl, kbins


def sph_bessel_func(l, k, r, method='default', r_min=None, r_max=None,
                    renorm=False, device=None):
    """
    Generate a spherical bessel radial basis function, g_l(k_n r)

    Parameters
    ----------
    l : int or float
        Integer angular l mode
    k : array_like
        k modes [cMpc^-1]
    r : array_like
        radial points to sample [cMpc]
    method : str, optional
        Method for basis functions.
        default : interval is 0 -> r_max, basis is
            j_l(kr), BC is j_l(k_ln r_max) = 0
        samushia : interval is r_min -> r_max, basis is
            g_ln = j_l(k_ln r) + A_ln y_l(k_ln r) and BC
            is g_ln(k r) = 0 for r_min and r_max (Samushia2019)
        gebhardt : interval is r_min -> r_max, basis is
            g_ln = j_l(k_ln r) + A_ln y_l(k_ln r)
            BC is potential field continuity (Gebhardt+2021)
            Not yet implemented.
    r_min, r_max : float, optional
        r_min and r_max of LIM survey. If None, will use
        min and max of r.
    renorm : bool, optional
        If True, renormalize amplitude of basis function
        such that inner product of r^1 g_l(k_n r) with
        itself equals pi/2 k^-2
    device : str, optional
        Device to place matrices on

    Returns
    -------
    array_like
        basis functions of shape (Nk, Nr)
    """
    # configure 
    Nk = len(k)
    if r_min is None:
        r_min = r.min()
    if r_max is None:
        r_max = r.max()

    j = torch.zeros(Nk, len(r), dtype=_float(), device=device)
    # loop over kbins and fill j matrix
    for i, _k in enumerate(k):
        if method == 'default':
            # just j_l(kr)
            j_i = special.jl(l, _k * r)

        elif method == 'samushia':
            # j_l(kr) + A y_l(kr)
            j_i = special.jl(l, _k * r)
            if _k > 0:
                A = -special.jl(l, _k * r_min) / special.yl(l, _k * r_min).clip(-1e50, np.inf)
                y_i = special.yl(l, _k * r).clip(-1e50, np.inf)
                j_i += A * y_i

        elif method == 'gebhardt':
            raise NotImplementedError

        j[i] = torch.as_tensor(j_i, dtype=_float(), device=device)

    # renormalize
    if renorm:
        rt = torch.as_tensor(r, device=device, dtype=_float())
        j *= torch.sqrt(np.pi/2 * k.clip(1e-3)**-2 / torch.sum(rt**2 * torch.abs(j)**2, axis=1))[:, None]

    return j


def sph_bessel_kln(l, r_min, r_max, kmax, dk_factor=5e-3, decimate=False,
                   method='default', filepath=None):
    """
    Get spherical bessel Fourier bins given method.

    Parameters
    ----------
    l : float
        Angular l mode
    r_min : float
        Survey starting boundary [cMpc]
    r_max : float
        Maximum survey radial extent [cMpc]
    kmax : float
        Maximum wavevector k [Mpc^-1] to compute
    dk_factor : float, optional
        The delta-k spacing in the k_array used for sampling
        for the roots of the boundary condition is given as
        dk = k_min * dk_factor where k_min = 2pi / (rmax-rmin)
        A smaller dk_factor leads to higher resolution in k
        when solving for roots, but is slower to compute.
    decimate : bool, optional
        If True, use every other zero
        starting at the second zero. This
        is consistent with Fourier k convention.
    method : str, optional
        Method for basis functions and for k_ln spectrum.
        default : interval is 0 -> r_max, basis is
            j_l(kr), BC is j_l(k_ln r_max) = 0
        samushia : interval is r_min -> r_max, basis is
            g_nl = j_l(k_ln r) + A_ln y_l(k_ln r) and BC
            is g_nl(k r) = 0 for r_min and r_max (Samushia2019)
        gebhardt : interval is r_min -> r_max, basis is
            g_nl = j_l(k_ln r) + A_ln y_l(k_ln r)
            BC is potential field continuity (Gebhardt+2021)
    filepath : str, optional
        filepath to csv of kbins [cMpc^-1] in form of
        l, 1st zero, 2nd zero, 3rd zero, ...
        This supercedes method if passed.

    Returns
    -------
    array
        Fourier modes k_n = [2pi / r_n]
    """
    # get pre-computed k bins
    if filepath is not None:
        k = np.loadtxt(filepath, delimiter=',')[l, 1:]
    else:
        # setup k_array of k samples to find roots
        kmin = 0.9 * (2 * np.pi / (r_max - r_min))  # give a 10% buffer to kmin
        dk = kmin * dk_factor
        k_arr = np.linspace(kmin, kmax, int((kmax-kmin)//dk)+1)

        if method == 'default':
            # BC is g_l(k_n r) = 0 at r_max
            y = special.jl(l, k_arr * r_max)

        elif method == 'samushia':
            # BC is g_l(k_n r) = 0 at r_min and r_max
            y = (special.jl(l, k_arr * r_min) * special.yl(l, k_arr * r_max).clip(-1e50, np.inf) \
                 - special.jl(l, k_arr * r_max) * special.yl(l, k_arr * r_min).clip(-1e50, np.inf))

        elif method == 'gebhardt':
            raise NotImplementedError

        # get roots
        k = get_zeros(k_arr, y)

    # decimate if desired
    if decimate:
        k = k[::2]

    return np.asarray(k)


def gen_poly_A(x, Ndeg, device=None, basis='direct', whiten=True):
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
    whiten : bool, optional
        If True, center (i.e. subtract mean) and scale (i.e. range of -1, 1) x.
        Useful when using orthogonal polynomial bases

    Returns
    -------
    torch tensor
        Polynomial design matrix (Nx, Ndeg)
    """
    # LEGACY
    #A = torch.as_tensor(torch.vstack([dfreqs**i for i in range(Ndeg)]),
    #                    dtype=_float(), device=device).T
    if whiten:
        x = x - x.mean()
        x = x / x.max()
    from emupy.linear import setup_polynomial
    A = setup_polynomial(x[:, None], Ndeg - 1, basis=basis)[0]
    return torch.as_tensor(A, dtype=_float(), device=device)


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


class PixInterp:
    """
    Sky pixel spatial interpolation object
    """
    def __init__(self, pixtype, nside=None, interp_mode='nearest',
                 theta=None, phi=None, Nnn=4, device=None,
                 leaf_size=40):
        """
        Interpolation is a weighted average of nearest neighbors.
        If pixtype is 'healpix', this is bilinear interpolation.
        If pixtype is 'other' use BallTree for nearest neighbor
        queries and use inverse distance as weighted mean.

        Parameters
        ----------
        pixtype : str
            Pixelization type. options are ['healpix', 'other']
        nside : int, optional
            nside of healpix map if pixtype == 'healpix'
        interp_mode : str, optional
            Spatial interpolation method, one of ['nearest'].
            Currently only a weighted sum of Nnn nearest
            neighbors is supported.
            interpolation on a rectangular grid.
        theta, phi : array_like
            Co-latitude and azimuth arrays [deg] of
            input params if pixtype is 'other'
        Nnn : int, optional
            Number of nearest neighbors to use
            for interp_mode of 'nearest'.
            Default is 4.
            If pixtype is healpix, Nnn must be 4.
        device : str, optional
            Device to place object on
        leaf_size : int, optional
            leaf size for BallTree
        """
        assert pixtype in ['healpix', 'other']
        if pixtype == 'other':
            assert theta is not None
            assert phi is not None
            assert import_sklearn
            X = np.array([np.pi / 2 - theta * D2R, phi * D2R]).T
            self.tree = BallTree(X, leaf_size=leaf_size, metric='haversine')
        else:
            self.tree = None
        self.pixtype = pixtype
        self.nside = nside
        self.interp_cache = {}
        self.interp_mode = interp_mode
        self.Nnn = Nnn
        self.theta, self.phi = theta, phi
        self.device = device

    def _clear_cache(self):
        """Clears interpolation cache"""
        self.interp_cache = {}

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
        h = ang_hash(zen), ang_hash(az)
        if h in self.interp_cache:
            # get interpolation if present
            interp = self.interp_cache[h]
        else:
            # otherwise generate it
            if self.pixtype == 'healpix':
                # get indices and weights for healpix interpolation
                assert self.Nnn == 4
                inds, wgts = healpy.get_interp_weights(self.nside,
                                                       tensor2numpy(zen) * D2R,
                                                       tensor2numpy(az) * D2R)

            elif self.pixtype == 'other':
                # get nearest neighbors and their interpolation weights
                zen, az = tensor2numpy(zen), tensor2numpy(az)
                dist, inds = self.tree.query(np.array([(90 - zen) * D2R, az * D2R]).T,
                                             k=self.Nnn, return_distance=True, sort_results=True)
                dist, inds = dist.T, inds.T
                # weight is inverse of distance from neighbor (not quite bilinear interp)
                wgts = 1 / dist.clip(1e-10, np.inf)**2

                # normalize weights
                wgts /= wgts.sum(axis=0, keepdims=True)

            # store it
            interp = (torch.as_tensor(inds, device=self.device),
                      torch.as_tensor(wgts, dtype=_float(), device=self.device))
            self.interp_cache[h] = interp

        return interp

    def interp(self, m, zen, az):
        """
        Interpolate a map m at zen, az points

        Parameters
        ----------
        m : array_like or tensor
            Map to interpolate. If Healpix map must be ring ordered.
        zen, az : array_like or tensor
            Zenith angle (co-latitude) and azimuth [deg]
            points at which to interpolate map
        """
        # get interpolation indices and weights
        inds, wgts = self.get_interp(zen, az)

        if self.interp_mode == 'nearest':
            # get nearest neighbors
            nearest = m[..., inds.T]

            # multiply by weights and sum
            out = torch.sum(nearest * wgts.T, axis=-1)

        else:
            raise ValueError("didn't recognize interp_mode {}".format(self.interp_mode))

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
        self.device = device
        for k in self.interp_cache:
            cache = self.interp_cache[k]
            self.interp_cache[k] = (cache[0], cache[1].to(device))


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

    def forward(self, inp=None, prior_cache=None, **kwargs):
        """
        The forward operator. Should have a kwarg for
        starting input to the model (inp) and a cache
        dictionary for holding the output of eval_prior

        Parameters
        ----------
        inp : object, optional
            Starting input for model
        prior_cache : dict, optional
            Cache for storing computed prior
            from self.cache_prior
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
        param = self[name].detach()
        del self[name]
        self[name] = param

    def set_param(self, name):
        """
        Set tensor "name" as a Parameter
        """
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
        priors_inp_params : optim.Log*Prior object or list
            Takes params as callable and returns
            scalar log prior. Can feed list of priors as well.
        priors_out_params : optim.Log*Prior object or list
            Takes the tensor output after pushing
            params through its response function
            as callable and returns scalar log prior.
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
        Shallow wrapper around self._eval_prior. Default
        behavior is to pass self.params and self.R(self.params)
        to self._eval_prior if self.name not in prior_cache.
        Non-standard subclasses should overload this method
        for proper usage.

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
                out_params = self.R(self.params)
            self._eval_prior(prior_cache, inp_params, out_params)

    def _eval_prior(self, prior_cache, inp_params=None, out_params=None):
        """
        Evaluate prior and insert into prior_cache.
        Will use self.name (default) or __class__.__name__ as key.
        If the key already exists, the prior will not be computed or stored.

        Parameters
        ----------
        prior_cache : dict
            Dictionary to hold computed prior, assigned as self.name
        inp_params, out_params : tensor, optional
            self.params and self.R(self.params), respectively
        """
        # append to cache
        if self.name not in prior_cache:
            prior_value = torch.as_tensor(0.0)
            if (hasattr(self, 'priors_inp_params') and
                inp_params is not None and
                self.priors_inp_params is not None):
                for prior in self.priors_inp_params:
                    prior_value = prior_value + prior(inp_params).to('cpu')

            if (hasattr(self, 'priors_out_params') and
                out_params is not None and
                self.priors_out_params is not None):
                for prior in self.priors_out_params:
                    prior_value = prior_value + prior(out_params).to('cpu')

            prior_cache[self.name] = prior_value

class Sequential(Module):
    """
    A minimal mirror of torch.nn.Sequential with added features.
    Inherits from bayeslim.utils.Module

    Instantiation takes a parameter dictionary as
    input and updates model before evaluation. e.g.

    .. code-block:: python

        S = Sequential(OrderedDict(model1=model1, model2=model2))

    where evaluation order is S(params) -> model2( model1( params ) )

    Note that the keys of the parameter dictionary
    must conform to nn.Module.named_parameters() syntax.
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
                   no_grad=True):
    """
    Set value to model as model.name

    If name is a torch.nn.Parameter, cast
    value as Parameter before setting.

    Parameters
    ----------
    model : utils.Module object
    name : str
    value : tensor
    clobber_param : bool, optional
        If True and key from pdict is an existing
        Parameter on self, del the param then assign
        it from pdict (this removes Parameter object
        but keeps memory address from pdict).
        If False (default), insert value from pdict
        into existing Parameter object, changing
        its memory address.
    no_grad : bool, optional
        If True, enter a torch.no_grad() context,
        otherwise enter a nullcontext.
    """
    context = torch.no_grad if no_grad else nullcontext
    with context():
        if isinstance(name, str):
            name = name.split('.')
        if len(name) == 1:
            # assign value to model as name
            device = None
            parameter = False
            if clobber_param:
                # if overwriting name del existing attr
                if hasattr(model, name[0]):
                    delattr(model, name[0])
            elif hasattr(model, name[0]):
                # if not clobber and this attr exists
                # get device and parameter status from attr
                device = getattr(model, name[0]).device
                parameter = isinstance(getattr(model, name[0]), torch.nn.Parameter)
                value = value.to(device)
            if parameter and not value.requires_grad:
                value = torch.nn.Parameter(value)
            setattr(model, name[0], value)
        else:
            set_model_attr(get_model_attr(model, '.'.join(name[:-1])),
                           name[-1], value, clobber_param=clobber_param)


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


def ang_hash(zen):
    """
    Hash sky angle (e.g. zen) by using its
    first value, last value and length as a
    unique identifier of the array.
    Note that if zen is a tensor, the device and
    require_grad values will affect the hash!
    Also note, normally array hash is not allowed.

    Parameters
    ----------
    zen : ndarray or tensor
        sky angle [arb. units]

    Returns
    -------
    hash object
    """
    return hash((float(tensor2numpy(zen[0])),
                 float(tensor2numpy(zen[-1])),
                 len(zen)))


def push(tensor, device, parameter=False):
    """
    Push a tensor to a new device. If the tensor
    is a parameter, it instantiates the parameter
    class on device.

    Parameters
    ----------
    tensor : tensor
        A pytorch tensor, optionally a pytorch Parameter
    device : str
        The device to push it to
    parameter : bool, optional
        Make new tensor a parameter. This is done
        by default if input is a Parameter

    Returns
    -------
    tensor
        The tensor on device
    """
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
            y.append(i)
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
    def __getitem__(self, k):
        return 0


def smi(name, fname='nvidia_mem.txt'):
    os.system("nvidia-smi -f {}".format(fname))
    with open(fname) as f: lines = f.readlines()
    date = lines[0].strip()
    output = []
    for i, line in enumerate(lines):
        if name in line:
            gpu = line[4]
            mems = lines[i+1].split('|')[2].strip().split('/')
            alloc = "{:>6} MiB".format("{:,}".format(int(mems[0].strip()[:-3])))
            total = "{:>6} MiB".format("{:,}".format(int(mems[1].strip()[:-3])))
            mem = "{} / {}".format(alloc, total)
            output.append("| GPU {} | {} |".format(gpu, mem))
    print('\n'.join([date] + ['o' + '-'*33 + 'o'] + output + ['o' + '-'*33 + 'o']))


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

