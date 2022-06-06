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

def _compute_lm_multiproc(job):
    args, kwargs = job
    return compute_lm(*args, **kwargs)

def compute_lm(phi_max, mmax, theta_min, theta_max, lmax, dl=0.1,
               mmin=0, high_prec=True, add_mono=True,
               add_sectoral=True, bc_type=2, real_field=True,
               Nrefine_iter=3, refine_dl=1e-7,
               Nproc=None, Ntask=5):
    """
    Compute associated Legendre function degrees l on
    the spherical stripe or cap given boundary conditions.

    For theta_min == 0 and theta_max < pi, we assume
    a spherical cap mask. For theta_min > 0 and
    theta_max < pi, we assume a spherical stripe mask.

    Boundary conditions on the polar axis (i.e. theta)
    are set by bc_type, setting the function (1)
    or its derivative (2) to zero as the polar boundary.
    The azimuthal boundary 

    bc_type == 1:
        dP_lm/dtheta(theta) = 0 when m == 0
        P_lm(theta) = 0 when m > 0
    bc_type == 2:
        dP_lm/dtheta(theta) = 0 for all m
    where theta is theta_max for cap, and is theta_min
    and theta_max for stripe.

    Azimuthal boundary condition is simply continuity,
    Phi(0) = Phi(phi_max).

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
        Minimum |m| to compute, default is 0.
    high_prec : bool, optional
        If True, use precise mpmath for hypergeometric
        calls, else use faster but less accurate scipy.
        Matters mostly for non-integer degree
    add_mono : bool, optional
        If True, include monopole mode (l = m = 0)
        if mmin = 0 even if it doesn't satisfy BCs.
    add_sectoral : bool, optional
        If True, include sectoral modes (l = m for l > 0)
        regardless of whether they satisfy BCs.
    bc_type : int, optional
        Boundary condition type on the polar axis (theta)
        for m > 0, either 1 or 2. 1 (Dirichlet) sets
        func. to zero at boundary and 2 (Neumann) sets
        its derivative to zero. Default = 2.
    real_field : bool, optional
        If True treat map as real valued so skip negative m modes.
    Nrefine_iter : int, optional
        Number of refinement interations (default = 2).
        Use finite difference to refine computation of
        degree l given boundary conditions.
    refine_dl : float, optional
        delta-l step size for refinement iterations.
    Nproc : int, optional
        If not None, launch multiprocessing of Nprocesses
    Ntask : int, optional
        Number of m-modes to solve for for each process

    Returns
    -------
    larr : ndarray
        Array of l values
    marr : ndarray
        Array of m values
    """
    # solve for m modes
    spacing = 2 * np.pi / phi_max
    assert np.isclose(spacing % 1, 0), "phi_max must evenly divide into 2pi"
    mmin = max([0, mmin])
    m = np.arange(mmin, mmax + 1.1, spacing)

    # run multiproc mode
    if Nproc is not None:
        # setup multiprocessing
        import multiprocessing
        Njobs = len(m) / Ntask
        if Njobs % 1 > 0:
            Njobs = np.floor(Njobs) + 1
        Njobs = int(Njobs)
        jobs = []
        for i in range(Njobs):
            marr = m[i*Ntask:(i+1)*Ntask]
            _mmin = marr.min()
            _mmax = marr.max() - 1
            jobs.append([(phi_max, _mmax, theta_min, theta_max, lmax),
                         dict(dl=dl, mmin=_mmin, high_prec=high_prec,
                              add_mono=add_mono, add_sectoral=add_sectoral,
                              bc_type=bc_type, Nproc=None,
                              Nrefine_iter=Nrefine_iter, refine_dl=refine_dl)
                         ])

        # run jobs
        try:
            pool = multiprocessing.Pool(Nproc)
            output = pool.map(_compute_lm_multiproc, jobs)
        finally:
            pool.close()
            pool.join()

        # combine
        larr, marr = [], []
        for out in output:
            larr.extend(out[0])
            marr.extend(out[1])
        larr = np.asarray(larr)
        marr = np.asarray(marr)

        return larr, marr


    # run single proc mode
    def get_y(larr, marr):
        if np.isclose(theta_min, 0):
            # spherical cap
            deriv = bc_type == 2 or _m == 0
            y = special.Plm(larr, marr, x_max, deriv=deriv, high_prec=high_prec, keepdims=True)

        else:
            # spherical stripe
            deriv = bc_type == 2
            y = special.Plm(larr, marr, x_min, deriv=deriv, high_prec=high_prec, keepdims=True, sq_norm=False) \
                * special.Qlm(larr, marr, x_max, deriv=deriv, high_prec=high_prec, keepdims=True, sq_norm=False) \
                - special.Plm(larr, marr, x_max, deriv=deriv, high_prec=high_prec, keepdims=True, sq_norm=False) \
                * special.Qlm(larr, marr, x_min, deriv=deriv, high_prec=high_prec, keepdims=True, sq_norm=False)

        return y

    assert bc_type in [1, 2]

    # solve for l modes
    assert theta_max < np.pi, "theta_max must be < pi for spherical cap or stripe"
    ls = {}
    x_min, x_max = np.cos(theta_min), np.cos(theta_max)
    m = np.atleast_1d(m)
    for _m in m:
        # construct array of test l's, skip l == m
        # larr goes out to lmax + dl (hence + 3)
        larr = _m + np.arange(1, (lmax - _m)//dl + 3) * dl
        marr = np.ones_like(larr) * _m
        if len(larr) < 1:
            continue

        # get y test values
        y = get_y(larr, marr).ravel()

        # look for zero crossings
        zeros = np.array(get_zeros(larr, y))

        # refinement steps
        if len(zeros) > 0:
            for i in range(Nrefine_iter):
                # x_new = x_1 - y_1 * delta-x / delta-y
                dx = refine_dl
                _marr = np.ones_like(zeros) * _m
                y1 = get_y(zeros, _marr).ravel()
                y2 = get_y(zeros + dx, _marr).ravel()
                dy = y2 - y1
                zeros = zeros - y1 * dx / dy

        # add sectoral
        if (_m == 0 and add_mono) or (_m > 0 and add_sectoral):
            zeros = [_m] + zeros.tolist()

        # append to l dictionary
        if len(zeros) > 0:
            ls[_m] = np.asarray(zeros)

    # unpack into arrays
    larr, marr = [], []
    for _m in ls:
        larr.extend(ls[_m])
        marr.extend([_m] * len(ls[_m]))

    larr, marr = np.array(larr), np.array(marr)

    return larr, marr


def _gen_sph2pix_multiproc(job):
    (l, m), args, kwargs = job
    Y, norm, alm_mult = gen_sph2pix(*args, **kwargs)
    Ydict = {(_l, _m): Y[i] for i, (_l, _m) in enumerate(zip(l, m))}
    return Ydict, norm, alm_mult


def gen_sph2pix(theta, phi, l, m, method='sphere', theta_max=None,
                Nproc=None, Ntask=10, device=None, high_prec=True,
                bc_type=2, real=False, m_phasor=False,
                renorm=False, **norm_kwargs):
    """
    Generate spherical harmonic forward model matrix.

    Note, this can take a _long_ time: dozens of minutes for a few hundred
    lm at tens of thousands of theta, phi points even with Nproc of 20.
    The code is limited by the internal high precision computation of the
    Legendre functions via mpmath, which enables stable evaluation of
    large, non-integer degree harmonics.
    It is advised to compute these once and store them.
    Also, pip installing the "gmpy" module can offer modest speed up.
    For computing integer degree spherical harmonics, high_prec is
    likely not needed.

    The orthonormalized cut-sky spherical harmonics are

    .. math::

        Y_{lm}(\theta,\phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}
                                e^{im\phi}(P_{lm} + A_{lm}Q_{lm})(\cos(\theta))

    Renormalization (optional) can be done to ensure the inner product 
    sums to one.

    Parameters
    ----------
    theta : array_like
        Co-latitude (i.e. zenith angle) [rad]
    phi : array_like
        Longitude (i.e. azimuth) [rad]
    l : array_like
        Integer or float array of spherical harmonic l modes
    m : array_like
        Integer array of spherical harmonic m modes
    method : str, optional
        Spherical harmonic mode ['sphere', 'stripe', 'cap']
        For 'sphere', l modes are integer
        For 'stripe' or 'cap', l modes are float
    theta_max : float, optional
        For method == 'stripe' or 'cap', this is the maximum theta
        boundary [radians] of the mask.
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
    bc_type : int, optional
        Boundary condition type on x for m > 0, either 1 or 2.
        1 (Dirichlet) sets func. to zero at boundary and
        2 (Neumann) sets its derivative to zero. Default = 2.
        Only needed for stripe method.
    real : bool, optional
        If True return real-valued Ylm (and update alm_mult accordingly)
        otherwise return complex Ylm (default).
    m_phasor : bool, optional
        If False, do nothing (default). If True, multiply
        all modes by exp(1j * phi) and update
        output alm_mult[m==0] to accomodate. This effectively
        boosts all m modes by 1. This is used when
        modeling polarized antenna Jones terms, which
        have a split symmetry that cannot be modeled
        without this additional sinusoidal component.
    renorm : bool, optional
        Re-normalize the spherical harmonics such that their
        inner product is 1 over their domain. This is done using
        the sampled theta, phi points as a numerical inner product
        approximation, see normalize_Ylm() for details.
    norm_kwargs : dict, optional
        Kwargs for renormalization see normalize_Ylm() for details.

    Returns
    -------
    Ylm : tensor
        An (Ncoeff, Npix) tensor encoding a spherical
        harmonic transform from a_lm -> map
    norm : tensor
        Normalization (Ncoeff,) divisor for each Ylm mode
    alm_mult : tensor
        A (Ncoeff) len tensor holding multiplicative factor
        for Ylm when taking forward transform. Needed when
        map is real-valued and we've truncated negative m modes.

    Notes
    -----
    The output dtype can be set using torch.set_default_dtype
    """
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
            args = (theta, phi, _l, _m)
            kwgs = dict(method=method, theta_max=theta_max,
                        high_prec=high_prec, m_phasor=m_phasor,
                        renorm=renorm, bc_type=bc_type, real=real)
            kwgs.update(norm_kwargs)
            jobs.append([(_l, _m), args, kwgs])

        # run jobs
        try:
            pool = multiprocessing.Pool(Nproc)
            output = pool.map(_gen_sph2pix_multiproc, jobs)
        finally:
            pool.close()
            pool.join()

        # combine
        Y = torch.zeros((len(l), len(theta)), dtype=_cfloat(), device=device)
        norm, alm_mult = [], []
        for (Ydict, nm, am) in output:
            for k in Ydict:
                _l, _m = k
                index = np.where((l == _l) & (m == _m))[0][0]
                Y[index] = Ydict[k].to(device)
            alm_mult.extend(am.numpy())
            norm.extend(nm.numpy())
        alm_mult = torch.as_tensor(alm_mult, dtype=_float())
        norm = torch.as_tensor(norm, dtype=_float())

        return Y, norm, alm_mult

    # run single proc mode
    if isinstance(l, (int, float)):
        l = np.array([l])
    if isinstance(m, (int, float)):
        m = np.array([m])
    l = l[:, None]
    m = m[:, None]
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    # get unique theta values and their indices in theta
    unq_theta, unq_idx = np.unique(theta, return_inverse=True)

    # compute assoc. legendre: note orthonorm is already in Plm and Qlm
    x = np.cos(unq_theta)
    if method == 'sphere':
        if theta_max is not None:
            assert np.isclose(theta_max, np.pi)
        else:
            theta_max = np.pi
    x_max = np.cos(theta_max)
    H_unq = legendre_func(x, l, m, method, x_max=x_max, high_prec=high_prec, bc_type=bc_type)

    # now broadcast across redundant theta values
    H = H_unq[:, unq_idx]

    # compute azimuthal fourier term
    Phi = np.exp(1j * m * phi)

    # combine into spherical harmonic
    Y = torch.as_tensor(H * Phi, dtype=_cfloat(), device=device)

    # apply additional m phasor
    if m_phasor:
        Y *= np.exp(1j * phi)

    # transform to real if needed
    if real:
        Y = Y.real

    if renorm:
        norm_kwargs['theta'] = theta
        Y, norm = normalize_Ylm(Y, **norm_kwargs)
    else:
        norm = torch.ones(len(Y))

    # get alm mult
    alm_mult = torch.ones(len(Y), dtype=_float())
    if not np.any(m < 0) and not real:
        alm_mult[m.ravel() > 0] *= 2
    if m_phasor and not real:
        # update m == 0 modes
        alm_mult[np.isclose(m.ravel(), 0)] *= 2

    return Y, norm, alm_mult


def normalize_Ylm(Ylm, norm=None, theta=None, dtheta=None, dphi=None,
                  hpix=True, pxarea=None, renorm_idx=None):
    """
    Normalize spherical harmonic Ylm tensor by diagonal of its
    inner product, or by a custom normalization.

    Parameters
    ----------
    Ylm : tensor
        Forward model tensor of shape (Ncoeff, Npix)
        encoding transfrom from a_lm -> map
    norm : tensor, optional
        (Ncoeff,) shaped tensor. Divide Ylm by norm along
        its 0th axis. Supercedes all other kwargs.
    theta : tensor, optional
        Co-latitude (i.e. zenith) points of Ylm [rad]. Needed
        for computing pixel areas of non-healpix grids. If not provided,
        assume pixel areas are 1.
    dtheta, dphi : tensor, optional
        Delta theta and delta phi differential sizes [rad] of each
        theta, phi sample. Needed for computing areas of non-healpix grids.
        If not provided, assume pixel areas are 1.
    hpix : bool, optional
        If True, Ylm is an equal-area healpix sampling
    pxarea : float, optional
        If hpix, this is the pixel area multiply inner product
        when computing normalization.
    renorm_idx : array, optional
        An indexing array along Ylm's Npix axis
        picking out Ylm samples to use in computing
        its inner product.

    Returns
    -------
    Ylm : tensor
        Normalized Ylm
    norm : tensor
        Computed normalization (divisor of input Ylm)
    """
    if norm is None:
        if renorm_idx is None:
            renorm_idx = slice(None)
        if pxarea is None:
            pxarea = 1.0
        if hpix:
            pxarea = torch.as_tensor([pxarea], device=Ylm.device)[None, :]
        else:
            if theta is None or dtheta is None or dphi is None:
                pxarea = torch.as_tensor([1.0], device=Ylm.device)[None, :]
            else:
                pxarea = torch.as_tensor(np.sin(theta) * dtheta * dphi, device=Ylm.device)[None, :]
        norm = torch.sqrt(torch.sum((torch.abs(Ylm)**2 * pxarea)[:, renorm_idx], axis=1))

    return Ylm / norm[:, None], norm


def legendre_func(x, l, m, method, x_crit=None, high_prec=True, bc_type=2, deriv=False):
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
    x_crit : float, optional
        If method is stripe, this is the x value for theta_max or theta_max,
        whichever yields more stable results (generally this is whichever
        is further from pi/2). Note that tests have shown that setting
        the stripe center > pi/2 tends to be more stable than < pi/2.
    high_prec : bool, optional
        If True, use arbitrary precision for Plm and Qlm
        otherwise use standard (faster) scipy method
    bc_type : int, optional
        Boundary condition type on x for m > 0, either 1 or 2.
        1 (Dirichlet) sets func. to zero at boundary and
        2 (Neumann) sets its derivative to zero. Default = 2.
        Only needed for stripe method.
    deriv : bool, optional
        If True return derivative, else return function (default)

    Returns
    -------
    H : array_like
        Legendre basis: P + A * Q
    """
    # compute assoc. legendre: orthonorm is already in Plm and Qlm
    P = special.Plm(l, m, x, high_prec=high_prec, keepdims=True, deriv=deriv, sq_norm=method!='stripe')
    if method == 'stripe':
        # spherical stripe: uses Plm and Qlm
        assert x_crit is not None
        # compute Qlms
        Q = special.Qlm(l, m, x, high_prec=high_prec, keepdims=True, deriv=deriv, sq_norm=False)
        # compute A coefficients
        A = -special.Plm(l, m, x_crit, high_prec=high_prec, keepdims=True, deriv=bc_type == 2, sq_norm=False) \
            / special.Qlm(l, m, x_crit, high_prec=high_prec, keepdims=True, deriv=bc_type == 2, sq_norm=False)

        # construct legendre func without sq_norm
        H = P + A * Q

        # set pixels close to zero to zero
        H2 = np.abs(P) + np.abs(A * Q)
        zero = np.abs(H / H2) < 1e-10  # double precision roundoff error (conservative)
        H[zero] = 0.0

        # add (1-x^2)^(-m/2) term in b/c it was left out due to roundoff errors in P + AQ
        if not isinstance(m, np.ndarray):
            m = np.atleast_1d(m)
        if m.ndim == 1:
            m = m[:, None]
        # multiply sq_norm back in
        H *= (1 - x**2)**(-np.atleast_1d(m)/2)

    else:
        H = P


    return H


def write_Ylm(fname, Ylm, angs, l, m, norm=None, alm_mult=None,
              theta_min=None, theta_max=None, phi_max=None,
              history='', overwrite=False):
    """
    Write a Ylm basis to HDF5 file

    Parameters
    ----------
    fname : str
        Filepath of output hdf5 file
    Ylm : array
        Ylm matrix of shape (Ncoeff, Npix)
    angs : array
        theta, phi sky positions of Ylm
        of shape (2, Npix), where theta is co-latitude
        and phi and azimuth [deg].
    l, m : array
        Ylm degree l and order m of len Ncoeff
    norm : array, optional
        Normalization (Ncoeff,) divisor for each Ylm mode
    alm_mult : array, optional
        alm coefficient multiplicative factor when
        taking forward transform of shape (Ncoeff,)
    theta_min : float, optional
        Minimum colatitude angle of Ylms [deg]
    theta_max : float, optional
        Maximum colatitude angle of Ylms [deg]
    phi_max : float, optional
        Maximum azimuthal angle of Ylms [deg]
    history : str, optional
        Notes about the Ylm modes
    overwrite : bool, optional
        Overwrite if file exists
    """
    if not os.path.exists(fname) or overwrite:
        with h5py.File(fname, 'w') as f:
            f.create_dataset('Ylm', data=Ylm)
            f.create_dataset('angs', data=np.array(angs))
            f.create_dataset('l', data=l)
            f.create_dataset('m', data=m)
            if norm is not None:
                f.create_dataset('norm', data=norm)
            if alm_mult is not None:
                f.create_dataset('alm_mult', data=alm_mult)
            if theta_min is not None:
                f.attrs['theta_min'] = theta_min
            if theta_max is not None:
                f.attrs['theta_max'] = theta_max
            if phi_max is not None:
                f.attrs['phi_max'] = phi_max
            f.attrs['history'] = history


def load_Ylm(fname, lmin=None, lmax=None, discard=None, cast=None,
             colat_min=None, colat_max=None, az_min=None, az_max=None,
             device=None, discard_sectoral=False, discard_mono=False,
             read_data=True, to_real=False):
    """
    Load an hdf5 file with Ylm tensors and metadata

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
        truncate Ylm response for colat <= colat_min [deg]
        assuming angs[0] is colatitude (zenith)
    colat_max : float, optional
        truncate Ylm response for colat >= colat_max [deg]
        assuming angs[0] is colatitude (zenith)
    az_min : float, optional
        truncate Ylm response for azimuth >= az_min [deg]
        assuming angs[1] is azimuth
    az_max : float, optional
        truncate Ylm response for azimuth <= az_max [deg]
        assuming angs[1] is azimuth    
    device : str, optional
        Device to place Ylm
    discard_sectoral : bool, optional
        If True, discard all modes where m == l (except for l == 0)
    discard_mono : bool, optional
        If True, discard monopole (i.e. l == m == 0)
    read_data : bool, optional
        If True, read and return Ylm
        else return None for Ylm
    to_real : bool, optional
        If Ylm is complex, cast it to real and
        update alm_mult metadata

    Returns
    -------
    Ylm, angs, l, m, info
    """
    import h5py
    with h5py.File(fname, 'r') as f:
        # load angles and all modes
        angs = f['angs'][:]
        l, m = f['l'][:], f['m'][:]
        if 'norm' in f:
            norm = f['norm'][:]
        else:
            norm = None
        if 'alm_mult' in f:
            alm_mult = f['alm_mult'][:]
        else:
            alm_mult = None
        info = {}
        for p in ['history', 'theta_max', 'theta_min', 'phi_max']:
            info[p] = f.attrs[p] if p in f.attrs else None

        # truncate modes
        keep = np.ones_like(l, dtype=bool)
        if lmin is not None:
            keep = keep & (l >= lmin)
        if lmax is not None:
            keep = keep & (l <= lmax + 1e-5)
        if discard is not None:
            cut_l, cut_m = discard
            for i in range(len(cut_l)):
                keep = keep & ~(np.isclose(l, cut_l[i], atol=1e-6) & np.isclose(m, cut_m[i], atol=1e-6))
        if discard_sectoral:
            keep = keep & ~((l == m) & (l > 0))
        if discard_mono:
            keep = keep & ~((l == 0) & (m == 0))

        # refactor keep as slicing if possible
        keep = np.where(keep)[0]
        if len(keep) == len(l):
            keep = slice(None)
        elif len(set(np.diff(keep))) == 1:
            keep = slice(keep[0], keep[-1]+1, keep[1] - keep[0])

        l, m = l[keep], m[keep]
        if alm_mult is not None:
            alm_mult = alm_mult[keep]
        if norm is not None:
            norm = norm[keep]
        if read_data:
            Ylm = f['Ylm'][keep, :]
        else:
            Ylm = None

        info['alm_mult'] = torch.as_tensor(alm_mult)
        info['norm'] = torch.as_tensor(norm)

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
    if az_min is not None:
        colat, az = angs
        keep = az >= az_min
        colat = colat[keep]
        az = az[keep]
        angs = colat, az
        if read_data:
            Ylm = Ylm[:, keep]
    if az_max is not None:
        colat, az = angs
        keep = az <= az_max
        colat = colat[keep]
        az = az[keep]
        angs = colat, az
        if read_data:
            Ylm = Ylm[:, keep]

    if to_real and np.iscomplexobj(Ylm):
        Ylm = Ylm.real
        info['alm_mult'][:] = 1.0

    if read_data:
        Ylm = torch.tensor(Ylm, device=device)
        if cast is not None:
            Ylm = Ylm.to(cast)

    return Ylm, angs, l, m, info


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


def gen_bessel2freq(l, freqs, cosmo, kbins=None, Nproc=None, Ntask=10,
                    device=None, method='shell', bc_type=2,
                    renorm=True, r_crit=None, **kln_kwargs):
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
        Frequency array [Hz]. If not passing kbins, assume
        first and last channel in freqs define radial mask.
    cosmo : Cosmology object
        For freq -> r [comoving Mpc] conversion
    kbins : dict, optional
        Use pre-computed k_ln bins. Keys are
        l modes, values are arrays of k_ln modes in cMpc^-1.
        No multiproc needed if passing kbins
    Nproc : int, optional
        If not None, enable multiprocessing with Nproc processes
        when solving for k_ln and g_l(kr)
    Ntask : int, optional
        Number of modes to compute per process
    device : str, optional
        Device to push g_l(kr) to.
    method : str, optional
        Radial mask method, ['ball', 'shell' (default)]
    bc_type : int, optional
        Type of boundary condition, 1 (Dirichlet) sets
        function to zero at edges, 2 (Neumann, default) sets
        its derivative to zero at edges.
    renorm : bool, optional
        If True (default), renormalize the g_l modes
        such that inner product of r^1 g_l(k_n r) with
        itself equals pi/2 k^-2
    r_crit : float, optional
        Either r_min or r_max for method == 'shell'.
        Used for normalization of 1st and 2nd sph bessel funcs.
    kln_kwargs : dict
        If kbins is not provided, compute them here.
        These are the args and kwargs fed to
        sph_bessel_kln().

    Returns
    -------
    gln : dict
        A dictionary holding a series of Nk x Nfreqs
        spherical Fourier Bessel transform matrices,
        one for each unique l mode.
        Keys are l mode floats, values are matrices.
    kln : dict
        A dictionary holding a series of k modes [Mpc^-1]
        for each l mode. same keys as gln
    """
    # convert frequency to LOS distance
    r = cosmo.f2r(freqs)
    ul = np.unique(l)

    # multiproc mode
    if Nproc is not None:
        assert kbins is None, "no multiproc necessary if passing kbins"
        import multiprocessing
        Njobs = len(ul) / Ntask
        if Njobs % 1 > 0:
            Njobs = np.floor(Njobs) + 1
        Njobs = int(Njobs)
        jobs = []
        for i in range(Njobs):
            _l = ul[i*Ntask:(i+1)*Ntask]
            args = (_l, freqs, cosmo)
            kwargs = dict(device=device, method=method, r_crit=r_crit,
                          renorm=renorm, bc_type=bc_type)
            kwargs.update(kln_kwargs)
            jobs.append([args, kwargs])

        # run jobs
        try:
            pool = multiprocessing.Pool(Nproc)
            output = pool.map(_gen_bessel2freq_multiproc, jobs)
        finally:
            pool.close()
            pool.join()

        # collect output
        gln, kln = {}, {}
        for out in output:
            gln.update(out[0])
            kln.update(out[1])

        return gln, kln

    # run single proc mode
    gln = {}
    kln = {}
    if kbins is None:
        kwgs = copy.deepcopy(kln_kwargs)
        r_min = kwgs.pop('r_min')
        r_max = kwgs.pop('r_max')
    for _l in ul:
        # get k bins for this l mode
        if kbins is None:
            k = sph_bessel_kln(_l, r_min, r_max, bc_type=bc_type, **kwgs)
        else:
            k = kbins[_l]
        # get basis function g_l
        gl = sph_bessel_func(_l, k, r, method=method, bc_type=bc_type,
                             renorm=renorm, device=device, r_crit=r_crit)
        # form transform matrix: sqrt(2/pi) k g_l
        rt = torch.as_tensor(r, device=device, dtype=_float())
        kt = torch.as_tensor(k, device=device, dtype=_float())
        gln[_l] = np.sqrt(2 / np.pi) * rt**2 * kt[:, None].clip(1e-3) * gl
        kln[_l] = k

    return gln, kln


def sph_bessel_func(l, k, r, method='shell', bc_type=2, r_crit=None, renorm=False, device=None):
    """
    Generate a spherical bessel radial basis function, g_l(k_n r)

    Parameters
    ----------
    l : float
        angular l mode
    k : array_like
        k modes to evaluate [cMpc^-1]
    r : array_like
        radial points to sample [cMpc]
    method : str, optional
        Radial mask method, ['ball', 'shell' (default)]
    bc_type : int, optional
        Only needed for method = shell.
        Type of boundary condition, 1 (Dirichlet) sets
        function to zero at edges, 2 (Neumann, default) sets
        its derivative to zero at edges. Only used to 
        solve for the proportionality constant between
        j_l and y_l if method is shell.
    r_crit : float, optional
        One of the radial edges [cMpc] if method is shell
        (aka rmin or rmax).
    renorm : bool, optional
        If True, renormalize amplitude of basis function
        such that inner product of r g_l(k_n r) with
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
    if method == 'shell':
        assert r_crit is not None

    j = torch.zeros(Nk, len(r), dtype=_float(), device=device)
    # loop over kbins and fill j matrix
    for i, _k in enumerate(k):
        if method == 'ball':
            # just j_l(kr)
            j_i = special.jl(l, _k * r)

        elif method == 'shell':
            # j_l(kr) + A y_l(kr)
            j_i = special.jl(l, _k * r)
            deriv = bc_type == 2
            if _k > 0:
                A = -special.jl(l, _k * r_crit, deriv=deriv) / \
                     special.yl(l, _k * r_crit, deriv=deriv).clip(-1e50, np.inf)
                y_i = special.yl(l, _k * r).clip(-1e50, np.inf)
                j_i += A * y_i

        else:
            raise ValueError("didn't recognize method {}".format(method))

        j[i] = torch.as_tensor(j_i, dtype=_float(), device=device)

    # renormalize
    if renorm:
        rt = torch.as_tensor(r, device=device, dtype=_float())
        j *= torch.sqrt(np.pi/2 * k.clip(1e-3)**-2 / torch.sum(rt**2 * torch.abs(j)**2, axis=1))[:, None]

    return j


def sph_bessel_kln(l, r_min, r_max, kmax=0.5, dk_factor=5e-3, decimate=False,
                   bc_type=2, add_kzero=False):
    """
    Get spherical bessel Fourier bins given r_min, r_max and
    boundary conditions. If r_min == 0, this is a ball.
    If r_min > 0, this is a shell.

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
    bc_type : int, optional
        Type of boundary condition, 1 (Dirichlet) sets
        function to zero at edges, 2 (Neumann, default) sets
        its derivative to zero at edges.
    add_kzero : bool, optional
        Add zeroth k mode.

    Returns
    -------
    array
        Fourier modes k_n = [2pi / r_n]
    """
    # setup k_array of k samples to find roots
    kmin = 0.9 * (2 * np.pi / (r_max - r_min))  # give a 10% buffer to kmin
    dk = kmin * dk_factor
    k_arr = np.linspace(kmin, kmax, int((kmax-kmin)//dk)+1)

    method = 'ball' if np.isclose(r_min, 0) else 'shell'
    deriv = bc_type == 2
    if method == 'ball':
        y = special.jl(l, k_arr * r_max, deriv=deriv)

    elif method == 'shell':
        y = (special.jl(l, k_arr * r_min, deriv=deriv) * \
             special.yl(l, k_arr * r_max, deriv=deriv).clip(-1e50, np.inf) - \
             special.jl(l, k_arr * r_max, deriv=deriv) * \
             special.yl(l, k_arr * r_min, deriv=deriv).clip(-1e50, np.inf))

    # get roots
    k = get_zeros(k_arr, y)

    # decimate if desired
    if decimate:
        k = k[::2]

    # add zeroth k mode
    if add_kzero:
        k = np.concatenate([[0.0], k])

    return np.asarray(k)


def gen_linear_A(linear_mode, A=None, x=None, whiten=True, x0=None, dx=None,
                 Ndeg=None, basis='direct', device=None, dtype=None, **kwargs):
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
        A = gen_poly_A(x, Ndeg, basis=basis, whiten=whiten, x0=x0, dx=dx)
    elif linear_mode == 'custom':
        A = torch.as_tensor(A)

    return A.to(dtype).to(device)


def gen_poly_A(x, Ndeg, device=None, basis='direct', whiten=True,
               x0=None, dx=None):
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
    x0 : float, optional
        If whiten, use this centering instead of x.mean()
    dx : float, optional
        If whiten, use this scaling instead of (x-x0).max()

    Returns
    -------
    A : tensor
        Polynomial design matrix (Nx, Ndeg)
    """
    # whiten the input if desired
    if whiten:
        x, x0, dx = whiten_xarr(x, x0, dx)

    # setup the polynomial
    from emupy.linear import setup_polynomial
    A = setup_polynomial(x[:, None], Ndeg - 1, basis=basis)[0]
    A = torch.as_tensor(A, dtype=_float(), device=device)

    return A


def whiten_xarr(x, x0=None, dx=None):
    """
    Whiten a monotonically increasing
    vector x to have a range of [-1, 1]
    for optimal polynomial orthogonality.
    For uniformly increasing x, the whitened
    range is [-1+dx/2, 1-dx/2].

    Parameters
    ----------
    x : tensor
        Monotonically increasing, but no
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
        dx = x.max() + (x[1] - x[0]) / 2
    x = x / dx

    return x, x0, dx


class LinearModel:
    """
    A linear model of

        y = Ax
    """
    def __init__(self, linear_mode, dim=0, **kwargs):
        """
        Parameters
        ----------
        linear_model : str
            The kind of model to generate: ['custom', 'poly'].
            See utils.gen_linear_A.
        dim : int, optional
            The dimension of the input params tensor to sum over.
        kwargs : dict
            keyword arguments for utils.gen_linear_A()
        """
        self.linear_mode = linear_mode
        self.dim = dim

        if self.linear_mode != 'custom':
            if kwargs.get('whiten', False):
                x = kwargs.get('x')
                _, x0, dx = whiten_xarr(x)
                if 'x0' not in kwargs:
                    kwargs['x0'] = x0
                if 'dx' not in kwargs:
                    kwargs['dx'] = dx

        self.kwargs = kwargs
        self.A = gen_linear_A(linear_mode, **kwargs)
        self.device = self.A.device

    def forward(self, params, A=None):
        """
        Forward pass parameter tensor through design matrix

        Parameters
        ----------
        params : tensor
        A : tensor, optional
            Use this (Nsamples, Nfeatures) design
            matrix instead of self.A

        Returns
        -------
        tensor
        """
        A = A if A is not None else self.A
        ndim = params.ndim
        if self.dim == 0:
            # trivial matmul
            return A @ params
        elif self.dim == ndim or self.dim == -1:
            # trivial transpose
            return params @ A.T
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
            return torch.einsum("{},{}->{}".format(t1, t2, out), A, params)

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
        return least_squares(self.A, y, **kwargs)

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
        self.A = self.A.to(device)
        self.device = device


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
                 theta_grid=None, phi_grid=None, device=None):
        """
        Interpolation is a weighted average of nearest neighbors.
        If pixtype is 'healpix', this is bilinear interpolation.
        If pixtype is 'rect' use interp_mode to set interpolation

        Parameters
        ----------
        pixtype : str
            Pixelization type. options are ['healpix', 'rect']
        nside : int, optional
            nside of healpix map if pixtype == 'healpix'
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
            should mark the pixel centers (i.e. theta_grid should
            generally not start at 0).
        device : str, optional
            Device to place object on
        """
        self.pixtype = pixtype
        self.nside = nside
        self.interp_cache = {}
        self.interp_mode = interp_mode
        self.theta_grid = theta_grid
        self.phi_grid = phi_grid
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
                xnew, ynew = tensor2numpy(az), tensor2numpy(zen)

                # get map indices
                inds, xyrel = bipoly_grid_index(xgrid, ygrid, xnew, ynew,
                                                degree[0]+1, degree[1]+1,
                                                wrapx=True, ravel=True)

                # get weights
                Ainv, Anew = setup_bipoly_interp(degree, dx, dy, xyrel[0], xyrel[1])
                wgts = Anew @ Ainv

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
        self.device = device
        for k in self.interp_cache:
            cache = self.interp_cache[k]
            self.interp_cache[k] = (cache[0].to(device),
                                    cache[1].to(device))


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


def bipoly_grid_index(xgrid, ygrid, xnew, ynew, Nx, Ny, wrapx=False, ravel=True):
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
    # get dx, dy
    dx, dy = np.median(np.diff(xgrid)), np.median(np.diff(ygrid))
    
    # wrap xgrid
    N = len(xgrid)
    if wrapx:
        xgrid = np.concatenate([xgrid[-Nx:]-N*dx, xgrid, xgrid[:Nx]+N*dx])

    # get xgrid and ygrid indices for each xynew
    xnn = np.sort(np.argsort(np.abs(xgrid - xnew[:, None]), axis=-1)[:, :Nx], -1)
    ynn = np.sort(np.argsort(np.abs(ygrid - ynew[:, None]), axis=-1)[:, :Ny], -1)

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
    AtAinvAt = np.linalg.pinv(A.T @ A) @ A.T
    
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
                    if prior is not None:
                        prior_value = prior_value + prior(inp_params).to('cpu')

            if (hasattr(self, 'priors_out_params') and
                out_params is not None and
                self.priors_out_params is not None):
                for prior in self.priors_out_params:
                    if prior is not None:
                        prior_value = prior_value + prior(out_params).to('cpu')

            prior_cache[self.name] = prior_value


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
    if isinstance(zen, torch.Tensor):
        zen = zen.numpy()
    return hash((zen[0], zen[-1], len(zen)))


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


def _list2slice(inds):
    """convert list indexing to slice if possible"""
    if isinstance(inds, list):
        diff = list(set(np.diff(inds)))
        if len(diff) == 1:
            if (inds[1] - inds[0]) > 0:
                # only return as slice if inds is increasing
                return slice(inds[0], inds[-1]+diff[0], diff[0])
    return inds

