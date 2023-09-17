"""
Utility module for all things spherical harmonics
and spherical fourier bessel
"""
import numpy as np
import torch
import copy
import os
import h5py

from . import special, version, utils, linalg


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
               Nproc=None, Ntask=5, use_pathos=False):
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
    use_pathos : bool, optional
        If multiprocessing (Nproc not None) and use_pathos = True,
        use the multiprocess module, otherwise use the more
        standard multiprocessing module (default).

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
    m = np.arange(mmin, mmax + .1, spacing)

    # run multiproc mode
    if Nproc is not None:
        # setup multiprocessing
        if use_pathos:
            import multiprocess as mproc
        else:
            import multiprocessing as mproc
        start_method = mproc.get_start_method(allow_none=True)
        if start_method is None: mproc.set_start_method('spawn') 
        Njobs = len(m) / Ntask
        if Njobs % 1 > 0:
            Njobs = np.floor(Njobs) + 1
        Njobs = int(Njobs)
        jobs = []
        for i in range(Njobs):
            marr = m[i*Ntask:(i+1)*Ntask]
            _mmin = marr.min()
            _mmax = marr.max()
            jobs.append([(phi_max, _mmax, theta_min, theta_max, lmax),
                         dict(dl=dl, mmin=_mmin, high_prec=high_prec,
                              add_mono=add_mono, add_sectoral=add_sectoral,
                              bc_type=bc_type, Nproc=None,
                              Nrefine_iter=Nrefine_iter, refine_dl=refine_dl)
                         ])

        # run jobs
        with mproc.Pool(Nproc) as pool:
            output = pool.map(_compute_lm_multiproc, jobs)

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
        zeros = np.array(utils.get_zeros(larr, y))

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
    if isinstance(Y, tuple):
        # this is separable
        Ydict = {(_l, _m): (Y[0][i], Y[1][i]) for i, (_l, _m) in enumerate(zip(l, m))}        
    else:
        Ydict = {(_l, _m): Y[i] for i, (_l, _m) in enumerate(zip(l, m))}
    return Ydict, norm, alm_mult


def gen_sph2pix(theta, phi, l, m, separable=False,
                method='sphere', theta_crit=None,
                Nproc=None, Ntask=10, device=None, high_prec=True,
                bc_type=2, real=False, m_phasor=False,
                renorm=False, use_pathos=False, **norm_kwargs):
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
    separable : bool, optional
        If True, separate polar and azimuthal transforms
        into Theta and Phi matrices. Note orthonorm is 
        put into Theta. Otherwise, compute one single Ylm
        transform matrix that performs both simultaneously.
    method : str, optional
        Spherical harmonic mode ['sphere', 'stripe', 'cap']
        For 'sphere', l modes are integer
        For 'stripe' or 'cap', l modes are float
    theta_crit : float, optional
        For method == 'stripe' or 'cap', this is a theta
        boundary [radians] of the mask.
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
    use_pathos : bool, optional
        If multiprocessing (Nproc not None) and use_pathos = True,
        use the multiprocess module, otherwise use the more
        standard multiprocessing module (default).

    Returns
    -------
    Ylm : tensor
        An (Ncoeff, Npix) tensor encoding a spherical
        harmonic transform from a_lm -> map.
        If separable, this is actually a tuple
        holding (Theta, Phi), each of which are of shape
        (Ncoeff, Npix)
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
        if use_pathos:
            import multiprocess as mproc
        else:
            import multiprocessing as mproc
        start_method = mproc.get_start_method(allow_none=True)
        if start_method is None: mproc.set_start_method('spawn') 
        Njobs = len(l) / Ntask
        if Njobs % 1 > 0:
            Njobs = np.floor(Njobs) + 1
        Njobs = int(Njobs)
        jobs = []
        for i in range(Njobs):
            _l = l[i*Ntask:(i+1)*Ntask]
            _m = m[i*Ntask:(i+1)*Ntask]
            args = (theta, phi, _l, _m)
            kwgs = dict(separable=separable,
                        method=method, theta_crit=theta_crit,
                        high_prec=high_prec, m_phasor=m_phasor,
                        renorm=renorm, bc_type=bc_type, real=real)
            kwgs.update(norm_kwargs)
            jobs.append([(_l, _m), args, kwgs])

        # run jobs
        with mproc.Pool(Nproc) as pool:
            output = pool.map(_gen_sph2pix_multiproc, jobs)

        # combine
        if separable:
            T = torch.zeros((len(l), len(theta)), dtype=utils._cfloat(), device=device)
            P = torch.zeros((len(l), len(phi)), dtype=utils._cfloat(), device=device)
            Y = (T, P)
        else:
            Y = torch.zeros((len(l), len(theta)), dtype=utils._cfloat(), device=device)

        norm, alm_mult = [], []
        for (Ydict, nm, am) in output:
            for k in Ydict:
                _l, _m = k
                index = np.where((l == _l) & (m == _m))[0][0]
                if separable:
                    Y[0][index] = Ydict[k][0].to(device)
                    Y[1][index] = Ydict[k][1].to(device)
                else:
                    Y[index] = Ydict[k].to(device)
            alm_mult.extend(am.numpy())
            norm.extend(nm.numpy())
        alm_mult = torch.as_tensor(alm_mult, dtype=utils._float())
        norm = torch.as_tensor(norm, dtype=utils._float())

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
        if theta_crit is None:
            theta_crit = np.pi
    assert theta_crit is not None

    x_crit = np.cos(theta_crit)
    H_unq = legendre_func(x, l, m, method, x_crit=x_crit, high_prec=high_prec, bc_type=bc_type)

    # now broadcast across redundant theta values
    H = H_unq[:, unq_idx].copy()

    # compute azimuthal fourier term
    Phi = np.exp(1j * m * phi)

    # apply additional m phasor
    if m_phasor:
        Phi *= np.exp(1j * phi)

    # transform to real if needed
    dtype = utils._float() if real else utils._cfloat()

    if separable:
        Y = (torch.as_tensor(H, dtype=dtype, device=device),
             torch.as_tensor(Phi, dtype=dtype, device=device))
    else:
        # combine into spherical harmonic
        Y = torch.as_tensor(H * Phi, dtype=dtype, device=device)

    if renorm:
        norm_kwargs['theta'] = theta
        Y, norm = normalize_Ylm(Y, **norm_kwargs)
    else:
        norm = torch.ones(len(l))

    # get alm mult
    alm_mult = torch.ones(len(l), dtype=utils._float())
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
    Ylm : tensor or tuple
        Forward model tensor of shape (Ncoeff, Npix)
        encoding transfrom from a_lm -> map.
        Can also be a tuple of two tensors (Theta, Phi)
        if generated with separable = True.
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
        if isinstance(Ylm, (list, tuple)):
            # this is separable
            T, P = Ylm
            # inflate
            Y = inflate_Ylm(Ylm)
            if theta is not None:
                theta = np.repeat(theta[:, None], P.shape[1], 1).ravel()
            if dtheta is not None and isinstance(dtheta, np.ndarray) and dtheta.size != theta.size:
                dtheta = np.repeat(dtheta[:, None], P.shape[1], 1).ravel()
        else:
            Y = Ylm
        if renorm_idx is None:
            renorm_idx = slice(None)
        if pxarea is None:
            pxarea = 1.0
        if hpix:
            pxarea = torch.as_tensor([pxarea], device=Y.device)[None, :]
        else:
            if theta is None or dtheta is None or dphi is None:
                pxarea = torch.as_tensor([1.0], device=Y.device)[None, :]
            else:
                pxarea = torch.as_tensor(np.sin(theta) * dtheta * dphi, device=Y.device)[None, :]
        norm = torch.sqrt(torch.sum((torch.abs(Y)**2 * pxarea)[:, renorm_idx], axis=1))

    if isinstance(Ylm, (list, tuple)):
        Ylm = (Ylm[0] / norm[:, None], Ylm[1])
    else:
        Ylm = Ylm / norm[:, None]

    return Ylm, norm


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
        If method is stripe, this is the x value for theta_min or theta_max,
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
        H *= (1 - x**2).clip(1e-40)**(-np.atleast_1d(m)/2)

    else:
        H = P

    return H


def write_Ylm(fname, Ylm, angs, l, m, norm=None, D=None, Dinv=None,
              alm_mult=None, theta_min=None, theta_max=None,
              phi_max=None, pxarea=None, history='', overwrite=False,
              idx=None, **kwargs):
    """
    Write a Ylm basis to HDF5 file

    Parameters
    ----------
    fname : str
        Filepath of output hdf5 file
    Ylm : array or tuple
        Ylm matrix of shape (Ncoeff, Npix)
        or a tuple of two matrices (Theta, Phi)
        of shape (Ncoeff, Ntheta) and (Ncoeff, Nphi)
    angs : tuple or array
        theta, phi sky positions of Ylm
        of shape (2, Npix), where theta is co-latitude
        and phi and azimuth [deg]. If theta and phi
        are of different lengths (e.g. for separable)
        then each are stored separately.
    l, m : array
        Ylm degree l and order m of len Ncoeff
    norm : array, optional
        Normalization (Ncoeff,) divisor for each Ylm mode
        when computing them (see normalize_Ylm)
    D : array, optional
        pre-computed least squares normalization matrix
        of shape (Ncoeff, Ncoeff), where D = (A.T Ninv A)^-1
    Dinv : array, optional
        This is the inner product of Ylm of shape (Ncoeff, Ncoeff),
        optionally weighted by Ninv.
    alm_mult : array, optional
        alm coefficient multiplicative factor when
        taking forward transform of shape (Ncoeff,)
    theta_min : float, optional
        Minimum colatitude angle of Ylms [deg]
    theta_max : float, optional
        Maximum colatitude angle of Ylms [deg]
    phi_max : float, optional
        Maximum azimuthal angle of Ylms [deg]
    pxarea : tensor, optional
        Pixel area of each pixel in the Ylm tensor.
        For healpix sampling use nside2pixarea().
        For uniform (theta, phi) grid use sin(theta) * d_theta * d_phi
    history : str, optional
        Notes about the Ylm modes
    overwrite : bool, optional
        Overwrite if file exists
    idx : tensor, optional
        Optional re-indexing tensor along Npix axis of Ylm
    kwargs : dict, optional
        Additional attributes to write as attrs, and
        added to the info dict upon read-in
        in as info
    """
    if not os.path.exists(fname) or overwrite:
        with h5py.File(fname, 'w') as f:
            # detect tuple
            if isinstance(Ylm, (tuple, list)):
                f.create_dataset('Theta', data=Ylm[0])
                f.create_dataset('Phi', data=Ylm[1])
            else:
                f.create_dataset('Ylm', data=Ylm)
            if isinstance(angs, (list, tuple)) and angs[0].shape != angs[1].shape:
                f.create_dataset('theta', data=angs[0])
                f.create_dataset('phi', data=angs[1])
            else:
                f.create_dataset('angs', data=np.asarray(angs))
            f.create_dataset('l', data=l)
            f.create_dataset('m', data=m)
            if D is not None:
                f.create_dataset('D', data=D)
            if Dinv is not None:
                f.create_dataset('Dinv', data=Dinv)
            if norm is not None:
                f.create_dataset('norm', data=norm)
            if alm_mult is not None:
                f.create_dataset('alm_mult', data=alm_mult)
            if pxarea is not None:
                f.create_dataset('pxarea', data=pxarea)
            if idx is not None:
                f.create_dataset('idx', data=idx)
            if theta_min is not None:
                f.attrs['theta_min'] = theta_min
            if theta_max is not None:
                f.attrs['theta_max'] = theta_max
            if phi_max is not None:
                f.attrs['phi_max'] = phi_max
            f.attrs['history'] = history
            for k in kwargs:
                f.attrs[k] = kwargs[k]


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
        if 'angs' in f:
            angs = f['angs'][()]
        elif 'theta' in f and 'phi' in f:
            theta = f['theta'][()]
            phi = f['phi'][()]
            angs = (theta, phi)
        l, m = f['l'][()], f['m'][()]
        if 'norm' in f:
            norm = f['norm'][()]
        else:
            norm = None
        if 'alm_mult' in f:
            alm_mult = f['alm_mult'][()]
        else:
            alm_mult = None
        info = {}
        for p in f.attrs:
            info[p] = f.attrs[p]
        for p in ['history', 'theta_max', 'theta_min', 'phi_max']:
            if p not in info:
                info[p] = None

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
            if 'Ylm' in f:
                Ylm = f['Ylm'][keep, :]
            elif 'Theta' in f and 'Phi' in f:
                Ylm = (f['Theta'][keep, :], f['Phi'][keep, :])
            if 'D' in f:
                D = f['D'][keep, :][:, keep]
            else:
                D = None
            if 'Dinv' in f:
                Dinv = f['Dinv'][keep, :][:, keep]
            else:
                Dinv = None
            if 'pxarea' in f:
                pxarea = f['pxarea'][()]
            else:
                pxarea = None
            if 'idx' in f:
                idx = f['idx'][()]
            else:
                idx = None
        else:
            Ylm = None
            D = None
            Dinv = None
            pxarea = None
            idx = None

        if alm_mult is not None:
            info['alm_mult'] = torch.as_tensor(alm_mult)
        if norm is not None:
            info['norm'] = torch.as_tensor(norm)

    # truncate sky
    if colat_min is not None:
        colat, az = angs
        keep = colat >= colat_min
        colat = colat[keep]
        az = az[keep]
        angs = colat, az
        if read_data:
            if isinstance(Ylm, tuple):
                Ylm = (Ylm[0][:, keep], Ylm[1][:, keep])
            else:
                Ylm = Ylm[:, keep]
            if pxarea is not None and not isinstance(pxarea, (int, float)):
                pxarea = pxarea[keep]
    if colat_max is not None:
        colat, az = angs
        keep = colat <= colat_max
        colat = colat[keep]
        az = az[keep]
        angs = colat, az
        if read_data:
            if isinstance(Ylm, tuple):
                Ylm = (Ylm[0][:, keep], Ylm[1][:, keep])
            else:
                Ylm = Ylm[:, keep]
            if pxarea is not None and not isinstance(pxarea, (int, float)):
                pxarea = pxarea[keep]
    if az_min is not None:
        colat, az = angs
        keep = az >= az_min
        colat = colat[keep]
        az = az[keep]
        angs = colat, az
        if read_data:
            if isinstance(Ylm, tuple):
                Ylm = (Ylm[0][:, keep], Ylm[1][:, keep])
            else:
                Ylm = Ylm[:, keep]
            if pxarea is not None and not isinstance(pxarea, (int, float)):
                pxarea = pxarea[keep]
    if az_max is not None:
        colat, az = angs
        keep = az <= az_max
        colat = colat[keep]
        az = az[keep]
        angs = colat, az
        if read_data:
            if isinstance(Ylm, tuple):
                Ylm = (Ylm[0][:, keep], Ylm[1][:, keep])
            else:
                Ylm = Ylm[:, keep]
            if pxarea is not None and not isinstance(pxarea, (int, float)):
                pxarea = pxarea[keep]

    if to_real:
        if np.iscomplexobj(Ylm) or (isinstance(Ylm, tuple) and np.iscomplexobj(Ylm[1])):
            if isinstance(Ylm, tuple):
                Ylm = (Ylm[0].real, Ylm[1].real)
            else:
                Ylm = Ylm.real
        if alm_mult is not None:
            info['alm_mult'][:] = 1.0

    if read_data:
        if isinstance(Ylm, tuple):
            Ylm = (torch.as_tensor(Ylm[0], device=device),
                   torch.as_tensor(Ylm[1], device=device))
        else:
            Ylm = torch.as_tensor(Ylm, device=device)
        if pxarea is not None:
            pxarea = torch.as_tensor(pxarea, device=device)
        if D is not None:
            D = torch.as_tensor(D, device=device)
        if Dinv is not None:
            Dinv = torch.as_tensor(Dinv, device=device)
        if cast is not None:
            if isinstance(Ylm, tuple):
                Ylm = (utils.push(Ylm[0], cast), utils.push(Ylm[1], cast))
            else:
                Ylm = utils.push(Ylm, cast)
            pxarea = utils.push(pxarea, cast)
            D = utils.push(D, cast)
            Dinv = utils.push(Dinv, cast)

    info['D'] = D
    info['Dinv'] = Dinv
    info['pxarea'] = pxarea
    info['idx'] = idx

    return Ylm, angs, l, m, info


def _gen_bessel2freq_multiproc(job):
    args, kwargs = job
    return gen_bessel2freq(*args, **kwargs)

 
def gen_bessel2freq(l, r, kbins=None, Nproc=None, Ntask=10,
                    device=None, dtype=None, method='shell', bc_type=2,
                    renorm=True, r_crit=None, use_pathos=False, **kln_kwargs):
    """
    Generate spherical Bessel forward model matrices sqrt(2/pi) r^2 k g_l(kr)
    from Fourier domain (k) to LOS distance or frequency domain (r or nu)

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
    r : array_like
        Line of sight comoving distances [Mpc] at which
        to evaluate spherical bessel functions.
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
        If True, renormalize the forward transform matrices such
        that their inner product is identity. Note
        this is not the same as renorm kwarg in sph_bessel_func.
    r_crit : float, optional
        Either r_min or r_max for method == 'shell'.
        Used for normalization of 1st and 2nd sph bessel funcs.
    kln_kwargs : dict
        If kbins is not provided, compute them here.
        These are the args and kwargs fed to
        sph_bessel_kln().
    use_pathos : bool, optional
        If multiprocessing (Nproc not None) and use_pathos = True,
        use the multiprocess module, otherwise use the more
        standard multiprocessing module (default).

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
    ul = np.unique(l)

    # multiproc mode
    if Nproc is not None:
        assert kbins is None, "no multiproc necessary if passing kbins"
        if use_pathos:
            import multiprocess as mproc
        else:
            from torch import multiprocessing as mproc
        start_method = mproc.get_start_method(allow_none=True)
        if start_method is None: mproc.set_start_method('spawn') 
        Njobs = len(ul) / Ntask
        if Njobs % 1 > 0:
            Njobs = np.floor(Njobs) + 1
        Njobs = int(Njobs)
        jobs = []
        for i in range(Njobs):
            _l = ul[i*Ntask:(i+1)*Ntask]
            args = (_l, r)
            kwargs = dict(device=device, dtype=dtype, method=method, r_crit=r_crit,
                          renorm=renorm, bc_type=bc_type)
            kwargs.update(kln_kwargs)
            jobs.append([args, kwargs])

        # run jobs
        with mproc.Pool(Nproc) as pool:
            output = pool.map(_gen_bessel2freq_multiproc, jobs)

        # collect output
        gln, kln = {}, {}
        for out in output:
            gln.update(out[0])
            kln.update(out[1])

        return gln, kln

    # run single proc mode
    dtype = dtype if dtype is not None else utils._float()
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
                             device=device, r_crit=r_crit, dtype=dtype)
        # form transform matrix: sqrt(2/pi) k g_l
        rt = torch.as_tensor(r, device=device, dtype=dtype)
        kt = torch.as_tensor(k, device=device, dtype=dtype)
        gln[_l] = np.sqrt(2 / np.pi) * rt**2 * kt[:, None].clip(1e-4) * gl
        kln[_l] = k

        if renorm:
            gln[_l] /= torch.sqrt((torch.abs(gln[_l])**2).sum(axis=1, keepdims=True).clip(1e-40))

    return gln, kln


def sph_bessel_func(l, k, r, method='shell', bc_type=2, r_crit=None, renorm=False,
                    device=None, dtype=None):
    """
    Generate a spherical bessel radial basis function, g_l(k_n r)

    g_l(k_ln r) = j_l(k_ln r) + A_ln y_l(k_ln r)

    where A_ln are coefficients determined by
    boundary conditions, and k_ln are wavevectors
    determined by boundary conditions.

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
        its derivative to zero at edges, and 3 (potential)
        sets the boundaries equal to the potential (see Gebhardt+21).
    r_crit : float, optional
        One of the radial edges [cMpc] if method is shell
        (aka rmin or rmax). If bc_type is 3 this should be
        r_min.
    renorm : bool, optional
        If True, renormalize amplitude of basis function
        such that inner product of r g_l(k_n r) with
        itself equals pi/2 k^-2
    device : str, optional
        Device to place matrices on
    dtype : dtype object, optional
        Data type for gln matrices

    Returns
    -------
    array_like
        basis functions of shape (Nk, Nr)
    """
    # configure 
    Nk = len(k)
    if method == 'shell':
        assert r_crit is not None

    dtype = dtype if dtype is not None else utils._float()
    j = torch.zeros(Nk, len(r), dtype=dtype, device=device)
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
                # get proportionality constant
                ell = l if bc_type < 3 else l + 1
                A = -special.jl(ell, _k * r_crit, deriv=deriv) / \
                     special.yl(ell, _k * r_crit, deriv=deriv).clip(-1e50, np.inf)
                # add in y_l
                y_i = special.yl(l, _k * r).clip(-1e50, np.inf)
                j_i += A * y_i

        else:
            raise ValueError("didn't recognize method {}".format(method))

        j[i] = torch.as_tensor(j_i, dtype=dtype, device=device)

    # renormalize
    if renorm:
        rt = torch.as_tensor(r, device=device, dtype=dtype)
        j *= torch.sqrt(np.pi/2 * k.clip(1e-4)**-2 / torch.sum(rt**2 * torch.abs(j)**2, axis=1))[:, None]

    return j


def sph_bessel_kln(l, r_min, r_max, kmax=0.5, dk_factor=0.5, decimate=False,
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
        dk = k_min * dk_factor where k_min = 1e-4
        A smaller dk_factor leads to higher resolution in k
        when solving for roots, but is slower to compute.
    decimate : bool, optional
        If True, use every other zero
        starting at the second zero. This
        is consistent with Fourier k convention.
    bc_type : int, optional
        Type of boundary condition, 1 (Dirichlet) sets
        function to zero at edges, 2 (Neumann, default) sets
        its derivative to zero at edges, 3 (potential)
        sets the edges equal to the potential (see Gebhardt+21).
    add_kzero : bool, optional
        Add a zero k mode if l == 0.

    Returns
    -------
    array
        Fourier modes k_n = [2pi / r_n]
    """
    # setup k_array of k samples to find roots
    assert bc_type in [1, 2, 3]
    kmin = 1e-4
    dk = kmin * dk_factor
    k_arr = np.linspace(kmin, kmax, int((kmax-kmin)//dk)+1)

    method = 'ball' if np.isclose(r_min, 0) else 'shell'
    deriv = bc_type == 2
    if method == 'ball':
        y = special.jl(l, k_arr * r_max, deriv=deriv)

    elif method == 'shell':
        l1 = l if bc_type < 3 else l - 1
        l2 = l if bc_type < 3 else l + 1
        y = (special.jl(l2, k_arr * r_min, deriv=deriv) * \
             special.yl(l1, k_arr * r_max, deriv=deriv).clip(-1e50, np.inf) - \
             special.jl(l1, k_arr * r_max, deriv=deriv) * \
             special.yl(l2, k_arr * r_min, deriv=deriv).clip(-1e50, np.inf))

    # get roots
    k = utils.get_zeros(k_arr, y)

    # decimate if desired
    if decimate:
        k = k[::2]

    # add zeroth k mode
    if add_kzero and np.isclose(l, 0, atol=1e-5):
        k = np.concatenate([[0.0], k])

    return np.asarray(k)


class AlmModel:
    """
    A general purpose spherical harmonic forward model

    f(theta, phi) = Sum_lm Y_lm(theta, phi) * a_lm

    The forward call takes a params tensor of shape
    (..., Ncoeff) and returns a map tensor of shape (..., Npix).

    Also supports Ylm matrix caching for forward modeling
    the same a_lm tensor to maps with different theta, phi
    sampling.

    Note: the last used or cached Ylm or (Theta, Phi) matrix
    is attached as self.Ylm or (self.Theta, self.Phi), but
    this can be changed by calling get_Ylm().
    """
    def __init__(self, l, m, default_kw=None, real_output=False, LM=None):
        """
        Parameters
        ----------
        l : array
            holds float "L" degrees of shape (Ncoeff,)
        m : array
            holds float "M" orders of shape (Ncoeff,)
        default_kw : dict, optional
            These are the default kwargs to use when generating Ylm
            from gen_sph2pix. 
        real_output : bool, optional
            If True, cast output to real before returning
        LM : LinearModel object, optional
            Pass the input params through this LinearModel
            object before passing through the response function.
        """
        self.l, self.m = l, m
        self.device = None
        self.default_kw = {} if default_kw is None else default_kw
        self.real_output = real_output
        self.LM = LM
        self.clear_Ylm_cache()
        self.clear_multigrid()

    def __call__(self, params, **kwargs):
        return self.forward_alm(params, **kwargs)

    def forward_alm(self, params, Ylm=None, alm_mult=None, ignoreLM=False):
        """
        Perform forward model from a_lm -> f(theta, phi)

        Will use transform matrices if passed, otherwise will
        use matrices attached to self.

        Parameters
        ----------
        params : tensor
            parameter tensor to forward model
            of shape (..., Ncoeff) unless passing
            as viewreal, in which case (..., Ncoeff, 2)
        Ylm : tensor or tuple, optional
            A forward model Ylm matrix (Ncoeff, Npix)
            or a tuple holding (Theta, Phi) if separable
        alm_mult : tensor, optional
            Multiply params by this before taking forward transform
        ignoreLM : bool, optional
            If True, ignore self.LM if it exists, otherwise pass
            params through it.
        """
        if self.LM is not None and not ignoreLM:
            params = self.LM(params)

        if Ylm is None and self.multigrid is not None:
            # iterate over multiple grids
            output = []
            for h in self.multigrid:
                # get the cache for this key
                c = self.Ylm_cache[h]
                Ylm, alm_mult = c['Ylm'], c['alm_mult']
                angs, separable = c['angs'], c['separable']
                output.append(self.forward_alm(params, Ylm=Ylm, alm_mult=alm_mult))

            # concatenate output along Npix
            out = torch.cat(output, dim=-1)

            if self._multigrid_idx is not None:
                # this is slightly slower than out[..., idx] on cpu, same speed on gpu
                # but backprop is faster on gpu w/ index_select
                out = torch.index_select(out, -1, self._multigrid_idx)

            return out

        # assume only forwarding one Ylm matrix
        if Ylm is None:
            Ylm = self.Ylm
            alm_mult = self.alm_mult
            separable = self.separable
        else:
            separable = isinstance(Ylm, (list, tuple))

        # convert params to complex if needed
        if separable:
            if torch.is_complex(Ylm[1]) and not torch.is_complex(params):
                params = utils.viewcomp(params)
        else:
            if torch.is_complex(Ylm) and not torch.is_complex(params):
                params = utils.viewcomp(params)

        # multiply by alm_mult if needed
        if alm_mult is not None:
            params = params * alm_mult

        if separable:
            # assume Theta and Phi are separable: (..., Ncoeff)
            Theta, Phi = Ylm
            # first broadcast self.Theta -> (..., Ncoeff, Ntheta)
            params = torch.einsum("ct,...c->...tc", Theta, params)
            # dot product into Phi --> (..., Nphi, Ntheta)
            params = torch.einsum("...tc,cp->...tp", params, Phi)
            # unravel to (..., Npix)
            shape = params.shape[:-2] + (Theta.shape[1] * Phi.shape[1],)
            out = params.reshape(shape)

        else:
            # full transform
            out = torch.einsum("...i,ij->...j", params, Ylm)

        if self.real_output:
            out = out.real

        return out

    @staticmethod
    def setup_angs(theta, phi, separable):
        """
        If separable, takes theta & phi grid points
        and meshes and flattens them. Otherwise return as is.

        Parameters
        ----------
        theta : array
            Polar (co-latitude) angles in degrees
            of shape (Npix,). If separable,
            this holds only the unique grid points.
        phi : array
            Azimuthal (longitude) angles in degrees
            of shape (Npix,), with a start convention
            of North of East. If separable,
            this holds only the unique grid points.
        separable: bool, optional
            Whether theta / phi are grid points or not.

        Returns
        -------
        theta, phi : array
            Samples for each pixel on the sky
        """
        if separable:
            if len(phi) > 10000 or len(theta) > 10000:
                print("Warning: the input phi or theta grid is too large")
            phi_arr, theta_arr = np.meshgrid(phi, theta, copy=False)
            phi = phi_arr.ravel()
            theta = theta_arr.ravel()

        return theta, phi

    def setup_Ylm(self, theta, phi, Ylm=None, alm_mult=None,
                  separable=False, generate=False, cache=True,
                  h=None, **kwargs):
        """
        Setup forward transform matrices.
        If Ylm is not provided, generate it.
        Sets Ylm, self.alm_mult

        Parameters
        ----------
        theta : array
            Polar (co-latitude) angles in degrees
            of shape (Npix,). If separable,
            this holds only the unique grid points.
        phi : array
            Azimuthal (longitude) angles in degrees
            of shape (Npix,), with a start convention
            of North of East. If separable,
            this holds only the unique grid points.
        Ylm : tensor, optional
            Full polar and azimuthal transform matrix
            of shape (Nmodes, Npix) if separable = False.
            If separable = True, then this is
            (Theta, Phi) where Theta is (Nmodes, Ntheta_uniq)
            and Phi is (Nmodes, Nphi_uniq).
            Default is to generate this given kwargs.
        alm_mult : tensor, optional
            Multiply parameter tensor by these coefficients
            before taking forward transform.
        separable : bool, optional
            If True, assume that theta and phi are sampled on a
            uniform rectangular grid, in which case theta and phi
            represent the unique, monotonically increasing grid values,
            stored as self.theta_grid and self.phi_grid, and theta and phi
            are converted to their full array shape
            as np.meshgrid(phi, theta).ravel().
            This can reduce memory overhead by over OOM.
            if separable = False.
            Otherwise, compute the full Ylm and perform both
            transforms simultaneously.
        generate : bool, optional
            If Ylm is passed as None and generate = True, use
            utils.gen_sph2pix to generate it given kwargs.
            Otherwise attach Ylm as None.
        cache : bool, optional
            If True, also store the Ylm or (Theta, Phi) matrices
            in the cache along with the sky angles hashed.
        h : int, optional
            Hash for the theta array. If not provided will
            compute it.
        kwargs : dict, optional
            Kwargs to pass to gen_sph2pix(). These supercede the
            default kwargs in self.default_kw
        """
        # set angles
        self.theta, self.phi = theta, phi
        if separable:
            self.theta_grid, self.phi_grid = theta, phi
            self.theta, self.phi = self.setup_angs(theta, phi, separable)

        # generate Ylm transform if needed
        if Ylm is None and generate:
            kw = copy.deepcopy(self.default_kw)
            kw.update(kwargs)
            if separable:
                th, ph = self.theta_grid, self.phi_grid
            else:
                th, ph = self.theta, self.phi
            Ylm, _, alm_mult = gen_sph2pix(th * utils.D2R,
                                           ph * utils.D2R,
                                           self.l, self.m,
                                           separable=separable,
                                           device=self.device,
                                           **kw)
        self.Ylm = Ylm
        self.alm_mult = alm_mult
        self.separable = separable
        if separable and self.Ylm is not None:
            assert isinstance(Ylm, (tuple, list))

        # cache if needed
        if cache:
            if separable:
                angs = (self.theta_grid, self.phi_grid)
            else:
                angs = (theta, phi)
            self.set_Ylm(Ylm, angs, alm_mult=alm_mult, h=h)

    def get_Ylm(self, theta, phi, separable=False, h=None):
        """
        Query cache for Y_lm matrix, otherwise generate it.

        Parameters
        ----------
        theta, phi : ndarrays
            Zenith angle (co-latitude) and
            azimuth (longitude) [deg] for
            all sky pixels. If separable,
            then these are the unique grid points.
        separable : bool
            Whether theta/phi represent grid points or not.
            If you are pulling from the cache, this kwarg
            doesn't matter. If you are setting the cache,
            then this matters.
        h : int, optional
            The arr_hash for theta used in the Ylm_cache.
            If not provided will compute it using
            utils.arr_hash()

        Returns
        -------
        Y : tensor
            Spherical harmonic tensor of shape
            (Nmodes, Npix)
        alm_mult : tensor
            Multiplication tensor to alm,
            if not stored return None
        """
        # get hash
        h = h if h is not None else utils.arr_hash(theta)
        if h in self.Ylm_cache:
            Ylm = self.Ylm_cache[h]['Ylm']
            theta, phi = self.Ylm_cache[h]['angs']
            if separable:
                self.theta_grid, self.phi_grid = theta, phi
                theta, phi = self.setup_angs(theta, phi, separable)
            alm_mult = self.Ylm_cache[h]['alm_mult']

        else:
            self.setup_Ylm(theta, phi, cache=True, h=h,
                           separable=separable)
            Ylm = self.Ylm
            alm_mult = self.alm_mult

        self.Ylm = Ylm
        self.theta, self.phi = theta, phi
        self.alm_mult = alm_mult
        self.separable = separable

        return Ylm, alm_mult

    def set_Ylm(self, Ylm, angs, alm_mult=None, h=None):
        """
        Insert a forward model Ylm matrix into the cache

        Parameters
        ----------
        Ylm : tensor
            Ylm forward model matrix of shape (Nmodes, Npix)
            or (Theta, Phi) tuple of shape (Nmodes, Ntheta)
            and (Nmodes, Nphi) if angs are separable.
        angs : tuple
            sky angles of Ylm pixels of shape (2, Npix)
            holding (zenith, azimuth) in [deg].
            If separable, this should be passed as
            (theta_grid, phi_grid).
        alm_mult : tensor, optional
            multiply this (Nmodes,) tensor into alm tensor
            before forward pass
        h : int, optional
            Hash for the provided theta array. Will compute
            if not provided.

        Returns
        -------
        h : int
            The key used to cache the inputs in Ylm_cache
        """
        theta, phi = angs
        h = h if h is not None else utils.arr_hash(theta)
        separable = isinstance(Ylm, (tuple, list))
        self.Ylm_cache[h] = dict(Ylm=Ylm, angs=angs, separable=separable,
                                 alm_mult=alm_mult)
        return h

    def clear_Ylm_cache(self):
        """
        Clear the self.Ylm_cache
        """
        self.Ylm_cache = {}

    def least_squares(self, y, cache_D=False, **kwargs):
        """
        Compute a least squares estimate of the params
        tensor given observations of shape (..., Npix).

        If self.separable, will take outer-product
        of self.Theta and self.Phi to form the full Ylm
        matrix in order to compute its inverse,
        which can be memory intensive.

        Parameters
        ----------
        y : tensor
            Map tensor of shape (..., Npix)
        cache_D : bool, optional
            If True, store the normalization D matrix
            as self._D to be used in future least_squares
            calls.
        kwargs : dict
            kwargs to pass to linalg.least_squares

        Returns
        -------
        tensor
        """
        # inflate Ylm to full (Ncoeff, Npix) shape
        if self.multigrid is not None:
            # collect all of the sub-grids if using multiple
            Ylm = torch.cat([inflate_Ylm(self.Ylm_cache[h]['Ylm']) for h in self.multigrid], dim=-1)
            if self._multigrid_idx is not None:
                Ylm = torch.index_select(Ylm, -1, self._multigrid_idx)
        else:
            Ylm = inflate_Ylm(self.Ylm)

        # get D if cached
        if hasattr(self, '_D') and 'D' not in kwargs:
            kwargs['D'] = self._D

        # compute least squares
        params, D = linalg.least_squares(Ylm, y, dim=-1, pretran=True, **kwargs)

        # store D
        if cache_D:
            self._D = D

        return params

    def make_closure(self, params, loss_fn, target, real=True):
        """
        Make and return a closure function used by
        optimization routines. Use this as an alternative
        to direct least_squares inversion if parameterization
        is large.

        Parameters
        ----------
        params : tensor
            alm tensor to optimize
        loss_fn : callable
            Loss function, takes (output, target)
        target : tensor
            Target to optimize against
        real : bool, optional
            Cast output and target as real-valued
            tensors before computing loss

        Returns
        -------
        callable
        """
        def closure(params=params, loss_fn=loss_fn,
                    target=target, real=real):
            if params.grad is not None:
                params.grad.zero_()
            out = self.forward_alm(params)
            if real:
                out, target = out.real, target.real
            L = loss_fn(out, target)
            L.backward()
            return L

        return closure

    def push(self, device):
        """
        Push items to new device.
        Note if self.Ylm exists in cache, this
        separates the pointers between self.Ylm and
        its corresponding value in self.Ylm_cache.
        """
        dtype = isinstance(device, torch.dtype)
        if hasattr(self, 'Ylm') and self.Ylm is not None:
            if isinstance(self.Ylm, tuple):
                self.Ylm = (utils.push(self.Ylm[0], device),
                            utils.push(self.Ylm[1], device))
            else:
                self.Ylm = utils.push(self.Ylm, device)

        if hasattr(self, 'alm_mult') and self.alm_mult is not None:
            self.alm_mult = utils.push(self.alm_mult, device)

        if self._multigrid_idx is not None and not dtype:
            self._multigrid_idx = self._multigrid_idx.to(device)

        for k in self.Ylm_cache:
            # push Ylms in cache
            Ylm = self.Ylm_cache[k]['Ylm']
            if isinstance(Ylm, tuple):
                self.Ylm_cache[k]['Ylm'] = (utils.push(Ylm[0], device),
                                            utils.push(Ylm[1], device))
            else:
                self.Ylm_cache[k]['Ylm'] = utils.push(Ylm, device)
            # push angs in cache
            angs = self.Ylm_cache[k]['angs']
            self.Ylm_cache[k]['angs'] = (angs[0].to(device), angs[1].to(device))
            # push alm_mults in cache
            am = self.Ylm_cache[k]['alm_mult']
            if am is not None:
                self.Ylm_cache[k]['alm_mult'] = utils.push(am, device)

        if not dtype:
            self.device = device

    def setup_multigrid_forward(self, thetas, phis, Ylms, alm_mults, idx=None):
        """
        Setup multiple Ylm matrices at distinct theta/phi points
        for the forward model. For a single call for self.forward_alm()
        each Ylm in Ylms will be called and the outputs will be
        concatenated across the Npix dimension. Sets self.multigrid

        Parameters
        ----------
        thetas : list
            List of theta arrays [deg]
        phis : list
            List of phi arrays [deg]
        Ylms : list
            List of Ylm matrices. Can have separable and non-separable
            Ylms together.
        alm_mults : list
            alm multiplies for each Ylm in Ylms
        idx : tensor, optional
            Re-indexing tensor of final output along Npix dim
        """
        # iterate over every element in Ylms
        self.multigrid = []
        self._multigrid_idx = idx
        for th, ph, Y, a in zip(thetas, phis, Ylms, alm_mults):
            h = self.set_Ylm(Y, (th, ph), alm_mult=a)
            self.multigrid.append(h)

    def clear_multigrid(self):
        """
        Remove multiple grids in the forward model
        """
        self.multigrid = None
        self._multigrid_idx = None

    def select(self, lm=None, lmin=None, lmax=None, mmin=None, mmax=None,
               other=None, atol=1e-10):
        """
        Down select on l and m modes. Operates in place.

        Parameters
        ----------
        lm : array, optional
            Array of l and m mode pairs to keep of shape
            (2, Nkeep), i.e. (l_keep, m_keep).
        lmin : float, optional
            Minimum l to keep
        lmax : float, optional
            Maximum l to keep
        mmin : float, optional
            Minimum m to keep
        mmax : float, optional
            Maximum m to keep
        other : array, optional
            Additional boolean array of shape (Nmodes,) to use
            when indexing.
        atol : float, optional
            tolerance to use when comparing l and m modes

        Returns
        -------
        s : tensor
            indexing tensor
        """
        # get indexing array
        s = np.ones(len(self.l), dtype=bool)
        if other is not None:
            s = s & other
        if lm is not None:
            _s = []
            for _l, _m in zip(*lm):
                _s.append((np.isclose(self.l, _l, atol=atol, rtol=1e-10) \
                          & np.isclose(self.m, _m, atol=atol, rtol=1e-10)).any())
            s = s & np.asarray(_s)
        if lmin is not None:
            s = s & (self.l >= lmin)
        if lmax is not None:
            s = s & (self.l <= lmax)
        if mmin is not None:
            s = s & (self.m >= mmin)
        if mmax is not None:
            s = s & (self.m <= mmax)

        def index_Ylm(Ylm, idx):
            if isinstance(Ylm, (tuple, list)):
                return (Ylm[0][idx], Ylm[1][idx])
            else:
                return Ylm[idx]

        # index relevant quantities
        self.l = self.l[s]
        self.m = self.m[s]
        if hasattr(self, 'Ylm') and self.Ylm is not None:
            self.Ylm = index_Ylm(self.Ylm, s)
        if hasattr(self, 'alm_mult') and self.alm_mult is not None:
            self.alm_mult = self.alm_mult[s]
        if hasattr(self, '_D') and self._D is not None:
            self._D = self._D[s, :][:, s]
        for k in self.Ylm_cache:
            self.Ylm_cache[k]['Ylm'] = index_Ylm(self.Ylm_cache[k]['Ylm'], s)
            if self.Ylm_cache[k]['alm_mult'] is not None:
                self.Ylm_cache[k]['alm_mult'] = self.Ylm_cache[k]['alm_mult'][s]

        return s


class SFBModel:
    """
    An object for handling the radial component of the
    spherical Fourier Bessel transform.

    a_lm(r) = Sum_n g_l(k_n r) * t_lmn

    The forward call takes a params tensor of shape
    (..., Nlmn) and returns a tensor of shape (..., Nlm, Nr),
    where Nlmn is the total number of sph.harm LM modes and
    their k_n modes, and Nr is the number of radial pixels.
    """
    def __init__(self, LM=None):
        """
        Parameters
        ----------
        LM : LinearModel object, optional
            Pass the input params through this LinearModel
            object before passing through the response function.
        """
        self.LM = LM

    def setup_gln(self, l, gln=None, kln=None, out_dtype=None,
                  r=None, m=None, **gln_kwargs):
        """
        Setup spherical Bessel forward transform
        gln matrices, t_lm(k_n) * g_l(k_n r) -> a_lm(r)

        Parameters
        ----------
        l : array
            The degree "L" of the output a_lm tensor along its
            Nlm dimension, of shape (Nlm,)
        gln : dict, optional
            Forward transform tensors of shape (Nk, Nradial)
            for each unique "L" in the l array. Key is "L" float,
            value is the g_l(k_n r) tensor.
        kln : dict, optional
            Dictionary holding the k wave-vector values for
            each entry in gln. Keys are the same as gln, values
            are 1D arrays holding k values for each gln mode
        r : array, optional
            Comoving radial bins to generate gln modes at, if
            gln is None
        m : array, optional
            An array of m modes of the same length as l. This
            is not strictly needed, but can be useful for
            debugging. If provided, this generates a self.m_arr
            array of the same shape as self.l_arr
        """
        # compute gln dictionary if needed
        if not gln:
            gln, kln = gen_bessel2freq(l, r, kbins=kln, dtype=out_dtype, **gln_kwargs)

        self.gln = gln
        self.kln = kln
        self.l = l
        self.m = m

        # get indexing from gln dict to params tensor
        # assumes gln is an ordered dict
        self.params_idx = {}
        self.alm_idx = {}
        self.alm_shape = {}
        self.k_arr = []
        self.l_arr = []
        self.m_arr = []
        Nlmn = 0
        for key in self.gln:
            Nk = len(self.gln[key])
            # get the indices in l for this particular key
            idx = np.where(np.isclose(l, key, atol=1e-6, rtol=1e-10))[0]
            Nl = len(idx)
            N = Nk * Nl
            # set indexing of input params tensor
            self.params_idx[key] = slice(Nlmn, Nlmn + N)
            # set indexing of output tensor
            self.alm_idx[key] = utils._list2slice(list(idx))
            self.alm_shape[key] = (self.gln[key].shape[1], len(idx))
            # populate arrays of k and l of the input params tensor
            self.k_arr.extend(list(self.kln[key]) * Nl) 
            self.l_arr.extend([key] * N)
            if m is not None:
                self.m_arr.extend([_m for _m in m[idx] for i in range(Nk)])
            Nlmn += N
    
        self.Nlmn = Nlmn
        self.k_arr = np.asarray(self.k_arr)
        self.l_arr = np.asarray(self.l_arr)
        self.m_arr = np.asarray(self.m_arr)
        self.Nr = self.gln[self.l_arr[0]].shape[1]
        self.Nlm = len(l)
        self.out_dtype = out_dtype if out_dtype is not None else utils._cfloat()
        self.device = self.gln[self.l_arr[0]].device

    def __call__(self, params, **kwargs):
        return self.forward_gln(params, **kwargs)

    def forward_gln(self, params, gln=None):
        """
        Perform radial forward model of a SFB
        params tensor t_lm(k_n) -> a_lm(r) 

        Parameters
        ----------
        params : tensor
            Of shape (..., Nlmn) where Nlmn
            is the sum total of row vectors
            in gln. Note that the ordering of l values
            along Nlmn must match that of self.alm_idx
        gln : dict, optional
            Use this gln dictionary instead
            of self.gln. Note that the matrices in gln must
            have the same shape as self.gln, and the key
            ordering of gln must match self.gln.

        Returns
        -------
        tensor
            Of shape (..., Nradial)
        """
        if self.LM is not None:
            params = self.LM(params)

        # get shapes
        pshape = params.shape[:-1]
        shape = pshape + (self.Nr, self.Nlm)
        out = torch.zeros(shape, dtype=self.out_dtype, device=self.device)
        for key, g in self.gln.items():
            new = pshape + (-1, self.alm_shape[key][1])  # (..., Nk, Nl)
            p = params[..., self.params_idx[key]].reshape(new)
            out[..., self.alm_idx[key]] = g.T @ p

        return out

    def push(self, device):
        """
        Push object to a new device (or dtype)
        """
        dtype = isinstance(device, torch.dtype)

        for key in self.gln:
            self.gln[key] = self.gln[key].to(device)

        if self.LM is not None:
            self.LM.push(device)

        if not dtype:
            self.device = device
        else:
            self.out_dtype = device

    def least_squares(self, y, **kwargs):
        """
        Compute a least squares estimate of the params
        tensor given observations of shape (..., Nr, Nlm).

        Parameters
        ----------
        y : tensor
            a_lm tensor of shape (..., Nr, Nlm)

        kwargs : dict
            kwargs to pass to linalg.least_squares

        Returns
        -------
        tensor
        """
        shape = y.shape[:-2] + (self.Nlmn,)
        params = torch.zeros(shape, dtype=self.out_dtype)

        # compute least squares
        for _l in np.unique(self.l):
            g = self.gln[_l]
            idx = self.alm_idx[_l]
            fit, D = linalg.least_squares(g, y[..., idx], dim=-2, pretran=True, **kwargs)
            params[..., self.params_idx[_l]] = fit.ravel()

        return params

    def make_closure(self, params, loss_fn, target, real=False):
        """
        Make and return a closure function used by
        optimization routines. Use this as an alternative
        to direct least_squares inversion if parameterization
        is large.

        Parameters
        ----------
        params : tensor
            SFB t_lmn tensor to optimize, shape (..., Nlmn)
        loss_fn : callable
            Loss function, takes (output, target)
        target : tensor
            Target a_lm tensor to optimize against (..., Nr, Nlm)
        real : bool, optional
            Cast output and target as real-valued
            tensors before computing loss

        Returns
        -------
        callable
        """
        def closure(params=params, loss_fn=loss_fn,
                    target=target, real=real):
            if params.grad is not None:
                params.grad.zero_()
            out = self.forward_gln(params)
            if real:
                out, target = out.real, target.real
            L = loss_fn(out, target)
            L.backward()
            return L

        return closure

    
def sfb_binning(params, k_arr, kbins, var=None, wgts=None, l_arr=None, lbins=None):
    """
    Bin and average a SFB tlmn params tensor along its last axis.

    Parameters
    ----------
    params : tensor
        A tlmn params tensor of shape (..., Nlmn)
    k_arr : array
        Array of k-values for params of shape (Nlmn,)
    kbins : array
        Array of k bin centers [Mpc^-1]
    var : array
        Variance of params tensor, of same shape.
    wgts : array, optional
        Binning weights, optional.
    l_arr : array, optional
        Array of l-values for params of shape (Nlmn,)
    lbins : array, optional
        Array of l bin centers for 2D k-l binning.
        Default is just k binning.

    Returns
    -------
    tensor
        output binned params
    tensor
        output binned var
    """
    # push bin centers forward by diff
    kdiff = np.diff(kbins)
    kdiff = np.concatenate([kdiff, kdiff[-1:]])
    kbins = kbins + kdiff / 2
    kinds = np.digitize(k_arr, kbins)
    Nk = len(kbins)
    if var is None:
        var = torch.ones_like(params)

    if lbins is not None:
        ldiff = np.diff(lbins)
        ldiff = np.concatenate([ldiff, ldiff[-1:]])
        lbins = lbins + ldiff / 2
        linds = np.digitize(l_arr, lbins)
        Nl = len(lbins)

    if wgts is None:
        wgts = torch.ones_like(params)

    if lbins is None:
        # 1D binning
        out = torch.zeros(params.shape[:-1] + (Nk,), dtype=params.dtype, device=params.device)
        vout = torch.zeros_like(out)

        # iterate over bins
        for i in range(Nk):
            idx = np.where(kinds == i)[0]
            w = wgts[idx]
            w /= torch.sum(w).clip(1e-40)
            out[..., i] = torch.sum(params[..., idx] * w, dim=-1)
            vout[..., i] = torch.sum(var[..., idx] * w**2, dim=-1)

    else:
        # 2D binning
        out = torch.zeros(params.shape[:-1] + (Nk, Nl), dtype=params.dtype, device=params.device)
        vout = torch.zeros_like(out)

        # iterate over bins
        for i in range(Nk):
            for j in range(Nl):
                idx = np.where((kinds == i) & (linds == j))[0]
                w = wgts[idx]
                w /= torch.sum(w).clip(1e-40)
                out[..., i, j] = torch.sum(params[..., idx] * w, dim=-1)
                vout[..., i, j] = torch.sum(var[..., idx] * w**2, dim=-1)

    return out, vout


def inflate_Ylm(Ylm):
    """
    If Ylm is separable, i.e. given as (Theta, Phi)
    inflate it to full Ylm size, otherwise
    return as is
    
    Parameters
    ----------
    Ylm : tensor or tuple
        If Ylm is (Ncoeff, Npix) tensor, return as-is,
        or if Ylm is (Theta, Phi) tuple take outer-product
        and return as (Ncoeff, Npix)

    Returns
    -------
    tensor
    """
    if isinstance(Ylm, (tuple, list)):
        # this is separable = True
        Theta, Phi = Ylm
        # take outer product to form Ylm
        Ylm = torch.einsum("ct,cp->ctp", Theta, Phi)
        Ylm = Ylm.view(-1, Theta.shape[1] * Phi.shape[1])

    return Ylm

