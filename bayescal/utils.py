"""
Utility module
"""
import numpy as np
import torch
from scipy.special import voigt_profile
from scipy.integrate import quad
import copy
import warnings

from . import special
from .data import DATA_PATH

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


########################################
######### Linear Algebra Tools #########
########################################

viewreal = torch.view_as_real
viewcomp = torch.view_as_complex
D2R = np.pi / 180


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
    Multiply two diagonal 1x1 or 2x2 matrices manually.
    This is generally faster than matmul or einsum
    for large, high dimensional stacks of 2x2 matrices.
    This drops the off-diagonal components of a and b.

    !! Note: this specifically ignores the off-diagonal for 2x2 matrices !!
    If you need off-diagonal components, you are
    better off using torch.matmul or torch.einsum directly.
    If you know off-diagonals are zero and are static, you can
    just use element-wise multiplication a * b.

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
        c[1, 1] = a[1, 1] * b[1, 1]
        return c
    else:
        raise ValueError("only 1x1 or 2x2 tensors")


def diag_inv(a):
    """
    Invert a diagonal 1x1 or 2x2 matrix manually.
    This is only beneficial for 2x2 matrices where
    you want to drop the off-diagonal terms.

    Parameters
    ----------
    a : tensor
        of shape (Nax, Nax, ...), where Nax = 1 or 2

    Returns
    -------
    c : tensor
        of shape (Nax, Nax, ...)
    """
    if a.shape[0] == 1:
        # 1x1: trivial
        return 1 / a
    elif a.shape[0] == 2:
        # 2x2
        c = torch.zeros_like(a)
        c[0, 0] = 1 / a[0, 0]
        c[1, 1] = 1 / a[1, 1]
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
    to cast to complex

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
                  high_prec=True, add_sectoral=True):
    """
    Compute associated Legendre function degrees l on
    the spherical cap or stripe given boundary conditions.

    theta BC:
        theta_min == 0 and theta_max < pi:
            This is a spherical cap, with boundary conditions
                P_lm(theta_max) = 0 and
                m == 0: d P_lm(theta_min) / d theta = 0
                m  > 0: P_lm(theta_min) = 0
        theta_min > 0 and theta_max < pi:
            This is a spherical stripe with BC
                P_lm(theta_min) = 0 and
                P_lm(theta_max) = 0

    phi BC:
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

    Notes
    -----

    """
    # solve for m modes
    spacing = 2 * np.pi / phi_max
    m = np.arange(0, mmax + 1.1, spacing)

    # solve for l modes
    assert theta_max < np.pi, "if theta_max must be < pi for spherical cap or stripe"
    ls = {}
    x_min, x_max = np.cos(theta_min), np.cos(theta_max)
    m = np.atleast_1d(m)
    for _m in m:
        # construct array of test l's, skip l == m
        l = _m + np.arange(1, (lmax - _m)//dl + 1) * dl
        if len(l) < 1:
            continue
        # boundary condition is derivative is zero for m == 0
        deriv = _m == 0
        if np.isclose(theta_min, 0):
            # spherical cap
            y = special.Plm(l, _m, x_max, deriv=deriv, high_prec=high_prec, keepdims=True)

        else:
            # spherical stripe
            y = special.Plm(l, _m, x_min, deriv=deriv, high_prec=high_prec, keepdims=True) \
                * special.Qlm(l, _m, x_max, deriv=deriv, high_prec=high_prec, keepdims=True) \
                - special.Plm(l, _m, x_max, deriv=deriv, high_prec=high_prec, keepdims=True) \
                * special.Qlm(l, _m, x_min, deriv=deriv, high_prec=high_prec, keepdims=True)

        y = y.ravel()
        zeros = get_zeros(l, y)
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


def gen_sph2pix(theta, phi, method='sphere', theta_min=None, l=None, m=None,
                lmax=None, real_field=True, Nproc=None, Ntask=10, device=None,
                high_prec=True, renorm=False):
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
    theta_min : float, optional
        For method == 'stripe', this is the minimum theta
        boundary [radians] of the mask.
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
            jobs.append([(_l, _m), (theta, phi), dict(method=method, theta_min=theta_min,
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
    P = special.Plm(l, m, x, high_prec=high_prec, keepdims=True)
    Phi = np.exp(1j * m * phi)

    if method == 'stripe':
        # spherical stripe: uses Plm and Qlm
        assert theta_min is not None
        # compute Qlms
        Q = special.Qlm(l, m, x, high_prec=high_prec, keepdims=True)
        # compute A coefficients
        x_min = np.cos(theta_min)
        A = -special.Plm(l, m, x_min, high_prec=high_prec, keepdims=True) \
            / special.Qlm(l, m, x_min, high_prec=high_prec, keepdims=True)
        # Use deriv = True for m == 0
        if 0 in m:
            mzero = np.ravel(m) == 0
            A[mzero] = -special.Plm(l[mzero], m[mzero], x_min, high_prec=high_prec, keepdims=True, deriv=True) \
                       / special.Qlm(l[mzero], m[mzero], x_min, high_prec=high_prec, keepdims=True, deriv=True)
        # construct Y
        Y = torch.as_tensor((P + A * Q) * Phi, dtype=_cfloat(), device=device)

    else:
        # spherical cap or full sky: uses Plm
        Y = torch.as_tensor(P * Phi, dtype=_cfloat(), device=device)

    # renormalize
    if renorm:
        # Note: theta and phi must be part of a HEALpix grid
        norm = torch.sqrt(torch.sum(torch.abs(Y)**2, axis=1))
        Y /= norm[:, None]

    return Y


def _gen_bessel2freq_multiproc(job):
    args, kwargs = job
    return gen_bessel2freq(*args, **kwargs)


def gen_bessel2freq(l, freqs, cosmo, kmax, method='default', kbin_file=None,
                    decimate=False, device=None, Nproc=None, Ntask=10, renorm=False):
    """
    Generate spherical Bessel forward model matrices sqrt(2/pi) k g_l(kr)
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
    decimate : bool, optional
        Use every other g_l(x) zero as k bins (i.e. DFT convention)
    device : str, optional
        Device to push g_l(kr) to.
    Nproc : int, optional
        If not None, enable multiprocessing mode with Nproc processes
    Ntask : int, optional
        Number of modes to compute per process
    renorm : bool, optional
        If True, renormalize g_l modes such that the numerically
        integrated inner product is equal to pi/2 k^-2

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
        import multiprocessing
        Njobs = len(ul) / Ntask
        if Njobs % 1 > 0:
            Njobs = np.floor(Njobs) + 1
        Njobs = int(Njobs)
        jobs = []
        for i in range(Njobs):
            _l = ul[i*Ntask:(i+1)*Ntask]
            jobs.append([(_l, freqs, cosmo, kmax), dict(method=method, decimate=decimate,
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
        k = sph_bessel_kln(_l, r_max, kmax, r_min=r_min, decimate=decimate,
                           method=method, filepath=kbin_file)
        # add monopole term if l = 0
        if _l == 0:
            k = np.concatenate([[0], k[:-1]])
        # get basis function g_l
        j = sph_bessel_func(_l, k, r, method=method, renorm=renorm, device=device)
        # form transform matrix: sqrt(2/pi) k g_l
        kt = torch.as_tensor(k, device=device, dtype=_float())
        jl[_l] = np.sqrt(2 / np.pi) * j * kt[:, None].clip(1e-3)
        kbins[_l] = k

    return jl, kbins


def sph_bessel_func(l, k, r, method='default', r_min=None, r_max=None,
                    renorm=False, device=None):
    """
    Generate a spherical bessel radial basis function

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
            g_nl = j_l(k_ln r) + A_ln y_l(k_ln r) and BC
            is g_nl(k r) = 0 for r_min and r_max (Samushia2019)
        gebhardt : interval is r_min -> r_max, basis is
            g_nl = j_l(k_ln r) + A_ln y_l(k_ln r)
            BC is potential field continuity (Gebhardt+2021)
    r_min, r_max : float, optional
        r_min and r_max of LIM survey. If None, will use
        min and max of r.
    renorm : bool, optional
        If True, renormalize amplitude of basis function
        such that inner product equals pi/2 k^-2
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
        j *= torch.sqrt(np.pi/2 * k.clip(1e-3)**-2 / torch.sum(torch.abs(j)**2, axis=1))[:, None]

    return j


def sph_bessel_kln(l, r_max, kmax, r_min=None, decimate=False,
                   method='default', filepath=None):
    """
    Get spherical bessel Fourier bins given method.

    Parameters
    ----------
    l : float
        Angular l mode
    r_max : float
        Maximum survey radial extent [cMpc]
    kmax : float
        Maximum wavevector k [Mpc^-1] to compute
    r_min : float, optional
        Survey starting boundary [cMpc]
        only used for special method
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
        if method == 'default':
            import mpmath
            k = 0.0
            zeros = []
            n = 1
            while k < kmax:
                k = float(mpmath.besseljzero(l+.5, n)) / r_max
                zeros.append(k)
                n += 1

        elif method == 'samushia':
            kmin = 2 * np.pi / (r_max - r_min)
            dk = kmin / 500
            k_arr = np.linspace(kmin, kmax, int((kmax-kmin)//dk)+1)
            y = (special.jl(l, k_arr * r_min) * special.yl(l, k_arr * r_max).clip(-1e50, np.inf) \
                 - special.jl(l, k_arr * r_max) * special.yl(l, k_arr * r_min).clip(-1e50, np.inf)) * k_arr**2
            k = get_zeros(k_arr, y)

        elif method == 'gebhardt':
            raise NotImplementedError

    # decimate if desired
    if decimate:
        k = k[1::2]

    return np.asarray(k)


def gen_poly_A(freqs, Ndeg, device=None):
    """
    Generate design matrix (A) for polynomial of Ndeg across freqs,
    with coefficient ordering

    .. math::

        a0 * x^0 + a1 * x^1 + a2 * x^2 + ...

    Parameters
    ----------
    freqs : ndarray
        Frequency bins [Hz]
    Ndeg : int
        Polynomial degree
    device : str
        device to send A matrix to

    Returns
    -------
    torch tensor
        Polynomial design matrix
    """
    ## TODO: implement this?
    #from emupy.linear import setup_polynomial
    #dfreqs = (freqs - freqs[0]) / 1e6  # In MHz
    #A = setup_polynomial(dfreqs[:, None], Ndeg-1, basis=basis)[0]
    #A = torch.as_tensor(A, dtype=dtype, device=device)
    dfreqs = (freqs - freqs.mean()) / 1e6  # In MHz
    A = torch.as_tensor(torch.vstack([dfreqs**i for i in range(Ndeg)]),
                        dtype=_float(), device=device).T
    return A


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
    def __init__(self, pixtype, npix, interp_mode='nearest',
                 device=None):
        """
        Parameters
        ----------
        pixtype : str
            Pixelization type. options = ['healpix', 'other']
        npix : int
            Number of sky pixels in the beam
        interp_mode : str, optional
            Spatial interpolation method. ['nearest', 'bilinear']
        device : str, optional
            Device to place object on
        """
        if pixtype != 'healpix':
            raise NotImplementedError("only supports healpix pixelization currently")
        self.pixtype = pixtype
        self.npix = npix
        self.interp_cache = {}
        self.interp_mode = interp_mode
        self.device = device

    def get_interp(self, zen, az):
        """
        Get bilinear or nearest interpolation

        Parameters
        ----------
        zen, az : zenith and azimuth angles [deg]

        Returns
        -------
        interp : tuple
            4 (1) nearest neighbor (indices, weights)
            for each entry in zen, az for bilinear (nearest)
            mode
        """
        # get hash
        h = ang_hash(zen), ang_hash(az)
        if h in self.interp_cache:
            # get interpolation if present
            interp = self.interp_cache[h]
        else:
            # otherwise generate it
            if self.pixtype == 'healpix':
                # get indices and weights for bilinear interpolation
                nside = healpy.npix2nside(self.npix)
                inds, wgts = healpy.get_interp_weights(nside,
                                                       tensor2numpy(zen) * D2R,
                                                       tensor2numpy(az) * D2R)
                # down select if using nearest interpolation
                if self.interp_mode == 'nearest':
                    wgts = np.argmax(wgts, axis=0)
                    inds = np.array([inds[wi, i] for i, wi in enumerate(wgts)])
                    wgts = 1.0
            else:
                raise NotImplementedError

            # store it
            interp = (torch.as_tensor(inds, device=self.device),
                      torch.as_tensor(wgts, dtype=_float(), device=self.device))
            self.interp_cache[h] = interp

        return interp

    def interp(self, m, zen, az):
        """
        Interpolate a healpix map m at zen, az points

        Parameters
        ----------
        m : array_like or tensor
            healpix map to interpolate
        zen, az : array_like or tensor
            Zenith angle (colatittude) and azimuth [deg]
            points at which to interpolate map
        """
        # get interpolation indices and weights
        inds, wgts = self.get_interp(zen, az)
        if self.interp_mode == 'nearest':
            # use nearest neighbor
            return m[..., inds]
        elif self.interp_mode == 'bilinear':
            # select out 4-nearest neighbor indices for each zen, az
            nearest = m[..., inds.T]
            # multiply by weights and sum
            return torch.sum(nearest * wgts.T, axis=-1)


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


#################################
######### Miscellaneous #########
#################################

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
        if tensor.device != 'cpu':
            tensor = tensor.cpu()
        if clone:
            tensor = tensor.clone()
        return tensor.numpy()
    return tensor


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
