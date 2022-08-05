"""
Module for special functions
"""
import numpy as np
from scipy.special import jv, jvp, yv, yvp, factorialk, gamma, gammaln
from scipy.integrate import quad
import copy
import warnings
import torch

from bayeslim.data import DATA_PATH


def Plm(l, m, x, deriv=False, dtheta=True, keepdims=False, high_prec=True,
        sq_norm=True):
    """
    Associated Legendre function of the first kind
    in hypergeometric form, aka Ferrers function
    DLMF 14.3.1 & 14.10.5 with interval -1 < x < 1.
    Note 1: this is numerically continued to |x| = 1
    Note 2: stable to integer l = m ~ 800, for all x

    .. math::

        P_{lm}(x) &= C\left(\frac{x+1}{x-1}\right)^{m/2}F(-l, l+1, 1-\mu, (1-x)/2) \\
        C &= \sqrt{\frac{2l+1}{4\pi}{(l-m)!}{(l+m)!}}

    Parameters
    ----------
    l : float or array_like
        Degree of the associated Legendre function
    m : int or array_like
        Order of the associated Legendre function
    x : float or array_like
        Argument of Legendre function, bounded by |x|<1
    deriv : bool, optional
        If True return derivative
    dtheta : bool, optional
        If True (default), return dP/dtheta instead 
        of dP/dx where x = cos(theta)
    keepdims : bool, optional
        If False and (l,m) or x is len 1
        then ravel the output
    high_prec : bool, optional
        If True, use precise mpmath for hypergeometric
        calls, else use faster but less accurate scipy.
        Matters mostly for non-integer degree
    sq_norm : bool, optional
        If True, return Plm with standard (1-x^2)^(-m/2)
        normalization (default). Otherwise withold it,
        which is needed for computing Plm - Qlm at
        high degree to prevent catastrophic cancellation.

    Returns
    -------
    array
        Legendre function of first kind at x
    """
    # reshape if needed
    l = np.atleast_1d(l)
    m = np.atleast_1d(m)
    assert np.all(m <= l+1e-5)
    if l.ndim == 1:
        l = l[:, None]
    if m.ndim == 1:
        m = m[:, None]
    assert m.shape == l.shape
    # avoid singularity
    x = np.atleast_1d(x).copy()
    s = np.isclose(np.abs(x), 1, rtol=1e-10)
    if np.any(s):
        dx = 1e-8
        x[s] *= (1 - dx)
    # compute Plm
    if not deriv:
        # compute hyper-geometric: DLMF 14.3.1
        # Note this previously used DLMF 14.3.11
        # but this was unstable at low theta. 14.3.1 works fine for Plm
        norm = ((1 + x) / (1 - x))**(m/2)
        a, b, c = l+1, -l, 1-m
        P = hypF(a, b, c, (1-x)/2, high_prec=high_prec, keepdims=True)
        isf = np.isfinite(norm)
        P[isf] *= norm[isf]
        # orthonormalize: sqrt[ (2l+1)/(4pi)*(l-m)!/(l+m)! ]
        C = _log_legendre_norm(l, m)
        # gammaln(c+1) comes from extra factor in hypF!
        P *= np.exp(C + gammaln(np.abs(c)+1))

        # remove (1-x^2)^(-m/2) term if requested: when combining with Qlm
        if not sq_norm:
            P /= (1-x**2)**(-m/2)

        # handle singularity: 1st order Euler
        if np.any(s):
            P[:, s] += Plm(l, m, x[s], deriv=True, keepdims=True) * dx
        if not keepdims:
            if 1 in P.shape:
                P = P.ravel()
            if P.size == 1:
                P = P[0]
        return P

    # compute derivative
    else:
        # DLMF 14.10.5
        norm = 1 / (1 - x**2)
        term1 = (m - l - 1) * Plm(l+1, m, x, keepdims=True, sq_norm=sq_norm, high_prec=high_prec)
        term1 *= np.exp(_log_legendre_norm(l, m) - _log_legendre_norm(l+1, m))
        term2 = (l+1) * x * Plm(l, m, x, keepdims=True, sq_norm=sq_norm, high_prec=high_prec)
        dPdx = norm * (term1 + term2)
        # handle singularity: 1st order Euler
        if np.any(s):
            dPdx[:, s] += (dPdx[:, s] - Plm(l, m, x[s] * (1 - dx), deriv=True, keepdims=True, sq_norm=sq_norm, high_prec=high_prec))
        # correct for change of variables if requested
        if dtheta:
            dPdx *= -np.sin(np.arccos(x))
        if not keepdims:
            if 1 in dPdx.shape:
                dPdx = dPdx.ravel()
            if dPdx.size == 1:
                dPdx = dPdx[0]
        return dPdx


def Qlm(l, m, x, deriv=False, dtheta=True, keepdims=False, high_prec=True, sq_norm=True):
    """
    Associated Legendre function of the second kind
    in hypergeometric form, aka Ferrers function
    DLMF 14.3.12 with interval -1 < x < 1.
    Note 1: this will return infs or nan at |x| = 1
    Note 2: stable to integer l = m ~ 800, for all x.

    Parameters
    ----------
    l : float or array_like
        Degree of the associated Legendre function
    m : int or array_like
        Order of the associated Legendre function
    x : float or array_like
        Argument of Legendre function, bounded by |x|<1
    deriv : bool, optional
        If True return derivative
    dtheta : bool, optional
        If True (default), return dQ/dtheta instead 
        of dQ/dx where x = cos(theta)
    keepdims : bool, optional
        If False and (l,m) or x is len 1
        then ravel the output
    high_prec : bool, optional
        If True, use precise mpmath for hypergeometric
        calls, else use faster but less accurate scipy.
        Matters mostly for non-integer degree
    sq_norm : bool, optional
        If True, return Plm with standard (1-x^2)^(-m/2)
        normalization (default). Otherwise withold it,
        which is needed for computing Plm - Qlm at
        high degree to prevent catastrophic cancellation.

    Returns
    -------
    array
        Legendre function of second kind at z
    """
    # reshape if needed
    l = np.atleast_1d(l).astype(float)
    m = np.atleast_1d(m).astype(float)
    if l.ndim == 1:
        l = l[:, None]
    if m.ndim == 1:
        m = m[:, None]
    # compute Qlm
    x = np.atleast_1d(x).copy()
    if not deriv:
        # compute ortho normalization in logspace
        C = _log_legendre_norm(l, m)
        # compute w1 and w2, multiply in normalization
        w1 = 2**m * hypF((-l-m)/2, (l-m+1)/2, .5, x**2, high_prec=high_prec)
        # gammaln(0.5+1) comes from extra factor in hypF!
        w1 *= np.exp(C + gammaln((l+m+1)/2) - gammaln((l-m+2)/2) + gammaln(0.5+1))
        w2 = 2**m * x * hypF((1-l-m)/2, (l-m+2)/2, 3./2, x**2, high_prec=high_prec)
        w2 *= np.exp(C + gammaln((l+m+2)/2) - gammaln((l-m+1)/2) + gammaln(3./2+1))
        Q = .5 * np.pi * (-np.sin(.5*(l+m)*np.pi) * w1 + np.cos(.5*(l+m)*np.pi) * w2)

        # add in sq_norm if desired
        if sq_norm:
            Q *= (1-x**2)**(-m/2)

        if not keepdims:
            if 1 in Q.shape:
                Q = Q.ravel()
            if Q.size == 1:
                Q = Q[0]
        return Q
    else:
        # DLMF 14.10.5
        norm = 1 / (1 - x**2)
        term1 = (m - l - 1) * Qlm(l+1, m, x, keepdims=True, high_prec=high_prec, sq_norm=sq_norm)
        term1 *= np.exp(_log_legendre_norm(l, m) - _log_legendre_norm(l+1, m))
        term2 = (l+1) * x * Qlm(l, m, x, keepdims=True, high_prec=high_prec, sq_norm=sq_norm)
        dQdx = norm * (term1 + term2)

        # correct for change of variables if requested
        if dtheta:
            dQdx *= -np.sin(np.arccos(x))

        if not keepdims:
            if 1 in dQdx.shape:
                dQdx = dQdx.ravel()
            if dQdx.size == 1:
                dQdx = dQdx[0]
        return dQdx


def _log_legendre_norm(l, m):
    """
    Compute log of ortho normalization for
    associated Legendre functions

    .. math::

        C = \sqrt{\frac{(2l+1)(l-m)!}{4\pi(l+m)!}}

    """
    return 0.5 * (np.log(2*l+1) - np.log(4*np.pi) + gammaln(l - m + 1) - gammaln(l + m + 1))


HYPF_KWGS = {'zeroprec': 1000}


def hypF(a, b, c, z, high_prec=True, keepdims=False):
    """
    Gauss hypergeometric function.
    Catches the case where c is <= 0
    and includes an extra normalization
    for numerical overflow
    DLMF 15.2.3_5

    .. math::

        F = \frac{_2F_1(a, b, c, z)}{\Gamma(c)\Gamma{c+1}} 

    Parameters
    ----------
    a, b : float
    c : int
    z : float
    high_prec : bool, optional
        If True, use mpmath (slow), else use scipy (fast)
    keepdims : bool, optional
        If True, keep full shape, otherwise squeeze
        along len-1 axes
    zeroprec : int, optional
        When using high_prec, if the hyp2f1 result
        is less than 2^-zeroprec mpmath calls it zero.
    
    Notes
    -----
    Note the extra factor of 1/Gamma(|c|+1), which
    is inserted artificially for numerical purposes,
    and must be re-normalized for standard use
    with spherical harmonics.
    Also note the HYPF_KWGS dictionary
    which controls mpmath.hyp2f1(*args, **kwgs)
    when using high_prec=True.
    """
    if high_prec:
        from mpmath import hyp2f1
        kg = HYPF_KWGS
    else:
        from scipy.special import hyp2f1
        kg = {}

    if not isinstance(a, np.ndarray):
        a = np.atleast_2d(a)
    if not isinstance(b, np.ndarray):
        b = np.atleast_2d(b)
    if not isinstance(c, np.ndarray):
        c = np.atleast_2d(c)
    maxlen = max([len(a), len(b), len(c)])
    # assumes a and b are the same shape
    if c.size < a.size:
        c = np.repeat(c, a.shape[0], axis=0)
    s = c.ravel() <= 0
    F = np.zeros((len(c), len(z)))
    if np.any(s):
        # compute assuming c <= 0: DLMF 15.2.3_5
        A = np.atleast_2d(a[s])
        B = np.atleast_2d(b[s])
        C = np.atleast_2d(c[s])
        n = -C
        # compute pochhammers in logspace as "norm"
        # and keep their sign, b/c of special case: DLMF 5.2.6
        norm = np.zeros_like(n, dtype=float)
        sign = np.ones_like(n, dtype=int)
        for inp in [A, B]:
            out = pochln(inp, n+1)
            sign *= out[0]
            norm += out[1]
        # divide by (n+1)! and divide by gamma(|c|+1)
        norm -= gammaln(n+2) + gammaln(n+1)
        # compute hypergeometric
        f21 = np.array(np.frompyfunc(lambda *a: float(hyp2f1(*a, **kg)), 4, 1)(A+n+1, B+n+1, n+2, z), dtype=float)
        # get the sign of f21 in real space
        f21_sign = np.sign(f21)
        # full term: wait to exp() until as late as possible, multiply by signs
        F[s] = sign * f21_sign * np.exp(np.log(np.abs(f21)) + np.log(z**(n+1)) + norm)
    if np.any(~s):
        # compute assuming c > 0
        A = np.atleast_2d(a[~s])
        B = np.atleast_2d(b[~s])
        C = np.atleast_2d(c[~s])
        f21 = np.array(np.frompyfunc(lambda *a: float(hyp2f1(*a, **kg)), 4, 1)(A, B, C, z), dtype=float)
        F[~s] = f21 / gamma(C) / gamma(C+1)

    if not keepdims and len(F) == 1:
        F = F[0]

    return F


def pochln(a, n):
    """
    Log pochhammer symbol

    .. math::

        ln(gamma(a + n)) - ln(gamma(n))

    Parameters
    ----------
    a : float
    n : float
    
    Returns
    -------
    int
        Sign of exp(ln(poch(a, n))), either 1 or -1 (DLMF 5.2.6)
    float
        natural log of |pochhammer(a, n)|
    """
    a = np.atleast_1d(a)
    n = np.atleast_1d(n)
    p =  np.zeros_like(a, dtype=float) 
    sign = np.ones_like(a, dtype=int)

    # negative integers
    integer = ((a % 1) == 0) & ((n % 1) == 0)
    s1 = integer & (a < 0) & (n < 0)
    if s1.any():
        sign[s1] = 1
        p[s1] = -np.inf

    # negative int and zero
    s2 = integer & (a < 0) & (n == 0)
    if s2.any():
        sign[s2] = 1
        p[s2] = 0.0

    # negative and positive int: DLMF 5.2.6
    s3 = integer & (a < 0) & (n > 0)
    if s3.any():
        sign[s3] = (-1)**n[s3]
        p[s3] = pochln(-a[s3]-n[s3]+1, n[s3])[1]

    # all others
    s4 = (~s1) & (~s2) & (~s3)
    if s4.any():
        p[s4] = gammaln(a[s4] + n[s4]) - gammaln(a[s4])
        # propagate negative signs for a < 0 and a+n < 0
        sign[s4 & (abs(a+n) % 2 < 1) & (a+n < 0)] *= -1
        sign[s4 & (abs(a) % 2 < 1) & (a < 0)] *= -1

    if a.size == 1:
        sign = sign[0]
        p = p[0]

    return sign, p


def jl(l, z, deriv=False, keepdims=False):
    """
    Spherical Bessel of the first kind.
    DLMF 10.47.3
    
    Parameters
    ----------
    l : float
        Non-negative degree
    z : float
        Argument
    deriv : bool, optional
        Return first derivative
    keepdims : bool, optional
        If False, ravel l or z axis
        if only 1 element, otherwise
        return as 2D

    Returns
    -------
    float
    """
    # reshape if needed
    l = np.atleast_1d(l)
    if l.ndim == 1:
        l = l[:, None]
 
    # avoid singularity
    z = np.atleast_1d(z).copy()
    s = np.isclose(z, 0, rtol=1e-10)
    if np.any(s):
        dz = 1e-8
        z[s] += dz

    if not deriv:
        j = np.sqrt(np.pi/2/z) * jv(l+.5, z)

        # handle singularity: 1st order Euler
        if np.any(s):
            j[:, s] -= jl(l, z[s], deriv=True) * dz

        if not keepdims:
            if 1 in j.shape:
                j = j.ravel()
            if j.size == 1:
                j = j[0]

        return j

    else:
        djdz = np.sqrt(np.pi/2) * (-.5 * z**(-3/2) * jv(l+.5, z) + z**-.5 * jvp(l+.5, z))

        # handle singularity: 1st order Euler
        if np.any(s):
            djdz[:, s] -= (djdz[:, s] - jl(l, z[s] + dz, deriv=True, keepdims=True))

        if not keepdims:
            if 1 in djdz.shape:
                djdz = djdz.ravel()
            if djdz.size == 1:
                djdz = djdz[0]

        return djdz


def yl(l, z, deriv=False, keepdims=False):
    """
    Spherical Bessel of the second kind.
    DLMF 10.47.4
    
    Parameters
    ----------
    l : float
        Non-negative degree
    z : float
        Argument
    deriv : bool, optional
        Return first derivative
    keepdims : bool, optional
        If False, ravel l or z axis
        if only 1 element, otherwise
        return as 2D

    Returns
    -------
    float
    """
    # reshape if needed
    l = np.atleast_1d(l)
    if l.ndim == 1:
        l = l[:, None]
 
    z = np.atleast_1d(z).copy()

    if not deriv:
        y = np.sqrt(np.pi/2/z) * yv(l+.5, z)

        if not keepdims:
            if 1 in y.shape:
                y = y.ravel()
            if y.size == 1:
                y = y[0]

        return y

    else:
        dydz = np.sqrt(np.pi/2) * (-.5 * z**(-3/2) * yv(l+.5, z) + z**-.5 * yvp(l+.5, z))

        if not keepdims:
            if 1 in dydz.shape:
                dydz = dydz.ravel()
            if dydz.size == 1:
                dydz = dydz[0]

        return dydz


def _bessel_integrand(x, tau, n=1):
    return torch.cos(n * tau - x * torch.sin(tau))


def j1(x, Ntau=100, brute_force=True):
    """
    Bessel function of the first kind,
    derived from trapezoidal integration
    of the Bessel integral
    https://en.wikipedia.org/wiki/Bessel_function
    (if brute_force), or can be done using
    scipy (which is not differentiable!)
    
    Parameters
    ----------
    x : tensor
        x values to evaluate
    Ntau : int, optional
        Bessel integral pixelization density
    brute_force : bool, optional
        If True, numerically integrate
        Bessel integral brute force.
        Makes this differentiable.
        Otherwise use a scipy routine.
        
    Returns
    -------
    tensor
    """
    if brute_force:
        # evaluate bessel integrand at fixed grid
        t = torch.linspace(0, np.pi, Ntau, device=x.device)
        diff = (t[1] - t[0])
        t = t.reshape((-1, ) + tuple(1 for i in range(x.ndim)))
        integrand = _bessel_integrand(x, t, n=1)
        
        # integrate using trapezoidal rule
        wgts = torch.ones_like(integrand)
        wgts[1:-1] = 2.0

        return torch.sum(wgts * integrand, dim=0) * diff / 2.0 / np.pi
    else:
        from scipy import special
        return special.j1(x)


def _Pmm_legacy(m, z):
    """
    Associated Legendre of the first kind with
    same integer order and degree for z in [-1, 1]
    valid up to m = 260
    
    .. math::
    
        P_m^m(z) &= C (-1)^m (2m - 1)!! (1 - z^2)^{m/2} \\
        C &= \sqrt{\frac{2m+1}{4\pi}{(m-m)!}{(m+m)!}}
        
    Cohl et al. 2020, doi:10.3390/sym12101598
    Eqn. 71

    Parameters
    ----------
    m : int
        Order and degree
    z : float
        Legendre argument [-1, 1]

    Notes
    -----
    The factorial
    
    .. math::
    
        \frac{(2m-1)!!}{\sqrt{(2m)!}}
        
    can be evaluated with double precision up to
    m = 260 by first computing
    
    .. math::
    
        (2m-1)!! &= (2m-1)!!!! (2m-3)!!!! \\
        (2m)! &= (2m)!!!! (2m-1)!!!! (2m-2)!!!! (2m-3)!!!!
    
    and interleaving the cancellation of multiplication and division.
    """
    # stable factorial computation of (2m-1)!! / sqrt((2m)!)
    n1 = float(factorialk(2*m-1, 4))
    n2 = float(factorialk(2*m-3, 4))
    d1 = float(factorialk(2*m-0, 4))
    d2 = float(factorialk(2*m-1, 4))
    d3 = float(factorialk(2*m-2, 4))
    d4 = float(factorialk(2*m-3, 4))
    f = (n1 / (np.sqrt(d1) * np.sqrt(d2))) * (n2 / (np.sqrt(d3) * np.sqrt(d4)))
    
    # compute normalization
    C = np.sqrt((2 * m + 1) / (4 * np.pi)) * f
    
    return (-1)**m * C * (1 - z**2)**(m/2)


def _Qmm_legacy(m, z, epsrel=1.49e-8):
    """
    Associated Legendre of the second kind with
    same integer order and degree for z in [-1, 1]
    valid up to m = 260
    
    .. math::
    
        Q_m^m(z) &= C (-1)^m (2m - 1)!! (1 - z^2)^{m/2} \\
        C &= \sqrt{\frac{2m+1}{4\pi}{(m-m)!}{(m+m)!}}
        
    Cohl et al. 2020, doi:10.3390/sym12101598
    Eqn. 99
        
    Parameters
    ----------
    m : int
        Order and degree
    z : float
        Legendre argument [-1, 1]
        If passed as an array, z must be
        ordered in ascending order
    epsrel : float
        Relative error for quad integration

    Notes
    -----
    A closed form expression is only available
    for half integer m. For integer m we perform
    numerical integration.
    """
    # Qmm normalization
    C = 2**m  * np.cos(np.pi * m)

    # ortho normalization
    C *= np.sqrt((2 * m + 1) / (4 * np.pi))

    # stable factorial normalization: m! / sqrt((2m)!)
    n1 = float(factorialk(m-0, 4))
    n2 = float(factorialk(m-1, 4))
    n3 = float(factorialk(m-2, 4))
    n4 = float(factorialk(m-3, 4))
    d1 = float(factorialk(2*m-0, 4))
    d2 = float(factorialk(2*m-1, 4))
    d3 = float(factorialk(2*m-2, 4))
    d4 = float(factorialk(2*m-3, 4))
    f = n1 / np.sqrt(d1)
    f *= n2 / np.sqrt(d2)
    f *= n3 / np.sqrt(d3)
    f *= n4 / np.sqrt(d4)
    C *= f

    # ensure z is an array
    if isinstance(z, (int, float)):
        z = np.atleast_1d(z)

    # perform w integration from [0, z]
    #@jit(nopython=True)
    def integrand(w, m):
        return 1 / (1 - w**2)**(m + 1)
    Q = np.zeros_like(z, dtype=float)

    # perform integral for z < 0
    neg_z = np.where(z < 0)[0]
    if len(neg_z) > 0:
        # order from 0 -> -1
        zs = z[neg_z][::-1]
        # iterate over zs
        z_start = 0
        _int = 0
        for i, z_end in enumerate(zs):
            _int += quad(integrand, z_start, z_end, args=(m,), epsrel=epsrel)[0]
            Q[neg_z[::-1][i]] = _int
            z_start = z_end
    
    # perform integral for z > 0
    pos_z = np.where(z > 0)[0]
    if len(pos_z) > 0:
        # order from 0 -> 1
        zs = z[pos_z]
        # iterate over zs
        z_start = 0
        _int = 0
        for i, z_end in enumerate(zs):
            _int += quad(integrand, z_start, z_end, args=(m,), epsrel=epsrel)[0]
            Q[pos_z[i]] = _int
            z_start = z_end

    # multiply normalization
    Q *= C * (1 - z**2)**(m/2)

    if len(Q) == 1:
        Q = Q[0]

    return Q

