"""
Module for special functions
"""
import numpy as np
from scipy.special import jv, jvp, yv, yvp, factorialk, gamma, gammaln
from mpmath import hyp2f1
from scipy.integrate import quad
import copy
import warnings


def Plm(l, m, z, deriv=False, keepdims=False):
    """
    Associated Legendre function of the first kind
    in hypergeometric form, aka Ferrers function
    DLMF 14.3.1 with interval -1 < z < 1.
    Note 1: this is numerically continued to |z| = 1
    Note 2: stable to integer l = m ~ 800, for all z
    Note 3: stable to float l = m ~ 200, for |z| < .9
    Note 4: the normalization C is actually put into F() for numerical ease

    .. math::

        P_{lm}(z) &= C\left(\frac{z+1}{z-1}\right)^{m/2}F(-l, l+1, 1-\mu, (1-z)/2) \\
        C &= \sqrt{\frac{2l+1}{4\pi}{(l-m)!}{(l+m)!}}

    Parameters
    ----------
    l : float or array_like
        Degree of the associated Legendre function
    m : int or array_like
        Order of the associated Legendre function
    z : float or array_like
        Argument of Legendre function, bounded by |z|<1
    deriv : bool, optional
        If True return derivative wrt z
    keepdims : bool, optional
        If False and (l,m) or z is len 1
        then ravel the output

    Returns
    -------
    array
        Legendre function of first kind at z
    """
    # reshape if needed
    l = np.atleast_1d(l)
    m = np.atleast_1d(m)
    if l.ndim == 1:
        l = l[:, None]
    if m.ndim == 1:
        m = m[:, None]
    # avoid singularity
    z = np.atleast_1d(z).copy()
    s = np.isclose(np.abs(z), 1, rtol=1e-10)
    if np.any(s):
        dz = 1e-8
        z[s] *= (1 - dz)
    # compute Plm
    if not deriv:
        # compute hyper-geometric
        norm = np.abs((1 + z) / (1 - z))**(m/2)
        a, b, c = -l, l+1, 1-m
        P = norm * hypF(a, b, c, (1-z)/2)
        # orthonormalize: sqrt[ (2l+1)/(4pi)*(l-m)!/(l+m)! ]
        C = 0.5 * (np.log(2*l+1) - np.log(4*np.pi) + gammaln(l - m + 1) - gammaln(l + m + 1))
        P *= np.exp(C + gammaln(np.abs(c)+1))
        # handle singularity: 1st order Euler
        if np.any(s):
            P[:, s] += Plm(l, m, z[s], deriv=True, keepdims=True) * dz
        if not keepdims:
            if 1 in P.shape:
                P = P.ravel()
            if P.size == 1:
                P = P[0]
        return P
    # compute derivative
    else:
        norm = 1 / (1 - z**2)
        dPdz = norm * ((m - l - 1) * Plm(l+1, m, z, keepdims=True) + (l+1) * z * Plm(l, m, z, keepdims=True))
        # handle singularity: 1st order Euler
        if np.any(s):
            dPdz[:, s] += (dPdz[:, s] - Plm(l, m, z[s] * (1 - dz), deriv=True, keepdims=True))
        if not keepdims:
            if 1 in dPdz.shape:
                dPdz = dPdz.ravel()
            if dPdz.size == 1:
                dPdz = dPdz[0]
        return dPdz


def Qlm(l, m, z, deriv=False, keepdims=False):
    """
    Associated Legendre function of the second kind
    in hypergeometric form, aka Ferrers function
    DLMF 14.3.12 with interval -1 < z < 1.
    Note 1: this will return infs or nan at |z| = 1
    Note 2: stable to integer l = m ~ 800, for all z
    Note 3: stable to float l = m ~ 200, for |z| < .9

    Parameters
    ----------
    l : float or array_like
        Degree of the associated Legendre function
    m : int or array_like
        Order of the associated Legendre function
    z : float or array_like
        Argument of Legendre function, bounded by |z|<1
    deriv : bool, optional
        If True return derivative wrt z
    keepdims : bool, optional
        If False and (l,m) or z is len 1
        then ravel the output

    Returns
    -------
    array
        Legendre function of second kind at z
    """
    # reshape if needed
    l = np.atleast_1d(l)
    m = np.atleast_1d(m)
    if l.ndim == 1:
        l = l[:, None]
    if m.ndim == 1:
        m = m[:, None]
    # compute Qlm
    z = np.atleast_1d(z).copy()
    if not deriv:
        # compute ortho normalization in logspace
        C = 0.5 * (np.log(2*l+1) - np.log(4*np.pi) + gammaln(l - m + 1) - gammaln(l + m + 1))
        # compute w1 and w2, multiply in normalization
        w1 = 2**m * (1-z**2)**(-m/2) * hypF((-l-m)/2, (l-m+1)/2, .5, z**2)
        w1 *= np.exp(C + gammaln((l+m+1)/2) - gammaln((l-m+2)/2))
        w2 = 2**m * z * (1-z**2)**(-m/2) * hypF((1-l-m)/2, (l-m+2)/2, 3/2, z**2)
        w2 *= np.exp(C + gammaln((l+m+2)/2) - gammaln((l-m+1)/2))
        Q = .5 * np.pi * (-np.sin(.5*(l+m)*np.pi) * w1 + np.cos(.5*(l+m)*np.pi) * w2)
        if not keepdims:
            if 1 in Q.shape:
                Q = Q.ravel()
            if Q.size == 1:
                Q = Q[0]
        return Q
    else:
        norm = 1 / (1 - z**2)
        dQdz = norm * ((m - l - 1) * Qlm(l+1, m, z, keepdims=True) + (l+1) * z * Qlm(l, m, z, keepdims=True))
        if not keepdims:
            if 1 in dQdz.shape:
                dQdz = dQdz.ravel()
            if dQdz.size == 1:
                dQdz = dQdz[0]
        return dQdz


def hypF(a, b, c, z):
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
    
    Notes
    -----
    Note the extra factor of 1/Gamma(|c|+1), which
    is inserted artificially for numerical purposes,
    and must be re-normalized for standard use
    with spherical harmonics.
    """
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
        # divide by (n+1)! and divide by 1/gamma(|c|+1), which are the same thing
        norm -= 2 * gammaln(n+1)
        # compute hypergeometric
        f21 = np.array(np.frompyfunc(lambda *a: float(hyp2f1(*a)), 4, 1)(A+n+1, B+n+1, n+2, z), dtype=float)
        # get the sign of f21 in real space
        f21_sign = np.sign(f21)
        # full term: wait to exp() until as late as possible, multiply by signs
        F[s] = sign * f21_sign * np.exp(np.log(np.abs(f21)) + np.log(z**(n+1)) + norm)
    if np.any(~s):
        # compute assuming c > 0
        A = np.atleast_2d(a[~s])
        B = np.atleast_2d(b[~s])
        C = np.atleast_2d(c[~s])
        f21 = np.array(np.frompyfunc(lambda *a: float(hyp2f1(*a)), 4, 1)(A, B, C, z), dtype=float)
        F[~s] = f21 / gamma(C) / gamma(C+1)

    if len(F) == 1:
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
    Spherical Bessel of the first kind
    
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
    Spherical Bessel of the second kind
    
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

