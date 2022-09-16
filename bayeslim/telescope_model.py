"""
Module for instrument models and relevant functions
"""
import torch
import numpy as np
from astropy import units, constants
from astropy.coordinates import AltAz, EarthLocation, ICRS
from astropy.time import Time
from scipy import special
import copy
import itertools

from . import utils, beam_model
from .utils import _float, _cfloat


D2R = utils.D2R


class TelescopeModel:

    def __init__(self, location, device=None):
        """
        A telescope model for performing
        coordinate conversions

        Parameters
        ----------
        location : tuple
            3-tuple location of the telescope in geodetic
            frame (lon, lat, alt), where lon, lat in degrees.
        """
        # setup telescope location in geocentric (ECEF) frame
        self.location = location
        self.tloc = EarthLocation.from_geodetic(*location)

        # setup coordinate conversion cache
        self.conv_cache = {}
        self.device = device

    def hash(self, time, ra, dec):
        """
        Create a hash from time, ra, and dec arrays

        Parameters
        ----------
        time : float
            Observation time in Julian Date
        ra : tensor or ndarray
            right ascension in degrees [J2000]
        dec : tensor or ndarray
            declination in degrees [J2000]

        Returns
        -------
        hash : int
            A unique integer hash
        """
        if isinstance(ra, torch.Tensor):
            ra = ra.numpy()
        if isinstance(dec, torch.Tensor):
            dec = dec.numpy()

        return hash((time, ra[0], dec[0], ra[-1], dec[-1]))

    def clear_cache(self, key=None):
        """Clear conversion cache, or just a single
        key from the cache

        Parameters
        ----------
        key : int, optional
            A unique hash to look for. If None,
            clear the whole cache
        """
        if key is None:
            del self.conv_cache
            self.conv_cache = {}
        else:
            del self.conv_cache[key]

    def eq2top(self, time, ra, dec, store=False):
        """
        Convert equatorial coordinates to topocentric (aka AltAz).
        Pull from the conv_cache if saved. Take care of how
        the hashing is performed in this function!

        Parameters
        ----------
        time : float
            Observation time in Julian Date
        ra : tensor or ndarray
            right ascension in degrees [J2000]
        dec : tensor or ndarray
            declination in degrees [J2000]
        store : bool, optional
            If True, store output to cache.

        Returns
        -------
        alt, az : array_like
            alitude angle and azimuth vectors [degrees]
            oriented along East-North-Up frame
        """
        # create hash
        h = self.hash(time, ra, dec)

        # pull from cache
        if h in self.conv_cache:
            return self.conv_cache[h]

        # if not present, perform conversion
        ra, dec = utils.tensor2numpy(ra), utils.tensor2numpy(dec)
        dtype = ra.dtype if isinstance(ra, torch.Tensor) else None
        angs = eq2top(self.tloc, time, ra, dec)
        angs = torch.as_tensor(np.asarray(angs), device=self.device, dtype=dtype)

        # and save to cache
        if store:
            self.conv_cache[h] = angs

        return angs

    def push(self, device):
        """push cache to device"""
        dtype = isinstance(device, torch.dtype)
        if not dtype:
            for key, angs in self.conv_cache.items():
                self.conv_cache[key] = angs.to(device)


class ArrayModel(utils.PixInterp, utils.Module):
    """
    A model for antenna layout and the baseline fringe

    Two kinds of caching are allowed
    1. caching the unit pointing s vector (default)
        This recomputes the fringes exactly
    2. caching the fringe on the sky
        This interpolates an existing fringe

    Note that for performance reasons in the RIME object,
    antpos is always kept on the CPU. If the model is pushed
    to the GPU, the antpos tensor becomes pinned to speed
    up CPU -> GPU transfer.
    """
    def __init__(self, antpos, freqs, pixtype='healpix', parameter=False,
                 device=None, cache_s=True, cache_f=False, cache_f_angs=None,
                 cache_blv=True, redtol=0.1, name=None, red_kwargs={}, pix_kwargs={}):
        """
        A model of an interferometric array

        Parameters
        ----------
        antpos : dict
            Dictionary of antenna positions.
            Keys are antenna integers, values
            are len-3 float arrays with antenna
            position in East-North-Up coordinates,
            centered at telescope location.
        freqs : tensor
            Frequencies to evaluate fringe [Hz]
        pixtype : str, optional
            If using the interpolation functionality of PixInterp
            parent class, this is the pixelization scheme.
            Only used if cache_f is True
        parameter : bool, optional
            If True, antenna positions become a parameter
            to be fitted. If False (default) they are held fixed.
        device : str, optional
            device to hold baseline vector and fringe term
            none is cpu.
        cache_s : bool, optional
            If True, cache the pointing unit vector s computation
            in the fringe-term for each sky model zen, az combination,
            but re-compute the fringe for each gen_fringe() call.
            If cache_f then cache_s is ignored.
        cache_f : bool, optional
            Cache the fringe response, shape (Nfreqs, Npix),
            for each baseline. Repeated calls to gen_fringe()
            for the same baseline interpolates the cached fringe.
            This only saves time and memory (if autodiff) if Ntimes >> 1.
        cache_f_angs : tensor, optional
            If cache_f, these are the sky angles (zen, az, [deg]) to
            evaluate the fringe at and then cache.
        cache_blv : bool, optional
            If True, cache the baseline vectors to prevent repeated
            calls to self.get_antpos(). Must be False if parameter=True.
        redtol : float, optional
            If parameter is False, then redundant baseline groups
            are built. This is the bl vector redundancy tolerance [m]
        name : str, optional
            Name for this object, stored as self.name
        red_kwargs : dict, optional
            Keyword arguments to pass to build_reds()
        pix_kwargs : dict, optional
            Keyword arguments to pass to PixInterp
        """
        # init Module
        super(utils.PixInterp, self).__init__(name=name)
        # init PixInterp
        npix = cache_f_angs.shape[-1] if cache_f else None
        super().__init__(pixtype, device=device, **pix_kwargs)
        # set location metadata
        self.ants = sorted(antpos.keys())
        self._ant_idx = {a: self.ants.index(a) for a in self.ants}
        self.antpos = torch.as_tensor([antpos[a] for a in self.ants], dtype=_float(), device='cpu')
        self.cache_s = cache_s if not cache_f else False
        self.cache_f = cache_f
        self.cache_f_angs = cache_f_angs
        self.cache_blv = cache_blv
        self.clear_cache()
        self.parameter = parameter
        self.redtol = redtol
        self.device = device
        self.set_freqs(freqs)
        if parameter:
            # make ant vecs a parameter if desired
            self.antpos = torch.nn.Parameter(self.antpos)

        # TODO: enable clearing of fringe cache if parameter is True
        if cache_f:
            assert parameter is False, "fringe caching not yet implemented for parameter = True"

        if parameter is False:
            # build redundant info
            (self.reds, self.redvecs, self.bl2red, self.bls, self.redlens, self.redangs,
             self.redtags) = build_reds(antpos, redtol=redtol, **red_kwargs)

        self.push(device)

    def get_antpos(self, ant):
        """
        Get antenna vector

        Parameters
        ----------
        ant : int
            antenna number in self.ants
        """
        if isinstance(ant, list):
            return self.antpos[[self._ant_idx[a] for a in ant]]
        else:
            return self.antpos[self._ant_idx[ant]]

    def match_bl_len(self, bl, bls):
        """
        Given a baseline and a list of baselines,
        figure out if any bls match bl in length
        within redtol. If so return the angle
        from bl to the bls match [deg], otherwise False.
        Note all baselines must appear in self.bls.

        Parameter
        ---------
        bl : tuple
            Antenna pair, e.g. (1, 2)
        bls : list
            List of ant-pairs, e.g. [(1, 2), (3, 5), ...]

        Returns
        -------
        float
            Angle from bl to match in bls [deg]
        bool or tuple
            bl tuple if match, else False
        """
        assert not self.parameter, "parameter must be False"
        match = False
        ang = 0
        bllen = self.redlens[self.bl2red[bl]]
        blang = self.redangs[self.bl2red[bl]]
        for _bl in bls:
            i = self.bl2red[_bl]
            l, a = self.redlens[i], self.redangs[i]
            if np.isclose(bllen, l, atol=self.redtol):
                ang = a - blang
                match = _bl
                break

        return ang, match

    def set_freqs(self, freqs):
        """set frequency array"""
        self.freqs = torch.as_tensor(freqs, dtype=_float(), device=self.device)
        self.clear_cache()

    def clear_cache(self):
        """
        Overloads PixInterp.clear_cache
        to clear both PixInterp and fringe_cache
        """
        # this is PixInterp cache
        self.interp_cache = {}
        # this is fringe cache
        self.cache = {}

    def _fringe(self, bl, zen, az, conj=False):
        """compute fringe term. Returns fringe tensor
        of shape (Nbls, Nfreqs, Nzen)
        """
        if not isinstance(bl, list):
            bl = [bl]
        zen, az = torch.as_tensor(zen), torch.as_tensor(az)
        # get angle and baseline hash
        s_h = utils.arr_hash(zen), utils.arr_hash(az)  # used for s caching
        b_h = utils.arr_hash(bl)
        key = (b_h, s_h)
        # s_h is used for pointing vector caching
        # b_h is used for bl_vec caching
        # key is used for complex fringe caching
        if s_h not in self.cache and key not in self.cache:
            # compute the pointing vector at each sky location
            _zen = zen * D2R
            _az = az * D2R
            s = torch.zeros(3, len(zen), dtype=_float(), device=self.device)
            # az is East of North
            s[0] = torch.sin(_zen) * torch.sin(_az)  # x
            s[1] = torch.sin(_zen) * torch.cos(_az)  # y
            s[2] = torch.cos(_zen)                   # z
            if self.cache_s:
                self.cache[s_h] = s
        elif self.cache_s:
            s = self.cache[s_h]
        else:
            # this is fringe caching
            pass

        if key not in self.cache:
            # get baseline vectors: shape (Nbls, 3)
            if self.cache_blv:
                # antpos is not a parameter, so cache the baseline vectors
                if b_h not in self.cache:
                    # construct bl vector
                    bl_vec = self.get_antpos([b[1] for b in bl]) - self.get_antpos([b[0] for b in bl])
                    bl_vec = bl_vec.to(self.device)
                    self.cache[b_h] = bl_vec
                else:
                    bl_vec = self.cache[b_h]
            else:
                # antpos is a parameter, so cache the antpos indexing tensors
                if b_h not in self.cache:
                    ant1 = torch.as_tensor([self.ants.index(b[0]) for b in bl])
                    ant2 = torch.as_tensor([self.ants.index(b[1]) for b in bl])
                    self.cache[b_h] = (ant1, ant2)
                else:
                    ant1, ant2 = self.cache[b_h]
                # generate bl_vector (note that antpos is on CPU always but is pinned)
                bl_vec = torch.index_select(self.antpos, 0, ant2) - torch.index_select(self.antpos, 0, ant1)
                bl_vec = bl_vec.to(self.device)

            # get fringe pattern: shape (Nbls, Nfreqs, Npix)
            sign = -2j if conj else 2j
            f = torch.exp(sign * np.pi * (bl_vec @ s)[:, None, :] / 2.99792458e8 * self.freqs[:, None])

            # if fringe caching, store the full fringe (this is large in memory!)
            if self.cache_f:
                self.cache[key] = f
        else:
            # interpolate cached fringe (not generally recommended)
            f = self.interp(self.cache[key], zen, az)

        return f

    def gen_fringe(self, bl, zen, az, conj=False):
        """
        Generate a fringe-response given a baseline and
        collection of sky angles

        Parameters
        ----------
        bl : 2-tuple or list of such
            Baseline tuple, specifying the participating
            antennas from self.ants, e.g. (1, 2)
        zen : tensor
            Zenith angle [degrees] of shape (Npix,).
        az : tensor
            Azimuth [degrees] of shape (Npix,).
        conj : bool, optional
            If True, conjugate complex fringe

        Returns
        -------
        fringe : tensor
            Fringe response of shape (Nbls, Nfreqs, Npix)
        """
        # do checks for fringe caching
        if self.cache_f:
            # first figure out if a bl with same len is cached
            if not self.parameter and not isinstance(bl, list):
                # this only works if feeding a single bl antpair
                ang, match = self.match_bl_len(bl, self.cache.keys())
                if match:
                    # found a match, so swap this bl w/ existing
                    # bl but rotate az angles
                    bl = match
                    az = (az + ang) % 360

            # if no cache for this bl, first generate it
            if bl not in self.cache:
                self._fringe(bl, *self.cache_f_angs, conj=conj)

        return self._fringe(bl, zen, az, conj=conj)

    def apply_fringe(self, fringe, sky):
        """
        Apply a fringe matrix to a representation
        of the sky.

        Parameters
        ----------
        fringe : tensor
            Holds the fringe response for a set of
            bls of shape (Nbls, Nfreqs, Npix)
        sky : tensor
            Holds the sky coherency matrix, which generally
            has a shape of (Nvec, Nvec, ..., Nfreqs, Npix)

        Returns
        -------
        psky : tensor
            perceived sky, having mutiplied fringe with sky
            of shape (Nvec, Nvec, Nbls, Nfreqs, Npix)
        """
        # move sky to fringe device
        sky = sky.to(self.device)

        # give sky Nbls dimension if not present
        if sky.ndim == 4:
            sky = sky[:, :, None]

        # multiply in fringe
        psky = sky * fringe

        return psky

    def push(self, device):
        """push model to a new device or dtype"""
        # setting antpos like this ensures it stays a Parameter
        # if it is to begin with
        dtype = isinstance(device, torch.dtype)
        if dtype:
            self.antpos = utils.push(self.antpos, device)
        if utils.device(device) != utils.device('cpu'):
            self['antpos'] = self.antpos.pin_memory()
        # use PixInterp push for its cache
        super().push(device)
        self.freqs = self.freqs.to(device)
        if not dtype: self.device = device
        # push fringe cache
        for k in self.cache:
            if isinstance(self.cache[k], torch.Tensor):
                self.cache[k] = utils.push(self.cache[k], device)

    def get_bls(self, uniq_bls=False, keep_autos=True,
                min_len=None, max_len=None,
                min_EW=None, max_EW=None, min_NS=None, max_NS=None,
                min_deg=None, max_deg=None):
        """
        Query all baselines associated with this array. Optionally
        select on baseline vector

        Parameters
        ----------
        uniq_bls : bool, optional
            If True, return only the first baseline
            in each redundant group. Otherwise return
            all physical baselines (default)
        keep_autos : bool, optional
            If True (default) return auto-correlations.
            Otherwise remove them.
        min_len : float, optional
            Sets minimum baseline length [m]
        max_len : float, Optional
            Sets maximum baseline length [m]
        min_EW : float, optional
            Sets min |East-West length| [m]
        max_EW : float, optional
            Sets max |East-West length| [m]
        min_NS : float, optional
            Sets min |North-South length| [m]
        max_NS : float, optional
            Sets max |North-South length| [m]
        min_deg : float, optional
            Sets min baseline angle (north of east) [deg]
        max_deg : float, optional
            Sets max baseline angle (north of east) [deg]

        Returns
        -------
        list
        """
        # get lists
        reds = copy.deepcopy(self.reds)
        redlens = copy.deepcopy(self.redlens)
        redangs = copy.deepcopy(self.redangs)
        redvecs = copy.deepcopy(self.redvecs)

        # pop autos
        if not keep_autos:
            match = np.isclose(redlens, 0, atol=self.redtol)
            assert match.any()
            i = np.where(match)[0][0]
            reds.pop(i)
            redlens.pop(i)
            redangs.pop(i)
            redvecs.pop(i)

        # baseline cuts
        keep = np.ones(len(reds), dtype=bool)
        if min_len is not None:
            keep = keep & (np.array(redlens) >= min_len)
        if max_len is not None:
            keep = keep & (np.array(redlens) <= max_len)
        if min_EW is not None:
            keep = keep & (np.abs(redvecs)[:, 0] >= min_EW)
        if max_EW is not None:
            keep = keep & (np.abs(redvecs)[:, 0] <= max_EW)
        if min_NS is not None:
            keep = keep & (np.abs(redvecs)[:, 1] >= min_NS)
        if max_NS is not None:
            keep = keep & (np.abs(redvecs)[:, 1] <= max_NS)
        if min_deg is not None:
            keep = keep & (np.array(redangs) >= min_deg)
        if max_deg is not None:
            keep = keep & (np.array(redangs) <= max_deg)

        keep = np.where(keep)[0]
        reds = [reds[i] for i in keep]

        # keep only uniq baselines
        if uniq_bls:
            reds = [red[:1] for red in reds]

        return utils.flatten(reds)


def eq2top(location, time, ra, dec):
    """
    Convert equatorial ICRS (J200) coordinates
    RA, Dec in degrees to topocentric coordinates
    alitude angle (alt) and azimuth in degrees

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation object
        Location of telescope
    time : float
        Observation Julian Date
    ra, dec : array
        Right ascension and declination [deg]

    Returns
    -------
    altitude, azimuth : array
        alitude angle and az in degrees

    Notes
    -----
    zenith angle (zen) is 90 - alt
    """
    # if ra/dec are tensors, then this is a lot slower
    if isinstance(ra, torch.Tensor):
        ra = ra.detach().numpy()
    if isinstance(dec, torch.Tensor):
        dec = dec.detach().numpy()
    altaz = AltAz(location=location, obstime=Time(time, format='jd'))
    icrs = ICRS(ra=ra * units.deg, dec=dec * units.deg)
    out = icrs.transform_to(altaz)

    return out.alt.deg, out.az.deg


def top2eq(location, time, alt, az):
    """
    Convert topocentric (AltAz) coordinates
    of altitude angle and azimuth [deg] to
    ICRS (J200) coordinates of RA, Dec [deg].

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation object
        Location of telescope
    time : float
        Observation Julian Date
    alt, az : array
        altitude and azimuth [deg]

    Returns
    -------
    ra, dec : array
        ra and dec in [deg]
    """
    # if alt/az are tensors, then this is a lot slower
    if isinstance(alt, torch.Tensor):
        alt = alt.detach().numpy()
    if isinstance(az, torch.Tensor):
        az = az.detach().numpy()
    altaz = AltAz(location=location, obstime=Time(time, format='jd'),
                  alt=alt * units.deg, az=az * units.deg)
    icrs = ICRS()
    out = altaz.transform_to(icrs)

    return out.ra.deg, out.dec.deg


def JD2RA(tloc, jd):
    """
    Convert JD to Equatorial J2000
    Right Ascension [deg] at telescope
    zenith pointing given a telescope
    location and JD

    Parameters
    ----------
    tloc : astropy.EarthLocation
    jd : float
        Julian Date

    Returns
    -------
    float
        right ascension at zenith [deg]
    """
    ra, dec = top2eq(tloc, jd, 90, 0)

    return ra


def JD2LST(jd, longitude):
    """
    Convert JD to Local Sidereal Time (LST)
    aka, the Right Ascension in the current
    epoch, rather than J2000 epoch.

    Parameters
    ----------
    jd : float
        Julian Date
    longitude : float
        Observers longitude [deg]

    Returns
    -------
    float
        LST [radian]
    """
    t = Time(jd, format='jd', scale='utc')
    return t.sidereal_time('apparent', longitude=longitude * units.deg).radian


def build_reds(antpos, bls=None, redtol=1.0, min_len=None, max_len=None,
               min_EW_len=None, exclude_reds=None):
    """
    Build redundant groups

    Parameters
    ----------
    antpos : dict
        Antenna positions in ENU frame [meters]
        keys are ant integers, values are len-3 arrays
        holding antenna position in ENU (east-north-up)
    bls : list, optional
        List of baselines to sort into redundant groups using
        antenna positions. Default is to compute all possible
        redundancies from antpos. Uses first baseline in numerical
        order as representative redundant baseline.
    redtol : float, optional
        Redunancy tolerance [m]
    min_len : float, optional
        Minimum baseline length cut
    max_len : float, optional
        Maximum baseline length cut
    min_EW_len : float, optional
        Minimum projected East-West baseline length cut
    exclude_reds : list, optional
        A list of baseline tuple(s) whose corresponding
        redundant type will be excluded from the output.
        E.g. For a standard HERA array, [(0, 1)] would exclude
        all 14.6-meter East-West baselines.

    Returns
    -------
    reds : list
        Nested set of redundant baseline lists
    redvecs : list
        Baseline vector for each redundant group
    bl2red : dict
        Maps baseline tuple to redundant group index
    bls : list
        List of all baselines
    redlens : list
        List of redundant group baseline lengths
    redangs : list
        List of redundant group baseline orientation, North of East [deg]
    redtags : list
        List of unique baseline length and angle str
    """
    # get antenna names and vectors
    ants = list(antpos.keys())
    antvec = [antpos[a] for a in ants]

    # get baselines
    if bls is None:
        bls = [(a, a) for a in ants] + list(itertools.combinations(ants, 2))

    # get excluded baseline vectors
    if exclude_reds is not None:
        exclude_reds = [utils.tensor2numpy(antpos[bl[1]] - antpos[bl[0]]) for bl in exclude_reds]

    # iterate over bls
    reds, rvec, bl2red = [], [], {}
    lens, angs, tags = [], [], []
    k = 0
    for bl in bls:
        # get baseline vector
        blvec = utils.tensor2numpy(antpos[bl[1]] - antpos[bl[0]])
        bllen = np.linalg.norm(blvec)

        # determine if we should skip this baseline
        if min_len is not None and bllen < min_len:
            continue
        if max_len is not None and bllen > max_len:
            continue
        if min_EW_len is not None and blvec[0] < min_EW_len:
            continue
        if exclude_reds is not None:
            # add and subtract to account for accidental conjugation
            match1 = [np.linalg.norm(blv - blvec) for blv in exclude_reds]
            match2 = [np.linalg.norm(blv + blvec) for blv in exclude_reds]
            if np.isclose(match1, 0, atol=redtol).any() or np.isclose(match2, 0, atol=redtol).any():
                continue

        # check if this is a unique bl
        rgroup = None
        for i, blv in enumerate(rvec):
            ## TODO: handle conjugated baselines
            if np.linalg.norm(blv - blvec) < redtol:
                rgroup = i
                break

        if rgroup is None:
            # this a unique group, append to lists
            reds.append([bl])
            rvec.append(blvec)
            # get unique baseline properties
            bllen = np.linalg.norm(blvec)
            # get angle [deg]
            blang = np.arctan2(*blvec[:2][::-1]) * 180 / np.pi
            # ensure angle is purely positive
            if blvec[1] < 0: blang += 180.0
            # if its close to 180 deg within redtol set it to zero
            if np.abs(blvec[1]) < redtol: blang = 0.0
            lens.append(bllen)
            angs.append(blang)
            tags.append("{:03.0f}_{:03.0f}".format(bllen, blang))
            k += 1

        else:
            # this falls into an existing redundant group
            reds[rgroup].append(bl)

    # resort by baseline length
    s = np.argsort(lens)
    reds = [sorted(reds[i]) for i in s]
    rvec = [rvec[i] for i in s]
    lens = [lens[i] for i in s]
    angs = [angs[i] for i in s]
    tags = [tags[i] for i in s]
    bls = utils.flatten(reds)

    # setup bl2red
    bl2red = {}
    for bl in bls:
        for i, rg in enumerate(reds):
            if bl in rg:
                bl2red[bl] = i
                break

    return reds, rvec, bl2red, bls, lens, angs, tags
