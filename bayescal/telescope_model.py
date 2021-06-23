"""
Module for instrument models and relevant functions
"""
import torch
import numpy as np
from astropy import units, time, constants
from astropy.coordinates import AltAz, EarthLocation, ICRS
from scipy import special
import copy

from . import utils, beam_model


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
            frame (lon, lat) in degrees.
        """
        # setup telescope location in geocentric (ECEF) frame
        self.location = location
        self.tloc = EarthLocation.from_geodetic(*location)

        # setup coordinate conversion cache
        self.conv_cache = {}
        self.device = device

    def hash(self, obs_jd, sky):
        """
        Create a hash from an observation time
        and sky model object. It is assumed that
        the ra, dec arrays of the sky model object
        are immutable.

        Parameters
        ----------
        obs_jd : float
            observation julian date (e.g. 2458101.245501)
        sky : sky model object
            A sky object (e.g. PointSourceModel, DiffuseModel)

        Returns
        -------
        hash : int
            A unique integer hash
        """
        return hash((obs_jd, sky))

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

    def eq2top(self, obs_jd, ra, dec, sky=None, store=False):
        """
        Convert equatorial coordinates to topocentric (aka AltAz).
        Pull from the conv_cache if saved.

        Parameters
        ----------
        obs_jd : float
            Observation time in Julian Date
        ra : tensor or ndarray
            right ascension in degrees [J2000]
        dec : tensor or ndarray
            declination in degrees [J2000]
        sky : SkyBase subclass or int
            The SkyBase subclass object, or some unique
            identifier of that specific instantiated
            sky model. Used for caching.
        store : bool, optional
            If True, store output to cache.

        Returns
        -------
        alt, az : array_like
            alitude angle and azimuth vectors [degrees]
            oriented along East-North-Up frame
        """
        # create hash
        h = self.hash(obs_jd, sky)

        # pull from cache
        if h in self.conv_cache:
            return self.conv_cache[h]

        # if not present, perform conversion
        ra, dec = utils.tensor2numpy(ra), utils.tensor2numpy(dec)
        dtype = ra.dtype if isinstance(ra, torch.Tensor) else None
        angs = eq2top(self.tloc, obs_jd, ra, dec)
        angs = torch.as_tensor(angs, device=self.device, dtype=dtype)

        # and save to cache
        if store:
            self.conv_cache[h] = angs

        return angs

    def push(self, device):
        """push cache to device"""
        for key, angs in self.conv_cache.items():
            self.conv_cache[key] = angs.to(device)


class ArrayModel(torch.nn.Module):
    """
    A model for antenna layout
    """
    def __init__(self, antpos, freqs, parameter=False, dtype=torch.float32, device=None,
                 cache_s=True):
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
        parameter : bool, optional
            If True, antenna positions become a parameter
            to be fitted. If False (default) they are held fixed.
        dtype : torch dtype
            Data type of baseline vectors in the fringe term
            The complex fringe is 2x the baseline dtype
            i.e. bl_vec [float32] -> fringe [complex64]
        device : str, optional
            device to hold baseline vector and fringe term
            none is cpu.
        cache_s : bool, optional
            If True, cache the pointing unit vector s computation
            in the fringe-term for each sky model zen, az combination
        """
        # init
        super().__init__()
        # set location metadata
        self.dtype = dtype
        self.device = device
        self.ants = sorted(antpos.keys())
        self.antpos = torch.as_tensor([antpos[a] for a in self.ants], dtype=dtype, device=device)
        self.freqs = torch.as_tensor(freqs, dtype=dtype, device=device)
        self.cache_s = cache_s
        self.cache = {}
        if parameter:
            # make ant vecs a parameter if desired
            self.antpos = torch.nn.Parameter(self.antpos)

    def get_antpos(self, ant):
        """
        Get antenna vector

        Parameters
        ----------
        ant : int
            antenna number in self.ants
        """
        return self.antpos[self.ants.index(ant)]

    def gen_fringe(self, bl, zen, az):
        """
        Compute a fringe term given a representation kind
        and an antenna pair (baseline).

        Parameters
        ----------
        bl : 2-tuple
            Baseline tuple, specifying the participating
            antennas from self.ants
        zen : tensor
            Zenith angle [degrees] of shape (Npix,).
            Used of kind of 'pixel' or 'point'
        az : tensor
            Azimuth [degrees] of shape (Npix,).
            Used for kind of 'pixel' or 'point'

        Returns
        -------
        fringe : tensor
            Fringe response of shape (Nfreqs, Npix)
            or (Nfreqs, Nalm)
        """
        # check hash if present
        key = (hash(bl), utils.zen_hash(zen))
        if self.cache_s and key in self.cache:
            s = self.cache[key]
        else:
            # compute the pointing vector at each sky location
            bl_vec = self.get_antpos(bl[1]) - self.get_antpos(bl[0])
            zen = zen * D2R
            az = az * D2R
            s = torch.zeros(3, len(zen), dtype=self.dtype, device=self.device)
            # az is East of North
            s[0] = torch.sin(zen) * torch.sin(az)  # x
            s[1] = torch.sin(zen) * torch.cos(az)  # y
            s[2] = torch.cos(zen)                  # z

        return torch.exp(2j * np.pi * (bl_vec @ s) / 2.99792458e8 * self.freqs[:, None])

    def apply_fringe(self, fringe, sky, kind):
        """
        Apply a fringe matrix to a representation
        of the sky.

        Parameters
        ----------
        fringe : tensor
            Holds the fringe response for a given baseline
            vector across frequency. If kind = 'point'
            or 'pixel' its shape is (Nfreqs, Npix)
        sky : tensor
            Holds the sky representatation. If kind = 'point'
            or 'pixel', its shape is (Npol, Npol, Nfreqs, Npix)
            elif 'alm', its shape is (Npol, Npol, Nfreqs, Nalm)
        kind : str
            Kind of fringe and sky model. 
            One of either ['point', 'pixel', 'alm']

        Returns
        -------
        psky : tensor
            perceived sky, having mutiplied fringe with sky
        """
        if kind in ['point', 'pixel']:
            psky = sky * fringe

        elif kind == 'alm':
            raise NotImplementedError

        else:
            raise ValueError("{} not recognized".format(kind))

        return psky

    def push(self, device):
        """push parameters to a new device"""
        self.antpos = utils.push(self.antpos, device)
        self.freqs = self.freqs.to(device)
        self.device = device


class RIME(torch.nn.Module):
    """
    Performs the sky integral of the radio interferometric
    measurement equation (RIME) to produce the interferometric
    visibilities between antennas p and q.

    .. math::

        V_{jk} = \int_{4\pi}d \Omega\ A_p(\hat{s})
                  I(\hat{s}) A_q^\dagger K_{pq}

    where K is the interferometric fringe term.
    """
    def __init__(self, telescope, beam, ant2model, array, bls, obs_jds, freqs,
                 vis_dtype=torch.cfloat, device=None):
        """
        RIME object. Takes as input a model
        of the sky brightness, passes it through
        a primary beam model (optional) and a
        fringe model, and then sums
        across the sky to produce the visibilities.

        If this is being used only for a forward model (i.e. no gradient
        calculation) you can reduce the memory load by either
        ensuring all params have parameter=False, or by running
        the forward() call in a torch.no_grad() context.

        Parameters
        ----------
        telescope : TelesscopeModel object
            Used to set the telescope location.
        beam : BeamModel object, optional
            A model of the directional and frequency response of the
            antenna primary beam. Default is a tophat response.
        ant2model : dict
            Dict of integers that map each antenna number in array.ants
            to a particular index in the beam model output from beam.
            E.g. {10: 0, 1: 0, 12: 0} for 3-antennas [10, 11, 12] with
            1 shared beam model or {10: 0, 11: 1, 12: 2} for 3-antennas
            [10, 11, 12] with different 3 beam models.
        array : ArrayModel object
            A model of the telescope location and antenna positions
        bls : list of 2-tuples
            A list of baseline tuples, or antenna-pair tuples,
            whose elements correspond to the participating antenna
            numbers in array.ants for each baseline.
            If array.ants = [1, 3, 5], then bls could be, for e.g.,
            bls = [(1, 3), (1, 5), (3, 5)]
        obs_jds : tensor
            Array of observational times in Julian Date
        freqs : tensor
            Array of observational frequencies [Hz]
        vis_dtype : torch dtype, optional
            Data type of output visibilities.
        """
        super().__init__()
        self.telescope = telescope
        self.beam = beam
        self.ant2model = ant2model
        self.array = array
        self.bls = bls
        self.obs_jds = obs_jds
        self.Ntimes = len(obs_jds)
        self.vis_dtype = vis_dtype
        self.device = device
        self.Nbls = len(self.bls)
        self.freqs = freqs
        self.Nfreqs = len(freqs)

    def forward(self, sky_components):
        """
        Forward pass sky components through the beam,
        the fringes, and then integrate to
        get the visibilities.

        Parameters
        ----------
        sky_components : list of sky component dictionaries
            Each dictionary is an output from a
            SkyBase subclass, containing:
                'kind' : str, kind of sky model
                'sky' : tensor, sky representation
                Extra kwargs given 'kind', including possibly
                    'angs' : tensor, optional, RA and Dec [deg] of pixels
                    'lms' : tensor, optional, l and m modes of a_lm coeffs

        Returns
        -------
        vis : tensor
            Measured visibilities, shape (Npol, Npol, Nbls, Ntimes, Nfreqs)
        """
        assert isinstance(sky_components, list)
        # initialize visibility tensor
        Npol = self.beam.Npol
        vis = torch.zeros((Npol, Npol, self.Nbls, self.Ntimes, self.Nfreqs),
                          dtype=self.vis_dtype, device=self.device)

        # clear pre-computed beam for YlmResponse type
        if self.beam.R.__class__ == beam_model.YlmResponse:
            self.beam.R.clear_beam()

        # iterate over sky components
        for i, sky_comp in enumerate(sky_components):

            # setup visibility for this sky component
            sky_vis = torch.zeros((Npol, Npol, self.Nbls, self.Ntimes, self.Nfreqs),
                                  dtype=self.vis_dtype, device=self.device)

            kind = sky_comp['kind']
            sky = sky_comp['sky']

            # iterate over observation times
            for j, obs_jd in enumerate(self.obs_jds):

                # get beam tensor
                if kind in ['pixel', 'point']:
                    # convert sky pixels from ra/dec to alt/az
                    ra, dec = sky_comp['angs']
                    alt, az = self.telescope.eq2top(obs_jd, ra, dec, sky=kind, store=True)

                    # evaluate beam response
                    zen = utils.colat2lat(alt, deg=True)
                    ant_beams, cut, zen, az = self.beam.gen_beam(zen, az)
                    cut_sky = sky[..., cut]

                elif kind == 'alm':
                    raise NotImplementedError

                # iterate over baselines
                for k, (ant1, ant2) in enumerate(self.bls):
                    self._prod_and_sum(self.beam, ant_beams, cut_sky, ant1, ant2,
                                       kind, zen, az, sky_vis, k, j)

            # explicit add needed for pytorch graph
            vis = vis + sky_vis

        return vis

    def _prod_and_sum(self, beam, ant_beams, cut_sky, ant1, ant2,
                      kind, zen, az, sky_vis, bl_ind, obs_ind):
        """
        Sky product and sum into sky_vis inplace
        """
        # get beam of each antenna
        beam1 = ant_beams[:, :, self.ant2model[ant1]]
        beam2 = ant_beams[:, :, self.ant2model[ant2]]

        # apply beam to sky
        psky = beam.apply_beam(beam1, cut_sky, beam2=beam2)

        # generate fringe
        fringe = self.array.gen_fringe((ant1, ant2), zen, az)

        # apply fringe
        psky = self.array.apply_fringe(fringe, psky, kind)

        # sum across sky
        sky_vis[:, :, bl_ind, obs_ind, :] = torch.sum(psky, axis=-1)


def eq2top(location, obs_jd, ra, dec):
    """
    Convert equatorial ICRS (J200) coordinates
    RA, Dec in degrees to topocentric coordinates
    alitude angle (alt) and azimuth in degrees

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation object
        Location of telescope
    obs_jd : float
        Observation Julian Date
    ra, dec : array
        Right ascension and declination [deg]

    Returns
    -------
    altitude, azimuth : array
        alitude angle and az in degrees

    Notes
    -----
    zenith angle (zen) is alt + 90
    """
    # if ra/dec are tensors, then this is a lot slower
    if isinstance(ra, torch.Tensor):
        ra = ra.detach().numpy()
    if isinstance(dec, torch.Tensor):
        dec = dec.detach().numpy()
    altaz = AltAz(location=location, obstime=time.Time(obs_jd, format='jd'))
    icrs = ICRS(ra=ra * units.deg, dec=dec * units.deg)
    out = icrs.transform_to(altaz)

    return out.alt.deg, out.az.deg


def top2eq(location, obs_jd, alt, az):
    """
    Convert topocentric (AltAz) coordinates
    of altitude angle and azimuth [deg] to
    ICRS (J200) coordinates of RA, Dec [deg].

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation object
        Location of telescope
    obs_jd : float
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
    altaz = AltAz(location=location, obstime=time.Time(obs_jd, format='jd'),
                  alt=alt * units.deg, az=az * units.deg)
    icrs = ICRS()
    out = altaz.transform_to(icrs)

    return out.ra.deg, out.dec.deg


