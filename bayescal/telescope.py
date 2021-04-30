"""
Module for instrument models and relevant functions
"""
import torch
import numpy as np
from astropy import units, time, constants
from astropy.coordinates import AltAz, EarthLocation, ICRS
from scipy import special
import healpy, mhealpy
import copy

from . import utils, beam


class TelescopeModel:

    def __init__(self, location, dtype=torch.float32):
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
        self.dtype = dtype

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
        ra : array_like
            right ascension in degrees [J2000]
        dec : array_like
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

        # if not, perform conversion
        angs = eq2top(self.tloc, obs_jd, ra, dec)
        angs = (torch.as_tensor(angs[0], dtype=self.dtype),
                torch.as_tensor(angs[1], dtype=self.dtype))

        # save cache
        if store:
            self.conv_cache[h] = angs

        return angs


class ArrayModel(torch.nn.Module):
    """
    A model for antenna layout
    """
    def __init__(self, antpos, parameter=False, dtype=torch.float32):
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
        parameter : bool, optional
            If True, antenna positions become a parameter
            to be fitted. If False (default) they are held fixed.
        """
        # init
        super().__init__()
        # set location metadata
        self.d2r = np.pi / 180
        self.antpos = antpos
        self.dtype = dtype
        if parameter:
            # make ant positions a parameter if desired
            for ant in self.antpos:
                self.antpos[ant] = torch.nn.Parameter(self.antpos[ant])
        self.ants = sorted(self.antpos.keys())
        self.ant_vecs = np.array([self.antpos[a] for a in self.ants])
        for i, a in enumerate(self.ants):
            setattr(self, '_ant{}_vec'.format(a), self.ant_vecs[i])

    def gen_fringe(self, bl, freqs, kind, zen=None, az=None):
        """
        Compute a fringe term given a representation kind
        and an antenna pair (baseline).

        Parameters
        ----------
        bl : 2-tuple
            Baseline tuple, specifying the participating
            antennas from self.ants
        freqs : tensor
            Frequencies [Hz]
        kind : str
            Kind of fringe model ['point', 'pixel', 'alm']
            Note that 'point' and 'pixel' are functionally
            the same thing.
        zen : array_like, optional
            Zenith angle [degrees] of shape (Npix,).
            Used of kind of 'pixel' or 'point'
        az : array_like, optional
            Azimuth [degrees] of shape (Npix,).
            Used for kind of 'pixel' or 'point'

        Returns
        -------
        fringe : tensor
            Fringe response of shape (Nfreqs, Npix)
            or (Nfreqs, Nalm)
        """
        if kind in ['pixel', 'point']:
            bl_vec = torch.as_tensor(self.antpos[bl[1]] - self.antpos[bl[0]], dtype=self.dtype)
            zen = torch.as_tensor(zen * self.d2r, dtype=self.dtype)
            az = torch.as_tensor(az * self.d2r, dtype=self.dtype)
            s = torch.zeros(3, len(zen), dtype=self.dtype)
            s[0] = torch.sin(zen) * torch.cos(az)
            s[1] = torch.sin(zen) * torch.sin(az)
            s[2] = torch.cos(zen)
            return torch.exp(2j * np.pi * (bl_vec @ s) / 2.99792458e8 * freqs[:, None])

        elif kind == 'alm':
            raise NotImplementedError

        else:
            raise ValueError("kind = {} not recognized".format(kind))

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
                 vis_dtype=torch.cfloat):
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
        ant2model : list
            List of integers that map each antenna in array.ants to
            a particular index in the beam model output from beam.
            E.g. [0, 0, 0] for 3-antennas with 1 beam model
            or [0, 1, 2] for 3-antennas with 3 beam models
            or [0, 0, 1] for 3-antennas with 2 beam models.
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
                    'lms' : tensor, optional, l and m ints of a_lm coeffs

        Returns
        -------
        vis : tensor
            Measured visibilities, shape (Npol, Npol, Nbls, Ntimes, Nfreqs)
        """
        assert isinstance(sky_components, list)
        # initialize visibility tensor
        Npol = self.beam.Npol
        vis = torch.zeros((Npol, Npol, self.Nbls, self.Ntimes, self.Nfreqs),
                          dtype=self.vis_dtype)

        # iterate over sky components
        for i, sky_comp in enumerate(sky_components):

            # setup visibility for this sky component
            sky_vis = torch.zeros((Npol, Npol, self.Nbls, self.Ntimes, self.Nfreqs),
                                  dtype=self.vis_dtype)

            kind = sky_comp['kind']
            sky = sky_comp['sky']

            # transform beam to appropriate sky representation
            beam_model = self.beam.transform_to(kind)

            # get beam response function
            beam_func = beam_model.R(beam_model.params)

            # iterate over observation times
            for j, obs_jd in enumerate(self.obs_jds):

                # get beam tensor
                if kind in ['pixel', 'point']:
                    # convert sky pixels from ra/dec to alt/az
                    alt, az = self.telescope.eq2top(obs_jd, sky_comp['angs'][0], sky_comp['angs'][1],
                                                    sky=kind, store=True)

                    # evaluate beam response
                    zen = 90 - alt
                    beam, cut = beam_model.gen_beam(None, zen, az, beam_func=beam_func)
                    zen, az = zen[cut], az[cut]
                    cut_sky = sky[..., cut]

                elif kind == 'alm':
                    raise NotImplementedError

                # iterate over baselines
                for k, (ant1, ant2) in enumerate(self.bls):
                    self._prod_and_sum(beam_model, beam, cut_sky, ant1, ant2,
                                       kind, zen, az, sky_vis, k, j)

            # explicit add needed for pytorch graph
            vis = vis + sky_vis

        return vis

    def _prod_and_sum(self, beam_model, beam, cut_sky, ant1, ant2,
                      kind, zen, az, sky_vis, bl_ind, obs_ind):
        """
        Sky product and sum into sky_vis inplace
        """
        # get beam of each antenna
        beam1 = beam[:, :, self.ant2model[ant1]]
        beam2 = beam[:, :, self.ant2model[ant2]]

        # apply beam to sky
        psky = beam_model.apply_beam(beam1, cut_sky, beam2=beam2)

        # generate fringe
        fringe = self.array.gen_fringe((ant1, ant2), self.freqs, kind, zen=zen, az=az)

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




