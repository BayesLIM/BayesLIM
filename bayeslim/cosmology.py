"""
Cosmology conversions module
"""
import numpy as np
from astropy import units, constants
from astropy.cosmology import FlatLambdaCDM, z_at_value

from . import utils


class Cosmology(FlatLambdaCDM):
    """
    Subclass of astropy.FlatLambdaCDM, with additional methods for 21cm intensity mapping.
    """
    def __init__(self, H0=67.7, Om0=0.3075, Ob0=0.0486):
        """
        Subclass of astropy.FlatLambdaCDM, with additional methods for 21cm intensity mapping.
        Default parameters are derived from the Planck2015 analysis.

        Parameters
        ----------
        H0 : float
            Hubble parameter at z = 0

        Om0 : float
            Omega matter at z = 0

        Ob0 : float
            Omega baryon at z = 0. Omega CDM is defined relative to Om0 and Ob0.
        """
        super().__init__(H0, Om0, Tcmb0=2.725, Ob0=Ob0, Neff=3.05, m_nu=[0., 0., 0.06] * units.eV)

        # 21 cm specific quantities
        self._f21 = 1.420405751e9  # frequency of 21cm transition in Hz
        self._w21 = 0.211061140542  # 21cm wavelength in meters

    def H(self, z):
        """
        Hubble parameter at redshift z

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            Hubble parameter km / sec / Mpc
        """
        return super().H(z).value

    def f2z(self, freq):
        """
        Convert frequency to redshift for the 21 cm line

        Parameters
        ----------
        freq : float
            frequency in Hz

        Returns
        -------
        float
            redshift
        """
        return self._f21 / freq - 1

    def z2f(self, z):
        """
        Convert redshift to frequency for the 21 cm line.

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            frequency in Hzv
        """
        return self._f21 / (z + 1)

    def f2r(self, f):
        """
        Convert frequency to LOS comoving distance

        Parameters
        ----------
        f : array
            Frequencies [Hz]
        
        Returns
        -------
        array
            LOS comoving distance [Mpc]
        """
        return self.comoving_distance(self.f2z(f)).value

    def r2f(self, r, **kwargs):
        """
        Convert LOS comoving distance [Mpc] to frequency [Hz.
        See astropy.cosmology.z_at_value for details
        and kwarg description.

        Parameters
        ----------
        r : array
            LOS comoving distance [Mpc]

        Returns
        -------
        array
            Frequencies [Hz]
        """
        if isinstance(r, (float, int)):
            return self.z2f(z_at_value(self.comoving_distance, r * units.Mpc, **kwargs))
        else:
            return np.array([self.r2f(_r) for _r in r])

    def dRperp_dtheta(self, z):
        """
        Conversion factor from angular size (radian) to transverse
        comoving distance (Mpc) at a specific redshift: [Mpc / radians]

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            comoving transverse distance [Mpc]
        """
        return self.comoving_transverse_distance(z).value

    def dRpara_df(self, z):
        """
        Conversion from frequency bandwidth to radial comoving distance at a 
        specific redshift: [Mpc / Hz]

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            Radial comoving distance [Mpc]
        """
        return (1 + z)**2.0 / self.H(z) * (constants.c.value / 1e3) / self._f21

    def X2Y(self, z):
        """
        Conversion from radians^2 Hz -> Mpc^3 at a specific redshift

        Parameters
        ----------
        z : float
            redshift
        
        Returns
        -------
        float
            Mpc^3 / (radians^2 Hz)
        """
        return self.dRperp_dtheta(z)**2 \
             * self.dRpara_df(z)

    def bl_to_kperp(self, z):
        """
        Conversion from baseline length [meters] to
        tranverse k_perp wavevector [Mpc^-1]

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            Conversion factor [Mpc^-1 / meters]
        """
        # Parsons 2012, Pober 2014, Kohn 2018
        return 2 * np.pi / (self.dRperp_dtheta(z) * (constants.c.value / self.z2f(z)))

    def tau_to_kpara(self, z):
        """
        Conversion from delay [seconds] to line-of-sight k_parallel
        wavevector [Mpc^-1]

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            Conversion factor [Mpc^-1 / seconds]
        """
        return 2 * np.pi / self.dRpara_df(z)


def gauss1d(x, scale=1.0, loc=0.0):
    """
    Generate a 1D Gaussian window function
    at provided points given window parameters

    Parameters
    ----------
    x : ndarray
        Locations to evaluate window. If 2D array
        of shape (Nfeatures, Nsamples), window
        normalization is along Nsamples dimension.
    scale : float, optional
        Standard deviation of Gaussian
    loc : float, optional
        Mean of Gaussian

    Returns
    -------
    win : ndarray
        Window values at provided radial points,
        normalized such that sum(win) = 1
    """
    w = np.atleast_2d(np.exp(-0.5 * (x - loc)**2 / scale**2))
    w /= w.sum(axis=1, keepdims=True)
    if w.size == 1:
        w = w[0, 0]

    return w


def cube2lcone(sims, sim_zs, freqs, sim_res, zinterp='nearest',
               interp='nearest', cosmo=None, nside=None, hpx=True,
               roll=None):
    """
    Project simulation cube onto a lightcone.

    Adapted from P. Kittisiwit's cosmotile
    github.com/piyanatk/cosmotile

    Parameters
    ----------
    sims: ndarray or str
        A 3D temperature simulation cube. This can be of shape
        (Nsim_freqs, Npix, Npix, Npix) or (Npix, Npix, Npix).
        This can also be a list of str filepaths to .npy boxes.
    sim_zs : ndarray
        Redshifts of simulation cubes.
        Must be ordered such that they are monotonically
        increasing or decreasing.
    freqs : float
        Frequencies to interpolate to [MHz]
    sim_res : float
        Simulation voxel resolution in cMpc
    zinterp : str, optional
        Line-of-sight (redshift) interpolation method
        ['nearest', 'linear', 'quadratic']
    interp : str, optional
        Spatial cube interpolation method ['nearest', 'linear'].
        See cube2map() for details.
    cosmo : Cosmology object, optional
    nside: integer, optional
        NSIDE of the output HEALPix image.
        Must be a valid NSIDE for HEALPix.
    hpx : bool, optional
        If True return a healpix map of shape
        (Nfreqs, Nhpix) otherwise return
        a box of shape (Nfreqs, Npix, Npix)
    roll : int or tuple, optional
        Before sampling the cube, roll along x, y, and/or z
        axis. If int, all x,y,z are rolled the same, or if
        tuple roll (x, y, z) separately.

    Returns
    -------
    array
        healpix or box lightcone
    array
        comoving distances at desired freqs
    """
    # get cosmology
    if cosmo is None:
        cosmo = Cosmology()

    # load data
    if isinstance(sims, str):
        # load single file
        sims = np.load(sims)

    # Determine the radial comoving distance r to the comoving shell at the
    # frequency of interest.
    f21 = 1420.40575177  # MHz
    if isinstance(freqs, (float, int)):
        freqs = np.array([freqs])
    zs = f21 / freqs - 1
    dcs = np.array([cosmo.comoving_distance(z).value for z in zs])
    # do the same for the simulation frequencies
    if isinstance(sim_zs, (float, int)):
        sim_zs = np.array([sim_zs])
    sim_dcs = np.array([cosmo.comoving_distance(z).value for z in sim_zs])

    # iterate over desired frequencies
    lcone = []
    for i in np.arange(len(freqs)):
        # interpolate frequency axis to get a cube
        if zinterp == 'nearest':
            cube = sims[np.argmin(np.abs(zs[i] - sim_zs))]

        elif zinterp == 'linear':
            # get two nearest cubes in ascending redshift order
            s1, s2 = sorted(np.arange(len(sim_zs))[np.argsort(np.abs(zs[i] - sim_zs))[:2]])

            # y = bx + c
            sim1, sim2 = sims[s1], sims[s2]
            z1, z2 = sim_zs[s1], sim_zs[s2]
            b = (sim2 - sim1) / (z2 - z1)
            c = sim1 - b * z1
            cube = b * zs[i] + c

        elif zinterp == 'quadratic':
            # get three nearest cubes in ascending redshift order
            s1, s2, s3 = sorted(np.arange(len(sim_zs))[np.argsort(np.abs(zs[i] - sim_zs))[:3]])

            # y = ax^2 + bx + c
            sim1, sim2, sim3 = sims[s2], sims[s2], sims[s3]
            z1, z2, z3 = sim_zs[s1], sim_zs[s2], sim_zs[s3]
            a = (sim3 - sim1 - (sim3 - sim2) * (z3 - z1) / (z3 - z2) ) \
                / (z3**2 - z1**2 - (z3 - z1) * (z3**2 - z2**2) / (z3 - z2))
            b = (sim3 - sim1 - a * (z3**2 - z2**2)) / (z3 - z2)
            c = sim1 - a * z1**2 - b * z1
            cube = a * zs[i]**2 + b * zs[i] + c

        # tile and sample onto a map
        m = cube2map(cube, dcs[i], sim_res, nside=nside, hpx=hpx,
                     roll=roll, interp=interp)
        lcone.append(m)

    return np.array(lcone), dcs


def cube2map(cube, dc, sim_res, nside=None, hpx=True, roll=None,
             interp='nearest'):
    """
    Tile a simulation cube and extract
    a map at a fixed comoving distance away
    from the observer using nearest neighbor interpolation

    Parameters
    ----------
    cube : ndarray
        3D temperature field of shape (Npix, Npix, Npix)
        aligned as (x, y, z) where z is the line-of-sight.
    dc : float
        Comoving distance away to sample box
    sim_res : float
        Resolution of cube in cMpc
    nside : int, optional
        NSIDE to sample HEALPix map if hpx is True
    hpx : bool, optional
        If True, sample the cube onto a healpix map,
        otherwise sample it on a box
    roll : int or tuple, optional
        Before sampling the cube, roll along x, y, and/or z
        axis. If int, all x,y,z are rolled the same, or if
        tuple roll (x, y, z) separately.
    interp : str, optional
        Spatial cube method ['nearest', 'linear']

    Returns
    -------
    ndarray
        cube sampled as a map
    """
    sim_size = cube.shape
    if roll is not None:
        if isinstance(roll, int):
            roll = (roll, roll, roll)
        cube = np.roll(cube, roll, axis=(0, 1, 2))

    if hpx:
        # tile onto a full sky healpix map
        assert utils.import_healpy
        # Get the vector coordinates (vx, vy, vz) of the HEALPIX pixels.
        vx, vy, vz = utils.healpy.pix2vec(nside, np.arange(utils.healpy.nside2npix(nside)))

        # translate hpx vector coords to comoving coords
        # and put into cube index units
        xr = vx * dc / sim_res
        yr = vy * dc / sim_res
        zr = vz * dc / sim_res

        # nearest neighbor interpolation
        if interp == 'nearest':
            # get nearest neighbors
            xi = np.mod(np.around(xr).astype(int), sim_size[0])
            yi = np.mod(np.around(yr).astype(int), sim_size[1])
            zi = np.mod(np.around(zr).astype(int), sim_size[2])
            out = np.asarray(cube[xi, yi, zi])

        # tri-linear interpolation
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        elif interp == 'linear':
            xd, yd, zd = xr % 1, yr % 1, zr % 1
            x0 = np.mod(np.floor(xr).astype(int), sim_size[0])
            x1 = np.mod(np.ceil(xr).astype(int), sim_size[0])
            y0 = np.mod(np.floor(yr).astype(int), sim_size[1])
            y1 = np.mod(np.ceil(yr).astype(int), sim_size[1])
            z0 = np.mod(np.floor(zr).astype(int), sim_size[2])
            z1 = np.mod(np.ceil(zr).astype(int), sim_size[2])
            c00 = cube[x0, y0, z0] * (1 - xd) + cube[x1, y0, z0] * xd
            c01 = cube[x0, y0, z1] * (1 - xd) + cube[x1, y0, z1] * xd
            c10 = cube[x0, y1, z0] * (1 - xd) + cube[x1, y1, z0] * xd
            c11 = cube[x0, y1, z1] * (1 - xd) + cube[x1, y1, z1] * xd
            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd
            out = c0 * (1 - zd) + c1 * zd

        else:
            raise ValueError

    else:
        # tile onto another cube
        if interp =='nearest':
            # nearest neighbor interpolation
            zi = np.mod(np.around(dc / sim_res).astype(int), sim_size[2])
            out = np.asarray(cube[..., zi])

        elif interp == 'linear':
            # linear interpolation
            zr = dc / sim_res
            zd = zr % 1
            z0 = np.mod(np.floor(zr).astype(int), sim_size[2])
            z1 = np.mod(np.ceil(zr).astype(int), sim_size[2])
            out = cube[..., z0] * (1 - zd) + cube[..., z1] * zd

        else:
            raise ValueError

    return out
