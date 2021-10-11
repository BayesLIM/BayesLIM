"""
Cosmology conversions module
"""
import numpy as np
from astropy import units, constants
from astropy.cosmology import FlatLambdaCDM, z_at_value


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


# try to import healpy
import warnings
try:
    import healpy as hp
    import_healpy = True
except ImportError:
    import_healpy = False
if not import_healpy:
    try:
        # note this will have more limited capability
        # than healpy, but can do what we need
        from astropy_healpix import healpy as hp
        import_healpy = True
    except ImportError:
        warnings.warn("could not import healpy")


def cube2hpx(simfile, freq, nside=1024, sim_res, sim_size,
             cosmo=None):
    """
    Project simulation cube onto a healpix map.

    Adapted from P. Kittisiwit's cosmotile
    github.com/piyanatk/cosmotile

    Parameters
    ----------
    simfile: ndarray or str
        A 3D temperature simulation cube or str path to .npy file
    freq: float
        Frequency of interest in MHz.
    nside: integer
        NSIDE of the output HEALPix image. Must be a valid NSIDE for HEALPix.
    sim_res : float
        Simulation voxel resolution in cMpc
    sim_size : tuple
        3-tuple containing Npixels for box along x,y,z axes
        e.g. (128, 128, 128)
    cosmo : Cosmology object, optional

    Returns
    -------
    array
        Healpix maps
    """
    if cosmo is None:
        cosmo = Cosmology()

    # Read in and interpolate simulation cubes to the redshift of interest.
    if isinstance(simfile, str):
        cube = np.load(simfile)
    else:
        cube = simfile

    # Determine the radial comoving distance r to the comoving shell at the
    # frequency of interest.
    f21 = 1420.40575177  # MHz
    z21 = f21 / freq - 1
    dc = cosmo.comoving_distance(z21).value

    # Get the vector coordinates (vx, vy, vz) of the HEALPIX pixels.
    vx, vy, vz = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))

    # Translate vector coordinates to comoving coordinates and determine the
    # corresponding cube indexes (xi, yi, zi). For faster operation, we will
    # use the mod function to determine the nearest neighboring pixels and
    # just grab the data points from those pixels instead of doing linear
    # interpolation.
    xi = np.mod(np.around(vx * dc / sim_res).astype(int), sim_size[0])
    yi = np.mod(np.around(vy * dc / sim_res).astype(int), sim_size[1])
    zi = np.mod(np.around(vz * dc / sim_res).astype(int), sim_size[2])
    out = np.asarray(cube[xi, yi, zi])

    return out


def cube2slice(simfile, freq, sim_res, sim_size, cosmo=None):
    """
    Project simulation cube onto a transverse slice
    orthogonal to the line-of-sight (e.g. for lightcones).

    Adapted from P. Kittisiwit's cosmotile
    github.com/piyanatk/cosmotile

    Parameters
    ----------
    simfile: ndarray or str
        A 3D temperature simulation cube or str path to .npy file
    freq: float
        Frequency of interest in MHz.
    sim_res : float
        Simulation voxel resolution in cMpc
    sim_size : tuple
        3-tuple containing Npixels for box along x,y,z axes
        e.g. (128, 128, 128)
    cosmo : Cosmology object, optional

    Returns
    -------
    array
        slice maps
    """
    if cosmo is None:
        cosmo = Cosmology()

    # Read in and interpolate simulation cubes to the redshift of interest.
    if isinstance(simfile, str):
        cube = np.load(simfile)
    else:
        cube = simfile

    # Determine the radial comoving distance r to the comoving slice at the
    # frequency of interest.
    f21 = 1420.40575177  # MHz
    z21 = f21 / freq - 1
    dc = cosmo.comoving_distance(z21).value

    zi = np.mod(np.around(dc / sim_res).astype(int), sim_size[2])
    out = np.asarray(cube[..., zi])

    return out
