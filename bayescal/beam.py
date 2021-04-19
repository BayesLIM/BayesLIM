"""
Module for primary beam modeling
"""
import torch
import numpy as np

from . import utils


class BeamBase(torch.nn.Module):
    """
    Base class for all antenna primary beam models,
    which relate the directional and frequency
    response of the sky to the "perceived" sky for
    a baseline between antennas p and q

    .. math::
        I_{pq}(\hat{s}, \nu) = A_p I A_q^\ast

    Note that this can be thought of as a direction-
    dependent Jones term which takes the form

    .. math::
        J_p = \\left[\\begin{array}{cc}J_{ee} & J_{en}\\\\
                    J_{ne} & J_{nn}\\end{array}\\right]

    where e and n index the East and North feed polarizations.
    The amplitude of the beam should be normalized to unity
    at boresight. Also, a single beam can be used for all
    antennas, or one can fit for per-antenna models.
    """
    def __init__(self, params, freqs, R, parameter, polmode='1pol', fov=180):
        """
        Instantiate BeamBase object.

        Parameters
        ----------
        params : tensor or list of tensor
            Initial beam parameterization.
        freqs : tensor
            Observational frequency bins [Hz]
        R : callabled
            Beam response function, mapping input params
            to output beam. This response function is
            unique for each subclass of BeamBase.
        parameter : bool
            Fit for params dynamically, otherwise
            keep it fixed to its input value
        polmode : str, ['1pol', '2pol', '4pol']
            Polarization mode. params must conform to polmode.
            1pol : single linear polarization
            2pol : two linear polarizations (diag of Jones)
            4pol : four linear and cross pol (2x2 Jones)
        fov : float, optional
            Total angular extent of the field-of-view in degrees, centered
            at the pointing center (alitude). Parts of the sky outside of fov
            are truncated from the sky integral.
            E.g. fov = 180 means we view the entire sky above the horizon,
            while fov = 90 means we view the sky withih 45 deg of zenith.
            Default is full sky above the horizon.
        """
        super().__init__()
        self.params = params
        if isinstance(params, (list, tuple)):
            _params = []
            for i in range(len(params)):
                p = params[i]
                if parameter:
                    p = torch.nn.Parameter(p)
                setattr(self, 'param{}'.format(i), p)
                _params.append(p)
            if parameter:
                self.params = _params

        else:
            if parameter:
                self.params = torch.nn.Parameter(self.params)
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.R = R
        self.polmode = polmode
        self.Npol = 1 if polmode == '1pol' else 2
        self.fov = fov
        self._transform = {}

    def set_transform(self, kind, transform):
        """
        Set the functions necessary to transform
        to a beam of "kind". These are put
        into self._transform, which can be called
        by self.transform_to().

        Parameters
        ----------
        kind : str
            Kind of beam model to transform to.
            ['pixel', 'point', 'alm'].
        transform : callable
            Takes self.params and returns
            (params, R) for the new beam model.
        """
        self._transform['kind'] = transform


class PixelBeam(BeamBase):
    """
    A generic beam model representation in
    ``configuration space'' or on the sky (i.e. not
    Fourier space). This takes sky models of 'point'
    and 'pixel' kind.
    """
    def __init__(self, params, freqs, R=None, parameter=True, polmode='1pol', fov=180):
        """
        A generic beam model evaluated on the sky

        Parameters
        ----------
        params : tensor or list of tensor
            Initial beam parameterization matched to response function.
            Given the default response function default_R(),
            params should be a list of two tensors, each of shape
            (Npol, Npol, Nmodel, Nfreqs). The first
            is the scale of the beam along the EW direction,
            the second is the scale along the NS direction.
        freqs : tensor
            Observational frequency bins [Hz]
        R : callable, optional
            Beam response function, mapping input params
            to output beam. This response function is
            unique for each subclass of BeamBase. It should return
            another callable that takes (zen [deg], az [deg], freqs [Hz])
            as arguments and returns the beam values of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix). Default is tophat beam.
        parameter : bool, optional
            If True, fit for params dynamically (default), otherwise
            keep it fixed to its input value
        polmode : str, ['1pol', '2pol', '4pol'], optional
            Polarization mode. params must conform to polmode.
            1pol : single linear polarization (default)
            2pol : two linear polarizations (diag of Jones)
            4pol : four linear and cross pol (2x2 Jones)
        fov : float, optional
            Total angular extent of the field-of-view in degrees, centered
            at the pointing center (alitude). Parts of the sky outside of fov
            are truncated from the sky integral.
            E.g. fov = 180 (default) means we view the entire sky above the horizon,
            while fov = 90 means we view the sky withih 45 deg of zenith.
            Default is full sky above the horizon.
        """
        if R is None:
            def R(params):
                def beam_func(zen, az, freqs):
                    return torch.ones(params.shape + (zen.size,), dtype=params.dtype)

        super(PixelBeam, self).__init__(params, freqs, R, parameter, polmode, fov)

    def gen_beam(self, params, zen, az, beam_func=None):
        """
        Generate a beam model given frequency and angles
        and the field-of-view (self.fov).

        Parameters
        ----------
        params : tensor or list of tensor
            Initial beam parameters
        zen, az : array
            zenith angle (co-latitude) and azimuth angle [deg]
        fov : float, optional
            Total angular extent of the field-of-view in degrees, centered
            at the pointing center (alitude). Parts of the sky outside of fov
            are truncated from the sky integral.
            E.g. fov = 180 means we view the entire sky above the horizon,
            while fov = 90 means we view the sky withih 45 deg of zenith.
        beam_func : callable, optional
            Output of R(params). Default is to use self.R(params)
            Should take (zen [deg], az [deg], freqs [Hz]) as argument.

        Returns
        -------
        beam : tensor
            A tensor beam model of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix)
        cut : array
            Indexing of Npix axis given fov cut
        """
        # get beam function
        if beam_func is None:
            beam_func = R(params)

        # enact fov cut
        cut = zen < self.fov / 2
        zen, az = zen[cut], az[cut]

        return beam_func(zen, az, self.freqs), cut

    def apply_beam(self, beam1, sky, beam2=None):
        """
        Apply a baseline beam matrix to a representation
        of the sky.

        Parameters
        ----------
        beam1 : tensor
            Holds the beam response for one of the antennas in a
            baseline, with shape (Npol, Npol, Nfreqs, Nsources).
        sky : tensor
            sky representation (Npol, Npol, Nfreqs, Nsources)
        beam2 : tensor, optional
            The beam response for the second antenna in the baseline.
            If None, use beam1 for both ant1 and ant2.

        Returns
        -------
        psky : tensor
            perceived sky, having mutiplied beam with sky, of shape
            (Npol, Npol, Nfreqs, Npix)
        """
        if beam2 is None:
            beam2 = beam1

        if self.polmode in ['1pol', '2pol']:
            psky = utils.diag_matmul(utils.diag_matmul(beam1, sky), beam2.conj())
        else:
            psky = torch.einsum("ab...,bc...,dc...->ad...", beam1,
                    sky, beam2.conj())

        return psky

    def forward(self, sky_comp, telescope, obs_jd, antpairs,
                params=None, beam_func=None):
        """
        Forward pass a single sky model through the beam
        at a single observation time.

        Parameters
        ----------
        sky_comp : dict
            Output of SkyBase subclass, holding
            'sky' : tensor, representation of the sky
            'kind' : str, kind of sky model, e.g. 'point', 'alm'
            and other keyword arguments specific to the model
            and expected by the beam response function.
        telescope : TelescopeModel object
            A model of the telescope location
        obs_jd : float
            Observation time in Julian Date (e.g. 2458101.23456)
        antpairs : list of 2-tuple
            A list of all unique antenna-pairs of the beam's
            "model" axis, with each 2-tuple indexing the
            unique model axis of the beam.
            For a beam with a single antenna model, or Nmodel=1,
            then antpairs = [(0, 0)].
            For a beam with 3 antenna models and a baseline
            list of [(ant1, ant2), (ant1, ant3), (ant2, ant3)],
            then antpairs = [(0, 1), (0, 2), (1, 2)]. Note that
            the following ArrayModel object should have a mapping
            of antpairs to the physical baseline list.
        params : tensor or list of tensor, optional
            Use these params instead of self.params
        beam_func : callable, optional
            Use this function to evaluate the beam, rather
            than the output of self.R(params)

        Returns
        -------
        psky_comp : dict
            Same input dictionary but with psky as 'sky' of shape
            (Npol, Npol, Nantpair, Nfreqs, Nsources), where
            roughly psky = beam1 * sky * beam2. The FoV cut has
            been applied to psky as well as the 'angs' key
        """
        _params = params
        if params is None:
            _params = self.params
        kind = sky_comp['kind']

        if kind in ['point', 'pixel']:

            # get coords
            alt, az = telescope.eq2top(obs_jd, sky_comp['angs'][0], sky_comp['angs'][1],
                                       sky=kind, store=False)
            zen = 90 - alt

            # get beam response function
            if beam_func is None:
                beam_func = self.R(_params)

            # evaluate beam
            beam, cut = self.gen_beam(None, zen, az, beam_func=beam_func)
            sky = sky_comp['sky'][..., cut]
            zen, alt, az = zen[cut], alt[cut], az[cut]

            # iterate over baselines
            shape = sky.shape
            psky = torch.zeros(shape[:2] + (len(antpairs),) + shape[2:], dtype=sky.dtype)
            for k, (ant1, ant2) in enumerate(antpairs):
                # get beam of each antenna
                beam1 = beam[:, :, ant1]
                beam2 = beam[:, :, ant2]

                # apply beam to sky
                psky[:, :, k] = self.apply_beam(beam1, sky, beam2=beam2)

            sky_comp['sky'] = psky
            sky_comp['angs'] = sky_comp['angs'][:, cut]
            sky_comp['altaz'] = torch.tensor([alt, az])

        else:
            raise NotImplementedError

        return sky_comp

    def transform_to(self, kind):
        """
        Transform beam to a different kind

        Parameters
        ----------
        kind : str
            Kind of beam model to transform to.
            ['point', 'pixel', 'alm']

        Returns
        -------
        new_beam : BeamBase subclass
            this beam but transformed to a new kind
        """
        if kind in ['point', 'pixel']:
            if self.__class__ == PixelBeam:
                # current beam conforms to requested beam
                return self
            elif self.__class__ == SphBeam:
                # transform SphBeam to PixelBeam
                params, R = self._transform['alm'](self.params)
                pix_beam = PixelBeam(params, freqs=self.freqs, R=R,
                                     parameter=True, polmode=self.polmode, fov=self.fov)

        elif kind == 'alm':
            if self.__class__ == SphBeam:
                # current beam conforms to requested beam
                return self
            if self.__class__ == PixelBeam:
                # transform PixelBeam to SphBeam
                params, R = self._transform['pixel'](self.params)
                sph_beam = SphBeam(params, freqs=self.freqs, R=R,
                                   parameter=True, polmode=self.polmode, fov=self.fov)
                return sph_beam


def Gauss_R(params):
    def beam_func(zen, az, freqs):
        zen, az = zen * np.pi / 180, az * np.pi / 180
        l = torch.as_tensor(np.sin(zen) * np.cos(az))[..., None]
        m = torch.as_tensor(np.sin(zen) * np.sin(az))[..., None]
        beam = torch.exp(-0.5 * ((l / params[0])**2 + (m / params[1])**2))
        return beam
  

class SphBeam(BeamBase):
    """
    A generic beam model representation in
    ``spherical harmonic space'' on the sky.
    This takes sky models of 'alm' kind.
    """
    def __init__(self, params, freqs, R, parameter=True, polmode='1pol', fov=180):
        """
        A generic beam model evaluated on the sky

        Parameters
        ----------
        params : tensor or list of tensor
            Initial beam parameterization matched to response function.
            Given the default response function default_R(),
            params should be a list of two tensors, each of shape
            (Npol, Npol, Nmodel, Nfreqs). The first
            is the scale of the beam along the EW direction,
            the second is the scale along the NS direction.
        freqs : tensor
            Observational frequency bins [Hz]
        R : callable
            Beam response function, mapping input params
            to output beam. This response function is
            unique for each subclass of BeamBase. It should return
            another callable that takes (zen [deg], az [deg], freqs [Hz])
            as arguments and returns the beam values of shape
            (Npol, Npol, Nmodel, Nfreqs, Npix).
        parameter : bool, optional
            If True, fit for params dynamically (default), otherwise
            keep it fixed to its input value
        polmode : str, ['1pol', '2pol', '4pol'], optional
            Polarization mode. params must conform to polmode.
            1pol : single linear polarization (default)
            2pol : two linear polarizations (diag of Jones)
            4pol : four linear and cross pol (2x2 Jones)
        fov : float, optional
            Total angular extent of the field-of-view in degrees, centered
            at the pointing center (alitude). Parts of the sky outside of fov
            are truncated from the sky integral.
            E.g. fov = 180 (default) means we view the entire sky above the horizon,
            while fov = 90 means we view the sky withih 45 deg of zenith.
            Default is full sky above the horizon.
        """
        super(SphBeam, self).__init__(params, freqs, R, parameter, polmode, fov)


