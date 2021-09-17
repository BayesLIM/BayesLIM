"""
Radio Interferometric Measurement Equation (RIME) module
"""
import torch
import numpy as np

from . import telescope_model, calibration, beam_model, sky_model, utils
from .utils import _float, _cfloat
from .visdata import VisData


class RIME(utils.Module):
    """
    Performs the sky integral component of the radio interferometric
    measurement equation (RIME) to produce the interferometric
    visibilities between antennas p and q

    .. math::

        V_{pq} = \int_{4\pi}d \Omega\ A_p(\hat{s})
                  I(\hat{s}) A_q^\dagger K_{pq}

    where K is the interferometric fringe term.
    This is performed for each pair of linear
    feed polarizations to form the matrix

    .. math::

        V_{pq} = \left(
            \begin{array}{cc}V^{ee} & V^{en} \\
            V^{ne} & V^{nn} \end{array}
        \right)

    for two dipole feeds east (e) and north (n).
    For the case of Npol = 1, then only a single
    element from the diagonal is used.
    """
    def __init__(self, sky, telescope, beam, ant2beam, array, sim_bls,
                 times, freqs, data_bls=None, device=None):
        """
        RIME object. Takes a model
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
        sky : SkyModel object
            A sky model or CompositeModel for multiple skies
        telescope : TelescopeModel object
            Used to set the telescope location.
        beam : BeamModel object, optional
            A model of the directional and frequency response of the
            antenna primary beam. Default is a tophat response.
        ant2beam : dict
            Dict of integers that map each antenna number in array.ants
            to a particular index in the beam model output from beam.
            E.g. {10: 0, 11: 0, 12: 0} for 3-antennas [10, 11, 12] with
            1 shared beam model or {10: 0, 11: 1, 12: 2} for 3-antennas
            [10, 11, 12] with different 3 beam models.
        array : ArrayModel object
            A model of the telescope location and antenna positions
        sim_bls : list of 2-tuples
            A list of baselines, i.e. antenna-pair tuples,
            to simulate, which hold the antenna numbers
            of each baseline in array.ants.
            If array.ants = [1, 3, 5], then bls could be, for e.g.,
            bls = [(1, 3), (1, 5), (3, 5)]. Note sim_bls must
            be a subset of array.bls.
        times : tensor
            Array of observation times in Julian Date
        freqs : tensor
            Array of observational frequencies [Hz]
        data_bls : list, optional
            List of baselines in the output visibilities.
            Default is just sim_bls. However, if simulating redundant
            baseline groups and array.antpos is not a parameter
            data_bls can contain bls not in sim_bls, and the
            redundant bl in sim_bls will be copied over to data_bls
            appropriately. Note this will not work if array.antpos
            is a parameter.
        """
        super().__init__()
        self.sky = sky
        self.telescope = telescope
        self.beam = beam
        self.ant2beam = ant2beam
        self.array = array
        self.times = times
        self.Ntimes = len(times)
        self.device = device
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.set_sim_bls(sim_bls, data_bls)

    def set_sim_bls(self, sim_bls, data_bls=None):
        """
        Set the active set of baselines for sim_bls,
        which is iterated over in forward().
        Sets self._bl2vis mapping.

        Parameters
        ----------
        sim_bls : list of 2-tuples
            A list of baselines, i.e. antenna-pair tuples,
            to simulate, which hold the antenna numbers
            of each baseline in array.ants.
            If array.ants = [1, 3, 5], then bls could be, for e.g.,
            bls = [(1, 3), (1, 5), (3, 5)]. Note sim_bls must
            be a subset of array.bls.
        data_bls : list, optional
            List of baselines in the output visibilities.
            Default is just sim_bls. However, if simulating redundant
            baseline groups and array.antpos is not a parameter
            data_bls can contain bls not in sim_bls, and the
            redundant bl in sim_bls will be copied over to data_bls
            appropriately. Note this will not work if array.antpos
            is a parameter. Also note that redundant baselines
            must be ordered next to each other in data_bls.
        """
        self.sim_bls = sim_bls
        self.Nbls = len(self.sim_bls)
        self.data_bls = None if self.array.parameter else data_bls
        if self.data_bls is None:
            self._bl2vis = {bl: i for i, bl in enumerate(self.sim_bls)}
            self.Ndata_bls = self.Nbls
        else:
            # get redundant group indices
            self.Ndata_bls = len(self.data_bls)
            sim_red_inds = [self.array.bl2red[bl] for bl in self.sim_bls]
            data_red_inds = [self.array.bl2red[bl] for bl in self.data_bls]
            assert set(sim_red_inds) == set(data_red_inds), "Found non-redundant baselines"
            self._bl2vis = {}
            for si, bl in zip(sim_red_inds, self.sim_bls):
                indices = [i for i, di in enumerate(data_red_inds) if di == si]
                assert np.all(np.diff(indices) == 1), "redundant bls must be adjacent each other in data_bls"
                self._bl2vis[bl] = slice(indices[0], indices[-1] + 1)

    def forward(self, sky_components=None):
        """
        Forward pass sky components through the beam,
        the fringes, and then integrate to
        get the visibilities.

        Parameters
        ----------
        sky_components : list of dictionaries
            Default is to use self.sky.forward()
            Each dictionary is an output from a
            SkyBase subclass, containing:
                'kind' : str, kind of sky model
                'sky' : tensor, sky representation
                Extra kwargs given 'kind', including possibly
                    'angs' : tensor, optional, RA and Dec [deg] of pixels
                    'lms' : tensor, optional, l and m modes of a_lm coeffs

        Returns
        -------
        vis : VisData object
            Measured visibilities, shape (Npol, Npol, Nbl, Ntimes, Nfreqs)
        """
        # get sky components
        if sky_components is None:
            sky_components = self.sky.forward()
        if not isinstance(sky_components, list):
            sky_components = [sky_components]

        # initialize visibility tensor
        Npol = self.beam.Npol
        vd = VisData()
        vis = torch.zeros((Npol, Npol, self.Ndata_bls, self.Ntimes, self.Nfreqs),
                          dtype=_cfloat(), device=self.device)

        # clear pre-computed beam for YlmResponse type
        if self.beam.R.__class__ == beam_model.YlmResponse:
            self.beam.R.clear_beam()

        # iterate over sky components
        for i, sky_comp in enumerate(sky_components):

            kind = sky_comp['kind']
            sky = sky_comp['sky']

            # iterate over observation times
            for j, time in enumerate(self.times):

                # get beam tensor
                if kind in ['pixel', 'point']:
                    # convert sky pixels from ra/dec to alt/az
                    ra, dec = sky_comp['angs']
                    alt, az = self.telescope.eq2top(time, ra, dec, sky=kind, store=True)

                    # evaluate beam response
                    zen = utils.colat2lat(alt, deg=True)
                    ant_beams, cut, zen, az = self.beam.gen_beam(zen, az)
                    cut_sky = sky[..., cut]

                elif kind == 'alm':
                    raise NotImplementedError

                # iterate over baselines
                for k, bl in enumerate(self.sim_bls):
                    bl_slice = self._bl2vis[bl]
                    self._prod_and_sum(self.beam, ant_beams, cut_sky, bl[0], bl[1],
                                       kind, zen, az, vis, bl_slice, j)

        history = utils.get_model_description(self)[0]
        vd.setup_telescope(self.telescope, self.array)
        vd.setup_data(self.sim_bls, self.times, self.freqs, pol=self.beam.pol,
                      data=vis, flags=None, cov=None, history=history)
        return vd

    def _prod_and_sum(self, beam, ant_beams, cut_sky, ant1, ant2,
                      kind, zen, az, vis, bl_slice, obs_ind):
        """
        Sky product and sum into vis inplace
        """
        # get beam of each antenna
        beam1 = ant_beams[:, :, self.ant2beam[ant1]]
        beam2 = ant_beams[:, :, self.ant2beam[ant2]]

        # generate fringe
        fringe = self.array.gen_fringe((ant1, ant2), zen, az)

        # apply fringe to beam and/or sky depending on params.
        # when using autograd, this can reduce memory of graph
        sky_is_param = cut_sky.requires_grad
        if sky_is_param:
            # first apply fringe to beam1
            beam1 = self.array.apply_fringe(fringe, beam1, kind)
            # then apply beam-weighted fringe to sky
            psky = beam.apply_beam(beam1, cut_sky, beam2=beam2)
        else:
            # first apply beam to sky
            psky = beam.apply_beam(beam1, cut_sky, beam2=beam2)
            # then apply fringe to beam-weighted sky
            psky = self.array.apply_fringe(fringe, psky, kind)

        # sum across sky
        vis[:, :, bl_slice, obs_ind, :] += torch.sum(psky, axis=-1)

