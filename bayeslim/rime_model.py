"""
Radio Interferometric Measurement Equation (RIME) module
"""
import torch
import numpy as np
from collections.abc import Iterable

from . import telescope_model, calibration, beam_model, sky_model, utils, io
from .utils import _float, _cfloat
from .dataset import VisData


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
    def __init__(self, sky, telescope, beam, array, sim_bls,
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
        array : ArrayModel object
            A model of the telescope location and antenna positions
        sim_bls : list of 2-tuples or list of lists
            A list of baselines, i.e. antenna-pair tuples,
            to simulate, which hold the antenna numbers of each baseline
            in array.ants. If array.ants = [1, 3, 5], then sim_bls could
            be, for e.g., sim_bls = [(1, 3), (1, 5), (3, 5)]. Note
            sim_bls must be a subset of array.bls.
            This can also be passed as a list of lists, specifying groups of 
            bls to simulate independently.
        times : tensor or list of tensors
            Array of observation times in Julian Date. Can also be passed
            as a list of tensors holding time groups to simulate
            independently (e.g. for minibatching).
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
        self.array = array
        self.device = device
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.setup_sim_bls(sim_bls, data_bls)
        self.setup_sim_times(times=times)

    def setup_sim_bls(self, sim_bls, data_bls=None):
        """
        Set the active set of baselines for sim_bls,
        which is iterated over in forward().
        Also sets self._bl2vis mapping, which maps a baseline
        tuple to its indices in the output visibility.

        Parameters
        ----------
        sim_bls : list of 2-tuples or list of lists
            See class init docstring.
        data_bls : list, optional
            See class init docstring.
        """
        self.bl_group_id = 0
        # turn sim_bls into a dictionary if necessary
        if not isinstance(sim_bls, dict):
            # sim_bls is either list of bl tuples, or list of list of bl tuples
            assert isinstance(sim_bls[0][0], int) or isinstance(sim_bls[0][0][0], int), \
                "sim_bls must be list of 2-tuples or list of list of 2-tuples"
            if isinstance(sim_bls[0], tuple):
                sim_bls = {0: sim_bls}
            elif isinstance(sim_bls[0], list):
                sim_bls = {i: sim_bls[i] for i in range(len(sim_bls))}

        # type checks
        for k in sim_bls:
            for i, bl in enumerate(sim_bls[k]):
                if not isinstance(bl, tuple):
                    sim_bls[k][i] = tuple(bl)
        if data_bls is not None:
            for i, bl in enumerate(data_bls):
                if not isinstance(bl, tuple):
                    data_bls[i] = tuple(bl)

        self.sim_bl_groups = sim_bls
        self.all_bls = [bl for k in self.sim_bl_groups for bl in self.sim_bl_groups[k]]
        data_bls = None if self.array.parameter else data_bls

        # _bl2vis: maps baseline tuple to output visibility indices along bl dimension
        if data_bls is None:
            # output visibility is same shape as sim_bls
            self._bl2vis = {k: {bl: i for i, bl in enumerate(self.sim_bl_groups[k])} for k in self.sim_bl_groups}
            self.data_bl_groups = self.sim_bl_groups
        else:
            # output visibility has extra redundant baselines
            self._bl2vis = {}
            self.data_bl_groups = {}
            for k in self.sim_bl_groups:
                if k not in self._bl2vis:
                    self._bl2vis[k] = {}
                sim_red_inds = [self.array.bl2red[bl] for bl in self.sim_bl_groups[k]]
                data_red_inds = [self.array.bl2red[bl] for bl in data_bls if self.array.bl2red[bl] in sim_red_inds]
                assert set(sim_red_inds) == set(data_red_inds), "Non-redundant baselines in data_bls wrt sim_bls"
                self.data_bl_groups[k] = [bl for i, bl in enumerate(data_bls) if data_red_inds[i] in sim_red_inds]
                # iterate over every bl in sim_bl_groups and compute its index in data_bls
                for si, bl in zip(sim_red_inds, self.sim_bl_groups[k]):
                    indices = [i for i, di in enumerate(data_red_inds) if di == si]
                    assert np.all(np.diff(indices) == 1), "redundant bls must be adjacent each other in data_bls"
                    self._bl2vis[k][bl] = slice(indices[0], indices[-1] + 1)

        self._set_group()

    def setup_sim_times(self, times):
        """
        Set the active times to simulate.

        Parameters
        ----------
        times : tensor or list of tensor
            See class init docstring
        """
        self.time_group_id = 0

        # turn into dict if neccessary
        if not isinstance(times, dict):
            if isinstance(times, list) or (isinstance(times, np.ndarray) and times.ndim > 1):
                # this is a list of time arrays
                times = {k: _times for k, _times in enumerate(times)}
            else:
                # this is just a time array
                times = {0: times}

        self.sim_time_groups = times
        self.all_times = torch.tensor([time for k in self.sim_time_groups for time in self.sim_time_groups[k]])
        self._set_group()

    @property
    def Nbatch(self):
        """Get the total number of batches in this model"""
        if hasattr(self, 'sim_bl_groups') and hasattr(self, 'sim_time_groups'):
            return len(self.sim_bl_groups) * len(self.sim_time_groups)
        else:
            return None

    @property
    def batch_idx(self):
        """Get the current batch index: time_group_id + bl_group_id"""
        if hasattr(self, 'bl_group_id') and hasattr(self, 'time_group_id'):
            return self.time_group_id + self.bl_group_id
        else:
            return None

    def set_batch_idx(self, idx):
        """
        Set the batch index

        Parameters
        ----------
        idx : int
            Index of current batch from 0 to Nbatch-1
        """
        assert idx < self.Nbatch and idx >= 0
        self.bl_group_id = int(np.floor(idx / len(self.sim_time_groups)))
        self.time_group_id = idx % len(self.sim_time_groups)

    def _set_group(self):
        """
        Given group_ids, set sim parameters
        """
        if hasattr(self, 'sim_bl_groups'):
            self.sim_bls = self.sim_bl_groups[self.bl_group_id]
            self.Nsim_bls = len(self.sim_bls)
            self.data_bls = self.data_bl_groups[self.bl_group_id]
            self.Ndata_bls = len(self.data_bls)

        if hasattr(self, 'sim_time_groups'):
            self.sim_times = self.sim_time_groups[self.time_group_id]
            self.Ntimes = len(self.sim_times)

    def forward(self, *args):
        """
        Forward pass sky components through the beam,
        the fringes, and then integrate to
        get the visibilities.

        Returns
        -------
        vis : VisData object
            Measured visibilities, shape (Npol, Npol, Nbl, Ntimes, Nfreqs)
        """
        # set sim group given batch index
        self._set_group()

        # get sky components
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
            ra, dec = sky_comp['angs']

            # iterate over observation times
            for j, time in enumerate(self.sim_times):

                # get beam tensor
                if kind in ['pixel', 'point']:
                    # convert sky pixels from ra/dec to alt/az
                    alt, az = self.telescope.eq2top(time, ra, dec, store=True,
                                                    sky=(float(ra[0]), float(ra[-1])))

                    # evaluate beam response
                    zen = utils.colat2lat(alt, deg=True)
                    ant_beams, cut, zen, az = self.beam.gen_beam(zen, az)
                    cut_sky = sky[..., cut]

                elif kind == 'alm':
                    raise NotImplementedError

                # iterate over baselines: generate and apply fringe, apply beam, sum
                for k, bl in enumerate(self.sim_bls):
                    bl_slice = self._bl2vis[self.bl_group_id][bl]
                    self._prod_and_sum(ant_beams, cut_sky, bl[0], bl[1],
                                       kind, zen, az, vis, bl_slice, j)

        history = io.get_model_description(self)[0]
        vd.setup_meta(self.telescope,
                      dict(zip(self.array.ants, self.array.antpos.cpu().detach().numpy())))
        vd.setup_data(self.data_bls, self.sim_times, self.freqs, pol=self.beam.pol,
                      data=vis, flags=None, cov=None, history=history)

        return vd

    def _prod_and_sum(self, ant_beams, cut_sky, ant1, ant2,
                      kind, zen, az, vis, bl_slice, obs_ind):
        """
        Sky product and sum into vis inplace
        """
        # get beam of each antenna
        beam1 = ant_beams[:, :, self.beam.ant2beam[ant1]]
        beam2 = ant_beams[:, :, self.beam.ant2beam[ant2]]

        # generate fringe
        fringe = self.array.gen_fringe((ant1, ant2), zen, az)

        # first apply beam to sky
        psky = self.beam.apply_beam(beam1, cut_sky, beam2=beam2)

        # then apply fringe to beam-weighted sky
        psky = self.array.apply_fringe(fringe, psky, kind)

        # LEGACY: this seems to consume more memory...
        #beam1 = self.array.apply_fringe(fringe, beam1, kind)
        #psky = self.beam.apply_beam(beam1, cut_sky, beam2=beam2)

        # sum across sky
        vis[:, :, bl_slice, obs_ind, :] += torch.sum(psky, axis=-1).to(self.device)
