"""
Radio Interferometric Measurement Equation (RIME) module
"""
import torch
import numpy as np
from collections.abc import Iterable
from datetime import datetime

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
                 times, freqs, data_bls=None, device=None, name=None,
                 cache_skycut=True, verbose=False):
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
            List of all baselines in the output visibility.
            Default is just sim_bls. However, if simulating redundant
            baseline groups and array.antpos is not a parameter
            data_bls can contain bls not in sim_bls, and the
            redundant bl in sim_bls will be copied over to data_bls
            appropriately. If a bl in data_bls does not have a redudant
            bl match in sim_bls, it is dropped. Baseline redundancies
            are computed by the array object.
        name : str, optional
            Name for this object, stored as self.name
        cache_skycut : bool, optional
            If True, cache the beam FOV cut indexing tensor
            on sky.device. This sidesteps need to move
            cut from beam.device to sky.device, which can
            be a perf bottleneck
        verbose : bool, optional
            If True, print simulation progress info
        """
        super().__init__(name=name)
        self.sky = sky
        self.telescope = telescope
        self.beam = beam
        self.array = array
        self.device = device
        self.cache_skycut = cache_skycut
        self.verbose = verbose
        self.setup_freqs(freqs)
        self.setup_sim_bls(sim_bls, data_bls)
        self.setup_sim_times(times=times)
 
    def setup_freqs(self, freqs):
        """
        Set frequency array

        Parameters
        ----------
        freqs : tensor
            Array of observational frequencies [Hz]
        """
        self.freqs = freqs
        self.Nfreqs = len(freqs)

    def setup_sim_bls(self, sim_bls, data_bls=None):
        """
        Configure how baselines are grouped, simulated, copied,
        and inserted into the output visibilities.

        Also sets self._sim2data dictionary, which maps a
        baseline group id to an indexing tensor expanding
        sim_bls to data_bls length.

        This sets self.sim_bl_groups, self.data_bl_groups dictionaries,
        the self._sim2data dictionary, and resets self.bl_group_id -> 0

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
                sim_bls = {i: sbl for i, sbl in enumerate(sim_bls)}

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
        self.all_sim_bls = utils.flatten(sim_bls.values())
        self.Nbl_groups = len(self.sim_bl_groups)
        data_bls = None if self.array.parameter else data_bls

        # setup _sim2data
        if data_bls is None:
            # output visibility is same shape as sim_bls
            self.data_bl_groups = self.sim_bl_groups
            self._sim2data = {i: None for i in range(len(self.sim_bl_groups))}

        else:
            # output visibility has extra redundant baselines
            self._sim2data = {}
            self.data_bl_groups = {}
            # iterate over sim baseline groups
            for i, blg in self.sim_bl_groups.items():
                # get redundant group indices for all sim_bls
                sim_red_inds = [self.array.bl2red[bl] for bl in blg]
                # reject any data_bls without a redundant type in sim_bls
                _data_bls = [bl for bl in data_bls if self.array.bl2red[bl] in sim_red_inds]
                # now get redundant group indices for data_bls
                data_red_inds = [self.array.bl2red[bl] for bl in _data_bls]
                # this ensures all simulated and data baselines have a redundant match
                assert set(sim_red_inds) == set(data_red_inds), "non-overlapping bl type(s) in data_bls and sim_bls"
                # this ensures that redundant baselines are grouped together in data_bls
                assert len(np.where(np.diff(data_red_inds)!=0)[0]) == len(blg) - 1
                self.data_bl_groups[i] = _data_bls
                # get sim_bls indexing tensor
                index = torch.as_tensor(
                    [sim_red_inds.index(i) for i in data_red_inds], 
                    device=self.device
                )
                # populate
                self._sim2data[i] = index

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
        self.all_times = np.asarray(utils.flatten(self.sim_time_groups))
        self.Ntime_groups = len(self.sim_time_groups)
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
            return self.time_group_id + self.bl_group_id * len(self.sim_time_groups)
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
        self._set_group()

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

    def forward(self, *args, prior_cache=None, **kwargs):
        """
        Forward pass sky components through the beam,
        the fringes, and then integrate to
        get the visibilities.

        Returns
        -------
        VisData
            model visibilities
        """
        # set sim group given batch index
        self._set_group()

        # get sky components
        sky_components = self.sky.forward(prior_cache=prior_cache)
        if not isinstance(sky_components, list):
            sky_components = [sky_components]

        # initialize visibility tensor
        Npol = self.beam.Npol
        if Npol == 1:
            pol = "{}{}".format(self.beam.pol, self.beam.pol)
        else:
            pol = None
        vd = VisData()
        vis = torch.zeros((Npol, Npol, self.Ndata_bls, self.Ntimes, self.Nfreqs),
                          dtype=_cfloat(), device=self.device)

        # clear pre-computed beam for YlmResponse type
        if hasattr(self.beam.R, 'clear_beam'):
            self.beam.R.clear_beam()

        # iterate over sky components
        start = datetime.now().timestamp()
        for i, sky_comp in enumerate(sky_components):

            kind = sky_comp['kind']
            sky = sky_comp['sky']
            ra, dec = sky_comp['angs']
            if 'prior' in sky_comp:
                out_dict['prior'] = out_dict['prior'] + sky_comp['prior']

            # iterate over observation times
            for j, time in enumerate(self.sim_times):

                # print info
                message = "{}/{} times for {}/{} sky model | {} elapsed"
                message = message.format(j+1, len(self.sim_times), i+1, len(sky_components),
                                         elapsed_time(start))
                log(message, verbose=self.verbose, style=1)

                # get beam tensor
                if kind in ['pixel', 'point']:
                    # convert sky pixels from ra/dec to alt/az
                    alt, az = self.telescope.eq2top(time, ra, dec, store=True)

                    # evaluate beam response
                    zen = utils.colat2lat(alt, deg=True)
                    ant_beams, cut, zen, az = self.beam.gen_beam(zen, az, prior_cache=prior_cache)
                    # cache a version of cut on sky.device: this prevents repeated
                    # calls to cut.to(sky.device) which can be a bottleneck if beam and
                    # sky are on different devices
                    if self.cache_skycut:
                        self.beam.set_sky_cut(zen, cut, device=sky.device)
                        cut = self.beam.query_cache(zen)
                    cut_sky = beam_model.cut_sky_fov(sky, cut)

                elif kind == 'alm':
                    raise NotImplementedError

                # apply beam and fringe for all bls to sky and sum into vis
                sim2data_idx = self._sim2data[self.bl_group_id]
                self._prod_and_sum(ant_beams, cut_sky, self.sim_bls,
                                   kind, zen, az, vis, sim2data_idx, j)

        history = io.get_model_description(self)[0]
        vd.setup_meta(self.telescope,
                      dict(zip(self.array.ants, self.array.antpos.cpu().detach().numpy())))
        vd.setup_data(self.data_bls, self.sim_times, self.freqs, pol=pol,
                      data=vis, flags=None, cov=None, history=history)

        return vd

    def _prod_and_sum(self, beam, cut_sky, bl, kind,
                      zen, az, vis, sim2data_idx, obs_ind):
        """
        Sky product and sum into vis inplace

        Parameters
        ----------
        beam : tensor
            Beam model output from gen_beam() of shape
            (Npol, Nvec, Nmodel, Nfreqs, Nsources)
        cut_sky : tensor
            Sky model (Nvec, Nvec, Nfreqs, Nsources)
        bl : tuple or list of tuple
            Baseline antenna pair e.g. (0, 1) or list
            of such to operate on simultaneously
        kind : str
            Kind of sky model ['point', 'pixel', 'alm']
        zen, az : tensor
            Zenith and azimuth angles of sky model [degrees]
        vis : tensor
            Visibility tensor to insert results
            (Npol, Npol, Nbls, Ntimes, Nfreqs)
        sim2data_idx : tensor
            Indexing tensor that expands sim_bls into data_bls
            for the current baseline group. If None, no expansion
            is applied.
        obs_ind : int
            Time index along Ntimes axis of vis
        """
        # first apply beam to sky: psky shape (Npol, Npol, Nbls, Nfreqs, Nsources)
        psky = self.beam.apply_beam(beam, bl, cut_sky)

        # generate fringe: (Nbls, Nfreqs, Npix)
        fringe = self.array.gen_fringe(bl, zen, az)

        # then apply fringe to beam-weighted sky
        psky = self.array.apply_fringe(fringe, psky, kind)

        # LEGACY: this seems to consume more memory...
        #beam1 = self.array.apply_fringe(fringe, beam1, kind)
        #psky = self.beam.apply_beam(beam1, cut_sky, beam2=beam2)

        # sum across sky
        sum_sky = torch.sum(psky, axis=-1).to(self.device)

        # copy sim_bls over to each redundant bl in visibility if needed
        if sim2data_idx is not None:
            sum_sky = torch.index_select(sum_sky, 2, sim2data_idx)

        # sum across sky
        vis[:, :, :, obs_ind, :] += sum_sky


def log(message, verbose=False, style=1):
    """
    Print message to stdout

    Parameters
    ----------
    message : str

    verbose : bool, optional
        If True print, otherwise silence

    style : int, optional
        Style of message
    """
    if verbose:
        if style == 1:
            print("{}".format(message))
        elif style == 2:
            print("{}\n{}".format(message, '-'*30))
        elif style == 3:
            print("\n{}\n{}\n{}".format('-'*30, message, '-'*30))


def elapsed_time(start):
    """
    Get elapsed time in seconds or minutes

    Parameters
    ----------
    start : float
        Start time in seconds, i.e.
        datetime.now().timestamp()

    Returns
    -------
    str
    """
    # get elapsed time in seconds
    t = datetime.now().timestamp() - start
    unit = 'sec'

    # if it is larger than 60000, convert to hours
    if t > 60000:
        t /= 3600
        unit = 'hrs'
    # if it is larger than 1000, convert to minutes
    elif t > 1000:
        t /= 60
        unit = 'min'

    return "{:.3f} {}".format(t, unit)
