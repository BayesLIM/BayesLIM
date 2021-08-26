"""
Interface to pyuvdata and a user-friendly visibility simulator
"""
import numpy as np
import torch
import warnings
import itertools
import copy

from . import utils, beam_model, sky_model, telescope_model, rime
from .utils import _float, _cfloat

try:
    from pyuvdata import UVData, UVCal, UVBeam, utils as uvutils
    import_pyuv = True
except ImportError:
    import_pyuv = False
    warnings.warn("could not import pyuvdata")


def run_rime_sim(sky, beam, uvd, ant2beam=None, partial_read={},
                 freq_interp='linear', array_kwargs={},
                 outfname=None, overwrite=False, partial_write=False):
    """
    Run a RIME visibility simulation given models
    for the sky, beam, and a UVData object.

    Parameters
    ----------
    sky : SkyBase subclass
        sky model object to simulate. can also
        be a sky_model.CompositeModel to simulate and
        then sum multiple sky models
    beam : UVBeam object or PixelBeam object
        A beam model
    uvd : UVData object or str
        A filename to read, or a UVData object
        to inform setup of telescope and array
    ant2beam : dict, optional
        Only used if more than 1 beam model in the array.
        Mapping of antenna number (key) to
        beam model index (value) in beam.params
    partial_raed : dict, optional
        kwargs for partial load of uvd if passed
        as a filename
    freq_interp : str, optional
        Frequency interpolation scheme for sky
        and beam if frequencies do not match uvd.
    array_kwargs : dict, optional
        Additional kwargs for ArrayModel
    outfname : str, optional
        Output filename to write simulation to file as UVH5
    overwrite : bool, optional
        Overwrite output if it exists
    partial_write : bool, optional
        Overwrite part of the file, if doing a partial read/write

    Returns
    -------
    UVData object
        Simulated data
    """
    # load uvdata
    infile = False
    if isinstance(uvd, str):
        infile = uvd
        uvd = UVData()
        uvd.read(infile, **partial_read)
    else:
        uvd = copy.deepcopy(uvd)

    # get metadata
    freqs = np.unique(uvd.freq_array)
    times = np.unique(uvd.time_array)
    sim_bls = uvd.get_antpairs()
    pols = uvd.get_pols()
    if uvd.x_orientation is None:
        # assume x_orientation = 'east', this has no effect
        # if the adopted beam also has x_orientation = None
        pols = [uvutils.polnum2str(uvutils.polstr2num(p), 'east') for p in pols]
    antpos, ants = uvd.get_ENU_antpos()
    antpos = dict(zip(ants, antpos))

    # determine polmode from uvd pols
    if len(pols) == 1:
        polmode = '1pol'
        assert pols[0][0] == pols[0][1], "data must be auto-pol for 1pol mode"
    elif len(pols) == 2:
        polmode = '2pol'
        assert pols[0][0] == pols[0][1], "data must be auto-pols for 2pol mode"
        assert pols[1][0] == pols[1][1], "data must be auto-pols for 2pol mode"
    else:
        polmode = '4pol'

    # read beamfits filepath to UVBeam
    if isinstance(beam, str):
        uvb = UVBeam()
        uvb.read_beamfits(beam)
        beam = uvb
    # convert beam to PixelBeam if necessary
    if isinstance(beam, UVBeam):
        assert ant2beam is None, "can only assign one beam model given UVBeam"
        assert uvb.beam_type == 'power', "if passing UVBeam, must be powerbeam"
        assert polmode in ['1pol', '2pol'], "if using powerbeam, data must be 1pol or 2pol"
        assert uvb.pixel_coordinate_system == 'healpix', 'if passing UVBeam, must be healpix'
        # get beam pols
        xorient = beam.x_orientation
        assert xorient == uvd.x_orientation, "uvdata and beam must have same x_orientation"
        if xorient is None:
            xorient = 'east'
        bpols = [uvutils.polnum2str(p, xorient) for p in beam.polarization_array]
        # get data
        bdata = uvb.data_array[0, 0]
        bfreqs = np.unique(beam.freq_array)
        bnpix = bdata.shape[-1]
        # sort beam polarizations
        if polmode == '1pol':
            bdata = torch.zeros(1, 1, 1, len(bfreqs), bnpix, dtype=_float(), device=sky.device)
            bdata[0, 0, 0] = torch.tensor(bdata[0], dtype=_float())

        elif polmode == '2pol':
            bdata = torch.zeros(2, 2, 1, len(bfreqs), bnpix, dtype=_float(), device=sky.device)
            bdata[0, 0, 0] = torch.tensor(bdata[bpols.index('ee')], dtype=_float())
            bdata[1, 1, 0] = torch.tensor(bdata[bpols.index('nn')], dtype=_float())

        # hard-code bilinear b/c nearest is insufficient
        # if higher order becomes available then expose kwarg
        beam = beam_model.PixelBeam(bdata, bfreqs, response=beam_model.PixelResponse,
                                    response_args=(bfreqs, 'healpix', bnpix, 'bilinear'),
                                    parameter=False, polmode=polmode,
                                    powerbeam=True, fov=180)

    assert sky.device == beam.device
    if len(pols) == 1:
        assert beam.polmode == '1pol'
    elif len(pols) == 2:
        assert beam.polmode == '2pol'
    else:
        assert beam.polmode == '4pol'

    # interpolate sky model to frequencies
    sky.freq_interp(freqs, kind=freq_interp)

    # interpolate sky model to frequencies
    beam.freq_interp(freqs, kind=freq_interp)

    # setup telescope and array models
    loc = uvd.telescope_location_lat_lon_alt_degrees
    tele = telescope_model.TelescopeModel((loc[1], loc[0], loc[2]),
                                          device=sky.device)
    arr = telescope_model.ArrayModel(antpos, freqs, device=sky.device, **array_kwargs)
    if ant2beam is None:
        ant2beam = {ant: 0 for ant in arr.ants}

    # setup RIME object
    RIME = rime.RIME(sky, tele, beam, ant2beam, arr, sim_bls, times, freqs,
                     device=sky.device)

    with torch.no_grad():
        # forward model sky, beam, and fringe to get visibilities
        vis = utils.tensor2numpy(RIME.forward(), clone=True)

    # flatten polarization axis, move to back
    vis = vis.reshape(-1, *vis.shape[2:])
    vis = np.moveaxis(vis, 0, -1)

    # get polarization indices
    vis_pols = np.array([['ee', 'en'], ['ne', 'nn']])
    if len(pols) == 1:
        pol_inds = np.array([0])
    elif len(pols) == 2:
        pol_inds = np.array([list(vis_pols.diagonal()).index(p) for p in pols])
    else:
        pol_inds = np.array([list(vis_pols.ravel()).index(p) for p in pols])

    # iterate over baselines and fill-in uvd
    for i, bl in enumerate(sim_bls):
        inds = uvd.antpair2ind(bl)
        uvd.data_array[inds, 0] = vis[i][:, :, pol_inds]

    # write to file
    if outfname:
        if overwrite or not os.path.exists(outfname):
            if partial_write:
                uvd.write_uvh5_part(outfname, uvd.data_array,
                                    uvd.flag_array, uvd.nsample_array,
                                    **partial_read)
            else:
                uvd.write_uvh5(outfname, clobber=True)

    return uvd


def setup_uvdata(antnums=None, antnames=None, antpos=None, bls=None, redundancy=False,
                 no_autos=False, telescope_location=None, telescope_name=None,
                 freq_array=None, time_array=None, pol_array=None,
                 x_orientation='north', make_data=True, run_check=True):
    """
    Create a UVData object given observatory
    and observation metadata.

    Parameters
    ----------
    antnums : array, dtype=int
        Array of antenna numbers
    antnames : array, dtype=str
        Array of antenna names
    antpos : array, dtype=float
        Array of antenna positions in ENU frame
        of shape (Nants, 3)
    bls : list of int 2-tuple
        A list of ant-num pairs for the data_array
        If None, will create all N(N-1)/2 pairs.
    redundancy : bool
        Only if bls is None. This creates only redundant
        baselines from antpos.
    no_autos : False
        If True and bls is None, don't compute auto-correlations
    telescope_location : float 3-tuple
        Telescope location in lat, lon, alt [deg, deg, m]
    telescope_name : str
        Name of the telescope
    freq_array : array, dtype=float
        Frequency bins (center of bin) [Hz]
    time_array : array, dtype=float
        Time bins (center of bin) in Julian Date
    pol_array : array, dtype=str
        Visibliity polarizations (e.g. 'xx', 'xy' or 'ee', 'en')
    x_orientation : str
        Direction of the 'X' feed, 'north' (default) or 'east'
    make_data : bool
        If True (default), create the data, flag and nsample_array
        otherwise leave them as None
    run_check : bool
        If True, run the uvdata.check() after completino

    Returns
    -------
    UVData object
    """
    raise NotImplementedError
    # copied from healvis
    # get tloc XYZ
    tloc = list(telescope_location)
    tloc[0] *= np.pi / 180
    tloc[1] *= np.pi / 180
    tloc_xyz = uvutils.XYZ_from_LatLonAlt(*tloc)

    # setup object
    uv_obj = UVData()
    uv_obj.telescope_location = tloc_xyz
    uv_obj.telescope_location_lat_lon_alt = (lat, lon, alt)
    uv_obj.telescope_location_lat_lon_alt_degrees = (
        np.degrees(lat),
        np.degrees(lon),
        alt
    )
    uv_obj.antenna_numbers = antnums
    uv_obj.antenna_names = antnames
    uv_obj.antenna_positions = uvutils.ECEF_from_ENU(antpos, *tloc) - tloc_xyz
    uv_obj.Nants_telescope = len(antnums)

    # get baseline info
    if bls is not None:
        if redundancy is not None:
            red_tol = redundancy
            reds, vec_bin_centers, lengths = uvutils.get_antenna_redundancies(
                anums, enu, tol=red_tol, include_autos=not bool(no_autos)
            )
            bls = [r[0] for r in reds]
            bls = [uvutils.baseline_to_antnums(bl_ind, Nants) for bl_ind in bls]

    bl_array = []
    for (a1, a2) in bls:
        bl_array.append(uvutils.antnums_to_baseline(a1, a2, 1))
    bl_array = np.asarray(bl_array)

    # fill in frequency info
    if freq_array is not None:
        uv_obj.freq_array = np.atleast_2d(freq_array)
        uv_obj.Nfreqs = freq_array.shape[-1]
        uv_obj.channel_width = np.mean(np.diff(freq_array.squeeze()))

    # fill in time info
    if time_array is not None:
        uv_obj

    if time_array is not None:
        if Ntimes is not None or start_time is not None or time_cadence is not None:
            raise ValueError(
                "Cannot specify time_array as well as Ntimes, start_time or time_cadence"
            )
        Ntimes = time_array.size
    else:
        time_dict = parse_time_params(
            dict(Ntimes=Ntimes, start_time=start_time, time_cadence=time_cadence)
        )
        time_array = time_dict["time_array"]

    uv_obj.freq_array = freq_array
    uv_obj.Nfreqs = Nfreqs
    uv_obj.Ntimes = Ntimes

    # fill in other attributes
    uv_obj.spw_array = np.array([0], dtype=int)
    uv_obj.Nspws = 1
    uv_obj.polarization_array = np.array(
        [uvutils.polstr2num(pol) for pol in pols], dtype=int
    )
    uv_obj.Npols = uv_obj.polarization_array.size
    if uv_obj.Nfreqs > 1:
        uv_obj.channel_width = np.diff(uv_obj.freq_array[0])[0]
    else:
        uv_obj.channel_width = 1.0
    uv_obj._set_drift()
    uv_obj.telescope_name = tele_dict["telescope_name"]
    uv_obj.instrument = "simulator"
    uv_obj.object_name = "zenith"
    uv_obj.vis_units = "Jy"
    uv_obj.history = ""

    if redundancy is not None:
        red_tol = redundancy
        reds, vec_bin_centers, lengths = uvutils.get_antenna_redundancies(
            antnums, antpos, tol=red_tol, include_autos=not bool(no_autos)
        )
        bls = [r[0] for r in reds]
        bls = [uvutils.baseline_to_antnums(bl_ind, Nants) for bl_ind in bls]

    # Setup array and implement antenna/baseline selections.
    bl_array = []
    _bls = [(a1, a2) for a1 in anums for a2 in anums if a1 <= a2]
    if bls is not None:
        if isinstance(bls, str):
            bls = ast.literal_eval(bls)
        bls = [bl for bl in _bls if bl in bls]
    else:
        bls = _bls
    if anchor_ant is not None:
        bls = [bl for bl in bls if anchor_ant in bl]

    if bool(no_autos):
        bls = [bl for bl in bls if bl[0] != bl[1]]
    if antenna_nums is not None:
        if isinstance(antenna_nums, str):
            antenna_nums = ast.literal_eval(antenna_nums)
        if isinstance(antenna_nums, int):
            antenna_nums = [antenna_nums]
        bls = [(a1, a2) for (a1, a2) in bls if a1 in antenna_nums or a2 in antenna_nums]
    bls = sorted(bls)
    for (a1, a2) in bls:
        bl_array.append(uvutils.antnums_to_baseline(a1, a2, 1))
    bl_array = np.asarray(bl_array)
    print("Nbls: {}".format(bl_array.size))
    if bl_array.size == 0:
        raise ValueError("No baselines selected.")
    uv_obj.time_array = time_array  # Keep length Ntimes
    uv_obj.baseline_array = bl_array  # Length Nbls

    if make_full:
        uv_obj = complete_uvdata(uv_obj, run_check=run_check)

    return uv_obj

def complete_uvdata(uvd):
    """
    Given an incomplete UVData without
    data_arrays, use metadata to populate them

    Parameters
    ----------
    uvd : UVData object

    Returns
    -------
    UVData object
    """
    raise NotImplementedError


def parse_params(tele_params, obs_params):
    """
    Parse a telescope and observation
    parameter file and return 
    a parameter dictionary

    Parameters
    ----------
    tele_params : str
        Telescope (or observatory)
        parameter file, containing
        array location, array layout
        antenna beam models etc.
    obs_params : str
        Observation strategy parameter
        file, containing metadata like
        frequency channelization, time
        integration, scan position and
        tracking strategy etc.
    """
    raise NotImplementedError


def get_params_from_uvdata(uvd):
    """
    Retrieve the metadata from uvd necessary to
    re-create a vanilla uvdata object

    Parameters
    ----------
    uvd : UVData object

    Returns
    -------
    dict
        metadata for setup_uvdata()
    """
    raise NotImplementedError

