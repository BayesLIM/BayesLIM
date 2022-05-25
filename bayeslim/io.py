"""
Module for generic input/output. See also utils.py
"""
import numpy as np
import torch
import os
import pickle

from . import utils, rime_model, telescope_model, sky_model, beam_model, dataset, optim
from .utils import _float, _cfloat
from .paramdict import ParamDict


def get_model_description(model):
    """
    Iterate through a torch Module or Sequential
    model and collect the tree structure and description
    of arguments, if provided

    Parameters
    ----------
    model : torch Module or Sequential

    Returns
    -------
    str
        High-level model tree structure
    dict
        Subdirectories containing sub-model
        arguments, if provided
    """
    # get str
    tree = str(model)
    # construct directory of model arguments
    name = model._get_name()
    model_args = {name: {}}
    args = getattr(model, '_args', None)
    if args is not None:
        model_args[name]['args'] = args
    # iterate over sub-modules
    for submod in model._modules:
        submodule = getattr(model, submod)
        subname = submodule._get_name()
        subargs = get_model_description(submodule)[1][subname]
        model_args[name][subname] = subargs

    return tree, model_args


def write_pkl(fname, model, overwrite=False):
    """
    Pickle an object as fname [.pkl]

    Parameters
    ----------
    fname : str
        filepath to output .pkl file
    model : object
    overwrite : bool, optional
        Overwrite output if it exists
    """
    if not os.path.exists(fname) or overwrite:
        with open(fname, 'wb') as f:
            pickle.dump(model, f, protocol=4)
    else:
        print("{} exists, not overwriting".format(fname))


def read_pkl(fname, pdict=None, device=None):
    """
    Load a .pkl file

    Parameters
    ----------
    fname : str
        Filepath to .pkl
    pdict : ParamDict, optional
        If fname is a Module subclass, replace
        its parameter values with this ParamDict
    device : str, optional
        If not None and if possible, send
        object to device

    Returns
    -------
    object
    """
    # load the model if necessary
    if isinstance(fname, str):
        with open(fname, 'rb') as f:
            model = pickle.load(f)
    else:
        model = fname

    # load and apply pdict
    if pdict is not None:
        assert isinstance(model, utils.Module), "fname must be a Module to apply a pdict"
        if isinstance(pdict, str):
            pdict = ParamDict.read_pkl(pdict)
        model.update(pdict)

    # move model, if possible
    if device is not None:
        if hasattr(model, 'push'):
            # for models with a push() method
            model.push(device)
        elif hasattr(model, 'device') and isinstance(model, utils.Module):
            # for models without push but with device (e.g. RIME)
            model.device = device
        elif isinstance(model, torch.Tensor):
            # for tensor
            model = model.to(device)

    return model


def del_model_attr(model, name):
    """
    Del "model.mod1.mod2.params"
    """
    if isinstance(name, str):
        name = name.split('.')
    if len(name) == 1:
        delattr(model, name[0])
    else:
        delattr(utils.get_model_attr(model, name[:-1]), name[-1])


def build_sky(multi=None, modfile=None, device=None, pdict=None,
              catfile=None, freqs=None, freq_interp='linear',
              set_param=None, unset_param=None, comp_kwargs={}):
    """
    Build a sky model. A few different methods
    for doing this are provided. For building a composite model,
    (or multiple composite models) use the multi kwarg.

    Parameters
    ----------
    multi : list of (str, dict), optional
        A list of 2-tuples holding (name, kwargs) for calls to
        build_sky(**kwargs), which are then inserted into
        a CompositeModel index by their names. This takes first
        precedence over all other kwargs in the function call.
    modfile : str, optional
        Filepath to a sky model saved as .pkl file.
        This takes second precedence over all other kwargs.
    device : str, optional
        Device to move model to.
    pdict : ParamDict or str, optional
        Update model with parameters in pdict
    catfile : str, optional
        Filepath to a catalogue .yaml file.
        This takes third precedence over all other kwargs.
    freqs : tensor, optional
        For sky models that don't have a frequency axis,
        use this.
    freq_interp : str, optional
        Kind of frequency interpolation on the model
        if freqs is not None.
    set_param : str, optional
        Attribute of model to set as a Parameter
    unset_param : str, optional
        Attribute of model to unset as a Parameter
    comp_kwargs : dict, optional
        Kwargs for CompositeModel when feeding multi kwarg.
    Returns
    -------
    SkyBase or CompositeModel object
    """
    # look for multiple files
    if multi is not None:
        models = {}
        for name, kwargs in multi:
            models[name] = build_sky(**kwargs)
        return sky_model.CompositeModel(models, **comp_kwargs)

    # model files
    if modfile is not None:
        model = read_pkl(modfile, device=device)

    # catalogue files
    elif catfile is not None:
        model = sky_model.read_catalogue(catfile, freqs=freqs, freq_interp=freq_interp,
                                         device=device)[0]

    if pdict is not None:
        if isinstance(pdict, str):
            pdict = ParamDict.read_pkl(pdict)
        model.update(pdict)

    if set_param is not None:
        if hasattr(model, set_param):
            model.set_param(set_param)

    if unset_param is not None:
        if hasattr(model, unset_param):
            model.unset_param(unset_param)

    return model


def build_beam(modfile=None, pdict=None, device=None):
    """
    Build a beam model

    Parameters
    ----------
    modfile : str, optional
        Filepath to a .pkl beam model
    pdict : ParamDict or str, optional
        Update model with parameters in pdict
    device : str, optional
        Device to move model to

    Returns
    -------
    PixelBeam
        beam model
    """
    if isinstance(modfile, str):
        beam = read_pkl(modfile, device=device)

    if device is not None:
        beam.push(device)

    if pdict is not None:
        if isinstance(pdict, str):
            pdict = ParamDict.read_pkl(pdict)
        beam.update(pdict)

    ### TODO: add more support for manual beams

    return beam


def build_telescope(modfile=None, location=None, device=None):
    """
    Build a telescope model

    Parameters
    ----------
    modfile : str, optional
        Filepath to a .pkl telescope model.
        This takes precedence over other kwargs.
    location : tuple
        (lon, lat, alt) in [deg, deg, m]
    """
    if isinstance(modfile, str):
        telescope = read_pkl(modfile, device=device)
    else:
        telescope = telescope_model.TelescopeModel(location)

    if device is not None:
        telescope.push(device)

    return telescope


def build_array(modfile=None, antpos=None, freqs=None, device=None,
                parameter=False, cache_f=False, cache_f_angs=None,
                interp_mode='bilinear', redtol=0.1):
    """
    Build an array model

    Parameters
    ----------
    modfile : str, optional
        Filepath to ArrayModel .pkl file.
        This takes precedence over other kwargs.
    antpos : dict, optional
        Dictionary of antenna number keys,
        ENU antpos vectors [meter] values.
        Remaining kwargs are for building an ArrayModel
        from antpos.
    freqs : tensor, optional
        Frequency array [Hz]
    device : str, optional
        Device for model
    parameter : bool, optional
        Cast antpos as a Parameter
    cache_f : bool, optional
        If True, cache fringe response
    cache_f_angs : tensor, optional
        Sky angles to evaluate and cache fringe
    interp_mode : str, optional
        Interpolation kind for fringe caching
    redtol : float, optional
        Baseline redundnacy tolerance [m]
    """
    if isinstance(modfile, str):
        return read_pkl(modfile, device=device)

    if isinstance(antpos, str):
        antpos = read_pkl(antpos)

    if isinstance(freqs, str):
        freqs = torch.as_tensor(read_pkl(freqs))

    if isinstance(cache_f_angs, str):
        cache_f_angs = toch.as_tensor(read_pkl(cache_f_angs))

    array = telescope_model.ArrayModel(antpos, freqs, parameter=parameter,
                                       device=device, cache_f=cache_f,
                                       cache_f_angs=cache_f_angs,
                                       interp_mode=interp_mode, redtol=redtol)

    return array


def build_rime(modfile=None, sky=None, beam=None, array=None,
               telescope=None, times=None, freqs=None, sim_bls=None,
               data_bls=None, device=None, pdict=None):
    """
    Build a RIME forward model object,

    Parameters
    ----------
    modfile : str, optional
        Filepath to .pkl RIME object.
        This takes precedence over other kwargs.
    sky : SkyBase, str or dict, optional
        Sky model to use, load, or build
    beam : PixelBeam, str or dict, optional
        Beam model to use, load or build
    array : ArrayModel, str, or dict, optional
        array model to use
    telesocpe : TelescopeModel, str, or dict, optional
        telescope model
    times : tensor or str, optional
        Observation times (julian date)
    freqs : tensor or str, optional
        Frequency bins [Hz]
    sim_bls ; list or str or dict, optional
        Baseline (ant-pair tuples) to simulate.
        This can be a list of tuples, a filepath to a pkl file,
        or a dict of kwargs to pass to array.get_bls()
    data_bls : list or str, optional
        Baseline (ant-pair tuples) to simulate.
        This can be a list of tuples, a filepath to a pkl file,
        or a dict of kwargs to pass to array.get_bls()
    device : str, optional
        Device of object
    pdict : ParamDict or str, optional
        parameter dictionary to update model

    Returns
    -------
    RIME object
    """
    if isinstance(modfile, str):
        return read_pkl(modfile, device=device)

    # build array
    if isinstance(array, telescope_model.ArrayModel):
        pass
    else:
        if isinstance(array, str):
            array = dict(modfile=array)
        array = build_array(**array)

    # setup metadata
    if isinstance(times, str):
        times = read_pkl(times)
    elif isinstance(times, list):
        times = np.array(times)

    if isinstance(freqs, str):
        freqs = torch.as_tensor(read_pkl(freqs))
    elif isinstance(freqs, (list, np.ndarray)):
        freqs = torch.tensor(freqs)

    if isinstance(sim_bls, str):
        sim_bls = read_pkl(sim_bls)
    elif isinstance(sim_bls, dict):
        sim_bls = array.get_bls(**sim_bls)

    if sim_bls is not None:
        if np.asarray(sim_bls).ndim > 2:
            # this is a list of sublists
            sim_bls = [[(int(bl[0]), int(bl[1])) for bl in sim_grp] for sim_grp in sim_bls]
        else:
            # this is just a list of bls
            sim_bls = [(int(bl[0]), int(bl[1])) for bl in sim_bls]

    if isinstance(data_bls, str):
        data_bls = read_pkl(data_bls)
    elif isinstance(sim_bls, dict):
        sim_bls = array.get_bls(**sim_bls)

    if data_bls is not None:
        data_bls = [(int(bl[0]), int(bl[1])) for bl in data_bls]

    # build sky
    if isinstance(sky, sky_model.SkyBase):
        pass
    else:
        if isinstance(sky, str):
            sky = dict(modfile=sky)
        sky = build_sky(**sky)

    # build beam
    if isinstance(beam, beam_model.PixelBeam):
        pass
    else:
        if isinstance(beam, str):
            beam = dict(modfile=beam)
        beam = build_beam(**beam)

    # build telescope
    if isinstance(telescope, telescope_model.TelescopeModel):
        pass
    else:
        if isinstance(telescope, str):
            telescope = dict(modfile=telescope)
        telescope = build_telescope(**telescope)

    # instantiate
    rime = rime_model.RIME(sky, telescope, beam, array, sim_bls,
                           times, freqs, data_bls=data_bls, device=device)

    # update parameters if desired
    if pdict is not None:
        if isinstance(pdict, str):
            pdict = ParamDict.read_pkl(pdict)
        rime.update(pdict)

    return rime


def build_calibration(modfile=None, device=None):
    """
    Build a direction-independent Jones model

    Parameters
    ----------
    modfile : str, optional
        Filepath to a .pkl JonesModel
    """
    ## TODO: add more support for manual calibration setup
    return read_pkl(modfile, device=device)


def build_sequential(modfile=None, order=None, kind=None, mdict=None):
    """
    Build a utils.Sequential forward model.
    See configs/model_setup.yaml for an example

    Parameters
    ----------
    modfile : str, optional
        Filepath to .pkl Sequential model. This supercedes all other
        kwargs.
    order : list, optional
        If building a Sequential object, this is a list of module
        block names in models dict in the order of their evaluation.
    kind : list, optional
        List of model types for each model in order.
        Can be one of ['sky', 'beam', 'telescope', 'array', 'rime',
        'calibration', 'sequential']
    mdict : dict, optional
        Holds model build dictionaries as values for each
        key in the "order" list

    Returns
    -------
    utils.Sequential object
    """
    if isinstance(modfile, str):
        return read_pkl(modfile)

    models = {}
    for mod, k in zip(order, kind):
        if k == 'sky':
            models[mod] = build_sky(**mdict[mod])
        elif k == 'beam':
            models[mod] = build_beam(**mdict[mod])
        elif k == 'telescope':
            models[mod] = build_telescope(**mdict[mod])
        elif k == 'array':
            models[mod] = build_array(**mdict[mod])
        elif k == 'rime':
            models[mod] = build_rime(**mdict[mod])
        elif k == 'calibration':
            models[mod] = build_calibration(**mdict[mod])
        elif k == 'sequential':
            models[mod] = build_sequential(**mdict[mod])

    return utils.Sequential(models)


def build_prob(modfile=None, seq_dict=None, data=None, start_inp=None, prior_dict=None,
               device=None, compute='post', negate=True):
    """
    Build a LogProb posterior probability object

    Parameters
    ----------
    modfile : str, optional
        Path to LobProb .pkl file. Takes precedence over
        all other kwargs.
    seq_dict : dict, optional
        Setup dictionary for sequential object
        that will be the main "model"
    data : str or list, optional
        Filepath to hdf5 data object of either
        VisData or MapData format, or list of strs
        for mini-batching.
    prior_dict : str or list, optional
        Prior dictionary to use instead of priors
        built-into the models
    device : str, optional
    compute : str, optional
        Compute posterior, likelihood or prior
        ['post', 'like', 'prior']
    negate : bool, optional
        If True return negative log-likelihood (gradient descent)
        otherwise return log-likelihood (MCMC)
    """
    if isinstance(modfile, str):
        return read_pkl(modfile, device=device)
    
    # build forward model
    model = build_sequential(**seq_dict)

    # load data, wrap as Dataset
    target = dataset.Dataset(dataset.load_data(data))

    # load others
    if isinstance(start_inp, str):
        start_inp = read_pkl(start_inp)
    if isinstance(prior_dict, str):
        prior_dict = read_pkl(prior_dict)

    prob = optim.LogProb(model, target, start_inp=start_inp,
                         prior_dict=prior_dict, device=device,
                         compute=compute, negate=negate)

    return prob


def load_yaml(yfile):
    """
    Load yaml dict

    Parameters
    ----------
    yfile : str

    Returns
    -------
    dict
    """
    import yaml
    with open(yfile) as f:
        out = yaml.load(f, Loader=yaml.FullLoader)

    return out
