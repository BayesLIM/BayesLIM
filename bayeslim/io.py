"""
Module for generic input/output. See also utils.py
"""
import numpy as np
import torch
import os

from . import utils, optim, rime_model, telescope_model, sky_model, beam_model
from .utils import _float, _cfloat


def model2pdict(model):
    """
    Build ParamDict from a model.
    Note that dict values are just pointers to the
    actual parameter tensors in the model.

    Only params from torch Modules in
    model.named_children() are returned.

    Parameters
    ----------
    model : torch.Module subclass

    Returns
    -------
    ParamDict object
    """
    d = {}
    for child in model.named_children():
        key, mod = child
        # append params from this module if exists
        if hasattr(mod, 'params'):
            ## TODO: this won't work for ArrayModel antpos params
            d[key + '.params'] = mod.params
        # add sub modules as well
        sub_d = model2pdict(mod)
        for sub_key in sub_d:
            d[key + '.' + sub_key] = sub_d[sub_key]

    return optim.ParamDict(d) 


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


def save_model(fname, model, overwrite=False):
    """
    Save a BayesLIM model as .npy

    Parameters
    ----------
    fname : str
        filepath to output .npy file
    model : torch.Module subclass
        A BayesLIM model object
    overwrite : bool, optional
        Overwrite output if it exists
    """
    if not os.path.exists(fname) or overwrite:
        np.save(fname, model)
    else:
        print("{} exists, not overwriting".format(fname))


def load_model(fname, pdict=None, device=None):
    """
    Load a BayesLIM model from a .npy file

    Parameters
    ----------
    fname : str
        Filepath to .npy holding model
    pdict : ParamDict or str, optional
        parameter dictionary (or .npy filepath
        to ParamDict) to initialize various
        model params with. Default is to 
        use existing params in model.
    device : str, optional
        If not None, device to model to

    Returns
    -------
    model : torch.Module subclass object
    """
    # load the model
    model = np.load(fname, allow_pickle=True)
    if isinstance(model, np.ndarray):
        if model.dtype == np.object_:
            model = model.item()

    # load pdict
    if pdict is not None:
        if isinstance(pdict, str):
            pdict = optim.ParamDict.load_npy(pdict)
        update_model_pdict(model, pdict)

    # move model
    if device is not None:
        model.push(device)

    return model


def update_model_pdict(model, pdict):
    """
    Update model parameters with pdict
    values inplace.

    Parameters
    ----------
    model : utils.Module subclass
    pdict : ParamDict
    """
    # iterate over keys in pdict and fill model
    for key in pdict:
        utils.set_model_attr(model, key, pdict[key])


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


def build_sky(multi=None, modfile=None, device=None,
              catfile=None, freqs=None, freq_interp='linear',
              set_param=None, unset_param=None):
    """
    Build a sky model. A few different methods
    for doing this are provided. For building a composite model,
    use the multi kwarg.

    Parameters
    ----------
    multi : list of (str, dict), optional
        A list of 2-tuples holding (name, kwargs) for
        build_sky(**kwargs), which are then inserted into
        a CompositeModel with their names. This takes first
        precedence over all other kwargs in the function call.
    modfile : str, optional
        Filepath to a sky model saved as .npy.
        This takes second precedence over all other kwargs.
    device : str, optional
        Device to move model to.
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

    Returns
    -------
    SkyBase or CompositeModel object
    """
    # look for multiple files
    if multi is not None:
        models = {}
        for name, kwargs in multi:
            models[name] = build_sky(**kwargs)
        return sky_model.CompositeModel(models)

    # model files
    if modfile is not None:
        model = load_model(modfile, device=device)

    # catalogue files
    elif catfile is not None:
        model = sky_model.read_catalogue(catfile, freqs=freqs, freq_interp=freq_interp,
                                         device=device)[0]

    if set_param is not None:
        if hasattr(model, set_param):
            setattr(model, set_param, torch.nn.Parameter(getattr(model, set_param)))
    if unset_param is not None:
        if hasattr(model, unset_param):
            setattr(model, unset_param, getattr(model, unset_param).detach())

    return model


def build_beam(modfile=None):
    """
    Build a beam model

    Parameters
    ----------
    modfile : str, optional
        Filepath to a .npy beam model

    Returns
    -------
    PixelBeam
        beam model
    """
    if isinstance(modfile, str):
        beam = load_model(modfile)

    ### TODO: add more support for manual beams

    return beam


def build_telescope(modfile=None, location=None, device=None):
    """
    Build a telescope model

    Parameters
    ----------
    modfile : str, optional
        Filepath to a .npy telescope model.
        This takes precedence over other kwargs.
    location : tuple
        (lon, lat, alt) in [deg, deg, m]

    """
    if isinstance(modfile, str):
        return load_model(modfile)

    if location is not None:
        telescope = telescope_model.TelescopeModel(location, device=device)

    return telescope


def build_array(modfile=None, antpos=None, freqs=None, device=None,
                parameter=False, cache_f=False, cache_f_angs=None,
                interp_mode='bilinear', redtol=0.1):
    """
    Build an array model

    Parameters
    ----------
    modfile : str, optional
        Filepath to ArrayModel .npy file.
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
        return load_model(modfile)

    if isinstance(antpos, str):
        antpos = load_model(antpos)

    if isinstance(freqs, str):
        freqs = torch.tensor(np.load(freqs))

    if isinstance(cache_f_angs, str):
        cache_f_angs = toch.tensor(np.load(cache_f_angs))

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
        Filepath to .npy RIME object.
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
    sim_bls ; list or str, optional
        Baseline (ant-pair tuples) to simulate
    data_bls : list or str, optional
        Baselines of the visibility, including redundancies
    device : str, optional
        Device of object
    pdict : ParamDict or str, optional
        parameter dictionary to update model

    Returns
    -------
    RIME object
    """
    if isinstance(modfile, str):
        return load_model(modfile)

    # setup basics
    if isinstance(times, str):
        times = np.load(times)
    elif isinstance(times, list):
        times = np.array(times)

    if isinstance(freqs, str):
        freqs = torch.tensor(np.load(freqs))
    elif isinstance(freqs, (list, np.ndarray)):
        freqs = torch.tensor(freqs)

    if isinstance(sim_bls, str):
        sim_bls = [tuple(bl) for bl in np.load(sim_bls).tolist()]
    elif isinstance(sim_bls, (list, np.ndarray)):
        sim_bls = [tuple(bl) for bl in sim_bls]

    if isinstance(data_bls, str):
        data_bls = [tuple(bl) for bl in np.load(data_bls).tolist()]
    elif isinstance(data_bls, (list, np.ndarray)):
        data_bls = [tuple(bl) for bl in data_bls]

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

    # build array
    if isinstance(array, telescope_model.ArrayModel):
        pass
    else:
        if isinstance(array, str):
            array = dict(modfile=array)
        array = build_array(**array)

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
            pdict = optim.ParamDict.load_npy(pdict)
        if isinstance(pdict, optim.ParamDict):
            for key, val in pdict.items():
                utils.set_model_attr(rime, key, val)

    return rime


def build_calibration(modfile=None):
    """
    Build a Jones model

    Parameters
    ----------
    modfile : str, optional
        Filepath to a .npy JonesModel
    """
    ## TODO: add more support for manual calibration setup

    return load_model(modfile)


def build_sequential(modfile=None, order=None, kind=None, mdict=None):
    """
    Build a optim.Sequential forward model.
    See configs/model_setup.yaml for an example

    Parameters
    ----------
    modfile : str, optional
        Filepath to .npy Sequential model
    order : list, optional
        List of module block names in models dict
        in order of evaluation
    kind : list, optional
        List of model types for each model in order.
        Can be one of ['sky', 'beam', 'telescope', 'array', 'rime',
        'calibration', 'sequential']
    mdict : dict, optional
        Holds model build dictionaries as values for each
        key in order

    Returns
    -------
    optim.Sequential object
    """
    if isinstance(modfile, str):
        return load_model(modfile)

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

    return optim.Sequential(models)


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
