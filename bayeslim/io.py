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


def load_model(fname, pdict=None):
    """
    Load a BayesLIM model from a .npy file

    Parameters
    ----------
    fname : str
        Filepath to .npy holding model
    pdict : ParamDict or str
        parameter dictionary (or .npy filepath
        to ParamDict) to initialize various
        model params with. Default is to 
        use existing params in model.

    Returns
    -------
    model : torch.Module subclass object
    """
    # load the model
    model = np.load(fname, allow_pickle=True).item()

    # load pdict
    if pdict is not None:
        if isinstance(pdict, str):
            pdict = optim.ParamDict.load_npy(pdict)
        # iterate over keys in pdict and fill model
        for key in pdict:
            set_model_attr(model, key, pdict[key])

    return model


def get_model_attr(model, name):
    """
    Get attr model.mod1.mod2.params
    """
    if isinstance(name, str):
        name = name.split('.')
    attr = getattr(model, name[0])
    if len(name) == 1:
        return attr
    else:
        return get_model_attr(attr, '.'.join(name[1:]))


def set_model_attr(model, name, value, inplace=True):
    """
    Set name "model.mod1.mod2.params"

    If name is a torch.nn.Parameter, cast
    value as Parameter before setting.
    If inplace, will try to update name inplace,
    i.e. keeps the same memory address.
    """
    if isinstance(name, str):
        name = name.split('.')
    if len(name) == 1:
        attr = getattr(model, name[0], None)
        value = value.to(attr.device)
        if isinstance(attr, torch.nn.Parameter):
            value = torch.nn.Parameter(value)
        setattr(model, name[0], value)
    else:
        set_model_attr(get_model_attr(model, '.'.join(name[:-1])),
                    name[-1], value, inplace=inplace)


def del_model_attr(model, name):
    """
    Del "model.mod1.mod2.params"
    """
    if isinstance(name, str):
        name = name.split('.')
    if len(name) == 1:
        delattr(model, name[0])
    else:
        delattr(get_model_attr(model, name[:-1]), name[-1])


def build_sky(sky, freqs=None):
    """
    Build a sky model

    Parameters
    ----------
    sky : str or dict
        Sky model .npy filepath or
        a dict
    kwargs : additional kwargs for certain modes
    """
    if isinstance(sky, str):
        # load sky
        sky = load_model(sky)
    elif isinstance(sky, dict):
        raise NotImplementedError
        # build sky
        models = {}
        for name in sky:
            mod = sky[name]
            if isinstance(mod, str):
                models[name] = load_model(mod)
                continue
            elif isinstance(mod, dict):
                if 'catfile' in mod:
                    models[name] = sky_model.read_catalogue(mod['catfile'], freqs,
                                                            device=mod['device'],
                                                            parameter=mod['parameter'],
                                                            freq_interp=mod['freq_interp'])
                    continue
                # load params
                p = np.load(mod['params']).item()
                params, angs = p['params'], p['angs']
                params = torch.tensor(params, device=mod['device'], dtype=_float())
                angs = torch.tensor(angs, device=mod['device'], dtype=_float())
                if mod['parameter']:
                    params = torch.nn.Parameter(params)
                models[name] = getattr(sky_model, mod['sky_type'])(params, angs, freqs)
        sky = sky_model.CompositeModel(models)

    return sky

def build_beam(beam):
    """
    Build a beam model

    Parameters
    ----------
    beam : str or dict
        Beam model .npy filepath or
        a dict
    """
    ant2beam = None
    if isinstance(beam, str):
        beam = load_model(beam)
    elif isinstance(beam, dict):        
        raise NotImplementedError
        ant2beam = beam['ant2beam']
        if isinstance(ant2beam, str):
            ant2beam = np.load(ant2beam, allow_pickle=True).item()

    return beam, ant2beam

def build_telescope(telescope):
    """
    Build a telescope model

    Parameters
    ----------
    telescope : str or dict
        Telescope model .npy filepath or
        a dict
    """
    if isinstance(telescope, str):
        telescope = load_model(telescope)
    elif isinstance(telescope, dict):
        raise NotImplementedError
        telescope = telescope_model.TelescopeModel(telescope['location'],
                                                   telescope['device'])

    return telescope

def build_array(array):
    """
    Build an array model

    Parameters
    ----------
    array : str or dict
        Array model .npy filepath or
        a dict
    """
    if isinstance(array, str):
        array = load_model(array)
    elif isinstance(array, dict):
        raise NotImplementedError
        if isinstance(array['antpos'], str):
            # ENU antpos dictionary
            array['antpos'] = np.load(array['antpos'], allow_pickle=True).item()
        if isinstance(array['cache_f_angs'], str):
            array['cache_f_angs'] = toch.tensor(np.load(array['cache_f_angs']), dtype=_float())
        array = telescope_model.ArrayModel(array['antpos'], freqs, parameter=array['parameter'],
                                           device=array['device'], cache_s=array['cache_s'],
                                           cache_f=array['cache_f'], cache_f_angs=array['cache_f_angs'],
                                           interp_mode=array['nearest'], redtol=array['redtol'])
    return array

def build_rime(mdict):
    """
    Build a RIME forward model object,
    See configs/rime_setup.yaml for an example.

    Parameters
    ----------
    mdict : dict
        model setup dictionary

    Returns
    -------
    RIME object
    """
    if isinstance(mdict, torch.nn.Module):
        return mdict

    # load yaml
    if isinstance(mdict, str):
        import yaml
        with open(mdict) as f:
            mdict = yaml.load(f, Loader=yaml.FullLoader)

    # setup basics
    times = mdict['times']
    if isinstance(times, str):
        times = np.load(times)
    elif isinstance(times, list):
        times = np.array(times)

    freqs = mdict['freqs']
    if isinstance(freqs, str):
        freqs = np.load(freqs)
    elif isinstance(freqs, list):
        freqs = torch.tensor(freqs, dtype=_float())

    sim_bls = mdict['sim_bls']
    if isinstance(sim_bls, str):
        sim_bls = [tuple(bl) for bl in np.load(sim_bls).tolist()]
    elif isinstance(sim_bls, list):
        sim_bls = [tuple(bl) for bl in sim_bls]

    data_bls = mdict['data_bls']
    if isinstance(data_bls, str):
        data_bls = [tuple(bl) for bl in np.load(data_bls).tolist()]
    elif isinstance(data_bls, list):
        data_bls = [tuple(bl) for bl in data_bls]

    # build components
    sky = build_sky(mdict['sky'], freqs=freqs)
    telescope = build_telescope(mdict['telescope'])
    array = build_array(mdict['array'], freqs=freqs)
    beam, ant2beam = build_beam(mdict['beam'])
    if ant2beam is None:
        ant2beam = {ant: 0 for ant in array.ants}

    # instantiate
    rime = rime_model.RIME(sky, telescope, beam, ant2beam, array, sim_bls,
                           times, freqs, data_bls=data_bls, device=device)

    # update parameters if desired
    if isinstance(mdict['pdict'], str):
        mdict['pdict'] = optim.ParamDict.load_npy(mdict['pdict'])
    if isinstance(mdict['pdict'], optim.ParamDict):
        for key, val in mdict['pdict'].items():
            set_model_attr(rime, key, val)

    return rime

def build_sequential_model(mdict):
    """
    Build a optim.Sequential forward model.
    See configs/model_setup.yaml for an example

    Parameters
    ----------
    mdict : dict
        Model build dictionary

    Returns
    -------
    optim.Sequential object
    """
    if isinstance(mdict, str):
        import yaml
        with open(mdict) as f:
            mdict = yaml.load(f, Loader=yaml.FullLoader)

    # get order
    order = mdict['order']

    # get model
    model = {}
    for block in order:
        model[block] = load_model(mdict[block])

    # get possible starting point
    start = mdict['starting_input']
    if isinstance(start, str):
        start = np.load(start)
        if isinstance(start, np.ndarray) and start.dtype == np.object_:
            start = start.item()

    return optim.Sequential(model, starting_input=start)

