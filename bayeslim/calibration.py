"""
Module for torch calibration models and relevant functions
"""
import torch
import numpy as np
import copy

from . import utils, linalg, dataset, telescope_model, linear_model, optim


class BaseResponse:
    """
    A base parameter response object for JonesModel, VisModel,
    and RedVisModel, taking a params tensor for JonesModel of shape
    (Npol, Npol, Nantenna, Ntimes, Nfreqs), and for (Red)VisModel
    of shape (Npol, Npol, Nbl, Ntimes, Nfreqs)
    """
    def __init__(self, freq_mode='channel', time_mode='channel', param_type='com',
                 device=None, freq_kwargs={}, time_kwargs={}, LM=None,
                 time_dim=3, freq_dim=4, projection_kwargs={}, base0=None):
        """
        Parameters
        ----------
        freq_mode : str, optional
            Frequency parameterization, ['channel', 'linear']
        time_mode : str, optional
            Time parameterization, ['channel', 'linear']
        param_type : str, optional
            dtype of input params. If 'com' push linear A matrices
            to complex type, and viewcomp params when input.
            options = ['com', 'real', 'amp', 'phs', 'amp_phs']
            If param_type is 'amp_phs', input params should be
            have (..., Ntimes, Nfreqs, 2) where the last dim holds
            (amp, phs) respectively.
        device : str, optional
            Device to place class attributes if needed
        freq_kwargs : dict, optional
            Keyword arguments for setup_freqs(). Note, must pass
            freqs [Hz] for dly param_type
        time_kwargs : dict, optional
            Keyword arguments for setup_times().
        LM : LinearModel object, optional
            Pass the input params through this LinearModel
            object before passing through the response function.
        time_dim : int, optional
            Dimension of time axis
        freq_dim : int, optional
            Dimension of frequency axis
        projection_kwargs : dict, optional
            If passed, this creates an operation on the
            fully complex parameters after passing through
            the response. See setup_projection() for details.
        base0 : tensor, optional
            Starting response to add to self(params). This redefines
            the forward model as a perturbation about base0. Must have
            the same shape as self(params) output.
        """
        self.freq_mode = freq_mode
        self.time_mode = time_mode
        self.param_type = param_type
        self.device = device
        self.freq_kwargs = freq_kwargs
        self.time_kwargs = time_kwargs
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.setup_freqs(**freq_kwargs)
        self.setup_times(**time_kwargs)
        self.setup_projection(**projection_kwargs)
        self.LM = LM
        self.base0 = base0

        # construct _args for str repr
        self._args = dict(freq_mode=self.freq_mode, time_mode=self.time_mode,
                          param_type=self.param_type)

    def setup_times(self, times=None, **kwargs):
        """
        Setup time parameterization. See required and optional
        params given time_mode.

        channel :
            None
        linear :
            linear_mode : str
            For more kwargs see linear_model.LinearModel()
            dim is hard-coded according to expected params shape
        """
        self.times = times
        if self.time_mode == 'channel':
            self.Ntime_params = None if times is None else len(times)

        elif self.time_mode == 'linear':
            # get linear A mapping wrt time
            kwgs = copy.deepcopy(kwargs)
            if 'LM' in kwgs:
                self.time_LM = kwgs['LM']
                self.time_LM.push(self.device)
            else:
                linear_mode = kwgs.pop('linear_mode')
                if times is not None:
                    kwgs['x'] = times
                kwgs['dtype'] = utils._cfloat() if self.param_type == 'com' else utils._float()
                self.time_LM = linear_model.LinearModel(linear_mode, dim=self.time_dim,
                                                        device=self.device, **kwgs)
            self.Ntime_params = self.time_LM.A.shape[1]

        else:
            raise ValueError("{} not recognized".format(self.time_mode))

    def setup_freqs(self, freqs=None, **kwargs):
        """
        Setup frequency parameterization. See required and optional
        params given freq_mode

        channel :
            None
        linear :
            linear_mode : str
            For more kwargs see linear_model.LinearModel()
            dim is hard-coded according to expected params shape
        """
        self.freqs = freqs
        if self.freq_mode == 'channel':
            self.Nfreq_params = None if freqs is None else len(freqs)

        elif self.freq_mode == 'linear':
            # get linear A mapping wrt freq
            kwgs = copy.deepcopy(kwargs)
            if 'LM' in kwgs:
                self.freq_LM = kwgs['LM']
                self.freq_LM.push(self.device)
            else:
                linear_mode = kwgs.pop('linear_mode')
                kwgs['dtype'] = utils._cfloat() if self.param_type == 'com' else utils._float()
                if freqs is not None:
                    kwgs['x'] = freqs
                self.freq_LM = linear_model.LinearModel(linear_mode, dim=self.freq_dim,
                                                        device=self.device, **kwgs)
            self.Nfreq_params = self.freq_LM.A.shape[1]

        else:
            raise ValueError("{} not recognized".format(self.freq_mode))

    def forward(self, params, **kwargs):
        """
        Forward pass params through response
        """
        # pass to device
        if not utils.check_devices(params.device, self.device):
            params = params.to(self.device)

        # pass through a LinearModel if requested
        if self.LM is not None:
            params = self.LM(params)

        # detect if params needs to be casted into complex
        if self.param_type == 'com' and not torch.is_complex(params):
            params = utils.viewcomp(params)

        # convert representation to full Ntimes, Nfreqs
        if self.freq_mode == 'channel':
            pass
        elif self.freq_mode == 'linear':
            params = self.freq_LM(params)

        if self.time_mode == 'channel':
            pass
        elif self.time_mode == 'linear':
            params = self.time_LM(params)

        if hasattr(self, 'base0') and self.base0 is not None:
            params = params + self.base0

        params = self.params2complex(params)

        # project out some component of the complex parameters
        params = self.projection(params)

        return params

    def params2complex(self, params):
        """
        Given param_type, convert params to complex form.

        Parameters
        ----------
        params : tensor
            shape (..., Ntimes, Nfreqs)

        Returns
        -------
        tensor
        """
        # convert to gains
        params = params2complex(params, self.param_type)

        return params

    def __call__(self, params, **kwargs):
        return self.forward(params, **kwargs)

    def push(self, device):
        """
        Push class attrs to new device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype: self.device = device
        if hasattr(self, 'base0') and self.base0 is not None:
            self.base0 = utils.push(base0, device)
        if hasattr(self, 'LM') and self.LM is not None:
            self.LM.push(device)
        if self.freq_mode == 'linear':
            self.freq_LM.push(device)
        if self.time_mode == 'linear':
            self.time_LM.push(device)

    def setup_projection(self, abs_amp_gain=False, phs_slope_gain=False,
                         wgts_gain=None, refant_idx=None):
        """
        Setup a block that projects the complex parameters onto a subspace
        after passing through the response.

        Parameters
        ----------
        abs_amp_gain : bool, optional
            If True, project out the antenna-averaged amplitude
            assuming params is complex gains
        phs_slope_gain : bool, optional
            If True, project out the antenna phase slope
            assuming params is complex gains
            Note this requires self.antpos to exist
        wgts_gain : tensor, optional
            1D tensor of shape (Nants,). If running abs_amp_gain or phs_slope_gain,
            these are the wgts to use in computing the degenerate parameters.
            Default is uniform weights.
        refant_idx : int, optional
            Rephase params to the index of this reference antenna
            assuming params is complex gains
        """
        self._proj_abs_amp_gain = abs_amp_gain
        self._proj_phs_slope_gain = phs_slope_gain
        self._proj_wgts_gain = wgts_gain
        if phs_slope_gain:
            assert self.antpos is not None, "phs_slope requires antpos"
        self._proj_refant_idx = refant_idx
        self._projection = abs_amp_gain or phs_slope_gain or refant_idx is not None

    def projection(self, params):
        """
        Project the complex params tensor onto a subspace.
        See self.setup_projection()

        Parameters
        ----------
        params : tensor
            Complex gain or visibility tensor

        Returns
        -------
        tensor
        """
        if not self._projection:
            return params

        # redcal degeneracies for gains
        if self._proj_abs_amp_gain or self._proj_phs_slope_gain:
            # if only need ants and antpos if running phs_slope
            ants = None if self.antpos is None else self.antpos.ants
            params = remove_redcal_degen(params, ants, self.antpos,
                                         abs_amp=self._proj_abs_amp_gain,
                                         phs_slope=self._proj_phs_slope_gain,
                                         wgts=self._proj_wgts_gain)[0]
        # refernce antenna for gains
        if self._proj_refant_idx is not None:
            idx = self._proj_refant_idx
            phs = torch.angle(params[:, :, idx:idx+1].detach())
            params /= torch.exp(1j*phs)

        return params


def params2complex(params, param_type):
    """
    Convert a parameter tensor of a gain or visibility
    into a complex form

    Parameters
    ----------
    params : tensor
        Gain or visibility in a "parameter" representation
        of type "param_type"
    param_type : str
        The type of params, ['com', 'real', 'amp', 'phs', 'amp_phs']
        If 'amp_phs', params must be of shape (..., 2) where
        the last two elements are (amp, phs) respectively.

    Returns
    -------
    tensor
    """
    if param_type == 'com':
        # assume params are already complex
        pass

    elif param_type == 'real':
        params = params + 0j

    elif param_type == 'amp':
        # assume params are amplitude
        params = torch.exp(params) + 0j

    elif param_type == 'phs':
        params = torch.exp(1j * params)

    elif param_type == 'amp_phs':
        params = torch.exp(params[..., 0] + 1j * params[..., 1])

    return params


def complex2params(data, param_type):
    """
    Convert a complex gain or visibility tensor into
    its parameter form given param_type

    Parameters
    ----------
    data : tensor
        A visibility or gain tensor of arbitrary shape
    param_type : str
        The type of params to convert to, one of
        ['com', 'real', 'amp', 'phs', 'amp_phs']

    Returns
    -------
    tensor
    """
    if param_type == 'com':
        # assume data are already complex
        pass

    elif param_type == 'real':
        data = data.real

    elif param_type == 'amp':
        # assume data are amplitude
        data = torch.log(torch.abs(data))

    elif param_type == 'phs':
        data = torch.angle(data)

    elif param_type == 'amp_phs':
        data = torch.cat([data.abs().log()[..., None], data.angle()[..., None]], dim=-1)

    return data


class IndexCache:
    """
    A class used by JonesModel, VisModel, and
    RedVisModel for time, baseline index
    caching, needed when minibatching over
    these dimensions. Assumes input to forward is
    of shape (..., Nbls, Ntimes, Nfreqs)
    """
    def __init__(self, times=None, bls=None, atol=1e-5):
        """
        Parameters
        ----------
        times : tensor, optional
            Total set of observation times
        bls : list, optional
            Total set of baseline tuples
        atol : float, optional
            Absolute tolerance for time indexing
        """
        self._times = times
        self._bls = bls
        self._atol = atol
        self.clear_time_cache()
        self.clear_bl_cache()

    def clear_time_cache(self):
        """
        Clear caching of time indices
        """
        self.cache_tidx = {}

    def get_time_idx(self, times):
        """
        Get time indices, and store in cache.
        This is used when minibatching over time axis.
        """
        if times is None or not hasattr(self, '_times'):
            return None
        h = utils.arr_hash(times)
        if h in self.cache_tidx:
            # query cache
            return self.cache_tidx[h]
        else:
            # compute time indices
            assert hasattr(self, '_times')
            idx = [torch.where(torch.isclose(self._times, t, atol=self._atol, rtol=1e-15))[0][0] for t in times]
            idx = utils._list2slice(idx)
            # store in cache and return
            self.cache_tidx[h] = idx
            return idx

    def clear_bl_cache(self):
        """
        Clear caching of bl indices
        """
        self.cache_bidx = {}

    def get_bl_idx(self, bls):
        """
        Get bl indices, and store in cache.
        This is used when minibatching over baseline axis.
        """
        if bls is None or not hasattr(self, '_bls'):
            return None
        h = utils.arr_hash(bls)
        if h in self.cache_bidx:
            # query cache
            return self.cache_bidx[h]
        else:
            # compute bl indices
            assert hasattr(self, '_bls')
            idx = [self._bls.index(bl) for bl in bls]
            idx = utils._list2slice(idx)
            # store in cache and return
            self.cache_bidx[h] = idx
            return idx

    def clear_cache(self):
        """clear all caches"""
        self.clear_time_cache()
        self.clear_bl_cache()

    def index_params(self, params, times=None, bls=None):
        if times is not None:
            idx = self.get_time_idx(times)
            if idx is None:
                params = params
            elif isinstance(idx, slice) and (idx.stop - idx.start) // idx.step == params.shape[-2]:
                params = params
            else:
                params = params[..., idx, :]
        if bls is not None:
            idx = self.get_bl_idx(bls)
            if idx is None:
                params = params
            elif isinstance(idx, slice) and (idx.stop - idx.start) // idx.step == params.shape[-3]:
                params = params
            else:
                params = params[..., idx, :, :]

        return params


class JonesModel(utils.Module, IndexCache):
    """
    A generic, antenna-based, direction-independent
    Jones term, relating the model (m) visibility to the
    data (d) visibility for antennas p and q
    and polarizations e and n.
    The Jones matrix for antenna p is constructed

    .. math::

        J_p = \\left[\\begin{array}{cc}J_{ee} & J_{en}\\\\
                    J_{ne} & J_{nn}\\end{array}\\right]

    and its application to the model visibility is

    .. math::

        V^d_{pq} = J_p \\cdot V^m_{pq} \\cdot J_q^\\dagger

    For 1-pol mode, :math:`J_p` is of shape (1, 1),
    For 2-pol mode it is diagonal of shape (2, 2),
    and 4-pol mode it is non-diagonal of shape (2, 2),
    where the off-diagonal are the so called "D-terms".
    """
    def __init__(self, params, ants, p0=None, refant=None, R=None,
                 parameter=True, polmode='1pol', single_ant=False, name=None,
                 vis_type='com', atol=1e-5):
        """
        Antenna-based Jones model.

        Parameters
        ----------
        params : tensor
            A tensor of the Jones parameters
            of shape (Npol, Npol, Nantenna, Ntimes, Nfreqs),
            where Nfreqs and Ntimes can be replaced by
            freq_Ncoeff and time_Ncoeff for sparse parameterizations.
        ants : list
            List of antenna numbers associated with an ArrayModel object
            with matched ordering to params' antenna axis, with the
            exception of single_ant mode.
        p0 : tensor, optional
            Starting params to sum with params before Response
            function. This reframes params as a perturbation about p0.
            Same shape and dtype as params. 
        refant : int, optional
            Reference antenna number from ants list for fixing the gain
            phase. Only needed if JonesResponse param_type is
            'com', 'phs', or 'dly'.
        R : callable, optional
            An arbitrary response function for the Jones parameters.
            This is a function that takes the params tensor and maps it
            into a (generally) higher dimensional space that can then
            be applied to the model visibilities. See JonesResponse()
        parameter : bool, optional
            If True, treat params as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        polmode : str, ['1pol', '2pol', '4pol'], optional
            Polarization mode. params must conform to polmode.
            1pol : single linear polarization (default)
            2pol : two linear polarizations (diag of Jones Mat)
            4pol : four linear and cross pol (2x2 Jones Mat)
        single_ant : bool, optional
            If True, solve for a single gain for all antennas.
            Nant of params must be one, but ants can still be
            the size of the array.
        name : str, optional
            Name for this object, stored as self.name
        vis_type : str, optional
            Type of visibility, complex or delay ['com', 'dly']
        atol : float, optional
            Absolute tolerance for time index caching
        """
        super().__init__(name=name)
        self.params = params
        self.device = params.device
        self.p0 = p0
        self.ants = list(ants)
        self.Nants = len(self.ants)
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        if R is None:
            # default response
            R = JonesResponse()
        self.R = R
        super(torch.nn.Module, self).__init__(times=R.times if hasattr(R, 'times') else None,
                                              atol=atol)
        self.polmode = polmode
        self.single_ant = single_ant
        self.vis_type = vis_type
        self.set_refant(refant)
        self.clear_cache()

        # construct _args for str repr
        self._args = dict(refant=refant, polmode=polmode)
        self._args[self.R.__class__.__name__] = getattr(self.R, '_args', None)

    def clear_cache(self):
        """clear all caches, some come from IndexCache object"""
        self.clear_time_cache()
        self.clear_bl_cache()
        self.clear_ant_cache()

    def clear_ant_cache(self):
        self.cache_aidx = {}

    def get_ant_idx(self, bls):
        """
        Query cache_aidx for ant -> bls mapping

        Parameters
        ----------
        bls : list of tuple
            List of baseline tuples e.g. [(0, 1), (2, 3), ...]

        Returns
        -------
        g1_idx : tensor
            len(bls) tensor indexing the Nants
            axis of params for each baseline
            for gain1 term
        g2_idx : tensor
            len(bls) tensor indexing the Nants
            axis of params for each baseline
            for gain2 term
        """
        h = utils.arr_hash(bls)
        if h not in self.cache_aidx:
            if self.single_ant:
                g1_idx = torch.as_tensor([0 for bl in bls], device=self.device)
                g2_idx = torch.as_tensor([0 for bl in bls], device=self.device)
            else:
                g1_idx = torch.as_tensor([self.ants.index(bl[0]) for bl in bls], device=self.device)
                g2_idx = torch.as_tensor([self.ants.index(bl[1]) for bl in bls], device=self.device)
            self.cache_aidx[h] = (g1_idx, g2_idx)
        else:
            g1_idx, g2_idx = self.cache_aidx[h]

        return g1_idx, g2_idx

    def set_refant(self, refant):
        """
        Set the reference antenna for phase calibration

        Parameters
        ----------
        refant : int
            Reference antenna number, to be indexed
            in the self.ants list, which should
            match the Nants ordering in self.params
        """
        self.refant, self.refant_idx = refant, None
        self.rephase_mode = None
        if refant is not None:
            assert self.refant in self.ants, "need a valid refant"
            self.refant_idx = self.ants.index(self.refant)
            if self.R.time_mode == 'channel' and self.R.freq_mode == 'channel':
                self.rephase_mode = 'rephase'
            else:
                self.rephase_mode = 'zero'
            self.fix_refant_phs()

    def fix_refant_phs(self):
        """
        Ensure that the reference antenna phase
        is set to zero: operates inplace.
        This only has an effect if the JonesResponse
        param_type is ['com', 'dly', 'phs'],
        otherwise params is unaffected.
        """
        with torch.no_grad():
            rephase_to_refant(self.params, self.R.param_type, self.refant_idx,
                              p0=self.p0, mode=self.rephase_mode, inplace=True)

    def forward(self, vd, undo=False, prior_cache=None, jones=None):
        """
        Forward pass vd through the Jones model.

        Parameters
        ----------
        vd : VisData
            Holds model visibilities of shape
            (Npol, Npol, Nbl, Ntimes, Nfreqs).
        undo : bool, optional
            If True, invert params and apply to vd. 
        prior_cache : dict, optional
            Cache for storing computed priors
        jones : tensor, optional
            Complex gains of shape
            (Npol, Npol, Nant, Ntimes, Nfreqs) to use
            instead of params attached to self.

        Returns
        -------
        VisData
            Predicted visibilities, having forwarded
            vd through the Jones parameters.
        """
        # fix reference antenna if needed
        if self.refant_idx is not None:
            self.fix_refant_phs()

        # push vd to self.device
        vd.push(self.device)

        # setup empty VisData for output
        vout = vd.copy()

        # add prior model for params
        if self.p0 is None:
            params = self.params
        else:
            params = self.params + self.p0

        # push through reponse function
        if jones is None:
            jones = self.R(params)

        # register gradient hooks if desired
        if hasattr(self, '_hook_registry') and self._hook_registry is not None:
            if jones.requires_grad:
                for r in self._hook_registry:
                    jones.register_hook(r)

        # evaluate priors on full jones tensor size
        self.eval_prior(prior_cache, inp_params=self.params, out_params=jones)

        # down select on times and freqs
        jones = self.index_params(jones, times=vd.times)

        # get g1 and g2 indexing
        g1_idx, g2_idx = self.get_ant_idx(vd.bls)

        # apply calibration and insert into output vis
        vout.data, _ = _apply_cal(vd.data, jones, g1_idx, g2_idx,
                                 cal_2pol=self.polmode=='2pol',
                                 vis_type=self.vis_type, undo=undo)

        return vout

    def push(self, device):
        """
        Push params and other attrs to new device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype:
            self.clear_cache()
            self.device = device
        self.params = utils.push(self.params, device)
        self.R.push(device)
        if self.p0 is not None:
            self.p0 = utils.push(self.p0, device)
        # push prior functions
        if self.priors_inp_params is not None:
            for pr in self.priors_inp_params:
                if pr is not None:
                    pr.push(device)
        if self.priors_out_params is not None:
            for pr in self.priors_out_params:
                if pr is not None:
                    pr.push(device)

    def to_CalData(self, pol=None, flags=None, cov=None, cov_axis=None,
                   telescope=None, antpos=None, history='', **kwargs):
        """
        Export gains as CalData object

        Parameters
        ----------
        pol : str, optional
            If 1pol mode, this is the feed pol string ('Jee' or 'Jnn').
        flags : tensor, optional
            Flag (boolean) tensor of data shape to attach to CalData
        cov : tensor, optional
            Covariance of data to attach to CalData
        cov_axis : tensor, optional
            Covariance shape of cov, see optim.apply_icov() for details.
        telescope : TelescopeModel object, optional
            TelescopeModel object associated with
            telescope to attach to CalData
        antpos : dict, optional
            Antenna position dictionry to attach to CalData,
            keys are antenna numbers, values are ENU vectors [meter]
        history : str, optional
            History to attach to CalData
        kwargs : extra attrs for CalData if not found on object

        Returns
        -------
        CalData object
        """
        with torch.no_grad():
            # setup object and metadata
            cd = dataset.CalData()
            cd.setup_meta(telescope=telescope, antpos=antpos)

            # get gains
            if self.p0 is not None:
                p = (self.params + self.p0).detach()
            else:
                p = self.params.detach()
            gains = self.R(p).cpu().clone()

            # try to get time and freq metadata
            if 'freqs' in self.R.freq_kwargs:
                freqs = self.R.freq_kwargs['freqs']
            else:
                try:
                    freqs = kwargs['freqs']
                except KeyError:
                    raise ValueError("freqs not found in self.R.freq_kwargs, must pass as kwarg")
            if 'times' in self.R.time_kwargs:
                times = self.R.time_kwargs['times']
            else:
                try:
                    times = kwargs['times']
                except KeyError:
                    raise ValueError("times not found in self.R.time_kwargs, must pass as kwarg")

            # setup object
            cd.setup_data(ants=self.ants, times=times,
                          freqs=freqs, pol=pol,
                          data=gains, flags=flags, cov=cov,
                          cov_axis=cov_axis,
                          history=history)

            return cd


class JonesResponse(BaseResponse):
    """
    A response object for JonesModel, subclass of BaseResponse
    taking params of shape (Npol, Npol, Nantenna, Ntimes, Nfreqs).

    Allows for polynomial parameterization across time and/or frequency,
    and for a gain type of complex, amplitude, phase, delay, EW & NS delay slope,
    and EW & NS phase slope (the latter two are relevant for redundant calibration) 
    """
    def __init__(self, freq_mode='channel', time_mode='channel', param_type='com',
                 vis_type='com', antpos=None, device=None,
                 freq_kwargs={}, time_kwargs={}, LM=None, base0=None):
        """
        Parameters
        ----------
        freq_mode : str, optional
            Frequency parameterization, ['channel', 'linear']
        time_mode : str, optional
            Time parameterization, ['channel', 'linear']
        param_type : str, optional
            Type of gain parameter. One of
            ['com', 'dly', 'amp', 'phs', 'dly_slope', 'phs_slope']
                'com' : complex gains
                'dly' : delay, g = exp(2i * pi * freqs * delay)
                'amp' : amplitude, g = exp(amp)
                'phs' : phase, g = exp(i * phs)
                '*_slope' : spatial gradient, [EastWest, NorthSouth]
        vis_type : str, optional
            Type of visibility, complex or delay ['com', 'dly']
        antpos : dict
            Antenna position dictionary for dly_slope or phs_slope
            Keys as antenna integers, values as 3-vector in ENU frame
        device : str, optional
            Device to place class attributes if needed
        freq_kwargs : dict, optional
            Keyword arguments for setup_freqs(). Note, must pass
            freqs [Hz] for dly param_type
        time_kwargs : dict, optional
            Keyword arguments for setup_times().
        LM : LinearModel object, optional
            Pass the input params through this LinearModel
            object before passing through the response function.

        Notes
        -----
        For param_type in ['phs_slope', 'dly_slope]
            antpos is required. params tensor is assumed
            to hold the [EW, NS] slope along its antenna axis.
        """
        super().__init__(freq_mode=freq_mode, time_mode=time_mode,
                         param_type=param_type, device=device,
                         freq_kwargs=freq_kwargs, time_kwargs=time_kwargs,
                         LM=LM, base0=base0)
        self.vis_type = vis_type
        self.antpos = antpos

        if self.param_type in ['dly_slope', 'phs_slope']:
            # setup antpos tensors
            assert antpos is not None, 'need antpos for dly_slope or phs_slope'
            EW = torch.as_tensor([antpos[a][0] for a in antpos], device=self.device)
            self.antpos_EW = EW[None, None, :, None, None]  
            NS = torch.as_tensor([antpos[a][1] for a in antpos], device=self.device)
            self.antpos_NS = NS[None, None, :, None, None]
        elif 'dly' in self.param_type:
            assert self.freqs is not None, 'need frequencies for delay gain type'

        assert self.param_type in ['com', 'amp', 'phs', 'dly', 'real',
                                   'amp_phs', 'phs_slope', 'dly_slope']

    def params2complex(self, jones):
        """
        Convert jones to complex gain given param_type.
        Note this should be after passing jones through
        its response function, such that the jones tensor
        is a function of time and frequency.

        Parameters
        ----------
        jones : tensor
            jones parameter of shape (Npol, Npol, Nant, Ntimes, Nfreqs)

        Returns
        -------
        tensor
            Complex gain tensor (Npol, Npol, Nant, Ntimes, Nfreqs)
        """
        jones = super().params2complex(jones)

        if self.param_type == 'dly':
            # assume jones are in delay [nanosec]
            if self.vis_type == 'dly':
                pass
            elif self.vis_type == 'com':
                jones = torch.exp(2j * np.pi * jones * torch.as_tensor(self.freqs / 1e9, dtype=jones.dtype))

        elif self.param_type == 'dly_slope':
            # extract EW and NS delay slopes: ns / meter
            EW = jones[:, :, :1]
            NS = jones[:, :, 1:]
            # get total delay per antenna
            tot_dly = EW * self.antpos_EW \
                      + NS * self.antpos_NS
            if self.vis_type == 'com':
                # convert to complex gains
                jones = torch.exp(2j * np.pi * tot_dly * self.freqs / 1e9)
            elif self.vis_type == 'dly':
                jones = tot_dly

        elif self.param_type == 'phs_slope':
            # extract EW and NS phase slopes: rad / meter
            EW = jones[:, :, :1]
            NS = jones[:, :, 1:]
            # get total phase per antenna
            tot_phs = EW * self.antpos_EW \
                      + NS * self.antpos_NS
            # convert to complex gains
            jones = torch.exp(1j * tot_phs)

        return jones

    def push(self, device):
        """
        Push class attrs to new device
        """
        super().push(device)
        if self.param_type in ['dly_slope', 'phs_slope']:
            self.antpos_EW = self.antpos_EW.to(device) 
            self.antpos_NS = self.antpos_NS.to(device)


class RedVisModel(utils.Module, IndexCache):
    """
    Redundant visibility model (r) relating the starting
    model visibility (m) to the data visibility (d)
    for antennas j and k.

    .. math::

        V^{d}_{jk} = V^{r} + V^{m}_{jk}

    """
    def __init__(self, params, bl2red, R=None, parameter=True, p0=None,
                 name=None, atol=1e-5):
        """
        Redundant visibility model

        Parameters
        ----------
        params : tensor
            Initial redundant visibility tensor
            of shape (Npol, Npol, Nredvis, Ntimes, Nfreqs) where Nredvis
            is the number of unique baseline types.
        bl2red : dict
            Maps a baseline tuple, e.g. (1, 3), to its corresponding redundant
            baseline index of self.params along its Nredvis axis.
            See telescope_model.build_reds()
        R : VisModelResponse object, optional
            A response function for the redundant visibility
            model parameterization. Default is freq and time channels.
        parameter : bool, optional
            If True, treat params as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        p0 : tensor, optional
            Starting params to sum with params before the response
            function. This reframes params as a perturbation about p0.
            Same shape and dtype as params.
        name : str, optional
            Name for this object, stored as self.name
        atol : float, optional
            Absolute tolerance for time index caching
        """
        super().__init__(name=name)
        self.params = params
        self.device = params.device
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        self.bl2red = bl2red
        if R is None:
            # default response is per freq channel and time bin
            R = VisModelResponse()
        self.R = R
        self.p0 = p0
        super(torch.nn.Module, self).__init__(times=R.times if hasattr(R, 'times') else None,
                                              atol=atol)
        self.clear_cache()

    def forward(self, vd, undo=False, prior_cache=None):
        """
        Forward pass vd through redundant
        model term.

        Parameters
        ----------
        vd : VisData
            Starting model visibilities of shape
            (Npol, Npol, Nredbl, Ntimes, Nfreqs). In the general case,
            this should be a zero tensor so that the
            predicted visibilities are simply the redundant
            model. However, if you have a model of per-baseline
            non-redundancies, these could be included by putting
            them into vd. If this holds zero tensor as data, then
            this is mainly used to propagate important metadata
            downstream.
        undo : bool, optional
            If True, push vd backwards through the model.
        prior_cache : dict, optional
            Cache for holding computed priors.

        Returns
        -------
        VisData
            The predicted visibilities, having pushed vd through
            the redundant visibility model.
        """
        # push to device
        if not utils.check_devices(self.device, vd.data.device):
            vd.push(self.device)

        # setup predicted visibility
        vout = vd.copy()

        # get unique visibilities
        if self.p0 is not None:
            params = self.params + self.p0
        else:
            params = self.params
        redvis = self.R(params)

        # register gradient hooks if desired
        if hasattr(self, '_hook_registry') and self._hook_registry is not None:
            if redvis.requires_grad:
                for r in self._hook_registry:
                    redvis.register_hook(r)

        # evaluate priors
        self.eval_prior(prior_cache, inp_params=self.params, out_params=redvis)

        # down select on time
        redvis = self.index_params(redvis, times=vd.times)

        # expand redvis to vis size if needed
        if redvis.shape[2] != vout.data.shape[2]:
            index = self.get_bl_idx(vd.bls)
            redvis = torch.index_select(redvis, 2, index)

        # apply redvis model
        if not undo:
            vout.data += redvis
        else:
            vout.data -= redvis

        return vout

    def get_bl_idx(self, bls):
        """
        Get indexing tensor that expands
        redvis to vis shape along Nbls axis.
        Overloads IndexCache.get_bl_idx
        """
        if not isinstance(bls, list):
            bls = [bls]
        h = utils.arr_hash(bls)
        if h not in self.cache_bidx:
            index = torch.as_tensor(
                [self.bl2red[bl] for bl in bls],
                device=self.device)
            self.cache_bidx[h] = index
        else:
            index = self.cache_bidx[h]

        return index

    def clear_cache(self):
        """clear all caches, some come from IndexCache object"""
        self.clear_time_cache()
        self.clear_bl_cache()

    def clear_bl_cache(self):
        self.cache_bidx = {}

    def push(self, device):
        """
        Push to a new device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype: self.device = device
        self.params = utils.push(self.params, device)
        self.R.push(device)
        if self.p0 is not None:
            self.p0 = utils.push(self.p0, device)
        if not dtype:
            for h in self.cache_bidx:
                self.cache_bidx[h] = self.cache_bidx[h].to(device)
        # push prior functions
        if self.priors_inp_params is not None:
            for pr in self.priors_inp_params:
                pr.push(device)
        if self.priors_out_params is not None:
            for pr in self.priors_out_params:
                pr.push(device)


class VisModel(utils.Module, IndexCache):
    """
    Visibility model (v) relating the starting
    model visibility (m) to the data visibility (d)
    for antennas j and k.

    .. math::

        V^{d}_{jk} = V^{v}_{jk} + V^{m}_{jk} 

    """
    def __init__(self, params, R=None, parameter=True, p0=None,
                 name=None, atol=1e-5):
        """
        Visibility model

        Parameters
        ----------
        params : tensor
            Visibility model parameter of shape
            (Npol, Npol, Nbl, Ntimes, Nfreqs). Ordering should
            match ordering of vd input to self.forward.
        R : callable, optional
            An arbitrary response function for the
            visibility model, mapping the parameters
            to the space of vd (input to self.forward).
            Default (None) is unit response.
            Note this must use torch functions.
        parameter : bool, optional
            If True, treat vis as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        p0 : tensor, optional
            Starting params to sum with params before Response
            function. This reframes params as a perturbation about p0.
            Same shape and dtype as params.
        name : str, optional
            Name for this object, stored as self.name
        atol : float, optional
            Absolute tolerance for time index caching
        """
        super().__init__(name=name)
        self.params = params
        self.device = params.device
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        if R is None:
            # default response is per freq channel and time bin
            R = VisModelResponse()
        self.R = R
        self.p0 = p0
        super(torch.nn.Module, self).__init__(times=R.times if hasattr(R, 'times') else None,
                                              bls=R.bls if hasattr(R, 'bls') else None,
                                              atol=atol)
        self.clear_cache()

    def forward(self, vd, undo=False, prior_cache=None, **kwargs):
        """
        Forward pass vd through visibility
        model term.

        Parameters
        ----------
        vd : VisData
            Starting model visibilities
            of shape (Npol, Npol, Nbl, Ntimes, Nfreqs). In the general case,
            this should be a zero tensor so that the
            predicted visibilities are simply the redundant
            model. However, if you have a model of per-baseline
            non-redundancies, these could be included by putting
            them into vd.
        undo : bool, optional
            If True, push vd backwards through the model.
        prior_cache : dict, optional
            Cache for storing computed priors

        Returns
        -------
        VisData
            The predicted visibilities, having summed vd
            with the visibility model.
        """
        vout = vd.copy()

        # forward model params
        if self.p0 is not None:
            params = self.params + self.p0
        else:
            params = self.params
        vis = self.R(params)

        # register gradient hooks if desired
        if hasattr(self, '_hook_registry') and self._hook_registry is not None:
            if vis.requires_grad:
                for r in self._hook_registry:
                    vis.register_hook(r)

        # evaluate priors
        self.eval_prior(prior_cache, inp_params=self.params, out_params=vis)

        # down select
        vis = self.index_params(vis, times=vd.times, bls=vd.bls)

        if not undo:
            vout.data = vout.data + vis
        else:
            vout.data = vout.data - vis

        return vout

    def push(self, device):
        """
        Push to a new device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype: self.device = device
        self.R.push(device)
        self.params = utils.push(self.params, device)
        if self.p0 is not None:
            self.p0 = utils.push(self.p0, device)
        # push prior functions
        if self.priors_inp_params is not None:
            for pr in self.priors_inp_params:
                pr.push(device)
        if self.priors_out_params is not None:
            for pr in self.priors_out_params:
                pr.push(device)


class VisModelResponse(BaseResponse):
    """
    A response object for VisModel and RedVisModel, subclass of BaseResponse
    taking params of shape (Npol, Npol, Nbls, Ntimes, Nfreqs)
    """
    def __init__(self, bls=None, freq_mode='channel', time_mode='channel',
                 param_type='real', device=None, time_dim=3, freq_dim=4,
                 freq_kwargs={}, time_kwargs={}, LM=None, base0=None):
        """
        Parameters
        ----------
        bls : list of 2-tuple, optional
            List of baseline tuples for baseline indexing (minibatching)
            e.g. [(0, 1), (1, 2). ...] along the Nbls params dimension
        freq_mode : str, optional
            Frequency parameterization, ['channel', 'linear']
        time_mode : str, optional
            Time parameterization, ['channel', 'linear']
        device : str, None
            Device for object
        param_type : str, optional
            Type of params ['com', 'real' 'amp', 'amp_phs']
            com : visibility represented as real and imag params
                where the last dim is [real, imag]
            amp_phs : visibility represented as amplitude and phase
                params, where the last dim of params is [amp, phs]
        LM : LinearModel object, optional
            Pass the input params through this LinearModel
            object before passing through the response function.
        """
        super().__init__(freq_mode=freq_mode, time_mode=time_mode, time_dim=time_dim,
                         freq_dim=freq_dim, param_type=param_type, device=device,
                         freq_kwargs=freq_kwargs, time_kwargs=time_kwargs, LM=LM,
                         base0=base0)

    def forward(self, params, bls=None, times=None, **kwargs):
        """
        Forward pass params through response to get
        complex visibility model per time and frequency
        """
        params = super().forward(params)

        return params


class VisCoupling(utils.Module, IndexCache):
    """
    A visibility coupling module, describing
    a linear transformation of visibilities across
    the baseline axis.

    Let V be an Nantenna x Nantenna hermitian matrix holding
    all N^2 visibilities, then single-path mutual coupling
    including 1st and 2nd order terms can be described as 

        Vc = X @ V @ X^T

    where Vc is the coupled visibility and X = I + E where E
    is the Nantenna x Nantenna matrix holding coupling
    coefficients between antennas.

    Note that because params should be of shape
    (Npol, Npol, Nantenna, Nantenna, Ntime_coeff, Nfreq_coeff),
    the Response object should have time_dim=4, freq_dim=5.

    Currently only supports single-pol coupling.
    """
    def __init__(self, params, freqs, antpos, bls, R=None, parameter=True,
                 p0=None, name=None, atol=1e-5, add_I=True, prod='both'):
        """
        Visibility coupling model. Note this does not support baseline
        minibatching (all baselines must exist in the input VisData object).
        Must run self.setup_coupling() before using this model.

        Parameters
        ----------
        params : tensor
            (Npol, Npol, Nantenna, Nantenna, Ntime_coeff, Nfreq_coeff) tensor
            describing coupling between antennas
        freqs : tensor
            Observing frequencies [Hz]
        antpos : dict
            Dictionary holding antenna positions in ENU coordiates. Keys
            are antenna integers, ordered according to the Nantenna
            axis of params, values are antenna vectors in ENU frame [meters]
        bls : list of tuples
            List of baselines along the Nbls axis of the input VisData
            object
        R : callable, optional
            Response object for params, mapping it from
            sparse basis to its (Npol, Npol, ... Nfreqs) shape
            Default is VisModelResponse
        parameter : bool, optional
            If True, treat params as differentiable
        p0 : tensor, optional
            Starting params to sum with params before Response
            function. This reframes params as a perturbation about p0.
            Same shape and dtype as params.
        name : str, optional
            Name for this module, default is class name
        atol : float, optional
            Absolute tolerance for time index caching
        add_I : bool, optional
            If True, (default) add identity to the coupling matrix (I + E)
        prod : str, optional
            If 'both' (default) form Vc = X V0 X^T
            if 'left' form Vc = X V0
            if 'right' form Vc = V0 X^T
        """
        super().__init__(name=name)
        ## TODO: support multi-pol coupling
        ## TODO: support frequency batching (for IndexCache and beam, sky objects too)
        ## TODO: support multi-path coupling
        ## TODO: support OWL-QN LBFGS optimization for sparse-priors on coupling terms
        self.params = params
        self.p0 = p0
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.antpos = antpos
        self.Nants = len(antpos)
        self.bls = bls
        self.c = 2.99792458e8
        self.device = params.device
        self.Npol = params.shape[0]
        assert self.Npol == 1, "multi-pol not currently supported"
        self.add_I = add_I
        self.prod = prod
        if parameter:
            self.params = torch.nn.Parameter(self.params)

        self.Nants = params.shape[2]

        if R is None:
            # default response is per freq channel and time bin
            R = VisModelResponse(time_dim=4, freq_dim=5)
        self.R = R

        # IndexCache.__init__
        super(torch.nn.Module, self).__init__(times=R.times if hasattr(R, 'times') else None,
                                              bls=R.bls if hasattr(R, 'bls') else None,
                                              atol=atol)
        self.clear_cache()

    def setup_coupling(self, bls=None):
        """
        Setup coupling forward model metadata (e.g. delay term and matrix indexing)

        Parameters
        ----------
        bls : list of ant-pair tuples
            The baselines of the input data
        """
        # setup time delay phasor between coupled antennas
        self.dly = torch.ones(1, 1, self.Nants, self.Nants, 1, self.Nfreqs,
                              dtype=utils._cfloat(), device=self.device)
        dfreqs = self.freqs - self.freqs[0]
        for i, ant1 in enumerate(self.antpos):
            for j, ant2 in enumerate(self.antpos):
                vec_len = torch.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
                self.dly[0, 0, i, j, 0] = torch.exp(2j*torch.pi*dfreqs/self.c*vec_len)

        # setup indexing of input VisData and its reshaping into square matrix and back
        ## TODO: this assumes ant2 >= ant1 ordering of input bls. relax this assumption...
        if bls is not None:
            self.bls = bls
        mat_bls = [[(ant1, ant2) for ant2 in self.antpos] for ant1 in self.antpos]
        flat_data_idx = []
        flat_data_null = []
        flat_unconj_idx = []
        flat_conj_idx = []
        k = 0
        for row in mat_bls:
            for mbl in row:
                if mbl[1] >= mbl[0]:
                    try:
                        idx = self.bls.index(mbl)
                        flat_data_null.append(False)
                    except ValueError:
                        # mbl not in self.bls
                        idx = 0  # placeholder
                        flat_data_null.append(True)
                    flat_unconj_idx.append(k)

                else:
                    try:
                        idx = self.bls.index(mbl[::-1])
                        flat_data_null.append(False)
                    except ValueError:
                        # mbl[::-1] not in self.bls
                        idx = 0  # placeholder
                        flat_data_null.append(True)
                    flat_conj_idx.append(k)

                flat_data_idx.append(idx)

                k += 1

        self.flat_data_idx = torch.tensor(flat_data_idx)
        self.flat_conj_idx = torch.tensor(flat_conj_idx)
        self.flat_unconj_idx = torch.tensor(flat_unconj_idx)
        self.flat_data_null = torch.tensor(flat_data_null)

        # now get indexing from square-flattened data back to bls ordering
        _argsort_idx = self.flat_data_idx.clone()
        _argsort_idx[self.flat_conj_idx] = 1e10  # set all conjugated elements to near-inf
        _argsort_idx[self.flat_data_null] = 1e10  # set all null elements to near-inf
        self.bls_idx = torch.argsort(_argsort_idx)[:len(self.bls)]

        # TODO: turn flat_data_idx, bls_idx, flat_data_null into slice objects if possible

        # setup identity matrix (Npol, Npol, Nant, Nant, Ntime, Nfreq)
        self.I = torch.eye(
            self.Nants, dtype=utils._float(), device=self.device
            )[None, None, :, :, None, None]

    def forward(self, vd, prior_cache=None, add_I=None, prod=None, **kwargs):
        """
        Forward pass vd through visibility coupling
        model term.

        Parameters
        ----------
        vd : VisData
            Starting model visibilities
            of shape (Npol, Npol, Nbls, Ntimes, Nfreqs).
        prior_cache : dict, optional
            Cache for storing computed priors on self.params

        Returns
        -------
        VisData
            The predicted visibilities, having pushed input
            through coupling matrix
        """
        vout = dataset.VisData()
        vout.setup_meta(telescope=vd.telescope, antpos=vd.antpos)

        # forward model
        if self.p0 is not None:
            params = self.params + self.p0
        else:
            params = self.params
        coupling = self.R(params)

        # register gradient hooks if desired
        if hasattr(self, '_hook_registry') and self._hook_registry is not None:
            if coupling.requires_grad:
                for r in self._hook_registry:
                    coupling.register_hook(r)

        # evaluate priors
        self.eval_prior(prior_cache, inp_params=self.params, out_params=coupling)

        # down select on times
        coupling = self.index_params(coupling, times=vd.times)
        Ntimes = coupling.shape[3]

        # multiply by delay term
        coupling *= self.dly
        Nfreqs = coupling.shape[5]

        # add identity
        add_I = add_I if add_I is not None else self.add_I
        if add_I:
            coupling += self.I

        # reshape input data along bls axis
        flat_data = torch.index_select(vd.data, 2, self.flat_data_idx)

        # for bls that don't exist in vd.data, null them out
        flat_data[:, :, self.flat_data_null] = 0.0

        # conjugate lower triangular component
        flat_data[:, :, self.flat_conj_idx] = flat_data[:, :, self.flat_conj_idx].conj()

        # reshape to (..., Nant, Nant, ...) size
        shape = vd.data.shape
        new_shape = shape[:2] + (self.Nants, self.Nants) + shape[3:]
        reshaped_data = flat_data.reshape(new_shape)

        # take product of coupling matrix with visibility matrix
        prod = prod if prod is not None else self.prod
        if prod in ['left', 'both']:
            reshaped_data = torch.einsum("ijkl...,ijl...->ijk...", coupling, reshaped_data)
        if prod in ['right', 'both']:
            reshaped_data = torch.einsum("ijkl...,ijml...->ijkm...", reshaped_data, coupling.conj())

        # unravel to flat along Nbls axis
        flat_coupled = reshaped_data.flatten(start_dim=2, end_dim=3)

        # pick out original bls
        data = torch.index_select(flat_coupled, 2, self.bls_idx)

        vout.setup_data(bls=vd.bls, times=vd.times, freqs=vd.freqs, pol=vd.pol,
                        data=data, flags=vd.flags, cov=vd.cov,
                        cov_axis=vd.cov_axis, icov=vd.icov, history=vd.history)

        return vout

    def index_params(self, params, times=None, bls=None):
        """overload IndexCache index_params b/c of time broadcast"""
        if times is not None:
            # if only 1 time bin in params, assume we want to broadcast
            if params.shape[-2] == 1:
                pass
            else:
                params = super(utils.Module, self).index_params(params, times=times)

        return super(utils.Module, self).index_params(params, bls=bls)

    def push(self, device):
        """
        Push to a new device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype and hasattr(self, "bls_idx"):
            self.device = device
            self.flat_data_idx = self.flat_data_idx.to(device)
            self.flat_conj_idx = self.flat_conj_idx.to(device)
            self.flat_unconj_idx = self.flat_unconj_idx.to(device)
            self.bls_idx = self.bls_idx.to(device)
        if hasattr(self, 'I'):
            self.I = utils.push(self.I, device)

        self.params = utils.push(self.params, device)
        self.dly = utils.push(self.dly, device)
        self.freqs = utils.push(self.freqs, device)
        self.R.push(device)
        if self.p0 is not None:
            self.p0 = utils.push(self.p0, device)
        # push prior functions
        if self.priors_inp_params is not None:
            for pr in self.priors_inp_params:
                pr.push(device)
        if self.priors_out_params is not None:
            for pr in self.priors_out_params:
                pr.push(device)


class RedVisCoupling(utils.Module, IndexCache):
    """
    A visibility coupling matrix assuming input visibilities
    are a redundant model and the output is a set of full visibilities.

    If Vr is a redundant visibility column vector, then the coupled
    (non-redundant visibilities) are 

        Vc = A @ Vr + B @ Vr.conj()

    where Vr is a column vector of length Nredbl and
    A, B are matrices of shape (Nbls, Nredbls).

    In contrast to VisCoupling, the params tensor of
    RedVisCoupling should be of shape
    (Npol, Npol, Ncoupling, Ntimes, Nfreqs), where
    Ncoupling is the total number of coupling terms
    one wants to model.

    Notation: eps_0_1 is the coupling of voltage v_1 -> v_0,
    such that v_0 <- v_0 + eps_0_1 * v_1
    """
    def __init__(self, params, freqs, antpos, coupling_terms, bls_in, bls_out,
                 coupling_idx=None, R=None, parameter=True, p0=None, name=None,
                 atol=1e-5):
        """
        Visibility coupling model. Note this does not support baseline
        minibatching (all baselines must exist in the input VisData object).
        Must run self.setup_coupling() before using this model.

        Parameters
        ----------
        params : tensor
            (Npol, Npol, Ncoupling, Ntime_coeff, Nfreq_coeff) tensor
            describing coupling between antennas
        freqs : tensor
            Observing frequencies [Hz]
        antpos : dict
            Antenna position dictionary, ant integer as key, ENU baseline vec
            as value
        coupling_terms : list
            A list of antenna-integer pairs denoting the coupling coefficients
            to be modeled, matching the ordering in params along its Ncoupling axis.
            e.g. [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 2), (2, 1)]
            where (0, 1) means antenna0 -> antenna1 (eps_0_1).
        bls_in : list
            The baseline ordering (ant-pair tuples) of the VisData object that is
            input to this forward pass. E.g. [(0, 1), (1, 2), (2, 3), ...]
        bls_out : list
            The baseline ordering (ant-pair tuples) of the VisData object that is
            output from the forward pass. E.g. [(0, 1), (1, 2), (2, 3), ...]
        coupling_idx : dict, optional
            Keys are coupling term tuples and values are the index of this term
            in the coupling_terms list. Default is to generate automatically.
            To operate in "redundant coupling" mode, this should be passed
            by the user. See gen_coupling_terms(..., compress_to_red=True)
        use_reds : bool
            If True, bls_in represents unique redundant model baselines, otherwise
            it represents all physical baselines.
        R : callable, optional
            Response object for params, mapping it from
            sparse basis to its (Npol, Npol, ... Nfreqs) shape
            Default is VisModelResponse
        parameter : bool, optional
            If True, treat params as differentiable
        p0 : tensor, optional
            Starting params to sum with params before Response
            function. This reframes params as a perturbation about p0.
            Same shape and dtype as params.
        name : str, optional
            Name for this module, default is class name
        atol : float, optional
            Absolute tolerance for time index caching
        """
        ## TODO: support multi-pol coupling
        ## TODO: support frequency batching (for IndexCache and beam, sky objects too)
        ## TODO: support multi-path coupling
        ## TODO: support OWL-QN LBFGS optimization for sparse-priors on coupling terms
        super().__init__()
        self.params = params
        self.p0 = p0
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.antpos = antpos
        self.Nants = len(antpos)
        self.c = 2.99792458e8
        self.device = params.device
        self.Npol = params.shape[0]
        assert self.Npol == 1, "multi-pol not currently supported"
        if parameter:
            self.params = torch.nn.Parameter(self.params)

        if R is None:
            # default response is per freq channel and time bin
            R = VisModelResponse(time_dim=3, freq_dim=4)
        self.R = R

        # IndexCache.__init__
        super(torch.nn.Module, self).__init__(times=R.times if hasattr(R, 'times') else None,
                                              bls=R.bls if hasattr(R, 'bls') else None,
                                              atol=atol)
        self.clear_cache()

        self.coupling_terms = coupling_terms
        if coupling_idx is None:
            coupling_idx = {c: i for i, c in enumerate(coupling_terms)}
        self.coupling_idx = coupling_idx
        self.Nterms = len(coupling_terms)
        self.bls_in = bls_in
        self.bls_out = bls_out
        self.Nants = len(antpos)

    def setup_coupling(self, use_reds=True, redtol=1.0,
                       include_second_order=True, min_len=None, max_len=None,
                       max_EW=None, max_NS=None, second_max_len=None, second_max_EW=None,
                       second_max_NS=None, Nproc=None, Ntask=5, use_pathos=True):
        """
        Setup fixed coupling parameters (e.g. delay term and matrix indexing).
        Assumes bl = (ant1, ant2) where ant1 <= ant2 ordering.

        Parameters
        ----------
        use_reds : bool, optional
            If True assume self.bls_in represents unique redundant baselines,
            otherwise assume it represents physical baselines
        redtol : float, optional
            Red tolerance for use_reds in meters
        include_second_order : bool, optional
            If True, include second order (single-path) coupling terms.
        min_len : float, optional
            Minimum ant-ant vector length for coupling terms
        max_len : float, optional
            Maximum ant-ant vector length for coupling terms
        max_EW : float, optional
            Maximum ant-ant East-West vector length for coupling terms
        max_NS: float, optional
            Maximum ant-ant North-South vector length for coupling terms
        second_max_* : float, optional
            Max length/EW/NS parameters for second order coupling terms, if
            include_second_order is True.
        Nproc : int, optional
            If not None, launch multiprocessing of Nprocesses
        Ntask : int, optional
            Number of jobs to solve for for each process
        use_pathos : bool, optional
            Use pathos multiproc instead of multiprocessing
        """
        if use_reds:
            # build redundancies
            reds, rvec, bl2red_idx, all_bls, lens, angs, _ = telescope_model.build_reds(self.antpos,
                bls=self.bls_out, redtol=redtol)
            bl2red = {}
            for k in bl2red_idx:
                bl2red[k] = reds[bl2red_idx[k]][0]
                bl2red[k[::-1]] = reds[bl2red_idx[k]][0][::-1]

        else:
            bl2red = None

        # setup time delay phasor between coupled antennas
        self.dly = torch.ones(1, 1, self.Nterms, 1, self.Nfreqs, dtype=utils._cfloat(), device=self.device)
        freqs = self.freqs.to(self.device)
        freqs = freqs - freqs[0]
        for i, (ant1, ant2) in enumerate(self.coupling_terms):
            bl_len = torch.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
            self.dly[0, 0, i, 0] = torch.exp(2j*np.pi*freqs/self.c*bl_len)

        # get the rows of the A matrix, mapping input bls to output bls
        Arows = configure_coupling_matrix_singlepath(self.antpos, bls=self.bls_out, bl2red=bl2red,
                                                     include_second_order=include_second_order,
                                                     min_len=min_len, max_len=max_len, max_EW=max_EW,
                                                     max_NS=max_NS,
                                                     second_max_len=second_max_len,
                                                     second_max_EW=second_max_EW, second_max_NS=second_max_NS,
                                                     Nproc=Nproc, Ntask=Ntask, use_pathos=use_pathos)
        if use_reds:
            self.red_bls = [bl2red[bl] for bl in self.bls_out]
        else:
            self.red_bls = self.bls_out

        # Create indexing lists for matrix operations that need to be performed:
        # 1. (coupling + coupling.conj + coupling * coupling.conj) @ vis
        # 2. (coupling + coupling.conj + coupling * coupling.conj) @ vis.conj

        # Below, the first two lists index the rows & cols of A matrix, respectively,
        # and the third list indexes coupling tensor along its Ncoupling axis.
        unconj_param_unconj_vis = ([], [], [])
        unconj_param_conj_vis = ([], [], [])
        conj_param_unconj_vis = ([], [], [])
        conj_param_conj_vis = ([], [], [])

        # Below the first two lists index the rows & cols of A matrix, respectively.
        # Then we have indices of unique elements in A_ij, and indices of coupling
        # and coupling.conj(), respectively:
        # 1. Index A_i
        # 2. Index A_j
        # 3. Counter of unique elements mapped to A_ij used in torch.index_add()
        # 4. indexing of coupling along Ncoupling dim for coupling
        # 5. indexing of coupling along Ncoupling dim for coupling.conj()
        sq_param_unconj_vis = ([], [], [], [], [])
        sq_param_conj_vis = ([], [], [], [], [])

        k, l = 0, 0
        unconj_inds = set()
        conj_inds = set()
        for i, blo in enumerate(self.bls_out):
            Arow = Arows[blo]
            for j, bli in enumerate(self.bls_in):
                # setup for unconj_vis bli
                if bli in Arow:
                    sq_appended = False
                    for eps in Arow[bli]:
                        if ',' in eps:
                            # this is a second order term: eps_0_1,eps_1_2_conj
                            terms = eps.split(',')
                            # turn eps_0_1 -> (0, 1), note that conj term is always second
                            c_id1 = tuple(int(a) for a in terms[0].split("_")[1:3])
                            c_id2 = tuple(int(a) for a in terms[1].split("_")[1:3])
                            if c_id1 not in self.coupling_idx or c_id2 not in self.coupling_idx:
                                continue
                            c_idx1 = self.coupling_idx[c_id1]
                            c_idx2 = self.coupling_idx[c_id2]
                            if (i, j) not in unconj_inds:
                                # append unique A_ij indices
                                unconj_inds.add((i, j))
                                sq_param_unconj_vis[0].append(i)
                                sq_param_unconj_vis[1].append(j)
                            sq_param_unconj_vis[2].append(k)
                            sq_param_unconj_vis[3].append(c_idx1)  # unconj term
                            sq_param_unconj_vis[4].append(c_idx2)  # conj term
                            sq_appended = True

                        else:
                            # this is a first order term
                            # there is only one conj & unconj first-order term per Arow[bli]
                            c_id = tuple(int(a) for a in eps.split("_")[1:3])  # turn eps_0_1 -> (0, 1)
                            if c_id not in self.coupling_idx:
                                continue
                            c_idx = self.coupling_idx[c_id]
                            if 'conj' in eps:
                                # this is conj_param
                                conj_param_unconj_vis[0].append(i)
                                conj_param_unconj_vis[1].append(j)
                                conj_param_unconj_vis[2].append(c_idx)
                            else:
                                # this is unconj_param
                                unconj_param_unconj_vis[0].append(i)
                                unconj_param_unconj_vis[1].append(j)
                                unconj_param_unconj_vis[2].append(c_idx)

                    if sq_appended:
                        k += 1

                # setup for conj_vis bli[::-1]
                if (bli[0] != bli[1]) and (bli[::-1] in Arow):
                    sq_appended = False
                    for eps in Arow[bli[::-1]]:
                        if ',' in eps:
                            # this is a second order term: eps_0_1,eps_1_2_conj
                            terms = eps.split(',')
                            # turn eps_0_1 -> (0, 1), note that conj term is always second
                            c_id1 = tuple(int(a) for a in terms[0].split("_")[1:3])
                            c_id2 = tuple(int(a) for a in terms[1].split("_")[1:3])
                            if c_id1 not in self.coupling_idx or c_id2 not in self.coupling_idx:
                                continue
                            c_idx1 = self.coupling_idx[c_id1]
                            c_idx2 = self.coupling_idx[c_id2]
                            if (i, j) not in conj_inds:
                                # append unique A_ij indices
                                conj_inds.add((i, j))
                                sq_param_conj_vis[0].append(i)
                                sq_param_conj_vis[1].append(j)
                            sq_param_conj_vis[2].append(l)
                            sq_param_conj_vis[3].append(c_idx1)  # unconj term
                            sq_param_conj_vis[4].append(c_idx2)  # conj term
                            sq_appended = True

                        else:
                            # this is a first order term
                            # there is only one conj & unconj first-order term per Arow[bli]
                            c_id = tuple(int(a) for a in eps.split("_")[1:3])  # turn eps_0_1 -> (0, 1)
                            if c_id not in self.coupling_idx:
                                continue
                            c_idx = self.coupling_idx[c_id]
                            if 'conj' in eps:
                                # this is conj_param
                                conj_param_conj_vis[0].append(i)
                                conj_param_conj_vis[1].append(j)
                                conj_param_conj_vis[2].append(c_idx)
                            else:
                                # this is unconj_param
                                unconj_param_conj_vis[0].append(i)
                                unconj_param_conj_vis[1].append(j)
                                unconj_param_conj_vis[2].append(c_idx)

                    if sq_appended:
                        l += 1


        self.unconj_param_unconj_vis = (torch.as_tensor(unconj_param_unconj_vis[0]),
                                        torch.as_tensor(unconj_param_unconj_vis[1]),
                                        torch.as_tensor(unconj_param_unconj_vis[2]),
                                        )
        self.unconj_param_conj_vis = (torch.as_tensor(unconj_param_conj_vis[0]),
                                      torch.as_tensor(unconj_param_conj_vis[1]),
                                      torch.as_tensor(unconj_param_conj_vis[2]),
                                      )
        self.conj_param_unconj_vis = (torch.as_tensor(conj_param_unconj_vis[0]),
                                      torch.as_tensor(conj_param_unconj_vis[1]),
                                      torch.as_tensor(conj_param_unconj_vis[2]),
                                      )
        self.conj_param_conj_vis = (torch.as_tensor(conj_param_conj_vis[0]),
                                    torch.as_tensor(conj_param_conj_vis[1]),
                                    torch.as_tensor(conj_param_conj_vis[2]),
                                    )
        self.sq_param_unconj_vis = (torch.as_tensor(sq_param_unconj_vis[0]),
                                    torch.as_tensor(sq_param_unconj_vis[1]),
                                    torch.as_tensor(sq_param_unconj_vis[2]),
                                    torch.as_tensor(sq_param_unconj_vis[3]),
                                    torch.as_tensor(sq_param_unconj_vis[4]),
                                    )
        self.sq_param_conj_vis = (torch.as_tensor(sq_param_conj_vis[0]),
                                  torch.as_tensor(sq_param_conj_vis[1]),
                                  torch.as_tensor(sq_param_conj_vis[2]),
                                  torch.as_tensor(sq_param_conj_vis[3]),
                                  torch.as_tensor(sq_param_conj_vis[4]),
                                  )

    def forward(self, vd, prior_cache=None, **kwargs):
        """
        Forward pass vd through visibility coupling
        model term.

        Parameters
        ----------
        vd : VisData
            Starting model visibilities
            of shape (Npol, Npol, Nbl, Ntimes, Nfreqs).
        prior_cache : dict, optional
            Cache for storing computed priors on self.params

        Returns
        -------
        VisData
            The predicted visibilities, having pushed input
            through coupling matrix
        """
        # this is the inflated 0th order visibilities
        vout = vd._inflate_by_redundancy(self.bls_out, self.red_bls)

        # forward model
        if self.p0 is not None:
            params = self.params + self.p0
        else:
            params = self.params
        coupling = self.R(params)

        # register gradient hooks if desired
        if hasattr(self, '_hook_registry') and self._hook_registry is not None:
            if coupling.requires_grad:
                for r in self._hook_registry:
                    coupling.register_hook(r)

        # evaluate priors
        self.eval_prior(prior_cache, inp_params=self.params, out_params=coupling)

        # down select on times
        coupling = self.index_params(coupling, times=vd.times)
        Ntimes = coupling.shape[3]

        # multiply by delay term
        coupling = coupling * self.dly
        Nfreqs = coupling.shape[4]

        # construct coupling matrix for unconjugated data
        if len(self.unconj_param_unconj_vis[0]) > 0 or len(self.conj_param_unconj_vis[0]) > 0:
            mat = torch.zeros(
                (1, 1, len(self.bls_out), len(self.bls_in), Ntimes, Nfreqs),
                dtype=vd.data.dtype,
                device=self.device
            )

            # add in first order unconjugated params and conjugated params
            if len(self.unconj_param_unconj_vis[0]) > 0:
                prms = torch.index_select(coupling, 2, self.unconj_param_unconj_vis[2])
                mat[:, :, self.unconj_param_unconj_vis[0], self.unconj_param_unconj_vis[1]] += prms

            if len(self.conj_param_unconj_vis[0]) > 0:
                prms = torch.index_select(coupling.conj(), 2, self.conj_param_unconj_vis[2])
                mat[:, :, self.conj_param_unconj_vis[0], self.conj_param_unconj_vis[1]] += prms

            # add in second order params
            if len(self.sq_param_unconj_vis[0]) > 0:
                # index all necesary squared combinations
                _prms = torch.index_select(coupling, 2, self.sq_param_unconj_vis[3]) \
                       * torch.index_select(coupling.conj(), 2, self.sq_param_unconj_vis[4])
                # index_add() to unique elements in A_ij
                prms = torch.zeros(
                    _prms.shape[:2] + self.sq_param_unconj_vis[0].shape + _prms.shape[3:],
                    device=_prms.device,
                    dtype=_prms.dtype
                )
                prms.index_add_(2, self.sq_param_unconj_vis[2], _prms)
                mat[:, :, self.sq_param_unconj_vis[0], self.sq_param_unconj_vis[1]] += prms

            # take product with vis and add to output
            vout.data += torch.einsum("ijkl...,ijl...->ijk...", mat, vd.data)

        # construct coupling matrix for conjugated data
        if len(self.unconj_param_conj_vis[0]) > 0 or len(self.conj_param_conj_vis[0]) > 0:
            mat = torch.zeros(
                (1, 1, len(self.bls_out), len(self.bls_in), Ntimes, Nfreqs),
                dtype=vd.data.dtype,
                device=self.device
            )

            # add in unconjugated params and conjugated params
            if len(self.unconj_param_conj_vis[0]) > 0:
                prms = torch.index_select(coupling, 2, self.unconj_param_conj_vis[2])
                mat[:, :, self.unconj_param_conj_vis[0], self.unconj_param_conj_vis[1]] += prms
            if len(self.conj_param_conj_vis[0]) > 0:
                prms = torch.index_select(coupling.conj(), 2, self.conj_param_conj_vis[2])
                mat[:, :, self.conj_param_conj_vis[0], self.conj_param_conj_vis[1]] += prms

            # add in second order params
            if len(self.sq_param_conj_vis[0]) > 0:
                # index all necesary squared combinations
                _prms = torch.index_select(coupling, 2, self.sq_param_conj_vis[3]) \
                       * torch.index_select(coupling.conj(), 2, self.sq_param_conj_vis[4])
                # index_add() to unique elements in A_ij
                prms = torch.zeros(
                    _prms.shape[:2] + self.sq_param_conj_vis[0].shape + _prms.shape[3:],
                    device=_prms.device,
                    dtype=_prms.dtype
                )
                prms.index_add_(2, self.sq_param_conj_vis[2], _prms)
                mat[:, :, self.sq_param_conj_vis[0], self.sq_param_conj_vis[1]] += prms

            # take product with vis and add to output
            vout.data += torch.einsum("ijkl...,ijl...->ijk...", mat, vd.data.conj())

        return vout

    def index_params(self, params, times=None, bls=None):
        """overload IndexCache index_params b/c of time broadcast"""
        if times is not None:
            # if only 1 time bin in params, assume we want to broadcast
            if params.shape[-2] == 1:
                pass
            else:
                params = super(utils.Module, self).index_params(params, times=times)

        return super(utils.Module, self).index_params(params, bls=bls)

    def push(self, device):
        """
        Push to a new device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype:
            self.device = device
            for k in ['unconj_param_unconj_vis',
                      'unconj_param_conj_vis',
                      'conj_param_unconj_vis',
                      'conj_param_conj_vis',
                      'sq_param_unconj_vis',
                      'sq_param_conj_vis']:
                arr = getattr(self, k)
                arr = tuple(a.to(device) for a in arr)
                setattr(self, k, arr)
        self.params = utils.push(self.params, device)
        self.dly = utils.push(self.dly, device)
        if self.p0 is not None:
            self.p0 = utils.push(self.p0, device)
        # push prior functions
        if self.priors_inp_params is not None:
            for pr in self.priors_inp_params:
                pr.push(device)
        if self.priors_out_params is not None:
            for pr in self.priors_out_params:
                pr.push(device)

    def get_coupling_hits(self):
        """
        Compute how many times each coupling term
        is used in the forward transform

        Returns
        -------
        dict
        """
        hits = {}
        for i, ct in enumerate(self.coupling_terms):
            Nhit = 0
            Nhit += len(np.where(self.conj_param_conj_vis[1] == i)[0])
            Nhit += len(np.where(self.conj_param_unconj_vis[1] == i)[0])
            Nhit += len(np.where(self.unconj_param_conj_vis[1] == i)[0])
            Nhit += len(np.where(self.unconj_param_unconj_vis[1] == i)[0])
            Nhit += len(np.where(self.sq_param_unconj_vis[1] == i)[0])
            Nhit += len(np.where(self.sq_param_unconj_vis[2] == i)[0])
            Nhit += len(np.where(self.sq_param_conj_vis[1] == i)[0])
            Nhit += len(np.where(self.sq_param_conj_vis[2] == i)[0])

            hits[ct] = Nhit

        return hits


class CouplingExpand:
    """
    Take a coupling vector and expand to an Nants x Nants
    coupling matrix.
    """
    def __init__(self, vecs, antpos, redtol=1.0):
        raise NotImplementedError
        self.vecs = vecs
        self.antpos = antpos
        self.ants = list(antpos.keys())
        Nants = len(antpos)
        self.redtol = redtol
        self.shape = (Nants, Nants)

        idx = torch.zeros(self.shape, dtype=torch.int64)
        zeros = torch.zeros(self.shape, dtype=torch.bool)

        # iterate over antenna-pairs
        for i in range(Nants):
            for j in range(Nants):
                vec = antpos[self.ants[j]] - antpos[self.ants[i]]
                norm = (vecs - vec).norm(dim=1).isclose(torch.tensor(0.), atol=redtol)
                if norm.any():
                    idx[i, j] = i * Nants + j
                else:
                    zeros[i, j] = True

        self.idx = idx.ravel()
        self.zeros = zeros.ravel()

    def __call__(self, params):
        # params = (Npol, Npol, Nvec, Ntimes, Nfreqs)
        shape = params.shape[:2] + self.shape + params.shape[-2:]

        # params = (Npol, Npol, Nant^2, Ntimes, Nfreqs)
        params = params[:, :, self.idx]

        # zero-out missing vectors
        params[:, :, self.zeros] = 0.0

        # params = (Npol, Npol, Nant, Nant, Ntimes, Nfreqs)
        params = params.reshape(shape)

        return params

    def push(self, device):
        self.idx = utils.push(self.idx, device)
        self.zeros = utils.push(self.zeros, device)


def apply_cal(vis, bls, gains, ants, cal_2pol=False, cov=None,
              vis_type='com', undo=False, inplace=False):
    """
    Apply calibration to a visibility tensor with a complex
    gain tensor. Default behavior is to multiply
    vis with gains, i.e. when undo = False. To divide
    vis by gains, set undo = True.
    
    Note: for a function that applies CalData objects
    see VisData.apply_cal()

    .. math::

        V_{12}^{\rm out} = g_1 V_{12}^{\rm inp} g_2^\ast

    Parameters
    ----------
    vis : tensor
        Visibility tensor of shape
        (Npol, Npol, Nbls, Ntimes, Nfreqs)
    bls : list
        List of baseline antenna-pair tuples e.g. [(0, 1), ...]
        of vis along Nbls dimension.
    gains : tensor
        Gain tensor of shape
        (Npol, Npol, Nants, Ntimes, Nfreqs)
    ants : list
        List of antenna integers of gains along Nants dimension
    cal_2pol : bool, optional
        If True, calibrate 4pol vis with diagonal of 4pol gains
        i.e., ignore off-diagonal gain terms. Only applicable if 
        vis and gains are 4pol (i.e. Npol=2)
    cov : tensor, optional
        The covariance of vis, to be updated by gains. Note this
        currently only works for a cov with a cov_axis of None,
        i.e. it is just the data variance of the same shape,
        for 1pol or 2pol mode, and for vis_type of 'com'.
    vis_type : str, optional
        Type of visibility and gain tensor. ['com', 'dly'].
        If 'com', vis and gains are complex (default).
        If 'dly', vis and gains are float delays.
    undo : bool, optional
        If True, divide vis by gains, otherwise
        (default) multiply vis by gains.
    inplace : bool, optional
        If True edit input vis inplace, otherwise make a copy

    Returns
    -------
    vout : tensor
        Output visibilities
    cov_out : tensor, optional
        Output vis covariance
    """
    g1_idx = torch.as_tensor([ants.index(bl[0]) for bl in bls], device=gains.device)
    g2_idx = torch.as_tensor([ants.index(bl[1]) for bl in bls], device=gains.device)

    return _apply_cal(vis, gains, g1_idx, g2_idx, cal_2pol=cal_2pol, cov=cov,
                      vis_type=vis_type, undo=undo, inplace=inplace)


def _apply_cal(vis, gains, g1_idx, g2_idx, cal_2pol=False, cov=None,
               vis_type='com', undo=False, inplace=False):
    """
    Apply calibration

    See apply_cal() for details on args and kwargs.

    g1_idx, g2_idx : tensor
        len(Nbls) tensor indexing the Nants dimension of gains for
        each antenna (g1, g2) participating in a given baseline in vis.

    Returns
    -------
    new_vis, new_cov
    """
    assert vis.shape[:2] == gains.shape[:2], "vis and gains must have same Npols"

    # get polmode
    polmode = '1pol' if vis.shape[:2] == (1, 1) else '4pol'
    if cal_2pol and polmode == '4pol':
        polmode = '2pol'

    # invert gains if necessary
    if undo:
        invgains = torch.zeros_like(gains)
        # iterate over antennas
        for i in range(gains.shape[2]):
            if polmode in ['1pol', '2pol']:
                if vis_type == 'com':
                    invgains[:, :, i] = linalg.diag_inv(gains[:, :, i])
                elif vis_type == 'dly':
                    invgains[:, :, i] = -gains[:, :, i]
            else:
                assert vis_type == 'com', 'must have complex vis_type for 4pol mode'
                invgains[:, :, i] = torch.pinv(gains[:, :, i])
        gains = invgains

    if inplace:
        vout = vis
    else:
        vout = torch.zeros_like(vis)
    if cov is not None:
        if inplace:
            cov_out = cov
        else:
            cov_out = torch.zeros_like(cov)
    else:
        cov_out = cov

    # legacy: haven't shown this is faster or more mem efficient, just is more verbose
    #g1 = torch.gather(gains, 2, g1_idx[None, None, :, None, None].expand_as(vis))
    #g2 = torch.gather(gains, 2, g2_idx[None, None, :, None, None].expand_as(vis))
    g1 = gains.index_select(2, g1_idx)
    g2 = gains.index_select(2, g2_idx)

    if polmode in ['1pol', '2pol']:
        # update visibilities
        if vis_type == 'com':
            G = g1 * g2.conj()
            vout = linalg.diag_matmul(G, vis)

            # update covariance
            if cov is not None:
                GG = G * G.conj()
                if torch.is_complex(GG):
                    GG = GG.real
                cov_out = linalg.diag_matmul(GG, cov)

        elif vis_type == 'dly':
            vout = vis + g1 - g2

    else:
        assert vis_type == 'com', "must have complex vis_type for 4pol mode"
        vout = torch.einsum("ab...,bc...,dc...->ad...", g1, vis, g2.conj())

    return vout, cov_out


def rephase_to_refant(params, param_type, refant_idx, p0=None, mode='rephase', inplace=False):
    """
    Rephase an antenna calibration parameter tensor such that
    the reference antenna has zero phase. In the default case,
    this will divide all antennas by the refant phasor for
    param_type = 'com'. For 'dly' or 'phs', the refant is subtracted.
    If mode = 'zero', then the refant imag component is simply set to zero
    and other antennas are not affected.

    Parameters
    ----------
    params : tensor
        Antenna gain parameters of shape
        (Npol, Npol, Nants, Ntimes, Nfreqs)
    param_type : str
        Type of params tensor ['com', 'phs', 'dly', 'amp', 'amp_phs']
        If 'amp_phs', params is (..., 2) ordered as (amp, phs) on last dim.
    refant_idx : int
        Reference antenna index in Nants axis of params
    p0 : tensor, optional
        Initial params tensor that is directly summed with params
        before solving for reference antenna phase. Must have
        same shape and dtype as params. Default is zeros.
        Note in inplace, this tensor is also updated inplace.
    mode : str, optional
        If 'rephase', divide all antennas by refant phase (default)
        or 'zero', just zero-out the imag component of the refant
        (for 'com') or zero-out its phase (for 'dly' or 'phs')
    inplace : bool, optional
        If True operate inplace otherwise return new copy.

    Returns
    -------
    params : tensor
        rephased (or zero'd) params tensor if not inplace
    p0 : tensor
        rephased (or zero'd) p0 tensor if not inplace
    """
    if refant_idx is None:
        return

    if p0 is None:
        p0 = torch.zeros_like(params)

    if not inplace:
        params = copy.deepcopy(params)
        p0 = copy.deepcopy(p0)

    if mode == 'rephase':
        # rephase all antennas to refant phase
        _p = params
        _p0 = p0

        if param_type == 'com':
            # divide out refant complex phasor
            if not torch.is_complex(params):
                _p = utils.viewcomp(params)
                _p0 = utils.viewcomp(p0)

            # get refant phasor and divide params by it
            phs = torch.angle((_p + _p0)[:, :, refant_idx:refant_idx+1]).detach().clone()
            phasor = torch.exp(1j * phs)
            _p /= phasor
            _p0 /= phasor

            if not torch.is_complex(params):
                # recast as view_real
                _p = utils.viewreal(_p)
                _p0 = utils.viewreal(_p0)

            params[:] = _p
            p0[:] = _p0

        elif param_type in ['dly', 'phs']:
            # subtract dly or phs of refant for all antennas
            params -= params[:, :, refant_idx:refant_idx+1].clone()
            p0 -= p0[:, :, refant_idx:refant_idx+1].clone()

        elif param_type == 'amp_phs':
            # subtract phs of refant
            params[..., 1] -= params[:, :, refant_idx:refant_idx+1, ..., 1].clone()
            p0[..., 1] -= p0[:, :, refant_idx:refant_idx+1, ..., 1].clone()

    elif mode == 'zero':
        # just zero-out refant imag component (or dly / phs)
        if param_type == 'com':
            # use zeros_like b/c scalar assignment on GPU breaks
            if not torch.is_complex(params):
                params[:, :, refant_idx:refant_idx+1, ..., 1] = torch.zeros_like(
                    params[:, :, refant_idx:refant_idx+1, ..., 1]
                )
                p0[:, :, refant_idx:refant_idx+1, ..., 1] = torch.zeros_like(
                    p0[:, :, refant_idx:refant_idx+1, ..., 1]
                )

            else:
                params.imag[:, :, refant_idx:refant_idx+1] = torch.zeros_like(
                    params.imag[:, :, refant_idx:refant_idx+1]
                )
                p0.imag[:, :, refant_idx:refant_idx+1] = torch.zeros_like(
                    p0.imag[:, :, refant_idx:refant_idx+1]
                )
        elif param_type in ['dly', 'phs']:
            params[:, :, refant_idx:refant_idx+1] = torch.zeros_like(
                params[:, :, refant_idx:refant_idx+1]
            )
            p0[:, :, refant_idx:refant_idx+1] = torch.zeros_like(
                p0[:, :, refant_idx:refant_idx+1]
            )
        elif param_type == 'amp_phs':
            params[:, :, refant_idx:refant_idx+1, ..., 1] = torch.zeros_like(
                params[:, :, refant_idx:refant_idx+1, ..., 1]
            )
            p0[:, :, refant_idx:refant_idx+1, ..., 1] = torch.zeros_like(
                p0[:, :, refant_idx:refant_idx+1, ..., 1]
            )

    if not inplace:
        return params, p0


def remove_redcal_degen(gains, ants, antpos, degen=None,
                        wgts=None, redvis=None, bls=None,
                        abs_amp=True, phs_slope=True):
    """
    Remove redcal degeneracies from a set of gains and model visibilities
    Note this currently only works for 1pol or 2pol gains.

    Parameters
    ----------
    gains : tensor
        Complex gain tensor of shape (Npol, Npol, Nants, Ntimes, Nfreqs)
    ants : list
        List of antenna numbers along Nants dim of gains
    antpos : AntposDict object
        Antenna position dictionary, ant num as key, ENU antvec [meter] as value
    degen : tensor, optional
        Complex gain tensor of new redcal degeneracy to insert into gains
    wgts : tensor, optional
        1D weight tensor of length Nants used for computing degeneracies
    redvis : tensor, optional
        Redundant model visibility to remove degenerate gains from
        of shape (Npol, Npol, Nbls, Ntimes, Nfreqs)
    bls : list, optional
        List of baseline tuples along Nbls dim of redvis
    abs_amp : bool, optional
        If True, compute absolute amplitude parameter (default)
    phs_slope : bool, optional
        If True, compute phase slope parameter (default)

    Returns
    -------
    new_gains : tensor
    new_vis : tensor
    degen_gains : tensor
    """
    # compute degenerate gains
    rd = compute_redcal_degen(gains.detach(), ants, antpos, wgts=wgts, abs_amp=abs_amp, phs_slope=phs_slope)
    degen_gains = redcal_degen_gains(ants=ants, antpos=antpos, abs_amp=rd[0], phs_slope=rd[1])

    if degen is not None:
        degen_gains /= degen

    # get new gains
    new_gains = gains / degen_gains

    # get new vis
    if redvis is None:
        new_vis = None
    else:
        new_vis = apply_cal(redvis, bls, degen_gains, ants, undo=False)[0]

    return new_gains, new_vis, degen_gains


def compute_redcal_degen(gains, ants, antpos, wgts=None, abs_amp=True, phs_slope=True):
    """
    Given a set of antenna gains compute the degeneracy
    parameters of redundant calibration, 1. the overall
    gain amplitude and 2. the antenna location phase gradient,
    where the antenna gains are related to the parameters as

    .. math::

        g^{\rm abs}_p = \exp[\eta^{\rm abs}]

    and

    .. math::

        g^{\rm phs}_p = \exp[i r_p \cdot \Phi]

    Parameters
    ----------
    gains : tensor
        Antenna gains of shape (Npol, Npol, Nant, Ntimes, Nfreqs)
    ants : list
        List of antenna numbers along the Nant axis
    antpos : AntposDict object
        Dictionary of ENU antenna vectors for each antenna number
    wgts : tensor, optional
        1D weight tensor to use in computing degenerate parameters
        of len(Nants). Normally, this should be the total number
        of visibilities used in redcal for each antenna.
        Default is uniform weighting.
    abs_amp : bool, optional
        If True, compute absolute amplitude parameter (default)
    phs_slope : bool, optional
        If True, compute phase slope parameter (default)

    Returns
    -------
    tensor
        absolute amplitude parameter of shape
        (Npol, Npol, 1, Ntimes, Nfreqs)
    tensor
        phase gradient parameter [rad / meter] of shape
        (Npol, Npol, 2, Ntimes, Nfreqs) where the two
        elements are the [East, North] gradients respectively
    """
    # get weights
    if wgts is None:
        w = 1.
        wsum = 1.
    else:
        w = wgts[:, None, None]
        wsum = torch.sum(wgts)

    # compute absolute amplitude parameter: average abs of squared gains
    abs_amp_param = None
    if abs_amp:
        abs_amp_param = torch.sum(torch.abs(gains)**2 * w, dim=2, keepdims=True) / wsum
        abs_amp_param = torch.log(torch.sqrt(abs_amp_param))

    ### LEGACY
    #eta = torch.log(torch.abs(gains))
    #abs_amp = torch.sum(eta * wgts, dim=2, keepdims=True) / wsum

    # compute phase slope parameter
    phs_slope_param = None
    if phs_slope:
        gain_phs = torch.angle(gains)
        A = antpos[[a for a in ants]][:, :2]
        if wgts is None:
            AtWAinvAtW = torch.pinverse(A.T @ A) @ A.T
        else:
            W = torch.eye(len(ants), device=gains.device) * wgts / wsum
            AtWAinvAtW = torch.pinverse(A.T @ W @ A) @ A.T @ W
        phs_slope_param = torch.einsum("ab,ijblm->ijalm", AtWAinvAtW, gain_phs)

    return abs_amp_param, phs_slope_param


def redcal_degen_gains(abs_amp=None, phs_slope=None, ants=None, antpos=None):
    """
    Given redcal degenerate parameters, transform to their complex gains

    Parameters
    ----------
    abs_amp : tensor, optional
        Absolute amplitude parameter of shape
        (Npol, Npol, 1, Ntimes, Nfreqs)
    phs_slope : tensor, optional
        Phase slope parameter of shape
        (Npol, Npol, 2, Ntimes, Nfreqs) where the two
        elements are the [East, North] gradients [rad / meter]
    ants : list
        List of antenna numbers for the output gains
    antpos : AntposDict object, optional
        Mapping of antenna number to antenna ENU vector [meters].
        Needed for phs_slope parameter

    Returns
    -------
    tensor
        Complex gains of shape (Npol, Npol, Nant, Ntimes, Nfreqs)
    """
    # fill unit gains
    device = None
    if abs_amp is not None:
        device = abs_amp.device
    elif phs_slope is not None:
        device = phs_slope.device
    gains = torch.ones(1, 1, 1, 1, 1, dtype=utils._cfloat(), device=device)

    # incorporate absolute amplitude
    if abs_amp is not None:
        gains = gains * torch.exp(abs_amp)

    # incorporate phase slope
    if phs_slope is not None:
        A = antpos[[a for a in antpos]][:, :2]
        phs = (phs_slope.moveaxis(2, -1) @ A.T).moveaxis(-1, 2)
        gains = gains * torch.exp(1j * phs)

    return gains


def compute_redcal_degen_vis(vd, wgts=None, abs_amp=True, phs_slope=True,
                             bls=None, antpos=None):
    """
    Given a VisData, compute the redcal degeneracies
    (absolute amplitude and X-Y phase slope) and return

    Parameters
    ----------
    vd : tensor or VisData object
        Visibility data to compute redcal degeneracies from
    wgts : tensor, optional
        Weight tensor to multiply into data along Nbls axis
        before computing degeneracies. shape (Nbls,)
    abs_amp : bool, optional
        If True, compute the absolute amplitude parameter
    phs_slope : bool, optional
        If True, compute the phase slope parameter
    bls : list, optional
        List of baseline antenna-pairs of the data along
        its Nbls axis, if vd is passed as a tensor
    antpos : AntposDict, optional
        Antenna number keys and ENU vectors [m] values if
        vd is passed as a tensor

    Returns
    -------
    abs_amp : tensor
        Baseline-averaged amplitude of shape (Npol, Npol, 1, Ntimes, Nfreqs)
    phs_slope : tensor
        Phase slope gradient [rad/m] of shape (Npol, Npol, 2, Ntimes, Nfreqs)
        where the 2nd axis holds (Phi_EW, Phi_NS).
    """
    data = vd.data if isinstance(vd, dataset.VisData) else vd
    # get weights
    if wgts is None:
        w = 1.
        wsum = 1.
    else:
        w = wgts[:, None, None]
        wsum = torch.sum(wgts)

    # compute absolute amplitude parameter: average abs of gains
    abs_amp_param = None
    if abs_amp:
        abs_amp_param = torch.sum(torch.abs(data) * w, dim=2, keepdims=True) / wsum
        abs_amp_param = torch.log(abs_amp_param)

    # compute phase slope parameter
    phs_slope_param = None
    if phs_slope:
        vis_phs = torch.angle(data)
        bls = vd.bls if isinstance(vd, dataset.VisData) else bls
        antpos = vd.antpos if isinstance(vd, dataset.VisData) else antpos
        ant1, ant2 = zip(*bls)
        A = (antpos[ant1] - antpos[ant2])[:, :2]
        if wgts is None:
            AtWAinvAtW = torch.pinverse(A.T @ A) @ A.T
        else:
            W = torch.eye(Nants, device=gains.device) * wgts / wsum
            AtWAinvAtW = torch.pinverse(A.T @ W @ A) @ A.T @ W
        phs_slope_param = torch.einsum("ab,ijblm->ijalm", AtWAinvAtW, vis_phs)

    return abs_amp_param, phs_slope_param


def redcal_degen_vis(abs_amp=None, phs_slope=None, vd=None, bls=None, antpos=None):
    """
    Given redcal degeneracies for a visibility, compute the
    degenerate visibilities

    Parameters
    ----------
    abs_amp : tensor, optional
        Absolute amplitude visibility parameter of shape
        (Npol, Npol, 1, Ntimes, Nfreqs)
    phs_slope : tensor, optional
        Phase slope visibility parameter
        (Npol, Npol, 2, Ntimes, Nfreqs)
    vd : VisData
        Holding the original visibilities of shape
        (Npol, Npol, Nbls, Ntimes, Nfreqs). If provided,
        the output will be a new VisData with the degnerate
        visibilities and the same metadata. Otherwise
        output is a tensor.
    bls : list, optional
        List of baseline antenna-pairs of the data along
        its Nbls axis, if vd is not passed
    antpos : AntposDict, optional
        Antenna number keys and ENU vectors [m] values if
        vd is not passed

    Returns
    -------
    VisData or tensor
    """
    if vd:
        out = dataset.VisData()
        out.setup_meta(telescope=vd.telescope, antpos=vd.antpos)

    if abs_amp is not None:
        amp_data = torch.exp(abs_amp)

    if phs_slope is not None:
        bls = vd.bls if vd else bls
        antpos = vd.antpos if isinstance(vd, dataset.VisData) else antpos
        ant1, ant2 = zip(*bls)
        A = (antpos[ant1] - antpos[ant2])[:, :2]
        phs = (phs_slope.moveaxis(2, -1) @ A.T).moveaxis(-1, 2)
        phs_data = torch.exp(1j * phs)

    data = None
    if abs_amp and phs_slope:
        data = amp_data * phs_data
    elif abs_amp and not phs_slope:
        data = amp_data
    elif phs_slope and not abs_amp:
        data = phs_data

    if vd:
        if data is None:
            data = torch.zeros_like(vd.data)
        out.setup_data(vd.bls, vd.times, vd.freqs, pol=vd.pol,
                       data=data, flags=vd.flags, cov=vd.cov, cov_axis=vd.cov_axis,
                       icov=vd.icov, history=vd.history)
    else:
        out = data

    return out


def vis2JonesModel(vis, param_type='com', freq_mode='channel', time_mode='channel',
                   freq_kwargs=None, time_kwargs=None, refant=None, single_ant=False):
    """
    Create a vanilla JonesModel object from
    a VisData object

    Parameters
    ----------
    vis : VisData
    kwargs : see JonesModel and JonesResponse for descriptions

    Returns
    -------
    JonesModel
    """
    time_kwargs = {} if time_kwargs is None else time_kwargs
    freq_kwargs = {} if freq_kwargs is None else freq_kwargs
    time_kwargs['times'] = vis.times
    freq_kwargs['freqs'] = vis.freqs
    R = JonesResponse(param_type=param_type, antpos=vis.antpos,
                      freq_mode=freq_mode, freq_kwargs=freq_kwargs,
                      time_mode=time_mode, time_kwargs=time_kwargs)
    ants = list(np.unique(np.concatenate([vis.ant1, vis.ant2])).astype(int))
    polmode = '1pol' if vis.Npol == 1 else '4pol'
    Nants = len(ants)
    if 'slope' in param_type:
        Nants = 2
    elif single_ant:
        Nants = 1
    if param_type == 'com':
        init_func = torch.ones
        dtype = utils._cfloat()
    else:
        init_func = torch.zeros
        dtype = utils._float()
    params = init_func(vis.Npol, vis.Npol, Nants, R.Ntime_params, R.Nfreq_params,
                       dtype=dtype)
    if param_type == 'com':
        params = utils.viewreal(params)
    return JonesModel(params, ants=ants, R=R, refant=refant,
                      polmode=polmode, single_ant=single_ant)


def vis2RedVisModel(vis, param_type='com', freq_mode='channel', time_mode='channel',
                    time_kwargs=None, freq_kwargs=None, redtol=1.0):
    """
    Create a vanilla RedVisModel object
    from a VisData object

    Parameters
    ----------
    vis : VisData object
    kwargs : see RedVisModel and VisModelResponse for details

    Returns
    -------
    RedVisModel
    """
    # get reds
    reds, rvecs, bl2red, bls, rl, ra, rt = telescope_model.build_reds(vis.antpos,
                                                                      bls=vis.bls,
                                                                      redtol=redtol)
    time_kwargs = {} if time_kwargs is None else time_kwargs
    freq_kwargs = {} if freq_kwargs is None else freq_kwargs
    time_kwargs['times'] = vis.times
    freq_kwargs['freqs'] = vis.freqs
    R = VisModelResponse(param_type=param_type,
                         freq_mode=freq_mode, freq_kwargs=freq_kwargs,
                         time_mode=time_mode, time_kwargs=time_kwargs)
    params = torch.zeros(vis.Npol, vis.Npol, len(reds), R.Ntime_params, R.Nfreq_params,
                         dtype=utils._cfloat())
    if param_type == 'com':
        params = utils.viewreal(params)
    return RedVisModel(params, bl2red, R=R)


def chisq(raw_data, forward_model, wgts, axis=None, dof=None, cov_axis=None, mode='vis'):
    """
    Compute chisq between two tensors.
    For a function that operates on VisData objects, see VisData.chisq().

    Parameters
    ----------
    raw_data : tensor
        Raw vis data used as target in optimization
    forward_model : tensor
        Model vis data used as start in optimization
        forward modeled through estimated gains
        (i.e. g_1 g_2^* V^mdl_12)
    wgts : tensor
        Optimization weights applied to raw data during
        optimization (i.e. the VisData.icov of the raw_data).
        This is generally the inverse noise variance, with
        the same shape as the raw_data (cov_axis=None).
        However, this can have other shapes (such as a dense
        inv covariance w/ off diagonal components), which
        necessitates specifying the covariance axis.
        See optim.apply_icov() for details on expected
        shape.
    axis : int or tuple, optional
        The axis over which to sum the chisq. Default is no summing.
        Note if wgts is supplied as a 2D covariance matrix then summing
        is already performed implicitly via the innerproduct.
    dof : float, optional
        Degrees of freedom in fit. If not provided, this is the
        un-normalized chisq, otherwise this is the reduced chisq.
        This is generally the number of data points - N parameters
    cov_axis : str, optional
        If wgts is the inverse variance used in optimization
        with the same shape as data, this should be None (default).
        Otherwise, see optim.apply_icov() for values this can take
        given more complex weighting schemes.
    mode : str, optional
        If supplying a covariance (w/ offdiagonal) as wgts, this is
        the kwarg input to optim.apply_icov().

    Returns
    -------
    tensor
    """
    # form residual
    res = raw_data - forward_model

    # apply wgts
    chisq = optim.apply_icov(res, wgts, cov_axis, mode=mode)

    # divide by dof
    if dof is not None:
        chisq /= dof

    # perform sum
    if axis is not None:
        chisq = torch.sum(chisq, axis=axis)

    return chisq


def _configure_coupling_matrix_singlepath_multiproc(job):
    args, kwargs = job
    return configure_coupling_matrix_singlepath(*args, **kwargs)


def configure_coupling_matrix_singlepath(antpos, bls, bl2red=None, no_auto_coupling=False,
                                         include_second_order=True, min_len=None, max_len=None,
                                         max_EW=None, max_NS=None, second_max_len=None,
                                         second_max_EW=None, second_max_NS=None,
                                         Nproc=None, Ntask=5, use_pathos=True):
    """
    Configure the visibility bl-to-bl coupling matrix for single path
    mutual coupling terms.

    For the voltage stream for ant0 : v_0, single-path coupling is defined as
    the mixing of neighboring antenna voltages:

    v_0 <- v_0 + eps_0_1*v_1 + eps_0_2*v_2 ...

    where eps_0_1 defines the mixing of ant1 into the ant0 voltage stream.
    If the (uncoupled) visibility is V = <v v^T>, then the coupled visibility
    matrix can be defined as

    Vc = <vc vc^T>

    where vc = v + X @ v and 

        | e_0_0  e_0_1  e_0_2 |
    X = | e_1_0  e_1_1  e_1_2 |
        | e_2_0  e_2_1  e_2_2 |

    for a 3-element array. Defining the matrix E = I + X,
    we can write this in compact form as

    Vc = E @ V @ E^T

    Parameters
    ----------
    antpos : dict
        Antenna position dictionary
    bls : list
        List of baseline tuples to compute coupling for (i.e. these
        are the output baselines)
    bl2red : dict, optional
        If provided, assume that the starting vis model is a redundant
        group, such that this is the bl2red mapping (see telescope_model.build_reds).
        Otherwise, we assume that the starting vis model are all physical baselines.
        Keys are physical baseline tuple, values are redundant baseline tuple.
        Note this should include conjugated physical bls w/ conjugated red bl values.
    no_auto_coupling : bool, optional
        If True, exclude ant_i -> ant_i coupling terms (i.e. signal chain reflections)
    include_second_order : bool, optional
        If True, include second order (single-path) coupling terms.
    min_len : float, optional
        If provided, cut all coupling terms with bl vector shorter than this.
    max_len : float, optional
        If provided, cut all coupling terms with bl vector longer than this.
    max_EW : float, optional
        If provided, cut all coupling terms with east-west length longer than this.
    max_NS : float, optional
        If provided, cut all coupling terms with north-south length longer than this.
    second_max_* : float, optional
        Max length/EW/NS parameters for second order coupling terms, if
        include_second_order is True.
    Nproc : int, optional
        If not None, launch multiprocessing of Nprocesses
    Ntask : int, optional
        Number of jobs to solve for for each process
    use_pathos : bool, optional
        Use pathos multiproc instead of multiprocessing

    Returns
    -------
    Arows : dict
        A dictionary holding the coupling matrix configuration.
        The keys are each row of the matrix indexed by the output
        baseline tuple. The values hold the appropriate coupling
        terms for each input baseline.
    """
    # run multiproc mode
    if Nproc is not None:
        # setup multiprocessing
        if use_pathos:
            import multiprocess as mproc
        else:
            import multiprocessing as mproc
        start_method = mproc.get_start_method(allow_none=True)
        if start_method is None: mproc.set_start_method('spawn') 
        Njobs = len(bls) / Ntask
        if Njobs % 1 > 0:
            Njobs = np.floor(Njobs) + 1
        Njobs = int(Njobs)
        jobs = []
        for i in range(Njobs):
            jobs.append([(antpos, bls[i*Ntask:(i+1)*Ntask]),
                         dict(bl2red=bl2red, no_auto_coupling=no_auto_coupling,
                              include_second_order=include_second_order, min_len=min_len, max_len=max_len,
                              max_EW=max_EW, max_NS=max_NS, second_max_len=second_max_len,
                              second_max_EW=second_max_EW, second_max_NS=second_max_NS,)
                         ])

        # run jobs
        with mproc.Pool(Nproc) as pool:
            output = pool.map(_configure_coupling_matrix_singlepath_multiproc, jobs)

        # combine
        keys, vals = [], []
        for out in output:
            keys.extend(out.keys())
            vals.extend(out.values())

        return dict(zip(keys, vals))

    # build rows of A for each bl output
    Arows = {}
    bl_vecs = {}
    for bl in bls:
        # for each baseline, iterate over coupling of all antennas
        # into each of the two antennas into this baseline.
        # For bl = (0, 1) and ant = i, define:
        # v_0 <- v_0 + eps_0_i v_i, mixing the visibility (i, 1)
        # v_1 <- v_1 + eps_1_i v_i, mixing the visibility (0, i)
        row = {}

        # iterate over first-order coupling terms
        for i, ant in enumerate(antpos):

            # first order coupling term into bl[0] antenna
            eps1 = (bl[0], ant)
            skip_eps = cut_bl(eps1, antpos, bl_vecs=bl_vecs, min_len=min_len,
                              max_len=max_len, max_EW=max_EW, max_NS=max_NS)
            skip_eps = skip_eps or (no_auto_coupling and eps1[0] == eps1[1])

            # get first order term eps1 with vis (ant, bl[1])
            if not skip_eps:
                coupled_vis = (ant, bl[1])
                if bl2red:
                    try:
                        coupled_vis = bl2red[coupled_vis]
                    except KeyError:
                        coupled_vis = None
                if coupled_vis:
                    if coupled_vis not in row:
                        row[coupled_vis] = []
                    coeff = "eps_{}_{}".format(*eps1)
                    row[coupled_vis].append(coeff)

            # get first order coupling term into bl[1] antenna
            eps2 = (bl[1], ant)
            skip_eps = cut_bl(eps2, antpos, bl_vecs=bl_vecs, min_len=min_len,
                               max_len=max_len, max_EW=max_EW, max_NS=max_NS)
            skip_eps = skip_eps or (no_auto_coupling and eps2[0] == eps2[1])

            # get first order term eps2 with vis (bl[0], ant)
            if not skip_eps:
                coupled_vis = (bl[0], ant)
                if bl2red:
                    try:
                        coupled_vis = bl2red[coupled_vis]
                    except KeyError:
                        coupled_vis = None
                if coupled_vis:
                    if coupled_vis not in row:
                        row[coupled_vis] = []
                    # this coeff is conjugated b/c ant -> bl[1]
                    coeff = "eps_{}_{}_conj".format(*eps2)
                    row[coupled_vis].append(coeff)

            if include_second_order:
                # iterate over second order coupling terms
                for j, ant2 in enumerate(antpos):
                    eps2 = (bl[1], ant2)
                    skip_eps1 = cut_bl(eps1, antpos, bl_vecs=bl_vecs, min_len=min_len,
                                       max_len=second_max_len, max_EW=second_max_EW,
                                       max_NS=second_max_NS)
                    skip_eps2 = cut_bl(eps2, antpos, bl_vecs=bl_vecs, min_len=min_len,
                                       max_len=second_max_len, max_EW=second_max_EW,
                                       max_NS=second_max_NS)
                    skip_eps = skip_eps1 or skip_eps2 or (no_auto_coupling and eps2[0] == eps2[1])

                    # get second order eps1 with eps2
                    if not skip_eps:
                        coupled_vis = (ant, ant2)
                        if bl2red:
                            try:
                                coupled_vis = bl2red[coupled_vis]
                            except KeyError:
                                coupled_vis = None
                        if coupled_vis:
                            if coupled_vis not in row:
                                row[coupled_vis] = []
                            coeff = "eps_{}_{},eps_{}_{}_conj".format(*eps1, *eps2)
                            row[coupled_vis].append(coeff)

        Arows[bl] = row

    return Arows


def gen_coupling_terms(antpos, min_len=None, max_len=None, max_EW=None, max_NS=None,
                       ants=None, no_auto_coupling=False, compress_to_red=False, redtol=1.0):
    """
    Given a dict of antennas and antenna vectors, generate a list of ant1->ant2 coupling
    pairs to model. Assumes single pol.

    Parameters
    ----------
    antpos : dict
        Antenna vector dictionary, ant_int key and ENU ant_vec value
    min_len : float, optional
        Minimum ant-ant vector length for coupling terms
    max_len : float, optional
        Maximum ant-ant vector length for coupling terms
    max_EW : float, optional
        Maximum ant-ant East-West vector length for coupling terms
    max_NS: float, optional
        Maximum ant-ant North-South vector length for coupling terms
    ants : list, optional
        Of all antennas listed in antpos, only generate coupling terms that
        end up in antennas in this ants list.
        E.g. if ants = [0], then only produce coupling terms [(1, 0), (2, 0), (3, 0), ...]
        Default is to use all antennas in antpos.
    no_auto_coupling : bool, optional
        If True, don't include ant_i -> ant_i coupling terms (default).
    compress_to_red : bool, optional
        If True, only return coupling terms for unique baseline vectors
        and modify coupling_idx to account for this redundancy. Note that
        unlike redundancy in baselines, the conjugate antenna-pair is not
        considered redundant with the unconjugate antenna-pair. E.g.
        (0, 1) != (1, 0)
    redtol : float, optional
        Redundancy tolerance in antenna pos units. 

    Returns
    --------
    coupling_terms : list
        List of coupling terms, e.g. [(0, 1), (1, 0), (0, 2), (2, 0), ...]
        where (0, 1) denotes coupling of ant1 -> ant0
    coupling_idx : dict
        Keys are coupling term tuples and values are the index of this term
        in the coupling_terms list.
    """
    assert isinstance(antpos, dict)
    coupling_terms = []
    for ant_i in antpos:
        for ant_j in antpos:
            if no_auto_coupling and ant_i == ant_j: continue
            if ants is not None and ant_j not in ants: continue
            vec = antpos[ant_j] - antpos[ant_i]
            vlen = torch.linalg.norm(vec)
            if min_len is not None and vlen < min_len:
                continue
            elif max_len is not None and vlen > max_len:
                continue
            elif max_EW is not None and abs(vec[0]) > max_EW:
                continue
            elif max_NS is not None and abs(vec[1]) > max_NS:
                continue
            coupling_terms.append((ant_i, ant_j))

    coupling_idx = {c: i for i, c in enumerate(coupling_terms)}

    if compress_to_red:
        # compress by redundancy
        red_vecs = []
        red_grps = []
        red_idx = []
        for ct in coupling_terms:
            c_vec = antpos[ct[1]] - antpos[ct[0]]

            if len(red_vecs) == 0:
                red_vecs.append(c_vec)
                red_grps.append([ct])
                red_idx.append(0)
                continue

            # look for a match for this vector
            match = torch.linalg.norm(torch.stack(red_vecs) - c_vec, dim=1) < redtol
            if match.any():
                # ct belongs to an existing red group
                k = torch.where(match)[0][0].item()
                red_grps[k].append(ct)
                red_idx.append(k)
            else:
                # ct is it a new group
                red_idx.append(len(red_vecs))
                red_vecs.append(c_vec)
                red_grps.append([ct])

        coupling_idx = {c: red_idx[i] for i, c in enumerate(coupling_terms)}
        coupling_terms = [rg[0] for rg in red_grps]

    return coupling_terms, coupling_idx


def cut_bl(bl, antpos, bl_vecs=None, min_len=None, max_len=None,
           min_EW=None, max_EW=None, min_NS=None, max_NS=None):
    """
    Determine whether a baseline should be cut from consideration based on its
    vector length

    Parameters
    ----------
    bl : tuple
        Antenna-pair tuple
    antpos : dict
        Antenna position dictionary
    bl_vecs : dict, optional
        Baseline vector cache. Keys are bl tuples, values
        are (bl_vec, bl_len), where bl_vec is ENU vector [meters]
        and bl_len is float.
    min_len : float, optional
        Minimum length
    max_len : float, optional
        Maximum length
    min_EW : float, optional
        Minimum East-West length
    max_EW : float, optional
        Maximum EW length
    min_NS : float, optional
        Minimum North-South length
    max_NS : float, optional
        Maximum NS length

    Returns
    -------
    bool
        True if this bl should be cut
    """
    if (min_len is None and max_len is None and min_EW is None) and \
       (max_EW is None and min_NS is None and max_NS is None):
        # no selection, just return False
        return False
    if bl_vecs is not None and bl in bl_vecs:
        vec, vec_len = bl_vecs[bl]
    else:
        vec = antpos[bl[1]] - antpos[bl[0]]
        vec_len = torch.linalg.norm(vec)
        if bl_vecs is not None:
            bl_vecs[bl] = vec, vec_len
    if min_len is not None and vec_len < min_len:
        return True
    if max_len is not None and vec_len > max_len:
        return True
    if min_EW is not None and abs(vec[0]) < min_EW:
        return True
    if max_EW is not None and abs(vec[0]) > max_EW:
        return True
    if min_NS is not None and abs(vec[0]) < min_NS:
        return True
    if max_NS is not None and abs(vec[0]) > max_NS:
        return True    
    
    return False 
