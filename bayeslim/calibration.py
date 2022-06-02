"""
Module for torch calibration models and relevant functions
"""
import torch
import numpy as np
import copy

from . import utils, linalg, dataset


class BaseResponse:
    """
    A base parameter response object for JonesModel taking
    params of shape (Npol, Npol, Nantenna, Ntimes, Nfreqs),
    and for VisModel taking params of shape
    (Npol, Npol, Nbl, Ntimes, Nfreqs)
    """
    def __init__(self, freq_mode='channel', time_mode='channel', param_type='com',
                 device=None, freq_kwargs={}, time_kwargs={}, atol=1e-4):
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
            options = ['com', 'real', 'amp', 'phs']
        device : str, optional
            Device to place class attributes if needed
        freq_kwargs : dict, optional
            Keyword arguments for setup_freqs(). Note, must pass
            freqs [Hz] for dly param_type
        time_kwargs : dict, optional
            Keyword arguments for setup_times().
        atol : float, optional
            Absolute tolerance used for time index caching.
        """
        self.freq_mode = freq_mode
        self.time_mode = time_mode
        self.param_type = param_type
        self.device = device
        self.freq_kwargs = freq_kwargs
        self.time_kwargs = time_kwargs
        self.setup_freqs(**freq_kwargs)
        self.setup_times(**time_kwargs)
        self.atol = atol

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
            For more kwargs see utils.LinearModel()
            dim is hard-coded according to expected params shape
        """
        self.times = times
        self.clear_cache()
        if self.time_mode == 'channel':
            pass  # nothing to do

        elif self.time_mode == 'linear':
            # get linear A mapping wrt time
            kwgs = copy.deepcopy(kwargs)
            linear_mode = kwgs.pop('linear_mode')
            if times is not None:
                kwgs['x'] = times
            kwgs['dtype'] = utils._cfloat() if self.param_type == 'com' else utils._float()
            self.time_LM = utils.LinearModel(linear_mode, dim=-2,
                                             device=self.device, **kwgs)

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
            For more kwargs see utils.LinearModel()
            dim is hard-coded according to expected params shape
        """
        self.freqs = freqs
        if self.freq_mode == 'channel':
            pass  # nothing to do

        elif self.freq_mode == 'linear':
            # get linear A mapping wrt freq
            kwgs = copy.deepcopy(kwargs)
            linear_mode = kwgs.pop('linear_mode')
            kwgs['dtype'] = utils._cfloat() if self.param_type == 'com' else utils._float()
            if freqs is not None:
                kwgs['x'] = freqs
            self.freq_LM = utils.LinearModel(linear_mode, dim=-1,
                                             device=self.device, **kwgs)

        else:
            raise ValueError("{} not recognized".format(self.freq_mode))

    def forward(self, params, times=None, **kwargs):
        """
        Forward pass params through response
        """
        # pass to device
        if utils.device(params.device) != utils.device(self.device):
            params = params.to(self.device)

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

        params = self.params2complex(params)

        # index time axis if needed
        if times is not None:
            tidx = self.get_time_idx(times)
            params = params[..., tidx, :]

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
        if self.param_type == 'com':
            # assume params are already complex
            pass

        elif self.param_type == 'real':
            params =  params + 0j

        elif self.param_type == 'amp':
            # assume params are amplitude
            params = torch.exp(params) + 0j

        elif self.param_type == 'phs':
            params = torch.exp(1j * params)

        return params

    def __call__(self, params, **kwargs):
        return self.forward(params, **kwargs)

    def push(self, device):
        """
        Push class attrs to new device
        """
        self.device = device
        if self.freq_mode == 'linear':
            self.freq_LM.push(device)
        if self.time_mode == 'linear':
            self.time_LM.push(device)

    def clear_cache(self):
        """
        Clear caching of time indices
        """
        self.cache_tidx = {}

    def hash_times(self, times):
        """get the hash of a times array"""
        return hash((times[0], times[-1], len(times)))

    def get_time_idx(self, times):
        """
        Get time indices, and store in cache.
        This is used when minibatching over time axis.
        """
        h = self.hash_times(times)
        if h in self.cache_tidx:
            # query cache
            return self.cache_tidx[h]
        else:
            # compute time indices
            idx = [np.where(np.isclose(self.times, t, atol=self.atol, rtol=1e-15))[0][0] for t in times]
            idx = utils._list2slice(idx)
            # store in cache and return
            self.cache_tidx[h] = idx
            return idx



class JonesModel(utils.Module):
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
    def __init__(self, params, ants, bls=None, refant=None, R=None, parameter=True,
                 polmode='1pol', single_ant=False, name=None, vis_type='com'):
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
        bls : list
            List of ant-pair tuples that hold the baselines of the
            input visibilities, matched ordering to baseline ax of V
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
        """
        super().__init__(name=name)
        self.params = params
        self.device = params.device
        self.refant, self.refant_idx = refant, None
        self.ants = ants
        if self.refant is not None:
            assert self.refant in ants, "need a valid refant"
            self.refant_idx = ants.index(self.refant)
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        if R is None:
            # default response
            R = JonesResponse()
        self.R = R
        self.polmode = polmode
        self.single_ant = single_ant
        self._setup(bls)
        self.vis_type = vis_type
        # construct _args for str repr
        self._args = dict(refant=refant, polmode=polmode)
        self._args[self.R.__class__.__name__] = getattr(self.R, '_args', None)

    def _setup(self, bls):
        bls = [tuple(bl) for bl in bls]
        self.bls = bls
        if not self.single_ant:
            self._vis2ants = {bl: (self.ants.index(bl[0]), self.ants.index(bl[1])) for bl in bls}
        else:
            # a single antenna for all baselines
            assert self.params.shape[2] == 1, "params must have 1 antenna for single_ant"
            self._vis2ants = {bl: (0, 0) for bl in bls}

    def fix_refant_phs(self):
        """
        Ensure that the reference antenna phase
        is set to zero: operates inplace.
        This only has an effect if the JonesResponse
        param_type is ['com', 'dly', 'phs'],
        otherwise params is unaffected.
        """
        with torch.no_grad():
            if self.R.param_type == 'com':
                # cast params to complex if needed
                if not torch.is_complex(self.params):
                    params = utils.viewcomp(self.params)
                else:
                    params = self.params

                # if time and freq mode are 'channel' then divide by phase
                if self.R.time_mode == 'channel' and self.R.freq_mode == 'channel':
                    phs = torch.angle(params[:, :, self.refant_idx:self.refant_idx+1]).detach().clone()
                    params /= torch.exp(1j * phs)
                # otherwise just set imag component to zero
                else:
                    params.imag[:, :, self.refant_idx:self.refant_idx+1] = torch.zeros_like(params.imag[:, :, self.refant_idx:self.refant_idx+1])

                if not torch.is_complex(self.params):
                    # recast as view_real
                    params = utils.viewreal(params)
                self.params[:] = params

            elif self.R.param_type in ['dly', 'phs']:
                self.params -= self.params[:, :, self.refant_idx:self.refant_idx+1].clone()

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
        self.fix_refant_phs()

        # configure data if needed
        if vd.bls != self.bls:
            self._setup(vd.bls)

        # push vd to self.device
        vd.push(self.device)

        # setup empty VisData for output
        vout = vd.copy()

        # push through reponse function
        if jones is None:
            jones = self.R(self.params, times=vd.times)

        # evaluate priors
        self.eval_prior(prior_cache, inp_params=self.params, out_params=jones)

        # apply calibration and insert into output vis
        vout.data, _ = apply_cal(vd.data, self.bls, jones, self._vis2ants,
                                 cal_2pol=self.polmode=='2pol',
                                 vis_type=self.vis_type, undo=undo)

        return vout

    def push(self, device):
        """
        Push params and other attrs to new device
        """
        self.device = device
        self.params = utils.push(self.params, device)
        self.R.push(device)

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
            gains = self.R(self.params)

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
                 freq_kwargs={}, time_kwargs={}):
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

        Notes
        -----
        For param_type in ['phs_slope', 'dly_slope]
            antpos is required. params tensor is assumed
            to hold the [EW, NS] slope along its antenna axis.
        """
        super().__init__(freq_mode=freq_mode, time_mode=time_mode,
                         param_type=param_type, device=device,
                         freq_kwargs=freq_kwargs, time_kwargs=time_kwargs)
        self.vis_type = vis_type

        if self.param_type in ['dly_slope', 'phs_slope']:
            # setup antpos tensors
            assert antpos is not None, 'need antpos for dly_slope or phs_slope'
            self.antpos = antpos
            EW = torch.as_tensor([antpos[a][0] for a in antpos], device=self.device)
            self.antpos_EW = EW[None, None, :, None, None]  
            NS = torch.as_tensor([antpos[a][1] for a in antpos], device=self.device)
            self.antpos_NS = NS[None, None, :, None, None]
        elif 'dly' in self.param_type:
            assert self.freqs is not None, 'need frequencies for delay gain type'

        assert self.param_type in ['com', 'amp', 'phs', 'dly', 'real', 'phs_slope', 'dly_slope']

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


class RedVisModel(utils.Module):
    """
    Redundant visibility model (r) relating the starting
    model visibility (m) to the data visibility (d)
    for antennas j and k.

    .. math::

        V^{d}_{jk} = V^{r} + V^{m}_{jk}

    """
    def __init__(self, params, bl2red, R=None, parameter=True, name=None):
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
        name : str, optional
            Name for this object, stored as self.name
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

    def forward(self, vd, undo=False, prior_cache=None):
        """
        Forward pass vd through redundant
        model term.

        Parameters
        ----------
        vd : VisData, optional
            Starting model visibilities of shape
            (Npol, Npol, Nbl, Ntimes, Nfreqs). In the general case,
            this should be a unit matrix so that the
            predicted visibilities are simply the redundant
            model. However, if you have a model of per-baseline
            non-redundancies, these could be included by putting
            them into vd.
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
        if vd is None:
            vd = dataset.VisData()
        # push to device
        vd.push(self.device)

        # setup predicted visibility
        vout = vd.copy()

        redvis = self.R(self.params, times=vd.times)

        # iterate through vis and apply redundant model
        for i, bl in enumerate(vout.bls):
            if not undo:
                vout.data[:, :, i] = vd.data[:, :, i] + redvis[:, :, self.bl2red[bl]]
            else:
                vout.data[:, :, i] = vd.data[:, :, i] - redvis[:, :, self.bl2red[bl]]

        # evaluate priors
        self.eval_prior(prior_cache, inp_params=self.params, out_params=redvis)

        return vout

    def push(self, device):
        """
        Push to a new device
        """
        self.params = utils.push(self.params, device)


class VisModel(utils.Module):
    """
    Visibility model (v) relating the starting
    model visibility (m) to the data visibility (d)
    for antennas j and k.

    .. math::

        V^{d}_{jk} = V^{v}_{jk} + V^{m}_{jk} 

    """
    def __init__(self, params, R=None, parameter=True, name=None):
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
        name : str, optional
            Name for this object, stored as self.name
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
        vis = self.R(self.params, times=vd.times)
        if not undo:
            vout.data = vout.data + vis
        else:
            vout.data = vout.data - vis

        # evaluate priors
        self.eval_prior(prior_cache, inp_params=self.params, out_params=vis)

        return vout

    def push(self, device):
        """
        Push to a new device
        """
        self.params = utils.push(self.params, device)


class VisModelResponse(BaseResponse):
    """
    A response object for VisModel and RedVisModel, subclass of BaseResponse
    taking params of shape (Npol, Npol, Nbl, Ntimes, Nfreqs)
    """
    def __init__(self, freq_mode='channel', time_mode='channel',
                 param_type='com', device=None,
                 freq_kwargs={}, time_kwargs={}):
        """
        Parameters
        ----------
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
        """
        super().__init__(freq_mode=freq_mode, time_mode=time_mode,
                         param_type=param_type, device=device,
                         freq_kwargs=freq_kwargs, time_kwargs=time_kwargs)

    def forward(self, params):
        """
        Forward pass params through response to get
        complex visibility model per time and frequency
        """
        params = super().forward(params)

        # detect if params needs to be casted into complex
        if self.param_type == 'amp_phs':
            params = torch.exp(params[..., 0] + 1j * params[..., 1])

        return params


class VisCoupling(utils.Module):
    """
    A visibility coupling module, describing
    a Nbls x Nbls coupling matrix
    """
    def __init__(self, params, bls, R=None, parameter=True, name=None):
        """
        Visibility coupling model

        Parameters
        ----------
        params : tensor
            (Npol, Npol, Nbls, Nbls, Ntimes, Nfreqs) tensor
            describing coupling between baselines
        bls : list
            List of antenna pair tuples (i.e. baselines) of
            params along its Nbls axis
        R : callable, optional
            Response object for params, mapping it from
            sparse basis to its (Npol, Npol, ... Nfreqs) shape
            Default is VisModelResponse
        parameter : bool, optional
            If True, treat params as differentiable
        name : str, optional
            Name for this module, default is class name
        """
        super().__init__(name=name)
        self.params = params
        self.device = params.device
        self.bls = bls
        self.Nbls = len(bls)
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        if R is None:
            # default response is per freq channel and time bin
            R = VisModelResponse()
        self.R = R

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
        vout = vd.copy(detach=False)
        coupling = self.R(self.params, times=vd.times)

        if vout.Nbls == self.Nbls:
            # if Nbls is the same, assume bl ordering matches!
            bl_idx = slice(None)
        else:
            # otherwise get relevant bls in vout for this coupling model
            bl_idx = vout._bl2ind(coupling.bls)

        # multiply coupling tensor
        vout.data[:, :, bl_idx] = torch.einsum("iijk...,iik...->iij...", coupling, vout.data)

        # evaluate priors
        self.eval_prior(prior_cache, inp_params=self.params, out_params=vis)

        return vout

    def push(self, device):
        """
        Push to a new device
        """
        self.params = utils.push(self.params, device)


def apply_cal(vis, bls, gains, vis2ants, cal_2pol=False, cov=None,
              vis_type='com', undo=False, inplace=False):
    """
    Apply calibration to a visibility tensor with a complex
    gain tensor. Default behavior is to multiply
    vis with gains, i.e. when undo = False. To divide
    vis by gains, set undo = True.

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
    vis2ants : dict
        Mapping between a baseline tuple in bls to the indices of
        the two antennas (g_1, g_2) in gains to apply.
        E.g. calibrating with Nants gains {(0, 1): (0, 1), (1, 3): (1, 3), ...}
        E.g. calibrating vis with 1 gain, {(0, 1): (0, 0), (1, 3): (0, 0), ...}
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

    # iterate through visibility and apply gain terms
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

    for i, bl in enumerate(bls):
        # pick out appropriate antennas
        g1 = gains[:, :, vis2ants[bl][0]]
        g2 = gains[:, :, vis2ants[bl][1]]

        if polmode in ['1pol', '2pol']:
            # update visibilities
            if vis_type == 'com':
                G = g1 * g2.conj()
                vout[:, :, i] = linalg.diag_matmul(G, vis[:, :, i])

                # update covariance
                if cov is not None:
                    GG = G * G.conj()
                    if torch.is_complex(GG):
                        GG = GG.real
                    cov_out[:, :, i] = linalg.diag_matmul(GG, cov[:, :, i])

            elif vis_type == 'dly':
                vout[:, :, i] = vis[:, :, i] + g1 - g2

        else:
            assert vis_type == 'com', "must have complex vis_type for 4pol mode"
            vout[:, :, i] = torch.einsum("ab...,bc...,dc...->ad...", g1, vis[:, :, i], g2.conj())

    return vout, cov_out


def compute_redcal_degen(params, ants, antpos, wgts=None):
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
    params : tensor
        Antenna gains of shape (Npol, Npol, Nant, Ntimes, Nfreqs)
    ants : list
        List of antenna numbers along the Nant axis
    antpos : dict
        Dictionary of ENU antenna vectors for each antenna number
    wgts : tensor, optional
        1D weight tensor to use in computing degenerate parameters
        of len(Nants). Normally, this should be the total number
        of visibilities used in redcal for each antenna.
        Default is uniform weighting.

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
    Nants = len(ants)
    if wgts is None:
        wgts = torch.ones(Nants, dtype=utils._float())
    wgts = wgts[:, None, None]
    wsum = torch.sum(wgts)

    # compute absolute amplitude parameter
    eta = torch.log(torch.abs(params))
    abs_amp = torch.sum(eta * wgts, dim=2, keepdims=True) / wsum

    # compute phase slope parameter
    phs = torch.angle(params).moveaxis(2, -1)
    A = torch.stack([torch.as_tensor(antpos[a][:2]) for a in ants])
    W = torch.eye(Nants) * wgts.squeeze()
    AtWAinv = torch.pinverse(A.T @ W @ A)
    phs_slope = (phs @ W @ A @ AtWAinv.T).moveaxis(-1, 2)

    return abs_amp, phs_slope


def redcal_degen_gains(ants, abs_amp=None, phs_slope=None, antpos=None):
    """
    Given redcal degenerate parameters, transform to their complex gains

    Parameters
    ----------
    ants : list
        List of antenna numbers for the output gains
    abs_amp : tensor, optional
        Absolute amplitude parameter of shape
        (Npol, Npol, 1, Ntimes, Nfreqs)
    phs_slope : tensor, optional
        Phase slope parameter of shape
        (Npol, Npol, 2, Ntimes, Nfreqs) where the two
        elements are the [East, North] gradients [rad / meter]
    antpos : dict, optional
        Mapping of antenna number to antenna ENU vector [meters].
        Needed for phs_slope parameter

    Returns
    -------
    tensor
        Complex gains of shape (Npol, Npol, Nant, Ntimes, Nfreqs)
    """
    # fill unit gains
    Nants = len(ants)
    gains = torch.ones(1, 1, Nants, 1, 1, dtype=utils._cfloat())

    # incorporate absolute amplitude
    if abs_amp is not None:
        gains = gains * torch.exp(abs_amp)

    # incorporate phase slope
    if phs_slope is not None:
        A = torch.stack([torch.as_tensor(antpos[a][:2]) for a in ants])
        phs = (phs_slope.moveaxis(2, -1) @ A.T).moveaxis(-1, 2)
        gains = gains * torch.exp(1j * phs)

    return gains
