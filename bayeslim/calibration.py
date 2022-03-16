"""
Module for torch calibration models and relevant functions
"""
import torch
import numpy as np

from . import utils, linalg, dataset


class BaseResponse:
    """
    A base parameter response object for JonesModel and VisModel
    """
    def __init__(self, freq_mode='channel', time_mode='channel', param_type='com',
                 device=None, freq_kwargs={}, time_kwargs={}):
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
        device : str, optional
            Device to place class attributes if needed
        freq_kwargs : dict, optional
            Keyword arguments for _setup_freqs(). Note, must pass
            freqs [Hz] for dly param_type
        time_kwargs : dict, optional
            Keyword arguments for _setup_times().
        """
        self.freq_mode = freq_mode
        self.time_mode = time_mode
        self.param_type = param_type
        self.device = device
        self.freq_kwargs = freq_kwargs
        self.time_kwargs = time_kwargs
        self._setup_freqs(**freq_kwargs)
        self._setup_times(**time_kwargs)

        # construct _args for str repr
        self._args = dict(freq_mode=self.freq_mode, time_mode=self.time_mode,
                          param_type=self.param_type)

    def _setup_times(self, times=None, linear_mode=None, A=None, t0=None,
                     Ndeg=None, basis='direct', whiten=False):
        """
        Setup time basis

        For time_mode == 'linear', see utils.gen_linear_A
        for info on kwargs.
        """
        self.times = times

        if self.time_mode == 'channel':
            pass  # nothing to do

        elif self.time_mode == 'linear':
            # get linear A mapping wrt time
            self.time_A = utils.gen_linear_A(linear_mode, A=A, x=times, x0=t0,
                                             Ndeg=Ndeg, basis=basis, whiten=whiten)
            if self.param_type == 'com':
                self.time_A = self.time_A.to(utils._cfloat())
            self.time_A = self.time_A.to(self.device)

    def _setup_freqs(self, freqs=None, linear_mode=None, A=None, f0=None,
                     Ndeg=None, basis='direct', whiten=False):
        """
        Setup freq basis

        For freq_mode == 'linear', see utils.gen_linear_A
        for info on kwargs
        """
        self.freqs = freqs

        if self.freq_mode == 'channel':
            pass  # nothing to do

        elif self.freq_mode == 'linear':
            # get linear A mapping wrt freq
            self.freq_A = utils.gen_linear_A(linear_mode, A=A, x=freqs, x0=f0,
                                             Ndeg=Ndeg, basis=basis, whiten=whiten)
            if self.param_type == 'com':
                self.freq_A = self.freq_A.to(utils._cfloat())
            self.freq_A = self.freq_A.to(self.device)

    def forward(self, params):
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
            params = params @ self.freq_A.T

        if self.time_mode == 'channel':
            pass
        elif self.time_mode == 'linear':
            params = (params.moveaxis(-2, -1) @ self.time_A.T).moveaxis(-1, -2)

        params = self.params2complex(params)

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

    def __call__(self, params):
        return self.forward(params)

    def push(self, device):
        """
        Push class attrs to new device
        """
        self.device = device
        if self.freq_mode == 'linear':
            self.freq_A = self.freq_A.to(device)
        if self.time_mode == 'linear':
            self.time_A = self.time_A.to(device)


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
            jones = self.R(self.params)

        # get time select (in case we are mini-batching over time axis)
        if vd.Ntimes != jones.shape[-2]:
            tselect = np.ravel([np.where(np.isclose(self.R.times, t, atol=1e-4, rtol=1e-10))[0] for t in vd.times]).tolist()
            diff = list(set(np.diff(tselect)))
            if len(diff) == 1:
                tselect = slice(tselect[0], tselect[-1]+diff[0], diff[0])
            jones = jones[..., tselect, :]


        # evaluate priors
        self.eval_prior(prior_cache, inp_params=self.params, out_params=jones)

        # calibrate and insert into output vis
        vout.data = vis_calibrate(vd.data, self.bls, jones, self._vis2ants, self.polmode,
                                  vis_type=self.vis_type, undo=undo)

        return vout

    def push(self, device):
        """
        Push params and other attrs to new device
        """
        self.device = device
        self.params = utils.push(self.params, device)
        self.R.push(device)


class JonesResponse(BaseResponse):
    """
    A response object for JonesModel, subclass of BaseResponse

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
            Keyword arguments for _setup_freqs(). Note, must pass
            freqs [Hz] for dly param_type
        time_kwargs : dict, optional
            Keyword arguments for _setup_times().

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

        redvis = self.R(self.params)

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
        vis = self.R(self.params)
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
        coupling = self.R(self.params)

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


class CalData:
    """
    Work in progress...

    An object for holding complex calibration
    solutions of shape (Npol, Npol, Nant, Ntimes, Nfreqs).
    Optionally, Ntimes and Nfreqs may be replaced by
    Ncoeff describing a sparse linear basis across
    those dimensions, which can be propagated to
    the time and/or frequency domain via the
    attached JonesResponse object, self.R.
    """
    def __init__(self):
        """
        """
        raise NotImplementedError

    def setup_response(self, freq_mode='channel', time_mode='channel', param_type='com',
                       vis_type='com', freqs=None, times=None, device=None, **setup_kwargs):
        """
        Setup response object for complex gains, mapping self.params to self.gains
        """
        self.R = JonesResponse(freq_mode=freq_mode, time_mode=time_mode,
                               param_type=param_type, vis_type=vis_type,
                               freqs=freqs, times=times, device=device, **setup_kwargs)

    def setup_data(self, ):
        """
        """
        pass

    def write_hdf5(self, ):
        """
        """
        pass

    def read_hdf5(self, ):
        """
        """
        pass

    def read_uvcal(self, ):
        """
        """
        pass


def vis_calibrate(vis, bls, gains, vis2ants, polmode, vis_type='com',
                  undo=False):
    """
    Calibrate a visibility tensor with a complex
    gain tensor. Default behavior is to multiply
    vis with gains, i.e. when undo = False.

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
    polmode : str
        Polarization mode of data ['1pol', '2pol', '4pol'].
        For 1pol data Npol = 1, for 2pol and 4pol data Npol = 2,
        but in 2pol mode off-diagonal pols are ignored.
    vis_type : str, optional
        Type of visibility and gain tensor. ['com', 'dly'].
        If 'com', vis and gains are complex (default).
        If 'dly', vis and gains are float delays.
    undo : bool, optional
        If True, divide vis by gains, otherwise
        (default) multiply vis by gains.
    """
    assert vis.shape[:2] == gains.shape[:2], "vis and gains must have same Npols"

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
    vout = torch.zeros_like(vis)
    for i, bl in enumerate(bls):
        # pick out appropriate antennas
        g1 = gains[:, :, vis2ants[bl][0]]
        g2 = gains[:, :, vis2ants[bl][1]]

        if polmode in ['1pol', '2pol']:
            if vis_type == 'com':
                vout[:, :, i] = linalg.diag_matmul(linalg.diag_matmul(g1, g2.conj()), vis[:, :, i])
            elif vis_type == 'dly':
                vout[:, :, i] = vis[:, :, i] + g1 - g2
        else:
            assert vis_type == 'com', "must have complex vis_type for 4pol mode"
            vout[:, :, i] = torch.einsum("ab...,bc...,dc...->ad...", g1, vis[:, :, i], g2.conj())

    return vout


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
    phs_slope = (phs @ W.T @ A @ AtWAinv.T).moveaxis(-1, 2)

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
