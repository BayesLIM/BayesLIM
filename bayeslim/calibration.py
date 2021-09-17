"""
Module for torch calibration models and relevant functions
"""
import torch
import numpy as np

from . import utils, linalg


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
    def __init__(self, params, ants, bls, refant, R=None, parameter=True,
                 polmode='1pol', single_ant=False):
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
        refant : int
            Reference antenna number from ants list for fixing the gain
            phase
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
        """
        super().__init__()
        self.params = params
        self.refant = refant
        assert refant in ants, "need a valid refant"
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        if R is None:
            # dummy function eval
            R = JonesResponse()
        self.R = R
        self.polmode = polmode
        self.ants = ants
        self.bls = bls
        if not single_ant:
            self._vis2ants = {bl: (ants.index(bl[0]), ants.index(bl[1])) for bl in bls}
        else:
            # a single antenna for all baselines
            assert self.params.shape[2] == 1, "params must have 1 antenna for single_ant"
            self._vis2ants = {bl: (0, 0) for bl in bls}
        # construct _args for str repr
        self._args = dict(refant=refant, polmode=polmode)
        self._args[self.R.__class__.__name__] = getattr(self.R, '_args', None)

    def forward(self, V_m, params=None, undo=False):
        """
        Forward pass V_m through the Jones model.

        Parameters
        ----------
        V_m : VisData
            Model visibilities of
            shape (Npol, Npol, Nbl, Nfreq, Ntime)
        params : tensor, optional
            If not None, use these jones parameters instead of
            self.params in the forward model. Default is None.
        undo : bool, optional
            If True, invert params and apply to V_m. 

        Returns
        -------
        V_p : VisData
            Predicted visibilities, having forwarded
            V_m through the Jones parameters.
        """
        # setup empty predicted visibility
        V_p = V_m.copy()

        # choose fed params or attached params
        if params is None:
            params = self.params

        # push through reponse function
        jones = self.R(params)

        # invert jones if necessary
        if undo:
            invjones = torch.zeros_like(jones)
            for i in range(jones.shape[2]):
                if self.polmode in ['1pol', '2pol']:
                    invjones[:, :, i] = linalg.diag_inv(jones[:, :, i])
                else:
                    invjones[:, :, i] = torch.pinv(jones[:, :, i])
            jones = invjones

        # iterate through visibility and apply Jones terms
        for i, bl in enumerate(self.bls):
            # pick out jones terms
            j1 = jones[:, :, self._vis2ants[bl][0]]
            j2 = jones[:, :, self._vis2ants[bl][1]]

            # set reference antenna phase to zero
            if bl[0] == self.refant:
                j1 = j1 / torch.exp(1j * torch.angle(j1))
            if bl[1] == self.refant:
                j2 = j2 / torch.exp(1j * torch.angle(j2))

            if self.polmode in ['1pol', '2pol']:
                V_p.data[:, :, i] = linalg.diag_matmul(linalg.diag_matmul(j1, V_m.data[:, :, i]), j2.conj())
            else:
                V_p.data[:, :, i] = torch.einsum("ab...,bc...,dc...->ad...", j1, V_m.data[:, :, i], j2.conj())

        return V_p


class JonesResponse:
    """
    A response object for JonesModel

    Allows for polynomial parameterization across time and/or frequency,
    and for a gain type of complex, amplitude, phase, delay, EW & NS delay slope,
    and EW & NS phase slope (the latter two are relevant for redundant calibration) 
    """
    def __init__(self, freq_mode='channel', time_mode='channel', gain_type='complex',
                 device=None, freqs=None, times=None, **setup_kwargs):
        """
        Parameters
        ----------
        freq_mode : str, optional
            Frequency parameterization, ['channel', 'poly']
        time_mode : str, optional
            Time parameterization, ['channel', 'poly']
        gain_type : str, optional
            Type of gain parameter
            ['complex', 'dly', 'amp', 'phs', 'dly_slope', 'phs_slope']
        device : str, optional
            Device to place class attributes if needed
        freqs : tensor, optional
            Frequency array [Hz], only needed for poly freq_mode
        times : tensor, optional
            Time array [arb. units], only needed for poly time_mode

        Notes
        -----
        Required setup_kwargs (see self._setup for details)
        if freq_mode == 'poly'
            f0 : float
                Anchor frequency [Hz]
            f_Ndeg : int
                Frequency polynomial degree
            freq_poly_basis : str
                Polynomial basis (see utils.gen_poly_A)

        if time_mode == 'poly'
            t0 : float
                Anchor time [arb. units]
            t_Ndeg : int
                Time polynomial degree
            time_poly_basis : str
                Polynomial basis (see utils.gen_poly_A)

        if gain_type == 'phs_slope' or 'dly_slope:
            antpos : dictionary
                Antenna vector in local ENU frame [meters]
                number as key, tensor (x, y, z) as value
            params tensor is assumed to hold the [EW, NS]
            slope along its antenna axis.
        """
        self.freq_mode = freq_mode
        self.time_mode = time_mode
        self.gain_type = gain_type
        self.device = device
        self.freqs = freqs
        self.times = times
        self.setup_kwargs = setup_kwargs
        self._setup(**setup_kwargs)

    def _setup(self, f0=None, f_Ndeg=None, freq_poly_basis='direct',
               t0=None, t_Ndeg=None, time_poly_basis='direct', antpos=None):
        """
        Setup the JonesResponse given the mode and type

        Parameters
        ----------
        f0 : float
            anchor frequency for poly [Hz]
        f_Ndeg : int
            Number of frquency degrees for poly
        freq_poly_basis : str
            Polynomial basis across freq (see utils.gen_poly_A)
        t0 : float
            anchor time for poly
        t_Ndeg : int
            Number of time degrees for poly
        time_poly_basis : str
            Polynomial basis across time (see utils.gen_poly_A)
        antpos : dict
            Antenna position dictionary for dly_slope or phs_slope
        """
        if self.freq_mode == 'channel':
            pass  # nothing to do
        elif self.freq_mode == 'poly':
            # get polynomial A matrix wrt freq
            assert f_Ndeg is not None, "need f_Ndeg for poly freq_mode"
            if f0 is None:
                f0 = self.freqs.mean()
            self.dfreqs = (self.freqs - f0) / 1e6  # MHz
            self.freq_A = utils.gen_poly_A(self.dfreqs, f_Ndeg,
                                           basis=freq_poly_basis, device=self.device)

        if self.time_mode == 'channel':
            pass  # nothing to do
        elif self.time_mode == 'poly':
            # get polynomial A matrix wrt times
            assert t_Ndeg is not None, "need t_Ndeg for poly time_mode"
            if t0 is None:
                t0 = self.times.mean()
            self.dtimes = self.times - t0
            self.time_A = utils.gen_poly_A(self.dtimes, t_Ndeg,
                                           basis=time_poly_basis, device=self.device)

        if self.gain_type in ['dly_slope', 'phs_slope']:
            # setup antpos tensors
            assert antpos is not None, 'need antpos for dly_slope or phs_slope'
            EW = torch.as_tensor([antpos[a][0] for a in self.ants], device=self.device)
            self.antpos_EW = EW[None, None, :, None, None]  
            NS = torch.as_tensor([antpos[a][1] for a in self.ants], device=self.device)
            self.antpos_NS = NS[None, None, :, None, None]  

        # construct _args for str repr
        self._args = dict(freq_mode=self.freq_mode, time_mode=self.time_mode,
                          gain_type=self.gain_type)

    def param2gain(self, params):
        """
        Convert parameter to complex gain given gain_type

        Parameters
        ----------
        params : tensor
            jones parameter of shape (Npol, Npol, Nant, Ntimes, Nfreqs)

        Returns
        -------
        tensor
            Complex gain tensor (Npol, Npol, Nant, Ntimes, Nfreqs)
        """
        if self.gain_type == 'complex':
            # assume params are complex gains
            return params

        elif self.gain_type == 'dly':
            # assume params are in delay [nanosec]
            return torch.exp(2j * np.pi * params * self.freqs / 1e9)

        elif self.gain_type == 'amp':
            # assume params are gain amplitude
            return torch.exp(params) + 0j

        elif self.gain_type == 'phs':
            return torch.exp(1j * params)

        elif self.gain_type == 'dly_slope':
            # extract EW and NS delay slopes: ns / meter
            EW = params[:, :, :1]
            NS = params[:, :, 1:]
            # get total delay per antenna
            tot_dly = EW * self.antpos_EW \
                      + NS * self.antpos_NS
            # convert to complex gains
            return torch.exp(2j * np.pi * tot_dly * self.freqs / 1e9)

        elif self.gain_type == 'phs_slope':
            # extract EW and NS phase slopes: rad / meter
            EW = params[:, :, :1]
            NS = params[:, :, 1:]
            # get total phase per antenna
            tot_phs = EW * self.antpos_EW \
                      + NS * self.antpos_NS
            # convert to complex gains
            return torch.exp(1j * tot_phs)

    def forward(self, params):
        """
        Forward pass params through response to get
        complex antenna gains per time and frequency
        """
        # convert sparse representation to full Ntimes, Nfreqs
        if self.freq_mode == 'channel':
            pass
        elif self.freq_mode == 'poly':
            params = params @ self.freq_A.T
        if self.time_mode == 'channel':
            pass
        elif self.time_mode == 'poly':
            params = (params.transpose(-1, -2) @ self.time_A.T).transpose(-1, -1)

        # convert gain types to complex gains
        params = self.param2gain(params)

        return params

    def __call__(self, params):
        return self.forward(params)


class RedVisModel(utils.Module):
    """
    Redundant visibility model (r) relating the starting
    model visibility (m) to the data visibility (d)
    for antennas j and k.

    .. math::

        V^{d}_{jk} = V^{r} + V^{m}_{jk}

    """
    def __init__(self, params, vis2red, R=None, parameter=True):
        """
        Redundant visibility model

        Parameters
        ----------
        params : tensor
            Initial redundant visibility tensor
            of shape (Npol, Npol, Nredvis, Ntimes, Nfreqs) where Nredvis
            is the number of unique baseline types.
        vis2red : list of int
            A list of length Nvis--the length of V_m input to
            self.forward()--whose elements index red.
        R : callable, optional
            An arbitrary response function for the redundant
            visibility model, mapping the parameters
            to the space of V_m (input to self.forward).
            Default (None) is unit response.
            Note this must use torch functions.
        parameter : bool, optional
            If True, treat params as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        """
        super().__init__()
        self.params = params
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        self.vis2red = vis2red
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R

    def forward(self, V_m, undo=False):
        """
        Forward pass V_m through redundant
        model term.

        Parameters
        ----------
        V_m : VisData
            Starting model visibilities of
            shape (Npol, Npol, Nbl, Ntimes, Nfreqs). In the general case,
            this should be a unit matrix so that the
            predicted visibilities are simply the redundant
            model. However, if you have a model of per-baseline
            non-redundancies, these could be included by putting
            them into V_m.
        undo : bool, optional
            If True, push V_m backwards through the model.

        Returns
        -------
        V_p : VisData
            The predicted visibilities, having pushed V_m through
            the redundant visibility model
        """
        # setup predicted visibility
        V_p = V_m.copy()

        params = self.R(self.params)

        # iterate through vis and apply redundant model
        for i in range(V_p.shape[2]):
            if not undo:
                V_p.data[:, :, i] = V_m.data[:, :, i] + params[:, :, self.vis2red[i]]
            else:
                V_p.data[:, :, i] = V_m.data[:, :, i] - params[:, :, self.vis2red[i]]

        return V_p


class VisModel(utils.Module):
    """
    Visibility model (v) relating the starting
    model visibility (m) to the data visibility (d)
    for antennas j and k.

    .. math::

        V^{d}_{jk} = V^{v}_{jk} + V^{m}_{jk} 

    """
    def __init__(self, params, R=None, parameter=True):
        """
        Visibility model

        Parameters
        ----------
        params : tensor
            Visibility model parameter of shape
            (Npol, Npol, Nbl, Ntimes, Nfreqs). Ordering should
            match ordering of V_m input to self.forward.
        R : callable, optional
            An arbitrary response function for the
            visibility model, mapping the parameters
            to the space of V_m (input to self.forward).
            Default (None) is unit response.
            Note this must use torch functions.
        parameter : bool, optional
            If True, treat vis as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        """
        super().__init__()
        self.params = params
        if parameter:
            self.params = torch.nn.Parameter(self.params)
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R

    def forward(self, V_m, undo=False):
        """
        Forward pass V_m through visibility
        model term.

        Parameters
        ----------
        V_m : VisData
            Starting model visibilities
            of shape (Npol, Npol, Nbl, Ntimes, Nfreqs). In the general case,
            this should be a zero tensor so that the
            predicted visibilities are simply the redundant
            model. However, if you have a model of per-baseline
            non-redundancies, these could be included by putting
            them into V_m.
        undo : bool, optional
            If True, push V_m backwards through the model.

        Returns
        -------
        V_p : VisData
            The predicted visibilities, having pushed V_m through
            the visibility model.
        """
        V_p = V_m.copy()
        params = self.R(self.params)
        if not undo:
            V_p.data = V_p.data + params
        else:
            V_p.data = V_p.data - params

        return V_p
