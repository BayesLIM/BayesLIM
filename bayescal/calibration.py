"""
Module for torch calibration models and relevant functions
"""
import torch
import numpy as np

from . import utils


class JonesModel(torch.nn.Module):
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
    def __init__(self, jones, ants, bls, R=None, parameter=True,
                 polmode='1pol'):
        """
        Antenna-based Jones model.

        Parameters
        ----------
        jones : tensor
            A tensor of the Jones parameters
            of shape (Npol, Npol, Nantenna, Ntimes, Nfreqs),
            where Nfreqs and Ntimes can be replaced by
            Nfreq_coeff and Ntime_coeff for sparse parameterization.
        ants : list
            List of antenna numbers associated with an ArrayModel object
        bls : list
            List of ant-pair tuples that hold the baselines of the
            input visibilities.
        R : callable, optional
            An arbitrary response function for the Jones parameters.
            This is a function that takes the jones tensor and maps it
            into a (generally) higher dimensional space that can then
            be applied to the model visibilities. An example is a
            mapping of jones parameters defined in delay space to
            the visibility space of frequency.
            Default (None) is a direct, unit response.
            Note this must use torch functions.
        parameter : bool, optional
            If True, treat jones as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        polmode : str, ['1pol', '2pol', '4pol'], optional
            Polarization mode. params must conform to polmode.
            1pol : single linear polarization (default)
            2pol : two linear polarizations (diag of Jones)
            4pol : four linear and cross pol (2x2 Jones)

        Examples
        --------
        1. Let V_m be of shape (1, 1, Nbl, Ntimes, Nfreq), and jones be of
        shape (1, 1, Nant, Ntimes, Nfreq), where Nbl is the number of
        baselines, and we are only using 1 feed polarization.
        Assume we have 3 antennas, [0, 1, 2], and three ant-pair
        visibility combinations [(0, 1), (1, 2), (0, 2)] in that order.
        Then vis2ants is simply [(0, 1), (1, 2), (0, 2)]
        """
        super().__init__()
        self.jones = jones
        if parameter:
            self.jones = torch.nn.Parameter(self.jones)
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R
        self.polmode = polmode
        self.ants = ants
        self.bls = bls
        self._vis2ants = {bl: (ants.index(bl[0]), ants.index(bl[1])) for bl in bls}

    def forward(self, V_m, jones=None, undo=False):
        """
        Forward pass V_m through the Jones model.

        Parameters
        ----------
        V_m : tensor
            Model visibilities of
            shape (Npol, Npol, Nbl, Nfreq, Ntime)
        jones : tensor, optional
            If not None, use these jones parameters instead of
            self.jones in the forward model. Default is None.
        undo : bool, optional
            If True, invert jones and apply to V_m. 

        Returns
        -------
        V_p : tensor
            Predicted visibilities, having forwarded
            V_m through the Jones parameters.
        """
        # setup empty predicted visibility
        V_p = torch.zeros_like(V_m)

        # choose fed jones or attached jones
        if jones is None:
            jones = self.jones

        # push through reponse function
        jones = self.R(jones)

        # invert jones if necessary
        if undo:
            invjones = torch.zeros_like(jones)
            for i in range(jones.shape[2]):
                if self.polmode in ['1pol', '2pol']:
                    invjones[:, :, i] = utils.diag_inv(jones[:, :, i])
                else:
                    invjones[:, :, i] = torch.pinv(jones[:, :, i])
            jones = invjones

        # iterate through visibility and apply Jones terms
        for i, bl in enumerate(self.bls):
            j1 = jones[:, :, self._vis2ants[bl][0]]
            j2 = jones[:, :, self._vis2ants[bl][1]]

            if self.polmode in ['1pol', '2pol']:
                V_p[:, :, i] = utils.diag_matmul(utils.diag_matmul(j1, V_m[:, :, i]), j2.conj())
            else:
                V_p[:, :, i] = torch.einsum("ab...,bc...,dc...->ad...", j1, V_m[:, :, i], j2.conj())

        return V_p


class RedVisModel(torch.nn.Module):
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
        V_m : tensor
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
        V_p : tensor
            The predicted visibilities, having pushed V_m through
            the redundant visibility model
        """
        # setup predicted visibility
        V_p = torch.zeros_like(V_m)

        params = self.R(self.params)

        # iterate through vis and apply redundant model
        for i in range(V_p.shape[2]):
            if not undo:
                V_p[:, :, i] = V_m[:, :, i] + params[:, :, self.vis2red[i]]
            else:
                V_p[:, :, i] = V_m[:, :, i] - params[:, :, self.vis2red[i]]

        return V_p


class VisModel(torch.nn.Module):
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
        V_m : tensor
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
        V_p : tensor
            The predicted visibilities, having pushed V_m through
            the visibility model.
        """
        params = self.R(self.params)
        if not undo:
            return V_m + params
        else:
            return V_m - params
