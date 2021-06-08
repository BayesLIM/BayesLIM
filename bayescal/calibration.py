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

    Notes
    -----
    2-real form means that the complex tensor of shape (3,)
        [1 + 0j, 2 + 1j, 1 - 2j]
    is stored as the real-valued tensor of shape (3, 2)
        [[1, 0], [2, 1], [1, -2]]
    which has an extra dimension along its last axis.
    See torch.view_as_complex() and torch.view_as_real().
    """
    def __init__(self, jones, vis2ants, R=None, parameter=True):
        """
        Antenna-based Jones model.

        Parameters
        ----------
        jones : tensor
            A 2-real tensor of the initial Jones parameters
            of shape (Npol, Npol, Nantenna, ..., 2).
            Npol=1 in 1-pol mode, or 2 for 2-pol or 4-pol mode.
        vis2ants : list of 2-tuples
            This is a list of length Nvis, whose elements index the
            0th axis of V_m, which is the input to self.forward().
            Each 2-tuple indexes the Nant axis of jones, and pick out
            the two antennas participating in this visibility.
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

        Examples
        --------
        1. Let V_m be of shape (1, 1, Nvis, Nfreq, 2), and jones be of
        shape (1, 1, Nant, Nfreq, 2), where Nvis is the number of
        visibilities, and we are only using 1 feed polarization.
        Assume we have 3 antennas, [0, 1, 2], and three ant-pair
        visibility combinations [(0, 1), (1, 2), (0, 2)] in that order.
        Then vis2ants is simply [(0, 1), (1, 2), (0, 2)]

        """
        super().__init__()
        self.jones = jones
        if parameter:
            self.jones = torch.nn.Parameter(self.jones)
        self.vis2ants = vis2ants
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R

    def forward(self, V_m, jones=None, undo=False):
        """
        Forward pass V_m through the Jones model.

        Parameters
        ----------
        V_m : tensor
            Model visibilities in 2-real form of
            shape (Npol, Npol, Nvis, ..., 2).
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
            _j = self.jones
        else:
            _j = jones

        # iterate through visibility and apply Jones terms
        for i in range(V_p.shape[2]):
            # push parameter through response
            j1 = self.R(_j[:, :, self.vis2ants[i][0]])
            j2 = self.R(_j[:, :, self.vis2ants[i][1]])

            # apply gains to input
            if not undo:
                #V_p[i] = utils.ceinsum("ij...,jk...,kl...->il...", j1, V_m[:, :, i],
                #                      torch.transpose(utils.cconj(j2), 0, 1))
                V_p[:, :, i] = utils.cmatmul(utils.cmatmul(j1, V_m[:, :, i]),
                                       torch.transpose(utils.cconj(j2), 0, 1))
            else:
                j1inv = utils.cinv(j1)
                j2inv = utils.cinv(j2)
                #V_p[i] = utils.ceinsum("ij...,jk...,kl...->il...", j1inv, V_m[:, :, i],
                #                      torch.transpose(utils.cconj(j2inv), 0, 1))
                V_p[:, :, i] = utils.cmatmul(utils.cmatmul(j1inv, V_m[:, :, i]),
                                       torch.transpose(utils.cconj(j2inv), 0, 1))

        return V_p


class RedVisModel(torch.nn.Module):
    """
    Redundant visibility model (r) relating the starting
    model visibility (m) to the data visibility (d)
    for antennas j and k.

    .. math::

        V^{d}_{jk} = V^{r} + V^{m}_{jk}

    """
    def __init__(self, red, vis2red, R=None, parameter=True):
        """
        Redundant visibility model

        Parameters
        ----------
        red : tensor
            Initial redundant visibility tensor in 2-real form,
            of shape (Npol, Npol, Nredvis, ..., 2) where Nredvis
            is the number of unique baseline types.
        vis2red : list of int
            A list of length Nvis--the length of V_m input to
            self.forward()--whose elements index red.
        R : callable, optional
            An arbitrary response function for the redundant
            visibility model, mapping the self.red parameters
            to the space of V_m (input to self.forward).
            Default (None) is unit response.
            Note this must use torch functions.
        parameter : bool, optional
            If True, treat red as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        """
        super().__init__()
        self.red = red
        if parameter:
            self.red = torch.nn.Parameter(self.red)
        self.vis2red = vis2red
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R

    def forward(self, V_m, red=None, undo=False):
        """
        Forward pass V_m through redundant
        model term.

        Parameters
        ----------
        V_m : tensor
            Starting model visibilities in 2-real form of
            shape (Npol, Npol, Nvis, ..., 2). In the general case,
            this should be a unit matrix so that the
            predicted visibilities are simply the redundant
            model. However, if you have a model of per-baseline
            non-redundancies, these could be included by putting
            them into V_m.
        red : tensor, optional
            If not None, use this as the redundant visibility model
            instead of self.red. Default is None.
        undo : bool, optional
            If True, push V_m backwards through the model.

        Returns
        -------
        V_p : tensor
            The predicted visibilities, having pushed V_m through
            the redundant visibility model in self.red.
        """
        # setup predicted visibility
        V_p = torch.zeros_like(V_m)

        # apply fed r or attr r
        if red is None:
            _r = self.red
        else:
            _r = red

        # iterate through vis and apply redundant model
        for i in range(V_p.shape[2]):
            if not undo:
                #V_p[i] = utils.cmult(V_m[i], _r[self.vis2red[i]])
                V_p[:, :, i] = V_m[:, :, i] + _r[:, :, self.vis2red[i]]
            else:
                #V_p[i] = utils.cdiv(V_m[i], _r[self.vis2red[i]])
                V_p[:, :, i] = V_m[:, :, i] - _r[:, :, self.vis2red[i]]

        return V_p


class VisModel(torch.nn.Module):
    """
    Visibility model (v) relating the starting
    model visibility (m) to the data visibility (d)
    for antennas j and k.

    .. math::

        V^{d}_{jk} = V^{v}_{jk} + V^{m}_{jk} 

    """
    def __init__(self, vis, R=None, parameter=True):
        """
        Visibility model

        Parameters
        ----------
        vis : tensor
            Initial visibility tensor in 2-real form,
            of shape (Nvis, ..., 2). Ordering should
            match ordering of V_m input to self.forward.
        R : callable, optional
            An arbitrary response function for the
            visibility model, mapping the self.vis parameters
            to the space of V_m (input to self.forward).
            Default (None) is unit response.
            Note this must use torch functions.
        parameter : bool, optional
            If True, treat vis as a parameter to be fitted,
            otherwise treat it as fixed to its input value.
        """
        super().__init__()
        self.vis = vis
        if parameter:
            self.vis = torch.nn.Parameter(self.vis)
        if R is None:
            # dummy function eval
            R = lambda x: x
        self.R = R

    def forward(self, V_m, vis=None, undo=False):
        """
        Forward pass V_m through visibility
        model term.

        Parameters
        ----------
        V_m : tensor
            Starting model visibilities in 2-real form
            of shape (Nvis, ..., 2). In the general case,
            this should be a unit matrix so that the
            predicted visibilities are simply the redundant
            model. However, if you have a model of per-baseline
            non-redundancies, these could be included by putting
            them into V_m.
        vis : tensor, optional
            If not None, use this as the visibility model
            instead of self.vis. Default is None.
        undo : bool, optional
            If True, push V_m backwards through the model.

        Returns
        -------
        V_p : tensor
            The predicted visibilities, having pushed V_m through
            the visibility model.
        """
        # setup predicted visibility
        V_p = torch.zeros_like(V_m)

        # apply fed r or attr r
        if vis is None:
            _v = self.vis
        else:
            _v = vis

        # iterate through vis and apply model
        for i in range(len(V_p.shape[2])):
            if not undo:
                V_p[:, :, i] = V_m[:, :, i] + _v[:, :, i]
            else:
                V_p[:, :, i] = V_m[i] - _v[:, :, i]

        return V_p

