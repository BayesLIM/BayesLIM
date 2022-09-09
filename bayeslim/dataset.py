"""
Module for visibility and map data formats, and a torch style data loader
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import copy
import h5py

from . import version, utils, telescope_model


class TensorData:
    """
    A shallow object for holding an arbitrary tensor data
    """
    def __init__(self):
        # init empty object
        self.setup_data()

    def setup_data(self, data=None, flags=None, cov=None,
                   cov_axis=None, icov=None, history=''):
        """
        Setup data tensors.

        Parameters
        ----------
        data : tensor, optional
            Data tensor of arbitrary shape
        flags : tensor, optional
            Boolean tensor holding data flags of same shape
        cov : tensor, optional
            Covariance of data tensor of multiple shapes
        cov_axis : str, optional
            None : (default) cov is same shape as data
                and represents data variance
            'full' : cov is N x N where N is data.size
                and represents covariance of data.ravel()
        icov : tensor, optional
            Inverse covariance. Must have same shape as cov.
        history : str, optional
            data history string
        """
        self.data = data
        self.flags = flags
        self.set_cov(cov, cov_axis, icov=icov)
        self.history = history

    def push(self, device):
        """
        Push data, flags, cov and icov to device
        """
        if self.data is not None:
            self.data = self.data.to(device)
        if self.flags is not None:
            self.flags = self.flags.to(device)
        if self.cov is not None:
            self.cov = self.cov.to(device)
        if self.icov is not None:
            self.icov = self.icov.to(device)

    def set_cov(self, cov, cov_axis, icov=None):
        """
        Set the covariance matrix as self.cov and
        compute covariance properties (ndim and log-det)

        Parameters
        ----------
        cov : tensor
            Covariance of data in a form that conforms
            to cov_axis specification. See optim.apply_icov
            for details on shape of cov.
        cov_axis : str
            The data axis along which the covariance is modeled.
            This specifies the type of covariance being supplied.
            Options are [None, 'full'].
            See optim.apply_icov() for details
        icov : tensor, optional
            pre-computed inverse covariance to set.
            Recommended to first set cov and then call compute_icov()
        """
        cov_logdet = None
        if cov is not None:
            # compute covariance log determinant
            if cov_axis is None:
                # cov is same shape as data and only holds variances
                cov_logdet = torch.sum(torch.log(cov))
            elif cov_axis == 'full':
                # cov is 2D (N x N)
                cov_logdet = torch.slogdet(cov).logabsdet
            else:
                # cov ndim > 2, but first two axes hold covariances
                cov_logdet = torch.tensor(0.)
                def recursive_logdet(cov, cov_logdet):
                    if cov.ndim > 2:
                        for i in range(cov.shape[2]):
                            recursive_logdet(cov[:, :, i], cov_logdet)
                    else:
                        cov_logdet += torch.slogdet(cov).logabsdet
                recursive_logdet(cov, cov_logdet)
            if torch.is_complex(cov_logdet):
                cov_logdet = cov_logdet.real

        # set covariance
        self.cov = cov
        self.icov = icov
        self.cov_axis = cov_axis
        self.cov_ndim = sum(self.data.shape) if self.data is not None else None
        self.cov_logdet = cov_logdet

    def compute_icov(self, pinv=True, rcond=1e-15):
        """
        Compute and set inverse covariance as self.icov.
        See optim.compute_cov and apply_icov() for shape.

        Parameters
        ----------
        pinv : bool, optional
            Use pseudo-inverse to compute covariance,
            otherwise use direct inversion
        rcond : float, optional
            rcond kwarg for pinverse
        """
        from bayeslim import optim
        self.icov = optim.compute_icov(self.cov, self.cov_axis, pinv=pinv, rcond=rcond)


class VisData(TensorData):
    """
    An object for holding visibility data of shape
    (Npol, Npol, Nbl, Ntimes, Nfreqs)
    """
    def __init__(self):
        # init empty object
        self.atol = 1e-10

    def setup_meta(self, telescope=None, antpos=None):
        """
        Set the telescope and antpos dict

        Parameters
        ----------
        telescope : TelescopeModel, optional
            Telescope location
        antpos : dict, optional
            Antenna position dictionary in ENU [meters].
            Antenna number integer as key, position vector as value
        """
        self.telescope = telescope
        self.antpos = antpos
        if antpos is not None:
            self.ants = np.array(list(antpos.keys()))
            self.antvec = np.array(list(antpos.values()))

    def setup_data(self, bls, times, freqs, pol=None,
                   data=None, flags=None, cov=None, cov_axis=None,
                   icov=None, history=''):
        """
        Setup metadata and optionally data tensors.

        Parameters
        ----------
        bls : list
            List of baseline tuples (antpairs) matching
            ordering of Nbl axis of data.
        times : tensor
            Julian date of unique times in the data.
        freqs : tensor
            Frequency array [Hz] of the data
        pol : str, optional
            If data Npol == 1, this is the name
            of the dipole polarization, one of ['ee', 'nn'].
            If None, it is assumed Npol == 2.
        data : tensor, optional
            Complex visibility data tensor of shape
            (Npol, Npol, Nbls, Ntimes, Nfreqs), where
            Npol is either 1 or 2 depending on the polmode.
            If Npol = 1, then pol must be fed.
        flags : tensor, optional
            Boolean tensor holding data flags
        cov : tensor, optional
            Tensor holding data covariance. If this is
            the same shape as data, then the elements are
            the data variance. Otherwise, this is of shape
            (Nax, Nax, ...) where Nax is the axis of the
            modeled covariance. All other axes should broadcast
            with other unmodeled axes.
            E.g. if
                cov_axis = 'bl': (Nbl, Nbl, Npol, Npol, Nfreqs, Ntimes)
                cov_axis = 'freq': (Nfreqs, Nfreqs, Npol, Npol, Nbl, Ntimes)
                cov_axis = 'time': (Ntimes, Ntimes, Npol, Npol, Nbl, Nfreqs)
        cov_axis : str, optional
            If cov represents on and off diagonal components, this is the
            axis over which off-diagonal is modeled.
            One of ['bl', 'time', 'freq'].
        icov : tensor, optional
            Inverse covariance. Must have same shape as cov.
            Recommended to call self.set_cov() and then self.compute_icov()
        history : str, optional
            data history string
        """
        self.Nbls = len(bls)
        self.ant1 = np.array([bl[0] for bl in bls])
        self.ant2 = np.array([bl[1] for bl in bls])
        self.times = times
        self.Ntimes = len(times)
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.pol = pol
        if isinstance(pol, str):
            assert pol.lower() in ['ee', 'nn'], "pol must be 'ee' or 'nn' for 1pol mode"
        self.Npol = 2 if self.pol is None else 1
        self.data = data
        self.flags = flags
        self.set_cov(cov, cov_axis, icov=icov)
        self.history = history

    @property
    def bls(self):
        return [(ant1, ant2) for ant1, ant2 in zip(self.ant1, self.ant2)]

    def copy(self, detach=True):
        """
        Copy and return self. This is equivalent
        to a detach and clone. Detach is optional
        """
        vd = VisData()
        vd.setup_meta(telescope=self.telescope, antpos=self.antpos)
        data = self.data.detach() if detach else self.data
        vd.setup_data(self.bls, self.times, self.freqs, pol=self.pol,
                      data=data.clone(),
                      flags=self.flags, cov=self.cov, icov=self.icov,
                      cov_axis=self.cov_axis, history=self.history)
        return vd

    def _bl2ind(self, bl):
        """
        Baseline(s) to index

        Parameters
        ----------
        bl : tuple or list of tuple
            Baseline antpair or list of such
        """
        if isinstance(bl, list):
            return [self._bl2ind(b) for b in bl]
        return np.where((self.ant1==bl[0])&(self.ant2==bl[1]))[0][0]

    def _bl2uniq_blpol(self, bl):
        """
        Given an antpair or antpair pol tuple "bl",
        or list of such, return unique bls and pol
        """
        if isinstance(bl, tuple):
            # this is an antpair or antpairpol
            if len(bl) == 2:
                bl, pol = [bl], None
            elif len(bl) == 3:
                bl, pol = [bl[:2]], bl[2]
        elif isinstance(bl, list):
            # this is a list of antpairs or antpairpols
            bl_list, pol_list = [], []
            for b in bl:
                _bl, _pol = self._bl2uniq_blpol(b)
                if _bl not in bl_list:
                    bl_list.extend(_bl)
                if _pol not in pol_list:
                    pol_list.append(_pol)
            bl = bl_list
            pol = pol_list
            if len(pol) > 1 and None in pol:
                pol.remove(None)
            assert len(pol) == 1, "can only index 1 pol at a time"
            pol = pol[0]

        return bl, pol

    def _time2ind(self, time):
        """
        Time(s) to index

        Parameters
        ----------
        time : float or list of floats
            Julian date times to index
        """
        iterable = False
        if isinstance(time, (list, np.ndarray)):
            iterable = True
        elif isinstance(time, torch.Tensor):
            if time.ndim == 1:
                iterable = True
        if iterable:
            return np.concatenate([self._time2ind(t) for t in time]).tolist()
        return np.where(np.isclose(self.times, time, atol=self.atol, rtol=1e-10))[0].tolist()

    def _freq2ind(self, freq):
        """
        Freq(s) to index

        Parameters
        ----------
        freq : float or list of floats
            frequencies [Hz] to index
        """
        iterable = False
        if isinstance(freq, (list, np.ndarray)):
            iterable = True
        elif isinstance(freq, torch.Tensor):
            if freq.ndim == 1:
                iterable = True
        if iterable:
            return np.concatenate([self._freq2ind(f) for f in freq]).tolist()
        return np.where(np.isclose(self.freqs, freq, atol=self.atol))[0].tolist()

    def _pol2ind(self, pol):
        """
        Polarization to index

        Parameters
        ----------
        pol : str
            Polarization to index
        """
        if isinstance(pol, list):
            assert len(pol) == 1
            pol = pol[0]
        assert isinstance(pol, str)
        if self.pol is not None:
            if pol.lower() != self.pol.lower():
                raise ValueError("cannot index pol from 1pol {}".format(self.pol))
            return slice(0, 1)
        if pol.lower() == 'ee':
            return slice(0, 1)
        elif pol.lower() == 'nn':
            return slice(1, 2)
        else:
            raise ValueError("cannot index cross-pols")

    def get_inds(self, bl=None, times=None, freqs=None, pol=None,
                 bl_inds=None, time_inds=None, freq_inds=None):
        """
        Given data selections, return data indexing list

        Parameters
        ----------
        bl : tuple or list of tuples, optional
            Baseline antpair, or list of such. Can also be
            antpair-pol, with limitations.
        times : tensor or float, optional
            Time(s) to index
        freqs : tensor or float, optional
            Frequencies to index
        pol : str, optional
            Polarization to index
        bl_inds : int or list of int, optional
            Instead of feeding bl, can feed a
            list of bl indices if these are
            already known given their location
            in self.ant1, self.ant2.
        time_inds : int or list of int, optional
            Instead of feeding times, can feed
            a list of time indices if these
            are already known given location
            in self.times.
        freq_inds : int or list of int, optional
            Instead of feeding freqs, can feed
            a list of freq indices if these
            are already known given location
            in self.freqs.

        Returns
        -------
        list
            A 5-len list holding slices along axes.
        """
        if bl is not None:
            assert bl_inds is None
            # special case for antpairpols
            bl, _pol = self._bl2uniq_blpol(bl)
            bl_inds = self._bl2ind(bl)
            if pol is not None:
                if _pol is not None:
                    assert _pol == pol
            pol = _pol
        elif bl_inds is not None:
            pass
        else:
            bl_inds = slice(None)

        if times is not None:
            assert time_inds is None
            time_inds = self._time2ind(times)
        elif time_inds is not None:
            pass
        else:
            time_inds = slice(None)

        if freqs is not None:
            assert freq_inds is None
            freq_inds = self._freq2ind(freqs)
        elif freq_inds is not None:
            pass
        else:
            freq_inds = slice(None)

        if pol is not None:
            pol_ind = self._pol2ind(pol)
        else:
            pol_ind = slice(None)

        inds = [pol_ind, pol_ind, bl_inds, time_inds, freq_inds]
        inds = tuple([utils._list2slice(ind) for ind in inds])
        slice_num = sum([isinstance(ind, slice) for ind in inds])
        assert slice_num > 3, "cannot fancy index more than 1 axis"

        return inds

    def get_data(self, bl=None, times=None, freqs=None, pol=None,
                 bl_inds=None, time_inds=None, freq_inds=None,
                 squeeze=True, data=None):
        """
        Slice into data tensor and return values.
        Only one axis can be specified at a time.

        Parameters
        ----------
        bl : tuple or list of tuples, optional
            Baseline antpair, or list of such, to return
        times : tensor or float, optional
            Time(s) to index
        freqs : tensor or float, optional
            Frequencies to index
        pol : str, optional
            Polarization to index
        bl_, time_, freq_inds : optional
            See self.get_inds() for details.
        squeeze : bool, optional
            If True, squeeze array before return
        data : tensor, optional
            Tensor to index. default is self.data
        """
        data = self.data if data is None else data
        if data is None:
            return None

        # get indexing
        inds = self.get_inds(bl=bl, times=times, freqs=freqs, pol=pol,
                             bl_inds=bl_inds, time_inds=time_inds,
                             freq_inds=freq_inds)
        data = data[inds]

        # squeeze if desired
        if squeeze:
            data = data.squeeze()

        return data

    def get_flags(self, bl=None, times=None, freqs=None, pol=None,
                  bl_inds=None, time_inds=None, freq_inds=None,
                  squeeze=True, flags=None):
        """
        Slice into flag tensor and return values.
        Only one axis can be specified at a time.

        Parameters
        ----------
        bl : tuple or list of tuples, optional
            Baseline antpair, or list of such, to return
        times : tensor or float, optional
            Time(s) to index
        freqs : tensor or float, optional
            Frequencies to index
        pol : str, optional
            Polarization to index
        bl_, time_, freq_inds : optional
            See self.get_inds() for details
        squeeze : bool, optional
            If True, squeeze array before return
        flags : tensor, optional
            flag array to index. default is self.flags
        """
        flags = self.flags if flags is None else flags
        if flags is None:
            return None

        # get indexing
        inds = self.get_inds(bl=bl, times=times, freqs=freqs, pol=pol,
                             bl_inds=bl_inds, time_inds=time_inds,
                             freq_inds=freq_inds)
        flags = flags[inds]

        # squeeze if desired
        if squeeze:
            flags = flags.squeeze()

        return flags

    def get_cov(self, bl=None, times=None, freqs=None, pol=None,
                bl_inds=None, time_inds=None, freq_inds=None,
                squeeze=True, cov=None):
        """
        Slice into cov tensor and return values.
        Only one axis can be specified at a time.
        See optim.apply_icov() for details on shape.

        Parameters
        ----------
        bl : tuple or list of tuples, optional
            Baseline antpair, or list of such, to return
        times : tensor or float, optional
            Time(s) to index
        freqs : tensor or float, optional
            Frequencies to index
        pol : str, optional
            Polarization to index
        bl_, time_, freq_inds : optional
            See self.get_inds() for details
        squeeze : bool, optional
            If True, squeeze array before return
        cov : tensor, optional
            Cov tensor to index. default is self.cov
        """
        cov = self.cov if cov is None else cov
        if cov is None:
            return None

        # get indexing
        inds = self.get_inds(bl=bl, times=times, freqs=freqs, pol=pol,
                             bl_inds=bl_inds, time_inds=time_inds,
                             freq_inds=freq_inds)

        if self.cov_axis is None:
            # cov is same shape as data
            cov = cov[inds]
        else:
            # cov is not the same shape as data
            if bl is not None or bl_inds is not None:
                if self.cov_axis == 'bl':
                    cov = cov[inds[2]][:, inds[2]]
                elif self.cov_axis in ['time', 'freq']:
                    cov = cov[:, :, :, :, inds[2]]
                elif self.cov_axis == 'full':
                    raise NotImplementedError
            elif times is not None or time_inds is not None:
                if self.cov_axis == 'time':
                    cov = cov[inds[3]][:, inds[3]]
                elif self.cov_axis == 'bl':
                    cov = cov[:, :, :, :, inds[3]]
                elif self.cov_axis == 'freq':
                    cov = cov[:, :, :, :, :, inds[3]]
                elif self.cov_axis == 'full':
                    raise NotImplementedError
            elif freqs is not None or freq_inds is not None:
                if self.cov_axis == 'freq':
                    cov = cov[inds[4]][:, inds[4]]
                elif self.cov_axis in ['bl', 'time']:
                    cov = cov[:, :, :, :, :, inds[4]]
                elif self.cov_axis == 'full':
                    raise NotImplementedError
            elif pol is not None:
                # pol-pol covariance not yet implemented
                cov = cov[:, :, inds[0], inds[0]]
            else:
                cov = cov[:]

        # squeeze
        if squeeze:
            cov = cov.squeeze()

        return cov

    def get_icov(self, bl=None, icov=None, **kwargs):
        """
        Slice into cached inverse covariance.
        Same kwargs as get_cov()
        """
        if icov is None:
            icov = self.icov
        return self.get_cov(bl=bl, cov=icov, **kwargs)

    def __getitem__(self, bl):
        return self.get_data(bl, squeeze=True)

    def __setitem__(self, bl, val):
        self.set(bl, val)

    def set(self, bl, val, arr='data'):
        """
        Set the desired sliced attribute "arr" as val

        Parameters
        ----------
        bl : tuple, optional
            Baseline(pol) to set
        kwargs : dict, optional
            Other parameters to set
        val : tensor
            Value to assign
        arr : str, optional
            Attribute to set
            ['data', 'flags', 'icov', 'cov']
        """
        # get slice indices
        inds = self.get_inds(bl=bl)

        # get desired attribute
        if arr == 'data':
            arr = self.data
        elif arr == 'flags':
            arr = self.flags
        elif arr == 'cov':
            # assert cov is shape of data
            assert self.cov_axis is None
            arr = self.cov
        elif arr == 'icov':
            # assert cov is shape of data
            assert self.cov_axis is None
            arr = self.icov

        arr[inds] = val

    def select(self, bl=None, times=None, freqs=None, pol=None,
               bl_inds=None, time_inds=None, freq_inds=None,
               inplace=True):
        """
        Downselect on data tensor.

        Parameters
        ----------
        bl : list, optional
            List of baselines to downselect
        times : tensor, optional
            List of Julian Date times to downselect
        freqs : tensor, optional
            List of frequencies [Hz] to downselect
        pol : str, optional
            Polarization to downselect
        bl_inds : int or list of int, optional
            Instead of feeding bl, can feed a
            list of bl indices if these are
            already known given their location
            in self.ant1, self.ant2.
        time_inds : int or list of int, optional
            Instead of feeding times, can feed
            a list of time indices if these
            are already known given location
            in self.times.
        freq_inds : int or list of int, optional
            Instead of feeding freqs, can feed
            a list of freq indices if these
            are already known given location
            in self.freqs.
        inplace : bool, optional
            If True downselect inplace, otherwise return
            a new VisData object
        """
        if inplace:
            obj = self
        else:
            obj = copy.deepcopy(self)

        if bl is not None or bl_inds is not None:
            assert not ((bl is not None) & (bl_inds is not None))
            data = obj.get_data(bl, bl_inds=bl_inds, squeeze=False)
            cov = obj.get_cov(bl, bl_inds=bl_inds, squeeze=False)
            icov = obj.get_icov(bl, bl_inds=bl_inds, squeeze=False)
            flags = obj.get_flags(bl, bl_inds=bl_inds, squeeze=False)
            if bl_inds is not None: bl = [obj.bls[i] for i in bl_inds]
            obj.setup_data(bl, obj.times, obj.freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if times is not None or time_inds is not None:
            assert not ((times is not None) & (time_inds is not None))
            data = obj.get_data(times=times, time_inds=time_inds, squeeze=False)
            cov = obj.get_cov(times=times, time_inds=time_inds, squeeze=False)
            icov = obj.get_icov(times=times, time_inds=time_inds, squeeze=False)
            flags = obj.get_flags(times=times, time_inds=time_inds, squeeze=False)
            if time_inds is not None: times = obj.times[time_inds]
            obj.setup_data(obj.bls, times, obj.freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if freqs is not None or freq_inds is not None:
            assert not ((freqs is not None) & (freq_inds is not None))
            data = obj.get_data(freqs=freqs, freq_inds=freq_inds, squeeze=False)
            cov = obj.get_cov(freqs=freqs, freq_inds=freq_inds, squeeze=False)
            icov = obj.get_icov(freqs=freqs, freq_inds=freq_inds, squeeze=False)
            flags = obj.get_flags(freqs=freqs, freq_inds=freq_inds, squeeze=False)
            if freq_inds is not None: freqs = obj.freqs[freq_inds]
            obj.setup_data(obj.bls, obj.times, freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if pol is not None:
            data = obj.get_data(pol=pol)
            flags = obj.get_flags(pol=pol)
            cov = obj.get_cov(pol=pol)
            icov = obj.get_icov(pol=pol)
            obj.setup_data(obj.bls, obj.times, obj.freqs, pol=pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if not inplace:
            return obj

    def apply_cal(self, cd, undo=False, inplace=True, cal_2pol=False):
        """
        Apply CalData to this VisData object.
        Default behavior is to multiply by gains.

        Parameters
        ----------
        cd : CalData object
            Calibration solutions to apply
        undo : bool, optional
            If True, multiply gains (default)
            otherwise divide by gains
        inplace : bool, optional
            If True edit self.data inplace
            otherwise copy and return a new VisData
        cal_2pol : bool, optional
            If True, calibrate 4pol vis with diagonal
            of 4pol gains. Only applicable if in 4pol mode. 

        Returns
        -------
        VisData
        """
        from bayeslim import calibration
        if inplace:
            vd = self
        else:
            vd = self.copy(detach=True)
        cal_data = cd.data
        if utils.device(vd.data.device) != utils.device(cal_data.device):
            cal_data = cal_data.to(vd.data.device)

        vd.data, vd.cov = calibration.apply_cal(vd.data, vd.bls, cal_data, cd.ants, cal_2pol=cal_2pol,
                                                cov=vd.cov, vis_type='com', undo=undo, inplace=inplace)

        return vd

    def inflate_by_redundancy(self, redtol=1.0, min_len=None, max_len=None):
        """
        If current data only includes unique redundant baseline types,
        copy over redundant types to all physical baselines and return
        a new copy of the object.

        Parameters
        ----------
        redtol : float, optional
            Redundancy tolerance in meters
        min_len : float, optional
            Minimum baseline length to keep in meters
        max_len : float, optional
            Maximum baseline length to keep in meters

        Returns
        -------
        VisData
        """
        # get setup an array object
        array = telescope_model.ArrayModel(self.antpos, self.freqs, redtol=redtol)
        # get redundant indices of current baselines
        redinds = [array.bl2red[bl] for bl in self.bls]
        
        # get all new baselines
        new_bls = array.get_bls(min_len=min_len, max_len=max_len)
        _bls = []
        for bl in new_bls:
            redidx = array.bl2red[bl]
            if redidx in redinds:
                _bls.append(self.bls[redinds.index(redidx)])
        
        # expand data across redundant baselines
        data = self.get_data(bl=_bls, squeeze=False)
        flags = self.get_flags(bl=_bls, squeeze=False)
        cov = self.get_cov(bl=_bls, squeeze=False)
        icov = self.get_icov(bl=_bls, squeeze=False)
        
        # setup new object
        new_vis = VisData()
        new_vis.setup_meta(telescope=self.telescope, antpos=self.antpos)
        new_vis.setup_data(new_bls, self.times, self.freqs, pol=self.pol,
                           data=data, flags=flags, cov=cov, icov=icov,
                           history=self.history)
        
        return new_vis

    def write_hdf5(self, fname, overwrite=False):
        """
        Write VisData to hdf5 file.

        Parameters
        ----------
        fname : str
            Output hdf5 filename
        overwrite : bool, optional
            If fname exists, overwrite it
        """
        import h5py
        from bayeslim import utils
        if not os.path.exists(fname) or overwrite:
            with h5py.File(fname, 'w') as f:
                # write data and metadata
                f.create_dataset('data', data=utils.tensor2numpy(self.data))
                if self.flags is not None:
                    f.create_dataset('flags', data=self.flags)
                if self.cov is not None:
                    f.create_dataset('cov', data=self.cov)
                if self.cov_axis is not None:
                    f.attr['cov_axis'] = self.cov_axis
                if self.icov is not None:
                    f.create_dataset('icov', data=self.icov)
                f.create_dataset('bls', data=self.bls)
                f.create_dataset('times', data=self.times)
                f.create_dataset('freqs', data=self.freqs)
                if self.pol is not None:
                    f.attrs['pol'] = self.pol
                f.attrs['history'] = self.history
                # write telescope and array objects
                f.attrs['tloc'] = self.telescope.location
                f.attrs['ants'] = self.ants
                f.attrs['antvec'] = self.antvec
                f.attrs['obj'] = 'VisData'
                f.attrs['version'] = version.__version__
        else:
            print("{} exists, not overwriting...".format(fname))

    def read_hdf5(self, fname, read_data=True, bl=None, times=None, freqs=None, pol=None,
                  suppress_nonessential=False):
        """
        Read HDF5 VisData object

        Parameters
        ----------
        fname : str
            File to read
        read_data : bool, optional
            If True, read data arrays as well as metadata
        bl, times, freqs, pol : read options. see self.select() for details
        suppress_nonessential : bool, optional
            If True, suppress reading-in flags and cov, as only data and icov
            are essential for inference.
        """
        import h5py
        from bayeslim import telescope_model
        with h5py.File(fname, 'r') as f:
            # load metadata
            assert str(f.attrs['obj']) == 'VisData', "not a VisData object"
            _bls = [tuple(_bl) for _bl in f['bls'][:]]
            _times = f['times'][:]
            _freqs = torch.as_tensor(f['freqs'][:])
            _pol = f.attrs['pol'] if 'pol' in f.attrs else None
            cov_axis = f.attrs['cov_axis'] if 'cov_axis' in f.attrs else None
            history = f.attrs['history'] if 'history' in f.attrs else ''

            # setup just full-size metadata
            self.setup_data(_bls, _times, _freqs, pol=_pol,
                            cov_axis=cov_axis)

            data, flags, cov, icov = None, None, None, None
            if read_data:
                data = self.get_data(bl=bl, times=times, freqs=freqs, pol=pol,
                                     squeeze=False, data=f['data'])
                data = torch.as_tensor(data)
                if 'flags' in f and not suppress_nonessential:
                    flags = self.get_flags(bl=bl, times=times, freqs=freqs, pol=pol,
                                          squeeze=False, flags=f['flags'])
                    flags = torch.as_tensor(flags)
                else:
                    flags = None
                if 'cov' in f and not suppress_nonessential:
                    cov = self.get_cov(bl=bl, times=times, freqs=freqs, pol=pol,
                                       squeeze=False, cov=f['cov'])
                    cov = torch.as_tensor(cov)
                else:
                    cov = None
                if 'icov' in f:
                    icov = self.get_icov(bl=bl, times=times, freqs=freqs, pol=pol,
                                         squeeze=False, icov=f['icov'])
                    icov = torch.as_tensor(icov)
                else:
                    icov = None

            # downselect metadata to selection
            self.select(bl=bl, times=times, freqs=freqs, pol=pol)

            # setup downselected metadata and data
            ants = f.attrs['ants'].tolist()
            antvec = f.attrs['antvec']
            antpos = dict(zip(ants, antvec))
            tloc = f.attrs['tloc']
            telescope = telescope_model.TelescopeModel(tloc)
            self.setup_meta(telescope, antpos)
            self.setup_data(self.bls, self.times, self.freqs, pol=self.pol,
                            data=data, flags=flags, cov=cov,
                            cov_axis=cov_axis, icov=icov, history=history)

    def read_uvdata(self, fname, run_check=True, **kwargs):
        """
        Read a UVH5 file into a UVData and transfer to
        a VisData object. Needs pyuvdata support.

        Parameters
        ----------
        fname : str or UVData
            Filename(s) to read, or UVData object
        run_check : bool, optional
            Run check after read
        kwargs : dict
            kwargs for UVData read
        """
        from pyuvdata import UVData
        from bayeslim.utils import _float, _cfloat
        from bayeslim import telescope_model
        # load uvdata
        if isinstance(fname, str):
            uvd = UVData()
            uvd.read_uvh5(fname, **kwargs)
        elif isinstance(fname, UVData):
            uvd = fname
        # extract data and metadata
        bls = uvd.get_antpairs()
        Nbls = uvd.Nbls
        times = torch.as_tensor(np.unique(uvd.time_array))
        Ntimes = uvd.Ntimes
        freqs = torch.tensor(uvd.freq_array[0], dtype=_float())
        Nfreqs = uvd.Nfreqs
        if uvd.x_orientation is None:
            uvd.x_orientation = 'north'  # IAU convention by default
        pols = uvd.get_pols()
        if len(pols) == 1:
            pol = pols[0]
        else:
            pol = None
        self.history = uvd.history
        Npol = 1 if len(pols) == 1 else 2
        data_pols = [[pol]] if Npol == 1 else [['ee', 'en'], ['ne', 'nn']]

        # organize data
        data = np.zeros((Npol, Npol, Nbls, Ntimes, Nfreqs), dtype=uvd.data_array.dtype)
        flags = np.zeros((Npol, Npol, Nbls, Ntimes, Nfreqs), dtype=bool)
        for i in range(len(data_pols)):
            for j in range(len(data_pols)):
                dpol = data_pols[i][j]
                for k, bl in enumerate(bls):
                    data[i, j, k] = uvd.get_data(bl + (dpol,))
                    flags[i, j, k] = uvd.get_flags(bl + (dpol,))
        data = torch.tensor(data, dtype=_cfloat())
        flags = torch.tensor(flags, dtype=torch.bool)

        # setup data
        self.setup_data(bls, times, freqs, pol=pol,
                        data=data, flags=flags, history=uvd.history)
        # configure telescope data
        antpos, ants = uvd.get_ENU_antpos()
        antpos = dict(zip(ants, antpos))
        loc = uvd.telescope_location_lat_lon_alt_degrees
        telescope = telescope_model.TelescopeModel((loc[1], loc[0], loc[2]))
        self.setup_meta(telescope, antpos)

        if run_check:
            self.check()

    def check(self):
        """
        Run checks on data
        """
        from bayeslim import telescope_model
        if self.telescope is not None:
            assert isinstance(self.telescope, telescope_model.TelescopeModel)
        if self.antpos is not None:
            assert isinstance(self.antpos, dict)
        if self.data is not None:
            assert isinstance(self.data, torch.Tensor)
            assert self.data.shape == (self.Npol, self.Npol, self.Nbls, self.Ntimes, self.Nfreqs)
        if self.flags is not None:
            assert isinstance(self.flags, torch.Tensor)
            assert self.data.shape == self.flags.shape
        if self.cov is not None:
            assert self.cov_axis is not None, "full data-sized covariance not implemented"
            if self.cov_axis == 'bl':
                assert self.cov.shape == (self.Nbls, self.Nbls, self.Npol, self.Npol,
                                          self.Ntimes, self.Nfreqs)
            elif self.cov_axis == 'time':
                assert self.cov.shape == (self.Ntimes, self.Ntimes, self.Npol, self.Npol,
                                          self.Nbls, self.Nfreqs)
            elif self.cov_axis == 'freq':
                assert self.cov.shape == (self.Nfreqs, self.Nfreqs, self.Npol, self.Npol,
                                          self.Nbls, self.Ntimes)
        for ant1, ant2 in zip(self.ant1, self.ant2):
            assert (ant1 in self.ants) and (ant2 in self.ants)


class MapData(TensorData):
    """
    An object for holding image or map data of shape
    (Npol, 1, Nfreqs, Npix)
    """
    def __init__(self):
        self.atol = 1e-10

    def setup_meta(self):
        """
        Setup metadata
        """
        pass

    def setup_data(self, freqs, df=None, pols=None, data=None, angs=None,
                   altaz=None, flags=None, cov=None, cov_axis=None, icov=None,
                   norm=None, history=''):
        """
        Setup data

        Parameters
        ----------
        freqs : tensor
            Frequency bins [Hz]
        df : tensor, optional
            Channel width of map at each frequency [Hz]
        pols : list of str, optional
            Polarizations of data along Npol axis
        data : tensor, optional
            Map data of shape (Npol, 1, Nfreqs, Npixels)
        angs : tensor, optional
            [RA, Dec] on the sky of the pixel centers in J2000 coords
            of shape (2, Npix) in degrees.
        altaz : tensor, optional
            [Altitude, Azimuth] on the sky of the pixel centers
            of shape (2, Npix) in degrees in topocentric coords.
        flags : tensor, optional
            Flags of bool dtype, shape of data
        cov : tensor, optional
            Covariance of maps either shape of data
            or (Nax, Nax, ...) where Nax is cov_axis
        cov_axis : str, optional
            Axis of covariance. If None assume cov is
            shape of data and is just variance. Otherwise
            can be ['freq', 'pixel'] and cov is shape
            (Nax, Nax, ...)
        icov : tensor, optional
            Inverse covariance. Same rules apply as cov.
            Recommended to set cov and then run self.compute_icov
        norm : tensor, optional
            Another tensor the shape of the map data holding
            normalization information (e.g. beam map).
        history : str, optional
        """
        self.freqs = freqs
        self.df = df
        self.angs = angs
        self.altaz = altaz
        self.Nfreqs = len(freqs)
        if pols is not None:
            if isinstance(pols, (torch.Tensor, np.ndarray)):
                pols = [p.lower() for p in pols.tolist()]
        self.pols = pols
        self.data = data
        self.flags = flags
        self.norm = norm
        self.set_cov(cov, cov_axis, icov=icov)
        self.history = history

    def copy(self, detach=True):
        """
        Copy and return self. This is equivalent
        to a detach and clone. Detach is optional
        """
        md = MapData()
        md.setup_meta()
        data = self.data.detach() if detach else self.data
        md.setup_data(self.freqs, df=self.df, pols=self.pols, data=data.clone(), norm=self.norm,
                      angs=self.angs, altaz=self.altaz, flags=self.flags, cov=self.cov,
                      icov=self.icov, cov_axis=self.cov_axis, history=self.history)
        return md

    def get_inds(self, angs=None, altaz=None, freqs=None, pols=None,
                 ang_inds=None, freq_inds=None, pol_inds=None):
        """
        Given data selections, return data indexing list

        Parameters
        ----------
        angs : tensor, optional
            J2000 [ra,dec] angles [deg] to index
        altaz : tensor, optional
            [alt, az] angles [deg] to index
        freqs : tensor or float, optional
            Frequencies to index
        pols : str or list, optional
            Polarization(s) to index
        ang_inds : int or list of int, optional
            Instead of feeding angs or altaz, can feed a
            list of indices along the Npix axis
            to index.
        freq_inds : int or list of int, optional
            Instead of feeding freqs, can feed
            a list of freq indices if these
            are already known given location
            in self.freqs.
        pol_inds : int or list of int, optional
            Instead of feeding pol str, feed list
            of pol indices

        Returns
        -------
        list
            A 4-len list holding slices along axes.
        """
        if angs is not None:
            assert ang_inds is None
            assert altaz is None
            ang_inds = self._ang2ind(angs, altaz=False)
        elif altaz is not None:
            assert ang_inds is None
            ang_inds = self._ang2ind(altaz, altaz=True)
        elif ang_inds is not None:
            pass
        else:
            ang_inds = slice(None)

        if freqs is not None:
            assert freq_inds is None
            freq_inds = self._freq2ind(freqs)
        elif freq_inds is not None:
            pass
        else:
            freq_inds = slice(None)

        if pols is not None:
            assert pol_inds is None
            pol_inds = self._pol2ind(pols)
        elif pol_inds is not None:
            pass
        else:
            pol_inds = slice(None)

        inds = [pol_inds, slice(None), freq_inds, ang_inds]
        inds = tuple([utils._list2slice(ind) for ind in inds])
        slice_num = sum([isinstance(ind, slice) for ind in inds])
        assert slice_num > 2, "cannot fancy index more than 1 axis"

        return inds

    def _ang2ind(self, angs, altaz=False):
        """
        Pixel angles to index. Note this is slow
        because we loop over all input angles
        and do an isclose().

        Parameters
        ----------
        angs : tensor
            Pixel centers in [ra, dec] of degrees to index
            of shape (2, Nindex). If altaz=True, angs is
            assumed ot be [alt, az] in degrees.
        altaz : bool, optional
            If True, assume angs input is [alt, az] in deg,
            otherwise assume its [ra, dec] in deg.
        """
        angs = torch.as_tensor(angs)
        _angs = self.angs if not altaz else self.altaz
        idx = []
        for ang1, ang2 in zip(*angs):
            match = np.isclose(ang1, _angs[0], atol=self.atol, rtol=1e-10) \
                    & np.isclose(ang2, _angs[1], atol=self.atol, rtol=1e-10)
            if match.any():
                idx.append(np.where(match)[0][0])

        return idx

    def _freq2ind(self, freq):
        """
        Freq(s) to index

        Parameters
        ----------
        freq : float or list of floats
            frequencies [Hz] to index
        """
        iterable = False
        if isinstance(freq, (list, np.ndarray)):
            iterable = True
        elif isinstance(freq, torch.Tensor):
            if freq.ndim == 1:
                iterable = True
        if iterable:
            return np.concatenate([self._freq2ind(f) for f in freq]).tolist()

        return np.where(np.isclose(self.freqs, freq, atol=self.atol))[0].tolist()

    def _pol2ind(self, pol):
        iterable = False
        if isinstance(pol, (list, np.ndarray)):
            iterable = True
        elif isinstance(pol, torch.Tensor):
            if pol.ndim == 1:
                iterable = True
        if iterable:
            return [self._pol2ind(p) for p in pol]

        return self.pols.index(pol.lower())

    def get_data(self, data=None, squeeze=True, angs=None,
                 altaz=None, freqs=None, pols=None, ang_inds=None,
                 freq_inds=None, pol_inds=None):
        """
        Get map data given selections

        Parameters
        ----------
        data : tensor, optional
            Data tensor to index, default is self.data
        squeeze : bool, optional
            If True squeeze output
        See get_inds for more details

        Returns
        -------
        tensor
        """
        data = self.data if data is None else data
        if data is None:
            return None

        # get indexing
        inds = self.get_inds(angs=angs, altaz=altaz, freqs=freqs, pols=pols,
                             ang_inds=ang_inds, freq_inds=freq_inds,
                             pol_inds=pol_inds)
        data = data[inds]

        # squeeze if desired
        if squeeze:
            data = data.squeeze()

        return data

    def get_flags(self, flags=None, squeeze=True, angs=None,
                 altaz=None, freqs=None, pols=None, ang_inds=None,
                 freq_inds=None, pol_inds=None):
        """
        Get flag data given selections

        Parameters
        ----------
        flags : tensor, optional
            Flag tensor to index. Default is self.flags.
        See self.get_data()
        """
        flags = self.flags if flags is None else flags
        if flags is None:
            return None

        # get indexing
        inds = self.get_inds(angs=angs, altaz=altaz, freqs=freqs, pols=pols,
                             ang_inds=ang_inds, freq_inds=freq_inds,
                             pol_inds=pol_inds)
        flags = flags[inds]

        # squeeze if desired
        if squeeze:
            flags = flags.squeeze()

        return flags

    def get_cov(self, cov=None, cov_axis=None, squeeze=True, angs=None,
                altaz=None, freqs=None, pols=None, ang_inds=None,
                freq_inds=None, pol_inds=None):
        """
        Index covariance given selections

        Parameters
        ----------
        cov : tensor, optional
            Covariance to index. Default is self.cov
        cov_axis : str, optional
            Covariance axis of cov. Default is self.cov_axis
        See get_data() for details
        """
        cov = self.cov if cov is None else cov
        cov_axis = self.cov_axis if cov_axis is None else cov_axis
        if cov is None:
            return None

        # get indexing
        inds = self.get_inds(angs=angs, altaz=altaz, freqs=freqs, pols=pols,
                             ang_inds=ang_inds, freq_inds=freq_inds,
                             pol_inds=pol_inds)

        if cov_axis is None:
            # cov is same shape as data
            cov = cov[inds]
        else:
            # cov is not the same shape as data
            if angs is not None or altaz is not None or ang_inds is not None:
                if cov_axis == 'pix':
                    cov = cov[inds[3]][:, inds[3]]
                elif cov_axis in ['freq']:
                    cov = cov[:, :, :, inds[-1]]
                elif cov_axis == 'full':
                    raise NotImplementedError
            elif freqs is not None or freq_inds is not None:
                if cov_axis == 'freq':
                    cov = cov[inds[2]][:, inds[2]]
                elif cov_axis in ['pix']:
                    cov = cov[:, :, :, inds[2]]
                elif cov_axis == 'full':
                    raise NotImplementedError
            elif pols is not None or pol_inds is not None:
                # pol-pol covariance not yet implemented
                if cov_axis in ['freq', 'pix']:
                    cov = cov[:, :, inds[0]]
                elif cov_axis == 'full':
                    raise NotImplementedError
            else:
                cov = cov[:]

        # squeeze
        if squeeze:
            cov = cov.squeeze()

        return cov

    def get_icov(self, icov=None, **kwargs):
        """
        Slice into cached inverse covariance.
        Same kwargs as get_cov()
        """
        if icov is None:
            icov = self.icov
        return self.get_cov(cov=icov, **kwargs)

    def select(self, angs=None, altaz=None, freqs=None, pols=None,
               ang_inds=None, freq_inds=None, pol_inds=None,
               inplace=True):
        """
        Downselect on data tensor.

        Parameters
        ----------
        angs : tensor, optional
            J2000 [ra,dec] angles [deg] to index
        altaz : tensor, optional
            [alt, az] angles [deg] to index
        freqs : tensor or float, optional
            Frequencies to index
        pols : str or list, optional
            Polarization(s) to index
        ang_inds : int or list of int, optional
            Instead of feeding angs or altaz, can feed a
            list of indices along the Npix axis
            to index.
        freq_inds : int or list of int, optional
            Instead of feeding freqs, can feed
            a list of freq indices if these
            are already known given location
            in self.freqs.
        pol_inds : int or list of int, optional
            Instead of feeding pol str, feed list
            of pol indices
        inplace : bool, optional
            If True downselect inplace, otherwise return
            a new VisData object
        """
        if inplace:
            obj = self
        else:
            obj = copy.deepcopy(self)

        if angs is not None or altaz is not None or ang_inds is not None:
            data = obj.get_data(angs=angs, altaz=altaz, ang_inds=ang_inds, squeeze=False)
            norm = obj.get_data(data=self.norm, angs=angs, altaz=altaz, ang_inds=ang_inds, squeeze=False)
            cov = obj.get_cov(angs=angs, altaz=altaz, ang_inds=ang_inds, squeeze=False)
            icov = obj.get_icov(angs=angs, altaz=altaz, ang_inds=ang_inds, squeeze=False)
            flags = obj.get_flags(angs=angs, altaz=altaz, ang_inds=ang_inds, squeeze=False)
            if ang_inds is not None:
                if self.angs is not None:
                    angs = self.angs[:, ang_inds]
                if self.altaz is not None:
                    altaz = self.altaz[:, ang_inds]
            obj.setup_data(obj.freqs, df=obj.df, pols=obj.pols, data=data, angs=angs,
                           altaz=altaz, flags=flags, cov=cov, cov_axis=obj.cov_axis, icov=icov,
                           norm=norm, history=obj.history)

        if freqs is not None or freq_inds is not None:
            data = obj.get_data(freqs=freqs, freq_inds=freq_inds, squeeze=False)
            norm = obj.get_data(data=self.norm, freqs=freqs, freq_inds=freq_inds, squeeze=False)
            cov = obj.get_cov(angs=angs, freqs=freqs, freq_inds=freq_inds, squeeze=False)
            icov = obj.get_icov(angs=angs, freqs=freqs, freq_inds=freq_inds, squeeze=False)
            flags = obj.get_flags(angs=angs, freqs=freqs, freq_inds=freq_inds, squeeze=False)
            if freq_inds is not None:
                freqs = obj.freqs[freq_inds]
                df = obj.df[freq_inds] if obj.df is not None else None
            else:
                df = obj.df[self._freq2ind(freqs)] if obj.df is not None else None
            obj.setup_data(freqs, df=df, pols=obj.pols, data=data, angs=obj.angs,
                           altaz=obj.altaz, flags=flags, cov=cov, cov_axis=obj.cov_axis, icov=icov,
                           norm=norm, history=obj.history)

        if pols is not None or pol_inds is not None:
            data = obj.get_data(pols=pols, pol_inds=pol_inds, squeeze=False)
            norm = obj.get_data(data=self.norm, pols=pols, pol_inds=pol_inds, squeeze=False)
            cov = obj.get_cov(angs=angs, pols=pols, pol_inds=pol_inds, squeeze=False)
            icov = obj.get_icov(angs=angs, pols=pols, pol_inds=pol_inds, squeeze=False)
            flags = obj.get_flags(angs=angs, pols=pols, pol_inds=pol_inds, squeeze=False)
            if pol_inds is not None: pols = [obj.pols[i] for i in pol_inds]
            obj.setup_data(obj.freqs, df=obj.df, pols=pols, data=data, angs=obj.angs,
                           altaz=obj.altaz, flags=flags, cov=cov, cov_axis=obj.cov_axis, icov=icov,
                           norm=norm, history=obj.history)

        if not inplace:
            return obj

    def write_hdf5(self, fname, overwrite=False):
        """
        Write MapData to hdf5 file.

        Parameters
        ----------
        fname : str
            Output hdf5 filename
        overwrite : bool, optional
            If fname exists, overwrite it
        """
        import h5py
        from bayeslim import utils
        if not os.path.exists(fname) or overwrite:
            with h5py.File(fname, 'w') as f:
                # write data and metadata
                f.create_dataset('data', data=utils.tensor2numpy(self.data))
                if self.norm is not None:
                    f.create_dataset('norm', data=self.norm)
                if self.flags is not None:
                    f.create_dataset('flags', data=self.flags)
                if self.cov is not None:
                    f.create_dataset('cov', data=self.cov)
                if self.cov_axis is not None:
                    f.attr['cov_axis'] = self.cov_axis
                if self.icov is not None:
                    f.create_dataset('icov', data=self.icov)
                if self.angs is not None:
                    f.create_dataset('angs', data=self.angs)
                if self.altaz is not None:
                    f.create_dataset('altaz', data=self.altaz)
                f.create_dataset('freqs', data=self.freqs)
                if self.df is not None:
                    f.create_dataset('df', data=self.df)
                if self.pols is not None:
                    f.attrs['pols'] = self.pols
                f.attrs['history'] = self.history
                f.attrs['obj'] = 'MapData'
                f.attrs['version'] = version.__version__
        else:
            print("{} exists, not overwriting...".format(fname))

    def read_hdf5(self, fname, read_data=True, suppress_nonessential=False, **kwargs):
        """
        Read HDF5 VisData object

        Parameters
        ----------
        fname : str
            File to read
        read_data : bool, optional
            If True, read data arrays as well as metadata
        suppress_nonessential : bool, optional
            If True, suppress reading-in flags, cov and norm, as only data and icov
            are essential for inference.
        kwargs : dict
            Additional select kwargs upon read-in
        """
        import h5py
        from bayeslim import telescope_model
        with h5py.File(fname, 'r') as f:
            # load metadata
            assert str(f.attrs['obj']) == 'MapData', "not a MapData object"
            _freqs = torch.as_tensor(f['freqs'][:])
            _df = torch.as_tensor(f['df'][:]) if 'df' in f else None
            _pols = f.attrs['pols'] if 'pols' in f.attrs else None
            _angs = f['angs'] if 'angs' in f else None
            _altaz = f['altaz'] if 'altaz' in f else None
            cov_axis = f.attrs['cov_axis'] if 'cov_axis' in f.attrs else None
            history = f.attrs['history'] if 'history' in f.attrs else ''

            # setup just full-size metadata
            self.setup_data(_freqs, df=_df, angs=_angs, altaz=_altaz, pols=_pols)

            # read-in data if needed
            data, norm, flags, cov, icov = None, None, None, None, None
            if read_data:
                data = self.get_data(data=f['data'], squeeze=False, **kwargs)
                data = torch.as_tensor(data)
                if 'flags' in f and not suppress_nonessential:
                    flags = self.get_flags(flags=f['flags'], squeeze=False, **kwargs)
                    flags = torch.as_tensor(flags)
                else:
                    flags = None
                if 'cov' in f and not suppress_nonessential:
                    cov = self.get_cov(cov=f['cov'], squeeze=False, **kwargs)
                    cov = torch.as_tensor(cov)
                else:
                    cov = None
                if 'icov' in f:
                    icov = self.get_icov(icov=f['icov'], squeeze=False, **kwargs)
                    icov = torch.as_tensor(icov)
                else:
                    icov = None
                if 'norm' in f and not suppress_nonessential:
                    norm = self.get_data(data=f['norm'], squeeze=False, **kwargs)
                    norm = torch.as_tensor(norm)
                else:
                    norm = None

            # downselect metadata to selection
            self.select(**kwargs)

            # setup downselected metadata and data
            self.setup_meta()
            self.setup_data(self.freqs, df=self.df, angs=self.angs, altaz=self.altaz,
                            pols=self.pols, data=data, flags=flags, cov=cov, norm=norm,
                            cov_axis=cov_axis, icov=icov, history=history)


class CalData(TensorData):
    """
    An object for holding calibration solutions
    of shape (Npol, Npol, Nant, Ntimes, Nfreqs)

    Implicit ordering of Npol dimension is

    .. math::

        [[J_{ee}, J_{en}], [J_{ne}, J_{nn}]]

    unless Npol = 1, in which case it is either
    Jee or Jnn as specified by pol.
    """
    def __init__(self):
        self.atol = 1e-10

    def setup_meta(self, telescope=None, antpos=None):
        """
        Set the telescope and antpos dict

        Parameters
        ----------
        telescope : TelescopeModel, optional
            Telescope location
        antpos : dict, optional
            Antenna position dictionary in ENU [meters].
            Antenna number integer as key, position vector as value
        """
        self.telescope = telescope
        self.antpos = antpos

    def setup_data(self, ants, times, freqs, pol=None,
                   data=None, flags=None, cov=None, cov_axis=None,
                   icov=None, history=''):
        """
        Setup metadata and optionally data tensors.

        Parameters
        ----------
        ants : list
            List of antenna integers matching
            ordering of Nant axis of data.
        times : tensor
            Julian date of unique times in the data.
        freqs : tensor
            Frequency array [Hz] of the data
        pol : str, optional
            If data Npol == 1, this is the name
            of the dipole polarization, one of ['Jee', 'Jnn'].
            If None, it is assumed Npol == 2
        data : tensor, optional
            Complex gain tensor of shape
            (Npol, Npol, Nant, Ntimes, Nfreqs), where
            Npol is either 1 or 2 depending on the polmode.
            If Npol = 1, then pol must be fed.
        flags : tensor, optional
            Boolean tensor holding data flags
        cov : tensor, optional
            Tensor holding data covariance. If this is
            the same shape as data, then the elements are
            the data variance. Otherwise, this is of shape
            (Nax, Nax, ...) where Nax is the axis of the
            modeled covariance. All other axes should broadcast
            with other unmodeled axes.
            E.g. if
                cov_axis = 'ant': (Nants, Nant, Npol, Npol, Nfreqs, Ntimes)
                cov_axis = 'freq': (Nfreqs, Nfreqs, Npol, Npol, Nants, Ntimes)
                cov_axis = 'time': (Ntimes, Ntimes, Npol, Npol, Nants, Nfreqs)
        cov_axis : str, optional
            If cov represents on and off diagonal components, this is the
            axis over which off-diagonal is modeled.
            One of ['ant', 'time', 'freq'].
        icov : tensor, optional
            Inverse covariance. Must have same shape as cov.
            Recommended to call self.set_cov() and then self.compute_icov()
        history : str, optional
            data history string
        """
        if not isinstance(ants, list):
            ants = ants.tolist()
        self.ants = ants
        self.Nants = len(ants)
        self.times = times
        self.Ntimes = len(times)
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.pol = pol
        if isinstance(pol, str):
            assert pol.lower() in ['jee', 'jee'], "pol must be 'ee' or 'nn' for 1pol mode"
        self.Npol = 2 if self.pol is None else 1
        self.data = data
        self.flags = flags
        self.set_cov(cov, cov_axis, icov=icov)
        self.history = history

    def copy(self, detach=True):
        """
        Copy and return self. This is equivalent
        to a detach and clone. Detach is optional
        """
        cd = CalData()
        cd.setup_meta(telescope=self.telescope, antpos=self.antpos)
        data = self.data.detach() if detach else self.data
        cd.setup_data(self.ants, self.times, self.freqs, pol=self.pol,
                      data=data.clone(),
                      flags=self.flags, cov=self.cov, icov=self.icov,
                      cov_axis=self.cov_axis, history=self.history)
        return cd

    def _ant2ind(self, ant):
        """
        Antenna(s) to index

        Parameters
        ----------
        ant : integer or list of ints
            Antenna number(s) to index
        """
        if isinstance(ant, list):
            return [self._ant2ind(a) for a in ant]
        return self.ants.index(ant)

    def _ant2uniq_antpol(self, ant):
        """
        Given a antenna or antenna-pol tuple "ant",
        or list of such, return unique ants and pols
        """
        pol = None
        if isinstance(ant, tuple):
            # this is a single ant or antpol
            if len(ant) == 1:
                ant, pol = [ant], None
            elif len(ant) == 2:
                ant, pol = [ant[0]], ant[1]
        elif isinstance(ant, list):
            # this is a list of ants or antpols
            ant_list, pol_list = [], []
            for a in ant:
                _ant, _pol = self._ant2uniq_antpol(a)
                if _ant not in ant_list:
                    ant_list.extend(_ant_list)
                if _pol not in pol_list:
                    pol_list.append(_pol)
            ant = ant_list
            pol = pol_list
            if len(pol) > 1 and None in pol:
                pol.remove(None)
            assert len(pol) == 1, "can only index 1 pol at a time"
            pol = pol[0]

        return ant, pol

    def _time2ind(self, time):
        """
        Time(s) to index

        Parameters
        ----------
        time : float or list of floats
            Julian date times to index
        """
        iterable = False
        if isinstance(time, (list, np.ndarray)):
            iterable = True
        elif isinstance(time, torch.Tensor):
            if time.ndim == 1:
                iterable = True
        if iterable:
            return np.concatenate([self._time2ind(t) for t in time]).tolist()
        return np.where(np.isclose(self.times, time, atol=self.atol, rtol=1e-10))[0].tolist()

    def _freq2ind(self, freq):
        """
        Freq(s) to index

        Parameters
        ----------
        freq : float or list of floats
            frequencies [Hz] to index
        """
        iterable = False
        if isinstance(freq, (list, np.ndarray)):
            iterable = True
        elif isinstance(freq, torch.Tensor):
            if freq.ndim == 1:
                iterable = True
        if iterable:
            return np.concatenate([self._freq2ind(f) for f in freq]).tolist()
        return np.where(np.isclose(self.freqs, freq, atol=self.atol))[0].tolist()

    def _pol2ind(self, pol):
        """
        Polarization to index

        Parameters
        ----------
        pol : str
            Polarization to index (either 'Jee' or 'Jnn')
        """
        if isinstance(pol, list):
            # can only index 1 pol at a time
            assert len(pol) == 1
            pol = pol[0]
        assert isinstance(pol, str)

        # first check if indexing from 1pol object (just return 1pol)
        if self.pol is not None:
            if pol.lower() != self.pol.lower():
                raise ValueError("cannot index pol from 1pol {}".format(self.pol))
            return slice(0, 1)
        if pol.lower() == 'jee':
            return slice(0, 1)
        elif pol.lower() == 'jnn':
            return slice(1, 2)
        else:
            raise ValueError("cannot index cross-pols")

    def get_inds(self, ant=None, times=None, freqs=None, pol=None):
        """
        Given data selections, return data indexing list

        Parameters
        ----------
        ant : int or list of ints, optional
            Antenna number, or list of such. Can also be
            ant-pol, with limitations.
        times : tensor or float, optional
            Time(s) to index
        freqs : tensor or float, optional
            Frequencies to index
        pol : str, optional
            Polarization to index

        Returns
        -------
        list
            A 5-len list holding slices along data axes.
        """
        if ant is not None:
            # special case for antpols
            ant, _pol = self._ant2uniq_antpol(ant)
            ant_inds = self._ant2ind(ant)
            if pol is not None:
                if _pol is not None:
                    assert _pol == pol
            pol = _pol
        else:
            ant_inds = slice(None)

        if times is not None:
            time_inds = self._time2ind(times)
        else:
            time_inds = slice(None)

        if freqs is not None:
            freq_inds = self._freq2ind(freqs)
        else:
            freq_inds = slice(None)

        if pol is not None:
            pol_ind = self._pol2ind(pol)
        else:
            pol_ind = slice(None)

        inds = [pol_ind, pol_ind, ant_inds, time_inds, freq_inds]
        inds = tuple([utils._list2slice(ind) for ind in inds])
        slice_num = sum([isinstance(ind, slice) for ind in inds])
        assert slice_num > 3, "cannot fancy index more than 1 axis"

        return inds

    def get_data(self, ant=None, times=None, freqs=None, pol=None,
                 squeeze=True, data=None):
        """
        Slice into data tensor and return values.
        Only one axis can be specified at a time.

        Parameters
        ----------
        ant : int or list of int, optional
            Antenna number, or list of such, to return
        times : tensor or float, optional
            Time(s) to index
        freqs : tensor or float, optional
            Frequencies to index
        pol : str, optional
            Polarization to index
        squeeze : bool, optional
            If True, squeeze array before return
        data : tensor, optional
            Tensor to index. default is self.data
        """
        data = self.data if data is None else data
        if data is None:
            return None

        # get indexing
        inds = self.get_inds(ant=ant, times=times, freqs=freqs, pol=pol)
        data = data[inds]

        # squeeze if desired
        if squeeze:
            data = data.squeeze()

        return data

    def get_flags(self, ant=None, times=None, freqs=None, pol=None,
                  squeeze=True, flags=None):
        """
        Slice into flag tensor and return values.
        Only one axis can be specified at a time.

        Parameters
        ----------
        ant : int or list of int, optional
            Antenna number, or list of such, to return
        times : tensor or float, optional
            Time(s) to index
        freqs : tensor or float, optional
            Frequencies to index
        pol : str, optional
            Polarization to index
        squeeze : bool, optional
            If True, squeeze array before return
        flags : tensor, optional
            flag array to index. default is self.flags
        """
        flags = self.flags if flags is None else flags
        if flags is None:
            return None

        # get indexing
        inds = self.get_inds(ant=ant, times=times, freqs=freqs, pol=pol)
        flags = flags[inds]

        # squeeze if desired
        if squeeze:
            flags = flags.squeeze()

        return flags

    def get_cov(self, ant=None, times=None, freqs=None, pol=None,
                squeeze=True, cov=None):
        """
        Slice into cov tensor and return values.
        Only one axis can be specified at a time.
        See optim.apply_icov() for details on shape.

        Parameters
        ----------
        ant : int or list of int, optional
            Antenna number, or list of such, to return
        times : tensor or float, optional
            Time(s) to index
        freqs : tensor or float, optional
            Frequencies to index
        pol : str, optional
            Polarization to index
        squeeze : bool, optional
            If True, squeeze array before return
        cov : tensor, optional
            Cov tensor to index. default is self.cov
        """
        cov = self.cov if cov is None else cov
        if cov is None:
            return None

        # get indexing
        inds = self.get_inds(ant=ant, times=times, freqs=freqs, pol=pol)

        if self.cov_axis is None:
            # cov is same shape as data
            cov = cov[inds]
        else:
            # cov is not the same shape as data
            if ant is not None:
                if self.cov_axis == 'ant':
                    cov = cov[inds[2]][:, inds[2]]
                elif self.cov_axis in ['time', 'freq']:
                    cov = cov[:, :, :, :, inds[2]]
                elif self.cov_axis == 'full':
                    raise NotImplementedError
            elif times is not None:
                if self.cov_axis == 'time':
                    cov = cov[inds[3]][:, inds[3]]
                elif self.cov_axis == 'ant':
                    cov = cov[:, :, :, :, inds[3]]
                elif self.cov_axis == 'freq':
                    cov = cov[:, :, :, :, :, inds[3]]
                elif self.cov_axis == 'full':
                    raise NotImplementedError
            elif freqs is not None:
                if self.cov_axis == 'freq':
                    cov = cov[inds[4]][:, inds[4]]
                elif self.cov_axis in ['ant', 'time']:
                    cov = cov[:, :, :, :, :, inds[4]]
                elif self.cov_axis == 'full':
                    raise NotImplementedError
            elif pol is not None:
                # pol-pol covariance not yet implemented
                cov = cov[:, :, inds[0], inds[0]]
            else:
                cov = cov[:]

        # squeeze
        if squeeze:
            cov = cov.squeeze()

        return cov

    def get_icov(self, ant=None, icov=None, **kwargs):
        """
        Slice into cached inverse covariance.
        Same kwargs as get_cov()
        """
        if icov is None:
            icov = self.icov
        return self.get_cov(ant=ant, cov=icov, **kwargs)

    def __getitem__(self, ant):
        return self.get_data(ant, squeeze=True)

    def __setitem__(self, ant, val):
        self.set(ant, val)

    def set(self, ant, val, arr='data'):
        """
        Set the desired sliced attribute "arr" as val

        Parameters
        ----------
        ant : int, optional
            antenna(pol) to set
        kwargs : dict, optional
            Other parameters to set
        val : tensor
            Value to assign
        arr : str, optional
            Attribute to set
            ['data', 'flags', 'icov', 'cov']
        """
        # get slice indices
        inds = self.get_inds(ant=ant)

        # get desired attribute
        if arr == 'data':
            arr = self.data
        elif arr == 'flags':
            arr = self.flags
        elif arr == 'cov':
            # assert cov is shape of data
            assert self.cov_axis is None
            arr = self.cov
        elif arr == 'icov':
            # assert cov is shape of data
            assert self.cov_axis is None
            arr = self.icov

        arr[inds] = val

    def select(self, ants=None, times=None, freqs=None, pol=None, inplace=True):
        """
        Downselect on data tensor.
        Operates in place.

        Parameters
        ----------
        ants : list, optional
            List of antenna numbers to downselect
        times : tensor, optional
            List of Julian Date times to downselect
        freqs : tensor, optional
            List of frequencies [Hz] to downselect
        pol : str, optional
            Polarization to downselect
        inplace : bool, optional
            If True downselect inplace, otherwise return a new object.
        """
        if inplace:
            obj = self
        else:
            obj = copy.deepcopy(self)

        if ants is not None:
            data = obj.get_data(ants, squeeze=False)
            cov = obj.get_cov(ants, squeeze=False)
            icov = obj.get_icov(ants, squeeze=False)
            flags = obj.get_flags(ants, squeeze=False)
            obj.setup_data(ants, obj.times, obj.freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if times is not None:
            data = obj.get_data(times=times, squeeze=False)
            cov = obj.get_cov(times=times, squeeze=False)
            icov = obj.get_icov(times=times, squeeze=False)
            flags = obj.get_flags(times=times, squeeze=False)
            obj.setup_data(obj.ants, times, obj.freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if freqs is not None:
            data = obj.get_data(freqs=freqs, squeeze=False)
            cov = obj.get_cov(freqs=freqs, squeeze=False)
            icov = obj.get_icov(freqs=freqs, squeeze=False)
            flags = obj.get_flags(freqs=freqs, squeeze=False)
            obj.setup_data(obj.ants, obj.times, freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if pol is not None:
            data = obj.get_data(pol=pol)
            flags = obj.get_flags(pol=pol)
            cov = obj.get_cov(pol=pol)
            icov = obj.get_icov(pol=pol)
            obj.setup_data(obj.ants, obj.times, obj.freqs, pol=pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if not inplace:
            return obj

    def rephase_to_refant(self, refant):
        """
        Rephase the complex gains to a reference antenna phase

        Parameters
        ----------
        refant : int
        """
        from bayeslim.calibration import rephase_to_refant
        idx = self.ants.index(refant)
        rephase_to_refant(self.data, 'com', idx, mode='rephase', inplace=True)

    def redcal_degens(self, wgts=None):
        """
        Compute redcal degenerate parameters

        Parameters
        ----------
        wgts : tensor
            1D weights of length Nants to use in
            computing degeracies. Should be number of visibilities
            per antenna used in redcal. Default is uniform weight.

        Returns
        -------
        abs_amp : tensor
            Shape (Npol, Npol, 1, Ntimes, Nfreqs)
        phs_slope : tensor
            Shape (Npol, Npol, 2, Ntimes, Nfreqs) where
            two elements are [East, North] phs gradient [rad / meter]
        """
        from bayeslim.calibration import compute_redcal_degen
        return compute_redcal_degen(self.data, self.ants, self.antpos, wgts=wgts)

    def redcal_degen_gains(self, wgts=None):
        """
        Compute redcal degenerate gains

        Parameters
        ----------
        wgts : tensor
            1D weights of length Nants to use in
            computing degeracies. Should be number of visibilities
            per antenna used in redcal. Default is uniform weight.

        Returns
        -------
        CalData
        """
        from bayeslim.calibration import redcal_degen_gains
        out = self.copy()
        rd = out.redcal_degens(wgts=wgts)
        out.data = redcal_degen_gains(out.ants, antpos=out.antpos,
                                      abs_amp=rd[0], phs_slope=rd[1])
        return out

    def remove_redcal_degen(self, redvis=None, degen=None, wgts=None):
        """
        Remove redcal degeneracies from gains and model visibility.
        Updates gains and model visibility inplace.

        Parameters
        ----------
        redvis : VisData object, optional
            Holds redcal model visibilities
        degen : tensor or CalData object
            New redcal degeneracies to insert into gains
        wgts : tensor, optional
            1D weights of length Nants to use in computing degeneracies
        """
        from bayeslim.calibration import remove_redcal_degen
        rvis = None if redvis is None else redvis.data
        bls = None if redvis is None else redvis.bls
        if isinstance(degen, CalData):
            degen = degen.data
        degen = None if degen is None else degen
        new_gain, new_vis, _ = remove_redcal_degen(self.data, self.ants,
                                                   self.antpos, degen=degen,
                                                   wgts=wgts,
                                                   redvis=rvis, bls=bls)
        self.data = new_gain
        if redvis is not None:
            redvis.data = new_vis

    def write_hdf5(self, fname, overwrite=False):
        """
        Write CalData to hdf5 file.

        Parameters
        ----------
        fname : str
            Output hdf5 filename
        overwrite : bool, optional
            If fname exists, overwrite it
        """
        import h5py
        from bayeslim import utils
        if not os.path.exists(fname) or overwrite:
            with h5py.File(fname, 'w') as f:
                # write data and metadata
                f.create_dataset('data', data=utils.tensor2numpy(self.data))
                if self.flags is not None:
                    f.create_dataset('flags', data=self.flags)
                if self.cov is not None:
                    f.create_dataset('cov', data=self.cov)
                if self.cov_axis is not None:
                    f.attr['cov_axis'] = self.cov_axis
                if self.icov is not None:
                    f.create_dataset('icov', data=self.icov)
                f.create_dataset('ants', data=self.ants)
                if self.antpos is not None:
                    antvec = np.array([self.antpos[a] for a in self.ants])
                    f.create_dataset('antvec', data=antvec)
                f.create_dataset('times', data=self.times)
                f.create_dataset('freqs', data=self.freqs)
                if self.pol is not None:
                    f.attrs['pol'] = self.pol
                f.attrs['history'] = self.history
                # write telescope and array objects
                if self.telescope is not None:
                    f.attrs['tloc'] = self.telescope.location
                f.attrs['obj'] = 'CalData'
                f.attrs['version'] = version.__version__
        else:
            print("{} exists, not overwriting...".format(fname))

    def read_hdf5(self, fname, read_data=True, ants=None, times=None, freqs=None, pol=None):
        """
        Read HDF5 CalData object

        Parameters
        ----------
        fname : str
            File to read
        read_data : bool, optional
            If True, read data arrays as well as metadata
        ants, times, freqs, pol : read options. see self.select() for details

        Returns
        -------
        CalData object
        """
        import h5py
        from bayeslim import telescope_model
        with h5py.File(fname, 'r') as f:
            # load metadata
            assert str(f.attrs['obj']) == 'CalData', "not a CalData object"
            _ants = [int(_ant) for _ant in f['ants'][:]]
            if 'antvec' in f:
                _antvec = np.asarray(f['antvec'][:])
                _antpos = dict(zip(_ants, _antvec))
            else:
                _antpos = None
            _times = f['times'][:]
            _freqs = torch.as_tensor(f['freqs'][:])
            _pol = f.attrs['pol'] if 'pol' in f.attrs else None
            cov_axis = f.attrs['cov_axis'] if 'cov_axis' in f.attrs else None
            history = f.attrs['history'] if 'history' in f.attrs else ''

            # setup just full-size data
            self.setup_data(_ants, _times, _freqs, pol=_pol,
                            cov_axis=cov_axis)

            data, flags, cov, icov = None, None, None, None
            if read_data:
                data = self.get_data(ant=ants, times=times, freqs=freqs, pol=pol,
                                     squeeze=False, data=f['data'])
                data = torch.as_tensor(data)
                if 'flags' in f:
                    flags = self.get_flags(ant=ants, times=times, freqs=freqs, pol=pol,
                                          squeeze=False, flags=f['flags'])
                    flags = torch.as_tensor(flags)
                else:
                    flags = None
                if 'cov' in f:
                    cov = self.get_cov(ant=ants, times=times, freqs=freqs, pol=pol,
                                       squeeze=False, cov=f['cov'])
                    cov = torch.as_tensor(cov)
                else:
                    cov = None
                if 'icov' in f:
                    icov = self.get_icov(ant=ants, times=times, freqs=freqs, pol=pol,
                                         squeeze=False, icov=f['icov'])
                    icov = torch.as_tensor(icov)
                else:
                    icov = None

            # downselect metadata to selection: note this doesn't touch data arrays
            self.select(ants=ants, times=times, freqs=freqs, pol=pol)

            # setup downselected metadata and data
            if 'tloc' in f.attrs:
                tloc = f.attrs['tloc']
                telescope = telescope_model.TelescopeModel(tloc)
            else:
                telescope = None
            self.setup_meta(telescope, _antpos)
            self.setup_data(self.ants, self.times, self.freqs, pol=self.pol,
                            data=data, flags=flags, cov=cov,
                            cov_axis=cov_axis, icov=icov, history=history)

    def read_uvcal(self, fname, run_check=True, **kwargs):
        """
        Read a pyuvdata UVCal file transfer to
        a CalData object. Needs pyuvdata support.

        Parameters
        ----------
        fname : str or UVCal
            Filename(s) to read, or UVCal object
        run_check : bool, optional
            Run check after read
        kwargs : dict
            kwargs for UVCal read
        """
        raise NotImplementedError

    def check(self):
        """
        Run basic checks on data
        """
        from bayeslim import telescope_model
        if self.telescope:
            assert isinstance(self.telescope, telescope_model.TelescopeModel)
        if self.antpos:
            assert isinstance(self.antpos, dict)
        if self.data:
            assert isinstance(self.data, torch.Tensor)
            assert self.data.shape == (self.Npol, self.Npol, self.Nants, self.Ntimes, self.Nfreqs)
        if self.flags:
            assert isinstance(self.flags, torch.Tensor)
            assert self.data.shape == self.flags.shape
        if self.cov:
            assert self.cov_axis is not None, "full data-sized covariance not implemented"
            if self.cov_axis == 'ant':
                assert self.cov.shape == (self.Nants, self.Nants, self.Npol, self.Npol,
                                          self.Ntimes, self.Nfreqs)
            elif self.cov_axis == 'time':
                assert self.cov.shape == (self.Ntimes, self.Ntimes, self.Npol, self.Npol,
                                          self.Nants, self.Nfreqs)
            elif self.cov_axis == 'freq':
                assert self.cov.shape == (self.Nfreqs, self.Nfreqs, self.Npol, self.Npol,
                                          self.Nants, self.Ntimes)

    def inflate_to_4pol(self):
        """
        If current Npol = 1, inflate the object to Npol = 2.
        This operation is inplace.
        """
        raise NotImplementedError


class Dataset(Dataset):
    """
    Dataset iterator for VisData, MapData, CalData, or TensorData
    """
    def __init__(self, data, read_fn=None, read_kwargs={}):
        """VisData or Mapdata Dataset object

        Parameters
        ----------
        data : list of str or VisData / MapData
            List of data objects to read and iterate over.
            Niter of data should match Niter of the
            posterior model. data passed as str will
            only be read when iterated on.
        read_fn : callable, optional
            Read function when iterating over data.
            If data is passed as pre-loaded VisData(s), use
            pass_data method for read_fn (default).
        read_kwargs : dict or list of dict
            If data is kept as a str, these are the
            kwargs passed to the read method. This can
            be a list of kwarg dicts for each file
            of data.
        """
        if isinstance(data, (str, VisData, MapData, CalData, TensorData)):
            data = [data]
        self.data = data
        self.Ndata = len(self.data)
        self.read_fn = read_fn if read_fn is not None else pass_data
        if isinstance(read_kwargs, dict):
            read_kwargs = [read_kwargs for d in self.data]
        self.read_kwargs = read_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.read_fn(self.data[idx], **self.read_kwargs[idx])


def concat_VisData(vds, axis, run_check=True):
    """
    Concatenate VisData objects together

    Parameters
    ----------
    vds : list of VisData
    axis : str, ['bl', 'time', 'freq']
        Axis to concatenate over. All other
        axes must match exactly in all vds.
    """
    if isinstance(vds, VisData):
        return vds
    Nvd = len(vds)
    assert Nvd > 0
    if Nvd == 1:
        return vds[0]
    vd = vds[0]
    out = VisData()
    flags, cov = None, None
    if axis == 'bl':
        dim = 2
        times = vd.times
        freqs = vd.freqs
        pol = vd.pol
        bls = []
        for o in vds:
            bls.extend(o.bls)
        if vd.cov is not None:
            raise NotImplementedError

    elif axis == 'time':
        dim = 3
        times = torch.cat([torch.as_tensor(o.times) for o in vds])
        freqs = vd.freqs
        pol = vd.pol
        bls = vd.bls
        if vd.cov is not None:
            raise NotImplementedError

    elif axis == 'freq':
        dim = 4
        times = vd.times
        freqs = torch.cat([torch.as_tensor(o.freqs) for o in vds])
        pol = vd.pol
        bls = vd.bls
        if vd.cov is not None:
            raise NotImplementedError

    data = torch.cat([o.data for o in vds], dim=dim)
    if vd.flags is not None:
        flags = torch.cat([o.flags for o in vds], dim=dim)

    out.setup_meta(vd.telescope, vd.antpos)
    out.setup_data(bls, times, freqs, pol=pol,
                   data=data, flags=flags, cov=cov,
                   cov_axis=vd.cov_axis, history=vd.history)

    if run_check:
        out.check()

    return out


def concat_MapData(mds, axis, run_check=True):
    """
    Concatenate MapData objects
    """
    raise NotImplementedError


def concat_CalData(cds, axis, run_check=True):
    """
    Concatenate CalData bojects
    """
    raise NotImplementedError


def load_data(fname, concat_ax=None, copy=False, **kwargs):
    """
    Load a VisData, MapData, or CalData HDF5 file(s)

    Parameters
    ----------
    fname : str or list of str
        Filepath to object. If list of str, concatenate
        data along concat_ax
    concat_ax : str, optional
        Concatenation axis if fname is a list. If None leave
        as a list.
    copy : bool, optional
        If True copy data before returning
    kwargs : dict
        Read kwargs. See VisData.select()
    
    Returns
    -------
    VisData, MapData, or CalData object
    """
    if isinstance(fname, (list, tuple)):
        dlist = [load_data(fn, **kwargs) for fn in fname]
        if concat_ax is not None:
            if isinstance(dlist[0], VisData):
                data = concat_VisData(dlist, concat_ax)
            elif isinstance(dlist[0], MapData):
                data = concat_MapData(dlist, concat_ax)
            elif isinstance(dlist[0], CalData):
                data = concat_CalData(dlist, concat_ax)
        else:
            data = dlist

    elif isinstance(fname, (VisData, MapData, CalData)):
        data = fname

    else:
        with h5py.File(fname) as f:
            obj = f.attrs['obj']

        if obj == 'VisData':
            data = VisData()
            data.read_hdf5(fname, **kwargs)
        elif obj == 'MapData':
            data = MapData()
            data.read_hdf5(fname, **kwargs)
        elif obj == 'CalData':
            data = CalData()
            data.read_hdf5(fname, **kwargs)

    if copy:
        data = copy.deepcopy(data)

    return data


def caldata_from_visdata(vd):
    """
    Initialize a CalData object using
    metadata from a VisData object

    Parameters
    ----------
    vd : VisData object

    Returns
    -------
    CalData object
    """
    raise NotImplementedError


def pass_data(fname, copy=False, **kwargs):
    """Dummy load function. Use this when storing data in memory,
    rather than performing dynamic IO"""
    if copy:
        return deepcopy(fname)
    else:
        return fname
