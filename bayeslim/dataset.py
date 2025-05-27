"""
Module for visibility and map data formats, and a torch style data loader
"""
import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import os
import copy
import h5py
from astropy.units import sday

from . import version, utils


class TensorData:
    """
    A shallow object for holding an arbitrary tensor data
    """
    def __init__(self):
        # init empty object
        self.device = None
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

    def push(self, device, return_obj=False):
        """
        Push data, flags, cov and icov to device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype: self.device = device
        if self.data is not None:
            self.data = utils.push(self.data, device)
        if self.flags is not None:
            if not dtype:
                self.flags = self.flags.to(device)
        if self.cov is not None:
            self.cov = utils.push(self.cov, device)
        if self.icov is not None:
            self.icov = utils.push(self.icov, device)
        if return_obj:
            return self

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
        if isinstance(cov, torch.Tensor):
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

        elif isinstance(icov, torch.Tensor):
            # only icov provided, try to get cov_logdet
            if cov_axis is None:
                cov_logdet = torch.sum(-torch.log(icov))

        # set covariance
        if isinstance(cov, torch.Tensor): cov = cov.clone()
        if isinstance(icov, torch.Tensor): icov = icov.clone()
        self.cov = cov
        self.icov = icov
        self.cov_axis = cov_axis
        self.cov_ndim = sum(self.data.shape) if self.data is not None else None
        self.cov_logdet = cov_logdet if cov_logdet is not None else torch.tensor(0.0)

    def compute_icov(self, inv='pinv', **kwargs):
        """
        Compute and set inverse covariance as self.icov.
        See optim.compute_cov and apply_icov() for output
        shapes, and linalg.invert_matrix() for expected kwargs.

        Parameters
        ----------
        inv : str, optional
            Inversion method. See linalg.invert_matrix()
        kwargs : dict, optional
            Keyword args for invert_matrix()
        """
        from bayeslim import optim
        self.icov = optim.compute_icov(self.cov, self.cov_axis, inv=inv, **kwargs)

    def copy(self, copydata=False, copymeta=False, detach=True):
        """
        Copy and return self.

        Parameters
        ----------
        copydata : bool, optional
            If True make a clone of the data.
            Default is False.
        copymeta : bool, optional
            If True make a new instantiation of metadata like
            times, freqs, flags, etc.
        detach : bool, optional
            If True (default) detach self.data for new object
            if copydata == True.
        """
        flags, cov, icov = self.flags, self.cov, self.icov
        history = self.history

        # clone data
        data = self.data
        if copydata:
            if isinstance(data, torch.Tensor):
                if data.requires_grad and detach:
                    data = data.detach()
                data = data.clone()

        if copymeta:
            if isinstance(flags, torch.Tensor): flags = flags.clone()
            if isinstance(cov, torch.Tensor): cov = cov.clone()
            if isinstance(icov, torch.Tensor): icov = icov.clone()

        td = TensorData()
        td.setup_data(data=data, flags=flags, cov=cov,
                      cov_axis=self.cov_axis, icov=icov, history=history)

        return td

    def get_data(self, **kwargs):
        """
        Return data
        """
        return self.data

    def get_flags(self, **kwargs):
        """
        Return flags
        """
        return self.flags

    def get_cov(self, **kwargs):
        """
        Return cov
        """
        return self.cov

    def get_icov(self, **kwargs):
        """
        Return icov
        """
        return self.icov

    def __add__(self, other):
        out = self.copy(copydata=False, copymeta=False)
        if isinstance(other, (float, int, complex, torch.Tensor)):
            out.data = out.data + other
        else:
            out.data = out.data + other.data
            self._propflags(out, other)
        return out

    def __iadd__(self, other):
        if isinstance(other, (float, int, complex, torch.Tensor)):
            self.data += other
        else:
            self.data += other.data
            self._propflags(self, other)
        return self

    def __sub__(self, other):
        out = self.copy(copydata=False, copymeta=False)
        if isinstance(other, (float, int, complex, torch.Tensor)):
            out.data = out.data - other
        else:
            out.data = out.data - other.data
            self._propflags(out, other)
        return out

    def __isub__(self, other):
        if isinstance(other, (float, int, complex, torch.Tensor)):
            self.data -= other
        else:
            self.data -= other.data
            self._propflags(self, other)
        return self

    def __mul__(self, other):
        out = self.copy(copydata=False, copymeta=False)
        if isinstance(other, (float, int, complex, torch.Tensor)):
            out.data = out.data * other
        else:
            out.data = out.data * other.data
            self._propflags(out, other)
        return out

    def __imul__(self, other):
        if isinstance(other, (float, int, complex, torch.Tensor)):
            self.data *= other
        else:
            self.data *= other.data
            self._propflags(self, other)
        return self

    def __truediv__(self, other):
        out = self.copy(copydata=False, copymeta=False)
        if isinstance(other, (float, int, complex, torch.Tensor)):
            out.data = out.data / other
        else:
            out.data = out.data / other.data
            self._propflags(out, other)
        return out

    def __itruediv__(self, other):
        if isinstance(other, (float, int, complex, torch.Tensor)):
            self.data /= other
        else:
            self.data /= other.data
            self._propflags(self, other)
        return self

    @staticmethod
    def _propflags(td1, td2):
        """Propagate flags from td2 into td1"""
        if td2.flags is not None:
            if td1.flags is None:
                td1.flags = td2.flags.copy()
            else:
                td1.flags += td2.flags


class VisData(TensorData):
    """
    An object for holding visibility data of shape
    (Npol, Npol, Nbl, Ntimes, Nfreqs)
    """
    def __init__(self):
        # init empty object
        self.data = None
        self.device = None
        self.atol = 1e-10
        self._file = None
        self.setup_meta()

    def push(self, device, return_obj=False):
        """
        Push data to a new device
        """
        dtype = isinstance(device, torch.dtype)
        # TensorData.push
        super().push(device)
        # and push antpos if needed
        if self.antpos:
            self.antpos.push(device)
        # push telescope if needed
        if self.telescope:
            self.telescope.push(device)
        self.freqs = utils.push(self.freqs, device)
        if not dtype:
            self._blnums = self._blnums.to(device)
            if hasattr(self._blnums, '_arr_hash'):
                del self._blnums._arr_hash
        if return_obj:
            return self

    def setup_meta(self, telescope=None, antpos=None):
        """
        Set the telescope and antpos dict.

        Parameters
        ----------
        telescope : TelescopeModel, optional
            Telescope location
        antpos : AntposDict, optional
            Antenna position dictionary in ENU [meters].
            Antenna number integer as key, position vector as value
        """
        self.telescope = telescope
        if antpos is not None:
            if isinstance(antpos, utils.AntposDict):
                pass
            else:
                antpos = utils.AntposDict(list(antpos.keys()),
                    list(antpos.values()))
        self.antpos = antpos
        self.ants = None
        if antpos is not None:
            self.ants = antpos.ants

    def setup_data(self, bls, times, freqs, pol=None,
                   data=None, flags=None, cov=None, cov_axis=None,
                   icov=None, history='', file=None):
        """
        Setup metadata and optionally data tensors.

        Parameters
        ----------
        bls : list or array
            List of baseline tuples (antpairs) matching
            ordering of Nbls axis of data, or a blnums
            int-array of shape (2, Nbls).
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
        file : HDF5 file handle, optional
            For lazy loading, this is an opened h5py file handle.
        """
        # set the data
        self.data = data
        # deal with baselines
        self._set_bls(bls)
        self.times = torch.as_tensor(times)
        self.Ntimes = len(times)
        self.freqs = torch.as_tensor(freqs)
        self.Nfreqs = len(freqs)
        self.pol = pol
        if isinstance(pol, str):
            assert pol.lower() in ['ee', 'nn'], "pol must be 'ee' or 'nn' for 1pol mode"
        self.Npol = 2 if self.pol is None else 1
        self.flags = flags
        self.set_cov(cov, cov_axis, icov=icov)
        self.history = history
        self._file = file

        if isinstance(data, torch.Tensor):
            if not utils.check_devices(data.device, self.device):
                self.push(data.device)

    def _set_bls(self, bls):
        """
        Set the blnums tensor for all
        baselines in the data.

        Note:
        self.blnums is numpy on cpu
        self._blnums is pytorch on self.device.

        Parameters
        ----------
        bls : list or array
            List of baseline tuples (antpairs) matching
            ordering of Nbls axis of data, or a blnums
            int-array of shape (2, Nbls).
        """
        if isinstance(bls, torch.Tensor):
            # bls is an blnums tensor
            self._blnums = bls.clone()
            if hasattr(self._blnums, '_arr_hash'):
                delattr(self._blnums, '_arr_hash')
        elif isinstance(bls, np.ndarray):
            # blnums in numpy
            self._blnums = torch.as_tensor(bls.copy())
        else:
            # bls is a tuple or list of tuples
            self._blnums = torch.as_tensor(utils.ants2blnum(bls))

        # check device
        if self.data is not None:
            if not utils.check_devices(self._blnums.device, self.device):
                self._blnums = self._blnums.to(self.device)

        self.blnums = self._blnums.cpu().numpy()
        self.Nbls = len(self.blnums)

    @property
    def bls(self):
        return utils.blnum2ants(self.blnums)

    def get_bls(self, uniq_bls=False, keep_autos=True,
                min_len=None, max_len=None,
                min_EW=None, max_EW=None, min_NS=None, max_NS=None,
                min_deg=None, max_deg=None, redtol=1.0):
        """
        Select a subset of baselines from the object.
        Note if uniq_bls = True this will need to call build_reds()
        which is slow.
        Note this ignores the z-component of the antennas.

        Parameters
        ----------
        uniq_bls : bool, optional
            If True, return only the first baseline
            in each redundant group. Otherwise return
            all physical baselines (default)
        keep_autos : bool, optional
            If True (default) return auto-correlations.
            Otherwise remove them.
        min_len : float, optional
            Sets minimum baseline length [m]
        max_len : float, Optional
            Sets maximum baseline length [m]
        min_EW : float, optional
            Sets min |East-West length| [m]
        max_EW : float, optional
            Sets max |East-West length| [m]
        min_NS : float, optional
            Sets min |North-South length| [m]
        max_NS : float, optional
            Sets max |North-South length| [m]
        min_deg : float, optional
            Sets min baseline angle (north of east) [deg]
        max_deg : float, optional
            Sets max baseline angle (north of east) [deg]
        redtol : float, optional
            Redundancy tolerance (meters) if uniq_bls = True 

        Returns
        -------
        list
            List of baseline antenna-pair tuples that fit selection
        """
        if uniq_bls:
            # Just call the ArrayModel method
            from bayeslim.telescope_model import ArrayModel
            array = ArrayModel(self.antpos, self.freqs, redtol=redtol, bls=self.bls)
            return array.get_bls(uniq_bls=uniq_bls, keep_autos=keep_autos, min_len=min_len,
                                 max_len=max_len, min_EW=min_EW, max_EW=max_EW, min_NS=min_NS,
                                 max_NS=max_NS, min_deg=min_deg, max_deg=max_deg)

        else:
            # generate bl_vec on the fly
            bls = self.bls
            bl_vecs = self.get_bl_vecs(bls).cpu().numpy()
            bl_lens = np.linalg.norm(bl_vecs, axis=1)
            bl_angs = np.arctan2(*bl_vecs[:, :2][:, ::-1].T) * 180 / np.pi
            bl_angs[bl_vecs[:, 1] < 0] += 180.0
            bl_angs[abs(bl_vecs[:, 1]) < redtol] = 0.0
            keep = np.ones(len(bl_vecs), dtype=bool)
            if not keep_autos:
                keep = keep & (bl_lens > redtol)
            if min_len is not None:
                keep = keep & (bl_lens >= min_len)
            if max_len is not None:
                keep = keep & (bl_lens <= max_len)
            if min_EW is not None:
                keep = keep & (abs(bl_vecs[0]) >= min_EW)
            if max_EW is not None:
                keep = keep & (abs(bl_vecs[0]) <= max_EW)
            if min_NS is not None:
                keep = keep & (abs(bl_vecs[1]) >= min_NS)
            if max_NS is not None:
                keep = keep & (abs(bl_vecs[1]) <= max_NS)
            if min_deg is not None:
                keep = keep & (bl_angs >= min_deg)
            if max_deg is not None:
                keep = keep & (bl_angs <= max_deg)

            return [bl for i, bl in enumerate(bls) if keep[i]]

    def get_bl_vecs(self, bls):
        """
        Return a tensor holding baseline
        vectors in ENU frame

        Parameters
        ----------
        bls : list or array
            List of baseline tuples e.g. [(0, 1), (1, 2), ...]
            to compute baseline vectors, or a blnums array.

        Returns
        -------
        tensor
        """
        if isinstance(bls, (np.ndarray, torch.Tensor)):
            ant1, ant2 = utils.blnum2ants(bls, separate=True)
        else:
            ant1, ant2 = zip(*bls)

        return self.antpos[ant2] - self.antpos[ant1]

    def copy(self, copydata=False, copymeta=False, detach=True):
        """
        Copy and return self.

        Parameters
        ----------
        copydata : bool, optional
            If True make a clone of the data.
            Default is False.
        copymeta : bool, optional
            If True make a new instantiation of metadata like
            telescope, antpos, times, freqs, flags, etc.
            Note that this drops things like telescope cache.
        detach : bool, optional
            If True (default) detach self.data for new object
            if copydata == True.
        """
        vd = VisData()
        telescope, antpos = self.telescope, self.antpos
        times, freqs, blnums = self.times, self.freqs, self.blnums
        flags, cov, icov = self.flags, self.cov, self.icov
        history = self.history

        # clone data
        data = self.data
        if copydata:
            if isinstance(data, torch.Tensor):
                if data.requires_grad and detach:
                    data = data.detach()
                data = data.clone()

        if copymeta:
            if telescope is not None:
                telescope = telescope.__class__(telescope.location, tloc=telescope.tloc,
                                                device=telescope.device)
            if antpos is not None:
                antpos = antpos.__class__(copy.deepcopy(antpos.ants), antpos.antvecs.clone())
            times = copy.deepcopy(times)
            freqs = copy.deepcopy(freqs)
            blnums = copy.deepcopy(blnums)
            if isinstance(flags, torch.Tensor): flags = flags.clone()
            if isinstance(cov, torch.Tensor): cov = cov.clone()
            if isinstance(icov, torch.Tensor): icov = icov.clone()

        vd.setup_meta(telescope=telescope, antpos=antpos)
        vd.setup_data(blnums, times, freqs, pol=self.pol,
                      data=data, flags=flags, cov=cov, icov=icov,
                      cov_axis=self.cov_axis, history=history)
        return vd

    def _bl2ind(self, bl):
        """
        Baseline(s) to index

        Parameters
        ----------
        bl : tuple, list of tuple, or array
            Antenna pair (1, 2) tuple, or lists of such,
            or blnums array.
        """
        if isinstance(bl, (list, np.ndarray, torch.Tensor)):
            if isinstance(bl, torch.Tensor):
                bl = bl.cpu().numpy()
            elif isinstance(bl, list):
                bl = utils.ants2blnum(bl)

            return [self._bl2ind(_bl) for _bl in bl]

        idx = np.where(self.blnums == bl)[0]
        if len(idx) == 0:
            raise ValueError("Couldn't find bl {}".format(bl))

        return idx[0]

    def _bl2blpol(self, bl):
        """
        Given an antpair of antpair pol tuple "bl",
        or a list of such, return all separated bls
        and pols
        """
        if isinstance(bl, tuple):
            # this is a single antpair or antpairpol
            return self._bl2uniq_blpol(bl)

        elif isinstance(bl, np.ndarray):
            # this is blnums array
            return bl, None

        elif isinstance(bl, torch.Tensor):
            # this is blnums tensor
            return bl.cpu().numpy(), None

        else:
            # list of tuples
            bls, pols = [], []
            for _bl in bl:
                b, p = self._bl2uniq_blpol(_bl)
                bls.extend(b)
                if p is not None:
                    pols.append(p)
            if len(pols) == 0:
                pols = None
            else:
                assert len(pols) == 1, "can only index 1 pol at a time"
                pols = pols[0]
            return bls, pols

    def _bl2uniq_blpol(self, bl):
        """
        Given an antpair or antpair pol tuple "bl",
        or list of such, return unique bls and pol
        """
        assert isinstance(bl, (tuple, list))
        if isinstance(bl, tuple):
            # this is an antpair or antpairpol
            if len(bl) == 2:
                bl, pol = [bl], None
            elif len(bl) == 3:
                bl, pol = [bl[:2]], bl[2]
        else:
            # bl is a list of antpairs or antpairpols
            hbl = [hash(_bl) for _bl in bl]
            uniq_h, uniq_idx = np.unique(hbl, return_index=True)
            ubls, upls = [], []  # uniq bls and uniq pols
            for i in sorted(uniq_idx):
                b, p = self._bl2uniq_blpol(bl[i])
                ubls.extend(b)
                if p is not None:
                    upls.append(p)
            bl = ubls
            pol = upls
            if len(pol) == 0:
                pol = None
            else:
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
        # rtol=1e-13 allows for 0.03 sec discrimination on JD values
        return np.where(np.isclose(self.times, time, atol=self.atol, rtol=1e-13))[0].tolist()

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

    def _pol2ind(self, pol, data=None):
        """
        Polarization to index.
        Does not support multi-pol indexing.

        Parameters
        ----------
        pol : str
            Dipole polarization to index ('ee' or 'nn')
        """
        ## TODO: support other polarizations (e.g. circular, etc)
        if isinstance(pol, list):
            # make sure we are only asking for 1 pol
            assert len(pol) == 1
            pol = pol[0]
        assert isinstance(pol, str)
        if self.pol is not None:
            if pol.lower() != self.pol.lower():
                raise ValueError("cannot index pol from 1pol {}".format(self.pol))
            return (slice(0, 1), slice(0, 1))
        if pol.lower() == 'ee':
            return (slice(0, 1), slice(0, 1))
        elif pol.lower() == 'nn':
            # check for special 2pol chase
            data = data if data is not None else self.data
            if data.shape[:2] == (2, 1):
                return (slice(1, 2), slice(0, 1))
            else:
                return (slice(1, 2), slice(1, 2))
        else:
            raise ValueError("cannot index cross-pols")

    def get_inds(self, bl=None, times=None, freqs=None, pol=None,
                 bl_inds=None, time_inds=None, freq_inds=None, data=None):
        """
        Given data selections, return data indexing list

        Parameters
        ----------
        bl : tuple, list, ndarray, optional
            Baseline number array, or list of antpair tuples.
            Can be antpair-pols, with only 1 unique pol.
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
            in self.blnums
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
        data : tensor, optional
            Use this data instead of self.data.
            Default is self.data. Only use this
            when reading from an HDF5 and passing
            the dataset handle.

        Returns
        -------
        list
            A 5-len list holding slices along axes.
        """
        data = data if data is not None else self.data
        if bl is not None:
            assert bl_inds is None
            # special case for antpairpols
            bl, _pol = self._bl2blpol(bl)
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
            pol_inds = self._pol2ind(pol, data=data)
        else:
            pol_inds = (slice(None), slice(None))

        inds = [pol_inds[0], pol_inds[1], bl_inds, time_inds, freq_inds]
        inds = tuple([utils._list2slice(ind) for ind in inds])
        slice_num = sum([isinstance(ind, slice) for ind in inds])
        assert slice_num > 3, "cannot fancy index more than 1 axis"

        return inds

    def get_data(self, bl=None, times=None, freqs=None, pol=None,
                 bl_inds=None, time_inds=None, freq_inds=None,
                 squeeze=True, data=None, try_view=False, **kwargs):
        """
        Slice into data tensor and return values.
        Only one axis can be specified at a time.

        Parameters
        ----------
        bl : tuple, list, ndarray, optional
            Baseline number array, or list of antpair tuples.
            Can be antpair-pols, with only 1 unique pol.
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
        try_view : bool, optional
            If True, if selections can be cast as slices
            then return a view of original data.
        """
        data = self.data if data is None else data
        if data is None:
            return None

        # get indexing
        inds = self.get_inds(bl=bl, times=times, freqs=freqs, pol=pol,
                             bl_inds=bl_inds, time_inds=time_inds,
                             freq_inds=freq_inds, data=data)
        data = data[inds]
        if not try_view and all([isinstance(ind, slice) for ind in inds]):
            data = data.clone()

        # squeeze if desired
        if squeeze:
            data = data.squeeze()

        return data

    def get_flags(self, bl=None, times=None, freqs=None, pol=None,
                  bl_inds=None, time_inds=None, freq_inds=None,
                  squeeze=True, flags=None, try_view=False, **kwargs):
        """
        Slice into flag tensor and return values.
        Only one axis can be specified at a time.

        Parameters
        ----------
        bl : tuple, list, ndarray, optional
            Baseline number array, or list of antpair tuples.
            Can be antpair-pols, with only 1 unique pol.
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
        try_view : bool, optional
            If True, if selections can be cast as slices
            then return a view of original data.
        """
        flags = self.flags if flags is None else flags
        if flags is None:
            return None

        # get indexing
        inds = self.get_inds(bl=bl, times=times, freqs=freqs, pol=pol,
                             bl_inds=bl_inds, time_inds=time_inds,
                             freq_inds=freq_inds, data=flags)
        flags = flags[inds]
        if not try_view and all([isinstance(ind, slice) for ind in inds]):
            flags = flags.clone()

        # squeeze if desired
        if squeeze:
            flags = flags.squeeze()

        return flags

    def get_cov(self, bl=None, times=None, freqs=None, pol=None,
                bl_inds=None, time_inds=None, freq_inds=None,
                squeeze=True, cov=None, try_view=False, **kwargs):
        """
        Slice into cov tensor and return values.
        Only one axis can be specified at a time.
        See optim.apply_icov() for details on shape.

        Parameters
        ----------
        bl : tuple, list, ndarray, optional
            Baseline number array, or list of antpair tuples.
            Can be antpair-pols, with only 1 unique pol.
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
        try_view : bool, optional
            If True, if selections can be cast as slices
            then return a view of original data.
        """
        cov = self.cov if cov is None else cov
        if cov is None:
            return None

        # get indexing
        inds = self.get_inds(bl=bl, times=times, freqs=freqs, pol=pol,
                             bl_inds=bl_inds, time_inds=time_inds,
                             freq_inds=freq_inds, data=cov)

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

        if not try_view and all([isinstance(ind, slice) for ind in inds]):
            cov = cov.clone()

        return cov

    def get_icov(self, bl=None, icov=None, try_view=False, **kwargs):
        """
        Slice into cached inverse covariance.
        Same kwargs as get_cov()
        """
        if icov is None:
            icov = self.icov
        return self.get_cov(bl=bl, cov=icov, try_view=try_view, **kwargs)

    def __getitem__(self, bl):
        return self.get_data(bl, squeeze=True)

    def __setitem__(self, bl, val):
        self.set(bl, val)

    def set(self, bl, val, arr='data'):
        """
        Set the desired sliced attribute "arr" as val

        Parameters
        ----------
        bl : tuple or int, optional
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
               inplace=True, try_view=False):
        """
        Downselect on data tensor.

        Parameters
        ----------
        bl : tuple, list, ndarray, optional
            Baseline number array, or list of antpair tuples.
            Can be antpair-pols, with only 1 unique pol.
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
            in self.blnums
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
        try_view : bool, optional
            If inplace=False and the requested indexing
            can be cast as slices, try to make the selected
            data a view of self.data.
        """
        if inplace:
            obj = self
            out = obj
            try_view = True
        else:
            obj = self
            out = VisData()
            out.setup_meta(telescope=self.telescope, antpos=self.antpos)

        if bl is not None or bl_inds is not None:
            assert not ((bl is not None) & (bl_inds is not None))
            data = obj.get_data(bl, bl_inds=bl_inds, squeeze=False, try_view=try_view)
            cov = obj.get_cov(bl, bl_inds=bl_inds, squeeze=False, try_view=try_view)
            icov = obj.get_icov(bl, bl_inds=bl_inds, squeeze=False, try_view=try_view)
            flags = obj.get_flags(bl, bl_inds=bl_inds, squeeze=False, try_view=try_view)
            if bl_inds is not None: bl = [obj.bls[i] for i in bl_inds]
            out.setup_data(bl, obj.times, obj.freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)
            obj = out

        if times is not None or time_inds is not None:
            assert not ((times is not None) & (time_inds is not None))
            data = obj.get_data(times=times, time_inds=time_inds, squeeze=False, try_view=try_view)
            cov = obj.get_cov(times=times, time_inds=time_inds, squeeze=False, try_view=try_view)
            icov = obj.get_icov(times=times, time_inds=time_inds, squeeze=False, try_view=try_view)
            flags = obj.get_flags(times=times, time_inds=time_inds, squeeze=False, try_view=try_view)
            if time_inds is not None: times = obj.times[time_inds]
            out.setup_data(obj.bls, times, obj.freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)
            obj = out

        if freqs is not None or freq_inds is not None:
            assert not ((freqs is not None) & (freq_inds is not None))
            data = obj.get_data(freqs=freqs, freq_inds=freq_inds, squeeze=False, try_view=try_view)
            cov = obj.get_cov(freqs=freqs, freq_inds=freq_inds, squeeze=False, try_view=try_view)
            icov = obj.get_icov(freqs=freqs, freq_inds=freq_inds, squeeze=False, try_view=try_view)
            flags = obj.get_flags(freqs=freqs, freq_inds=freq_inds, squeeze=False, try_view=try_view)
            if freq_inds is not None: freqs = obj.freqs[freq_inds]
            out.setup_data(obj.bls, obj.times, freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)
            obj = out

        if pol is not None:
            data = obj.get_data(pol=pol, try_view=try_view, squeeze=False)
            flags = obj.get_flags(pol=pol, try_view=try_view, squeeze=False)
            cov = obj.get_cov(pol=pol, try_view=try_view, squeeze=False)
            icov = obj.get_icov(pol=pol, try_view=try_view, squeeze=False)
            out.setup_data(obj.bls, obj.times, obj.freqs, pol=pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if not inplace:
            return out

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
            vd = self.copy(copymeta=True)
        cal_data = cd.data
        if not utils.check_devices(vd.data.device, cal_data.device):
            cal_data = cal_data.to(vd.data.device)

        vd.data, vd.cov = calibration.apply_cal(vd.data, vd.bls, cal_data, cd.ants, cal_2pol=cal_2pol,
                                                cov=vd.cov, vis_type='com', undo=undo, inplace=inplace)

        return vd

    def chisq(self, other_vis=None, dof=None, icov=None, cov_axis=None, axis=None):
        """
        Compute the chisquare statistic between this visibility data
        and another set of visibilities, optionally weighted by degrees-of-freedom.

        Parameters
        ----------
        other_vis : VisData, optional
            Other visibility data to form residual with and weight
            by icov. E.g. if self is raw_data, other_vis should be
            the model data forward passed through calibration model.
            Should have the same shape and ordering as self.data.
            Default is to take chisq of self.data.
        dof : float, optional
            Degrees-of-freedom of fit to divide by. If provided
            the output is the reduced chisq.
        icov : tensor, optional
            If provided, weight residual with this icov as oppossed to
            self.icov (default).
        cov_axis : str, optional
            cov_axis parameter for provided icov, see optim.apply_icov() for details.
        axis : int or tuple, optional
            The axis over which to sum the chisq. Default is no summing.
            Note if icov is a 2D covariance matrix then summing
            is already performed implicitly via the innerproduct.

        Returns
        -------
        tensor
        """
        from bayeslim import calibration
        icov = self.icov if icov is None else icov
        cov_axis = self.cov_axis if icov is None else cov_axis
        other_data = other_vis.data if other_vis is not None else torch.zeros_like(self.data)
        return calibration.chisq(self.data, other_data, icov, axis=axis, cov_axis=cov_axis, dof=dof)

    def bl_average(self, reds=None, wgts=None, redtol=1.0, inplace=False):
        """
        Average baselines together. Note this drops all baselines not
        present in reds from the object.

        Parameters
        ----------
        reds : list, optional
            List of baseline groups to average in either blnums format
            or antpair tuple format.
            E.g. [[(0, 1), (1, 2)], [(2, 5), (3, 6), (4, 7)], ...]
            Default is to automatically build redundant groups.
        wgts : tensor, optional
            Weights to use, with the same shape as the data. Default
            is to use diagonal component of self.icov, if it exists.
        redtol : float, optional
            Tolerance [meters] in building redundant groups
        inplace : bool, optional
            If True, edit arrays inplace, otherwise return a deepcopy
        """
        from bayeslim import optim

        # setup reds
        if reds is None:
            from bayeslim import telescope_model
            red_info = telescope_model.build_reds(
                self.antpos,
                bls=self.bls,
                redtol=redtol,
            )
            reds, bl2red = red_info[0], red_info[2]
        else:
            bl2red = {}
            for i, red in enumerate(reds):
                for bl in red:
                    bl2red[bl] = i

        # setup indexing tensor along Nbls axis
        bls = self.bls if isinstance(list(bl2red.keys())[0], tuple) else self.blnums
        Nmax = len(reds)
        index = torch.as_tensor(
            [bl2red.get(bl, Nmax) for bl in bls],
            device=self.data.device
        )
        Nout_bls = index.unique().numel()
        truncate = Nmax in index

        # get weights
        if wgts is None and self.icov is not None and self.cov_axis is None:
            wgts = self.icov

        # get cov
        cov = None
        if self.cov_axis is None:
            if self.cov is not None:
                cov = self.cov
            elif self.icov is not None:
                cov = 1 / self.icov.clip(1e-60)

        # take data average
        avg_data, sum_wgts, avg_cov = average_data(
            self.data,
            -3,
            index,
            Nout_bls,
            wgts=wgts,
            cov=cov,
            truncate=truncate
        )

        # get avg_flags
        avg_flags = None
        if self.flags is not None:
            avg_flags = torch.zeros_like(
                avg_data,
                dtype=bool,
                device=avg_data.device
            )
            avg_flags.index_add_(-3, index, ~self.flags)
            avg_flags = ~avg_flags

        # get avg_icov
        avg_icov = None
        if self.icov is not None:
            avg_icov = 1 / avg_cov.clip(1e-60)

        if inplace:
            vout = self
        else:
            vout = self.copy(copydata=False, copymeta=False)

        vout.setup_data([red[0] for red in reds], vout.times, vout.freqs, pol=self.pol,
                        data=avg_data, flags=avg_flags, cov=avg_cov,
                        icov=avg_icov, cov_axis=None, history=self.history)

        return vout

    def time_average(self, time_inds=None, wgts=None, rephase=False, inplace=True):
        """
        Average time integrations together, weighted by inverse covariance. Note
        this drops all time indices not present in time_inds list.

        Parameters
        ----------
        time_inds : list of tensors, optional
            List of time indices from self.times to average together. Default is to
            average all times. E.g. [(0,1,2,3,4,...,N)]
        wgts : tensor, optional
            Weights to use, with the same shape as the data. Default
            is to use diagonal component of self.icov, if it exists.
        rephase : bool, optional
            If True, assume data are in drift-scan mode. Rephase the data to bin center in LST
            before averaging. Note: makes a copy of rephased data if inplace = False.
        inplace : bool, optional
            If True, edit arrays inplace, otherwise return a deepcopy
        """
        # setup times
        if time_inds is None:
            time_inds = [torch.arange(self.Ntimes)]

        # get averaging index tensor
        Nmax = len(time_inds)
        index = torch.ones(self.Ntimes, dtype=torch.int64, device=self.data.device) * Nmax
        for i, tinds in enumerate(time_inds):
            tinds = utils._list2slice(tinds)
            time_inds[i] = tinds
            index[tinds] = i

        Nout_times = index.unique().numel()
        truncate = Nmax in index

        # get weights
        if wgts is None and self.icov is not None and self.cov_axis is None:
            wgts = self.icov

        # get cov
        cov = None
        if self.cov_axis is None:
            if self.cov is not None:
                cov = self.cov
            elif self.icov is not None:
                cov = 1 / self.icov.clip(1e-60)

        # get avg_times
        avg_times = torch.zeros(Nout_times)
        avg_times.index_add_(0, index, self.times)
        sum_wgts = torch.zeros(Nout_times)
        sum_wgts.index_add_(0, index, torch.ones_like(self.times))
        avg_times /= sum_wgts

        # rephase the data
        if rephase:
            # get dLST for each time integration
            from bayeslim import telescope_model
            dtimes = (avg_times[index.cpu()] - self.times)
            dLST = dtimes * 2 * np.pi / sday.to('day')
            phs = telescope_model.vis_rephase(
                dLST,
                self.telescope.location[1],
                self.get_bl_vecs(self.bls),
                self.freqs
            )
            data = self.data
            if inplace:
                data *= phs
            else:
                data = data * phs
        else:
            data = self.data

        # take data average
        avg_data, sum_wgts, avg_cov = average_data(
            data,
            -2,
            index,
            Nout_times,
            wgts=wgts,
            cov=cov,
            truncate=truncate
        )

        # get avg_flags
        avg_flags = None
        if self.flags is not None:
            avg_flags = torch.zeros_like(
                avg_data,
                dtype=bool,
                device=avg_data.device
            )
            avg_flags.index_add_(-2, index, ~self.flags)
            avg_flags = ~avg_flags

        # get avg_icov
        avg_icov = None
        if self.icov is not None:
            avg_icov = 1 / avg_cov.clip(1e-60)

        if inplace:
            vout = self
        else:
            vout = self.copy(copydata=False, copymeta=False)

        if truncate:
            avg_times = avg_times[:-1]

        vout.setup_data(self.blnums, avg_times, vout.freqs, pol=self.pol,
                        data=avg_data, flags=avg_flags, cov=avg_cov,
                        icov=avg_icov, cov_axis=None, history=self.history)

        return vout

    def _inflate_by_redundancy(self, new_bls, red_bl_inds, try_view=False):
        """
        Inflate data by redundancies and return a new object

        Parameters
        ----------
        new_bls : list, ndarray
            List of new baseline tuples for inflated data
            e.g. [(0, 1), (1, 2), (1, 3), ...]
            or blnums ndarray.
        red_bl_inds : list, ndarray
            List of baseline indices in redundant dataset
            for each bl in new_bls.
            e.g. [0, 0, 1, 2, 2, ...]
        try_view : bool, optional
            If True try to make inflated data a view of red data.

        Returns
        -------
        VisData
        """
        # expand data across redundant baselines
        data = self.get_data(bl_inds=red_bl_inds, squeeze=False, try_view=try_view)
        flags = self.get_flags(bl_inds=red_bl_inds, squeeze=False, try_view=try_view)
        cov = self.get_cov(bl_inds=red_bl_inds, squeeze=False, try_view=try_view)
        icov = self.get_icov(bl_inds=red_bl_inds, squeeze=False, try_view=try_view)

        # setup new object
        new_vis = VisData()
        new_vis.setup_meta(telescope=self.telescope, antpos=self.antpos)
        new_vis.setup_data(new_bls, self.times, self.freqs, pol=self.pol,
                           data=data, flags=flags, cov=cov, icov=icov,
                           history=self.history)

        return new_vis

    def inflate_by_redundancy(self, bls=None, bl2red=None, **kwargs):
        """
        If current data only includes unique redundant baseline types,
        copy over redundant types to all physical baselines and return
        a new copy of the object. Note this only inflates to redundant
        groups that currently exist in the data.

        Parameters
        ----------
        bls : list, ndarray, optional
            If provided, only inflate to these physical baselines. Note
            all baselines in bls must have a redundant dual in self.bls,
            otherwise its dropped. Default is to use all baselines in bl2red.
        bl2red : dict, optional
            A {bl: int} dictionary mapping a physical baseline
            to its redundant group index. If not passed, build this
            using self.antpos.
        kwargs : dict, optional
            Keyword arguments to pass to telescope_model.build_reds() when
            building the baseline redundancy groups if bl2red is not passed.

        Returns
        -------
        VisData
        """
        # check if bl2red is passed
        if bl2red is None:
            from bayeslim.telescope_model import build_reds
            bl2red = build_reds(self.antpos, red_bls=self.bls, **kwargs)[2]

        # get all new baselines
        if bls is None:
            bls = list(bl2red.keys())

        new_bls, red_bl_inds = utils.inflate_bls(self.bls, bl2red, bls)

        return self._inflate_by_redundancy(new_bls, red_bl_inds)

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
                f.create_dataset('blnums', data=self.blnums)
                f.create_dataset('times', data=self.times)
                f.create_dataset('freqs', data=self.freqs)
                if self.pol is not None:
                    f.attrs['pol'] = self.pol
                f.attrs['history'] = self.history
                # write telescope and array objects
                f.attrs['tloc'] = self.telescope.location
                f.attrs['ants'] = self.ants
                antvecs = None if self.antpos is None else self.antpos.antvecs
                f.attrs['antvecs'] = antvecs
                f.attrs['obj'] = 'VisData'
                f.attrs['version'] = version.__version__
        else:
            print("{} exists, not overwriting...".format(fname))

    def read_hdf5(self, fname, read_data=True,
                  bl=None, times=None, freqs=None, pol=None,
                  time_inds=None, freq_inds=None,
                  suppress_nonessential=False,
                  lazy_load=False):
        """
        Read HDF5 VisData object

        Parameters
        ----------
        fname : str
            File to read
        read_data : bool, optional
            If True, read data arrays as well as metadata
        bl, times, freqs, pol : read options. see self.select() for details
        time_inds, freq_inds : read options. see self.select() for details
        suppress_nonessential : bool, optional
            If True, suppress reading-in flags and cov, as only data and icov
            are essential for inference.
        lazy_load : bool, optional
            If True, load an HDF5 handle just for data-shaped tensors
            and keep the keep the HDF5 file open. Default = False.
            If True, cannot downselect on any indices during the read.
            If True, sets read_data = False.
        """
        from bayeslim import telescope_model
        f = h5py.File(fname, mode='r')  # safe to keep file open in mode='r'
        if lazy_load:
            file = f
            assert bl == times == freqs == pol == time_inds == freq_inds == None
            read_data = False
        else:
            file = None
        # load metadata
        assert str(f.attrs['obj']) == 'VisData', "not a VisData object"
        if 'blnums' in f:
            _blnums = f['blnums'][:]
        elif 'bls' in f:
            _blnums = utils.ants2blnum(list(zip(*f['bls'][:].T)))
        _times = f['times'][:]
        _freqs = torch.as_tensor(f['freqs'][:])
        _pol = f.attrs['pol'] if 'pol' in f.attrs else None
        cov_axis = f.attrs['cov_axis'] if 'cov_axis' in f.attrs else None
        history = f.attrs['history'] if 'history' in f.attrs else ''

        # setup just full-size metadata
        self.setup_data(_blnums, _times, _freqs, pol=_pol,
                        cov_axis=cov_axis)

        data, flags, cov, icov = None, None, None, None
        if read_data:
            data = self.get_data(bl=bl, times=times, freqs=freqs, pol=pol,
                                 time_inds=time_inds, freq_inds=freq_inds,
                                 squeeze=False, data=f['data'], try_view=True)
            data = torch.as_tensor(data, device=self.device)
            if 'flags' in f and not suppress_nonessential:
                flags = self.get_flags(bl=bl, times=times, freqs=freqs, pol=pol,
                                       time_inds=time_inds, freq_inds=freq_inds,
                                       squeeze=False, flags=f['flags'], try_view=True)
                flags = torch.as_tensor(flags, device=self.device)
            if 'cov' in f and not suppress_nonessential:
                cov = self.get_cov(bl=bl, times=times, freqs=freqs, pol=pol,
                                   time_inds=time_inds, freq_inds=freq_inds,
                                   squeeze=False, cov=f['cov'], try_view=True)
                cov = torch.as_tensor(cov, device=self.device)
            if 'icov' in f:
                icov = self.get_icov(bl=bl, times=times, freqs=freqs, pol=pol,
                                     time_inds=time_inds, freq_inds=freq_inds,
                                     squeeze=False, icov=f['icov'], try_view=True)
                icov = torch.as_tensor(icov, device=self.device)

        elif lazy_load:
            data = HDF5tensor(f['data'], device=self.device)
            if 'flags' in f and not suppress_nonessential:
                flags = HDF5tensor(f['flags'], device=self.device)
            if 'cov' in f and not suppress_nonessential:
                cov = HDF5tensor(f['cov'], device=self.device)
            if 'icov' in f:
                icov = HDF5tensor(f['icov'], device=self.device)

        # downselect metadata to selection
        self.select(bl=bl, times=times, freqs=freqs, pol=pol,
                    time_inds=time_inds, freq_inds=freq_inds)

        # setup downselected metadata and data
        ants = f.attrs['ants'].tolist()
        antvecs = f.attrs['antvecs']
        antpos = utils.AntposDict(ants, antvecs)
        tloc = f.attrs['tloc']
        telescope = telescope_model.TelescopeModel(tloc)
        self.setup_meta(telescope, antpos)
        self.setup_data(self.blnums, self.times, self.freqs, pol=self.pol,
                        data=data, flags=flags, cov=cov, cov_axis=cov_axis,
                        icov=icov, history=history, file=file)

        if not lazy_load:
            f.close()

    def check(self):
        """
        Run checks on data
        """
        from bayeslim import telescope_model
        if self.telescope is not None:
            assert isinstance(self.telescope, telescope_model.TelescopeModel)
        if self.data is not None:
            if self._file is None: assert isinstance(self.data, torch.Tensor)
            assert self.data.shape[-3:] == (self.Nbls, self.Ntimes, self.Nfreqs)
        if self.flags is not None:
            if self._file is None: assert isinstance(self.flags, torch.Tensor)
            assert self.flags.shape == self.data.shape
        for arr in ['cov', 'icov']:
            cov = getattr(self, arr)
            if cov is not None:
                assert self.cov_axis != 'full', "full data-sized covariance not implemented"
                if self.cov_axis is None:
                    assert cov.shape == self.data.shape
                elif self.cov_axis == 'bl':
                    assert cov.shape == (self.Nbls, self.Nbls, self.Npol, self.Npol,
                                              self.Ntimes, self.Nfreqs)
                elif self.cov_axis == 'time':
                    assert cov.shape == (self.Ntimes, self.Ntimes, self.Npol, self.Npol,
                                              self.Nbls, self.Nfreqs)
                elif self.cov_axis == 'freq':
                    assert cov.shape == (self.Nfreqs, self.Nfreqs, self.Npol, self.Npol,
                                              self.Nbls, self.Ntimes)
        for (ant1, ant2) in self.bls:
            assert (ant1 in self.ants) and (ant2 in self.ants)


class MapData(TensorData):
    """
    An object for holding image or map data of shape
    (Npol, 1, Nfreqs, Npix)
    """
    def __init__(self):
        self.data = None
        self.atol = 1e-10

    def setup_meta(self, name=None):
        """
        Setup metadata

        Parameters
        ----------
        name : str
            Name of map data
        """
        self.name = name

    def setup_data(self, freqs, df=None, pols=None, data=None, angs=None,
                   flags=None, cov=None, cov_axis=None, icov=None,
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
        md.setup_meta(name=self.name)
        data = self.data.detach() if detach else self.data
        md.setup_data(self.freqs, df=self.df, pols=self.pols, data=data.clone(), norm=self.norm,
                      angs=self.angs, flags=self.flags, cov=self.cov,
                      icov=self.icov, cov_axis=self.cov_axis, history=self.history)
        return md

    def get_inds(self, angs=None, freqs=None, pols=None,
                 ang_inds=None, freq_inds=None, pol_inds=None):
        """
        Given data selections, return data indexing list

        Parameters
        ----------
        angs : tensor, optional
            J2000 [ra,dec] angles [deg] to index
        freqs : tensor or float, optional
            Frequencies to index
        pols : str or list, optional
            Polarization(s) to index
        ang_inds : int or list of int, optional
            Instead of feeding angs, can feed a
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
            ang_inds = self._ang2ind(angs)
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

    def _ang2ind(self, angs):
        """
        Pixel angles to index. Note this is slow
        because we loop over all input angles
        and do an isclose().

        Parameters
        ----------
        angs : tensor
            Pixel centers in [ra, dec] of degrees to index
            of shape (2, Nindex).
        """
        angs = torch.as_tensor(angs)
        _angs = self.angs
        idx = []
        for ang1, ang2 in zip(*angs):
            match = np.isclose(ang1, _angs[0], atol=self.atol) \
                    & np.isclose(ang2, _angs[1], atol=self.atol)
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
                 freqs=None, pols=None, ang_inds=None,
                 freq_inds=None, pol_inds=None,
                 try_view=False, **kwargs):
        """
        Get map data given selections

        Parameters
        ----------
        data : tensor, optional
            Data tensor to index, default is self.data
        squeeze : bool, optional
            If True squeeze output
        angs : tensor, optional
            See get_inds for more details
        try_view : bool, optional
            If True, if selections can be cast as slices
            then return a view of original data.

        Returns
        -------
        tensor
        """
        data = self.data if data is None else data
        if data is None:
            return None

        # get indexing
        inds = self.get_inds(angs=angs, freqs=freqs, pols=pols,
                             ang_inds=ang_inds, freq_inds=freq_inds,
                             pol_inds=pol_inds)
        data = data[inds]
        if not try_view and all([isinstance(ind, slice) for ind in inds]):
            data = data.clone()

        # squeeze if desired
        if squeeze:
            data = data.squeeze()

        return data

    def get_flags(self, flags=None, squeeze=True, angs=None,
                  freqs=None, pols=None, ang_inds=None,
                  freq_inds=None, pol_inds=None,
                  try_view=False, **kwargs):
        """
        Get flag data given selections

        Parameters
        ----------
        flags : tensor, optional
            Flag tensor to index. Default is self.flags.
        See self.get_data()
        try_view : bool, optional
            If True, if selections can be cast as slices
            then return a view of original data.
        """
        flags = self.flags if flags is None else flags
        if flags is None:
            return None

        # get indexing
        inds = self.get_inds(angs=angs, freqs=freqs, pols=pols,
                             ang_inds=ang_inds, freq_inds=freq_inds,
                             pol_inds=pol_inds)
        flags = flags[inds]
        if not try_view and all([isinstance(ind, slice) for ind in inds]):
            flags = flags.clone()

        # squeeze if desired
        if squeeze:
            flags = flags.squeeze()

        return flags

    def get_cov(self, cov=None, cov_axis=None, squeeze=True, angs=None,
                freqs=None, pols=None, ang_inds=None,
                freq_inds=None, pol_inds=None, try_view=False, **kwargs):
        """
        Index covariance given selections

        Parameters
        ----------
        cov : tensor, optional
            Covariance to index. Default is self.cov
        cov_axis : str, optional
            Covariance axis of cov. Default is self.cov_axis
        See get_data() for details
        try_view : bool, optional
            If True, if selections can be cast as slices
            then return a view of original data.
        """
        cov = self.cov if cov is None else cov
        cov_axis = self.cov_axis if cov_axis is None else cov_axis
        if cov is None:
            return None

        # get indexing
        inds = self.get_inds(angs=angs, freqs=freqs, pols=pols,
                             ang_inds=ang_inds, freq_inds=freq_inds,
                             pol_inds=pol_inds)

        if cov_axis is None:
            # cov is same shape as data
            cov = cov[inds]
        else:
            # cov is not the same shape as data
            if angs is not None or ang_inds is not None:
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

        if not try_view and all([isinstance(ind, slice) for ind in inds]):
            cov = cov.clone()

        return cov

    def get_icov(self, icov=None, try_view=False, **kwargs):
        """
        Slice into cached inverse covariance.
        Same kwargs as get_cov()
        """
        if icov is None:
            icov = self.icov
        return self.get_cov(cov=icov, try_view=try_view, **kwargs)

    def select(self, angs=None, freqs=None, pols=None,
               ang_inds=None, freq_inds=None, pol_inds=None,
               inplace=True, try_view=False):
        """
        Downselect on data tensor.

        Parameters
        ----------
        angs : tensor, optional
            J2000 [ra,dec] angles [deg] to index
        freqs : tensor or float, optional
            Frequencies to index
        pols : str or list, optional
            Polarization(s) to index
        ang_inds : int or list of int, optional
            Instead of feeding angs, can feed a
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
        try_view : bool, optional
            If inplace=False and the requested indexing
            can be cast as slices, try to make the selected
            data a view of self.data.
        """
        if inplace:
            obj = self
            out = obj
            try_view = True
        else:
            obj = self
            out = MapData()
            out.setup_meta(name=self.name)

        if angs is not None or ang_inds is not None:
            data = obj.get_data(angs=angs, ang_inds=ang_inds, squeeze=False, try_view=try_view)
            norm = obj.get_data(data=self.norm, angs=angs, ang_inds=ang_inds, squeeze=False, try_view=try_view)
            cov = obj.get_cov(angs=angs, ang_inds=ang_inds, squeeze=False, try_view=try_view)
            icov = obj.get_icov(angs=angs, ang_inds=ang_inds, squeeze=False, try_view=try_view)
            flags = obj.get_flags(angs=angs, ang_inds=ang_inds, squeeze=False, try_view=try_view)
            if ang_inds is not None:
                if self.angs is not None:
                    angs = self.angs[:, ang_inds]
            out.setup_data(obj.freqs, df=obj.df, pols=obj.pols, data=data, angs=angs,
                           flags=flags, cov=cov, cov_axis=obj.cov_axis, icov=icov,
                           norm=norm, history=obj.history)
            obj = out

        if freqs is not None or freq_inds is not None:
            data = obj.get_data(freqs=freqs, freq_inds=freq_inds, squeeze=False, try_view=try_view)
            norm = obj.get_data(data=self.norm, freqs=freqs, freq_inds=freq_inds, squeeze=False, try_view=try_view)
            cov = obj.get_cov(angs=angs, freqs=freqs, freq_inds=freq_inds, squeeze=False, try_view=try_view)
            icov = obj.get_icov(angs=angs, freqs=freqs, freq_inds=freq_inds, squeeze=False, try_view=try_view)
            flags = obj.get_flags(angs=angs, freqs=freqs, freq_inds=freq_inds, squeeze=False, try_view=try_view)
            if freq_inds is not None:
                freqs = obj.freqs[freq_inds]
                df = obj.df[freq_inds] if obj.df is not None else None
            else:
                df = obj.df[self._freq2ind(freqs)] if obj.df is not None else None
            out.setup_data(freqs, df=df, pols=obj.pols, data=data, angs=obj.angs,
                           flags=flags, cov=cov, cov_axis=obj.cov_axis, icov=icov,
                           norm=norm, history=obj.history)
            obj = out

        if pols is not None or pol_inds is not None:
            data = obj.get_data(pols=pols, pol_inds=pol_inds, squeeze=False, try_view=try_view)
            norm = obj.get_data(data=self.norm, pols=pols, pol_inds=pol_inds, squeeze=False, try_view=try_view)
            cov = obj.get_cov(angs=angs, pols=pols, pol_inds=pol_inds, squeeze=False, try_view=try_view)
            icov = obj.get_icov(angs=angs, pols=pols, pol_inds=pol_inds, squeeze=False, try_view=try_view)
            flags = obj.get_flags(angs=angs, pols=pols, pol_inds=pol_inds, squeeze=False, try_view=try_view)
            if pol_inds is not None: pols = [obj.pols[i] for i in pol_inds]
            out.setup_data(obj.freqs, df=obj.df, pols=pols, data=data, angs=obj.angs,
                           flags=flags, cov=cov, cov_axis=obj.cov_axis, icov=icov,
                           norm=norm, history=obj.history)

        if not inplace:
            return out

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
            cov_axis = f.attrs['cov_axis'] if 'cov_axis' in f.attrs else None
            history = f.attrs['history'] if 'history' in f.attrs else ''

            # setup just full-size metadata
            self.setup_data(_freqs, df=_df, angs=_angs, pols=_pols)

            # read-in data if needed
            data, norm, flags, cov, icov = None, None, None, None, None
            if read_data:
                data = self.get_data(data=f['data'], squeeze=False, try_view=True, **kwargs)
                data = torch.as_tensor(data)
                if 'flags' in f and not suppress_nonessential:
                    flags = self.get_flags(flags=f['flags'], squeeze=False, try_view=True, **kwargs)
                    flags = torch.as_tensor(flags)
                else:
                    flags = None
                if 'cov' in f and not suppress_nonessential:
                    cov = self.get_cov(cov=f['cov'], squeeze=False, try_view=True, **kwargs)
                    cov = torch.as_tensor(cov)
                else:
                    cov = None
                if 'icov' in f:
                    icov = self.get_icov(icov=f['icov'], squeeze=False, try_view=True, **kwargs)
                    icov = torch.as_tensor(icov)
                else:
                    icov = None
                if 'norm' in f and not suppress_nonessential:
                    norm = self.get_data(data=f['norm'], squeeze=False, try_view=True, **kwargs)
                    norm = torch.as_tensor(norm)
                else:
                    norm = None

            # downselect metadata to selection
            self.select(**kwargs)

            # setup downselected metadata and data
            self.setup_meta(name=self.name)
            self.setup_data(self.freqs, df=self.df, angs=self.angs,
                            pols=self.pols, data=data, flags=flags, cov=cov, norm=norm,
                            cov_axis=cov_axis, icov=icov, history=history)

    def push(self, device, return_obj=False):
        """
        Push data, flags, cov and icov to device
        """
        dtype = isinstance(device, torch.dtype)
        if self.data is not None:
            self.data = utils.push(self.data, device)
        if self.flags is not None:
            if not dtype:
                self.flags = self.flags.to(device)
        if self.cov is not None:
            self.cov = utils.push(self.cov, device)
        if self.icov is not None:
            self.icov = utils.push(self.icov, device)
        if self.norm is not None:
            self.norm = utils.push(self.norm, device)
        if return_obj:
            return self


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
        self.data = None
        self.atol = 1e-10

    def setup_meta(self, telescope=None, antpos=None):
        """
        Set the telescope and antpos dict

        Parameters
        ----------
        telescope : TelescopeModel, optional
            Telescope location
        antpos : AntposDict, optional
            Antenna position dictionary in ENU [meters].
            Antenna number integer as key, position vector as value
        """
        self.telescope = telescope
        if antpos is not None:
            if isinstance(antpos, utils.AntposDict):
                pass
            else:
                antpos = utils.AntposDict(list(antpos.keys()),
                    list(torch.vstack(antpos.values())))
        self.antpos = antpos
        self.ants = None
        if antpos is not None:
            self.ants = antpos.ants

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

    def copy(self, deepcopy=False, detach=True):
        """
        Copy and return self. This is equivalent
        to a detach and clone. Detach is optional

        Parameters
        ----------
        deepcopy : bool, optional
            If True (default) also make a copy of metadata
            like telescope, antpos, times, freqs, flags, etc.
            Note that this also copies things like the telescope cache,
            which can be large in memory. Otherwise, only make a clone
            of data and make all other (meta)data a pointer to self.
        detach : bool, optional
            If True (default) detach self.data for new object
        """
        cd = CalData()
        telescope, antpos = self.telescope, self.antpos
        times, freqs, ants = self.times, self.freqs, self.ants
        flags, cov, icov = self.flags, self.cov, self.icov
        history = self.history

        # clone data
        data = self.data
        if data is not None:
            if detach:
                data = data.detach()
            data = data.clone()

        if deepcopy:
            if telescope is not None:
                telescope = telescope.__class__(telescope.location, tloc=telescope.tloc,
                                                device=telescope.device)
            if antpos is not None:
                antpos = antpos.__class__(copy.deepcopy(antpos.ants), antpos.antvecs.clone())
            times = copy.deepcopy(times)
            freqs = copy.deepcopy(freqs)
            ants = copy.copy(ants)
            if flags is not None: flags = flags.clone()
            if cov is not None: cov = cov.clone()
            if icov is not None: icov = icov.clone()

        cd.setup_meta(telescope=telescope, antpos=antpos)
        cd.setup_data(ants, times, freqs, pol=self.pol,
                      data=data, flags=flags, cov=cov, icov=icov,
                      cov_axis=self.cov_axis, history=history)
        return cd

    def _ant2ind(self, ant):
        """
        Antenna(s) to index

        Parameters
        ----------
        ant : integer or list of ints
            Antenna number(s) to index
        """
        if isinstance(ant, (int, np.integer)):
            return self.ants.index(ant)
        else:
            return [self._ant2ind(a) for a in ant]

    def _ant2uniq_antpol(self, ant):
        """
        Given a antenna or antenna-pol tuple "ant",
        or list of such, return unique ants and pols

        Parameters
        ----------
        ant : int, tuple, list
            Antenna int e.g. 2
            Ant-pol pair e.g. (2, 'Jee')
            or list of such e.g. [2, ...] or [(2, 'Jee'), ...]

        Returns
        -------
        ant : list of int
        pol : str
        """
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
                # [ant], pol
                _ant, _pol = self._ant2uniq_antpol(a)
                if _ant[0] not in ant_list:
                    ant_list.append(_ant[0])
                if _pol not in pol_list:
                    pol_list.append(_pol)
            ant = ant_list
            pol = pol_list
            if len(pol) > 1 and None in pol:
                pol.remove(None)
            assert len(pol) == 1, "can only index 1 pol at a time"
            pol = pol[0]
        else:
            # ant is already an integer
            ant = [ant]
            pol = None

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
        # rtol=1e-13 allows for 0.03 sec discrimination on JD values
        return np.where(np.isclose(self.times, time, atol=self.atol, rtol=1e-13))[0].tolist()

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
            ant-pol e.g. (2, 'Jee') or [(2, 'Jee'), ...]
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
            # separate ant from pol
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
                 squeeze=True, data=None, try_view=False, **kwargs):
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
        try_view : bool, optional
            If True, if selections can be cast as slices
            then return a view of original data.
        """
        data = self.data if data is None else data
        if data is None:
            return None

        # get indexing
        inds = self.get_inds(ant=ant, times=times, freqs=freqs, pol=pol)
        data = data[inds]
        if not try_view and all([isinstance(ind, slice) for ind in inds]):
            data = data.clone()

        # squeeze if desired
        if squeeze:
            data = data.squeeze()

        return data

    def get_flags(self, ant=None, times=None, freqs=None, pol=None,
                  squeeze=True, flags=None, try_view=False, **kwargs):
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
        try_view : bool, optional
            If True, if selections can be cast as slices
            then return a view of original data.
        """
        flags = self.flags if flags is None else flags
        if flags is None:
            return None

        # get indexing
        inds = self.get_inds(ant=ant, times=times, freqs=freqs, pol=pol)
        flags = flags[inds]
        if not try_view and all([isinstance(ind, slice) for ind in inds]):
            flags = flags.clone()

        # squeeze if desired
        if squeeze:
            flags = flags.squeeze()

        return flags

    def get_cov(self, ant=None, times=None, freqs=None, pol=None,
                squeeze=True, cov=None, try_view=False, **kwargs):
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
        try_view : bool, optional
            If True, if selections can be cast as slices
            then return a view of original data.
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

        if not try_view and all([isinstance(ind, slice) for ind in inds]):
            cov = cov.clone()

        return cov

    def get_icov(self, ant=None, icov=None, try_view=False, **kwargs):
        """
        Slice into cached inverse covariance.
        Same kwargs as get_cov()
        """
        if icov is None:
            icov = self.icov
        return self.get_cov(ant=ant, cov=icov, try_view=try_view, **kwargs)

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

    def select(self, ants=None, times=None, freqs=None, pol=None, inplace=True, try_view=False):
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
        try_view : bool, optional
            If inplace=False and the requested indexing
            can be cast as slices, try to make the selected
            data a view of self.data.
        """
        if inplace:
            obj = self
            out = obj
            try_view = True
        else:
            obj = self
            out = CalData()
            out.setup_meta(telescope=self.telescope, antpos=self.antpos)

        if ants is not None:
            data = obj.get_data(ants, squeeze=False, try_view=try_view)
            cov = obj.get_cov(ants, squeeze=False, try_view=try_view)
            icov = obj.get_icov(ants, squeeze=False, try_view=try_view)
            flags = obj.get_flags(ants, squeeze=False, try_view=try_view)
            out.setup_data(ants, obj.times, obj.freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)
            obj = out

        if times is not None:
            data = obj.get_data(times=times, squeeze=False, try_view=try_view)
            cov = obj.get_cov(times=times, squeeze=False, try_view=try_view)
            icov = obj.get_icov(times=times, squeeze=False, try_view=try_view)
            flags = obj.get_flags(times=times, squeeze=False, try_view=try_view)
            out.setup_data(obj.ants, times, obj.freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)
            obj = out

        if freqs is not None:
            data = obj.get_data(freqs=freqs, squeeze=False, try_view=try_view)
            cov = obj.get_cov(freqs=freqs, squeeze=False, try_view=try_view)
            icov = obj.get_icov(freqs=freqs, squeeze=False, try_view=try_view)
            flags = obj.get_flags(freqs=freqs, squeeze=False, try_view=try_view)
            out.setup_data(obj.ants, obj.times, freqs, pol=obj.pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)
            obj = out

        if pol is not None:
            data = obj.get_data(pol=pol, try_view=try_view)
            flags = obj.get_flags(pol=pol, try_view=try_view)
            cov = obj.get_cov(pol=pol, try_view=try_view)
            icov = obj.get_icov(pol=pol, try_view=try_view)
            out.setup_data(obj.ants, obj.times, obj.freqs, pol=pol, 
                            data=data, flags=obj.flags, cov=cov, icov=icov,
                            cov_axis=obj.cov_axis, history=obj.history)

        if not inplace:
            return out

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
        out = self.copy(copymeta=True)
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
                    antvecs = np.array([self.antpos[a] for a in self.ants])
                    f.create_dataset('antvecs', data=antvecs)
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
            if 'antvecs' in f:
                _antvec = np.asarray(f['antvecs'][:])
                _antpos = utils.AntposDict(_ants, _antvec)
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
                                     squeeze=False, data=f['data'], try_view=True)
                data = torch.as_tensor(data)
                if 'flags' in f:
                    flags = self.get_flags(ant=ants, times=times, freqs=freqs, pol=pol,
                                          squeeze=False, flags=f['flags'], try_view=True)
                    flags = torch.as_tensor(flags)
                else:
                    flags = None
                if 'cov' in f:
                    cov = self.get_cov(ant=ants, times=times, freqs=freqs, pol=pol,
                                       squeeze=False, cov=f['cov'], try_view=True)
                    cov = torch.as_tensor(cov)
                else:
                    cov = None
                if 'icov' in f:
                    icov = self.get_icov(ant=ants, times=times, freqs=freqs, pol=pol,
                                         squeeze=False, icov=f['icov'], try_view=True)
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


class HDF5tensor:
    def __init__(self, hdf5_dataset, dtype=None, device=None):
        self.hdf5_dataset = hdf5_dataset
        self.out_dtype = dtype
        self.device = device

    def __getitem__(self, idx):
        # slice into hdf5 handle to get numpy array
        data = self.hdf5_dataset[idx]

        # convert to tensor on self.device
        data = torch.as_tensor(data, device=self.device)

        # change dtype if needed
        if self.out_dtype is not None:
            data = utils.push(data, self.out_dtype)

        return data        

    def __repr__(self):
        return self.hdf5_dataset.__repr__()

    def __str__(self):
        return self.hdf5_dataset.__str__()

    @property
    def shape(self):
        return self.hdf5_dataset.shape

    @property
    def size(self):
        return self.hdf5_dataset.size

    @property
    def dtype(self):
        return self.hdf5_dataset.dtype

    def __len__(self):
        return len(self.hdf5_dataset)

    def push(self, device):
        dtype = isinstance(device, torch.dtype)
        if not dtype:
            self.device = device
        else:
            self.out_dtype = device



class Dataset(TorchDataset):
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


class RedVisAvg(utils.Module):
    """
    VisData redundant averaging block.

    Currently only supports diagonal covariances.
    """
    def __init__(self, reds, wgts=None, inplace=False, device=None):
        """
        Parameters
        ----------
        reds : list of lists
            Redundant baseline groups to average
            together
        wgts : tensor, optional
            Visibility weights to use when averaging
        inplace : bool, optional
            Average inplace, default = False
        device : bool, optional
            device to push to
        """
        super().__init__()
        self.reds = reds
        self.wgts = wgts
        self.inplace = inplace
        self.device = device

    def __call__(self, vd, **kwargs):
        vd = vd.bl_average(
            reds=self.reds,
            wgts=self.wgts,
            inplace=self.inplace
        )

        return vd

    def forward(self, vd, **kwargs):
        return self(vd, **kwargs)

    def push(self, device):
        """
        Push to device
        """
        self.wgts = utils.push(self.wgts, device)
        dtype = isinstance(device, torch.dtype)
        if not dtype:
            self.device = device

class RedVisInflate(utils.Module):
    """
    VisData redundant inflation block.
    """
    def __init__(self, new_bls, red_bl_inds):
        """
        Parameters
        ----------
        new_bls : list
            new baselines of inflated data.
        red_bl_inds : tensor
            Indicies of redundantly compressed data for each
            bl in new_bls.
        """
        super().__init__()
        self.new_bls = new_bls
        self.red_bl_inds = red_bl_inds
        self.device = None

    def __call__(self, vd, **kwargs):
        vd = vd._inflate_by_redundancy(
            new_bls=self.new_bls,
            red_bl_inds=self.red_bl_inds,
        )

        return vd

    def forward(self, vd, **kwargs):
        return self(vd, **kwargs)

    def push(self, device):
        """
        Push to device
        """
        dtype = isinstance(device, torch.dtype)
        if not dtype:
            self.red_bl_inds = utils.push(self.red_bl_inds, device)
            self.device = device


def concat_VisData(vds, axis, run_check=True, interleave=False):
    """
    Concatenate VisData objects together

    Parameters
    ----------
    vds : list of VisData
    axis : str, ['bl', 'time', 'freq']
        Axis to concatenate over. All other
        axes must match exactly in all vds.
    interleave : bool, optional
        If True, interleave the data along
        the desired axis. Note this is probably
        slow so its not recommended for repeated
        forward passes.
    """
    if isinstance(vds, VisData):
        return vds
    Nvd = len(vds)
    assert Nvd > 0
    if Nvd == 1:
        return vds[0]
    vd = vds[0]
    out = VisData()
    flags, cov, icov = None, None, None

    if axis == 'bl':
        dim = 2
        times = vd.times
        freqs = vd.freqs
        pol = vd.pol
        if interleave:
            bls = utils._tensor_concat([torch.as_tensor(o.bls) for o in vds], interleave=interleave)
            bls = [tuple(bl) for bl in bls.tolist()]
        else:
            bls = []
            for o in vds:
                bls.extend(o.bls)

    elif axis == 'time':
        dim = 3
        freqs = vd.freqs
        pol = vd.pol
        bls = vd.bls
        times = utils._tensor_concat([torch.as_tensor(o.times) for o in vds], interleave=interleave)

    elif axis == 'freq':
        dim = 4
        times = vd.times
        pol = vd.pol
        bls = vd.bls
        freqs = utils._tensor_concat([torch.as_tensor(o.freqs) for o in vds], interleave=interleave)

    # stack data and flags
    data = utils._tensor_concat([o.data for o in vds], dim=dim, interleave=interleave)
    if vd.flags is not None:
        flags = utils._tensor_concat([o.flags for o in vds], dim=dim, interleave=interleave)

    # stack cov and icov
    if vd.cov_axis is not None:
        raise NotImplementedError
    if vd.cov is not None:
        cov = utils._tensor_concat([o.cov for o in vds], dim=dim, interleave=interleave)
    if vd.icov is not None:
        icov = utils._tensor_concat([o.icov for o in vds], dim=dim, interleave=interleave)

    out.setup_meta(vd.telescope, vd.antpos)
    out.setup_data(bls, times, freqs, pol=pol,
                   data=data, flags=flags, cov=cov, icov=icov,
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


def average_TensorData(objs, wgts=None):
    """
    Average multiple TensorData subclasses together,
    assuming they share the exact same shape and metadata
    ordering.

    Parameters
    ----------
    objs : list of TensorData objects
        TensorData objects to average together
    wgts : list of tensor
        Weights for each TensorData.data tensor.
        Default is to use their self.icov tensors.

    Returns
    -------
    TensorData object
    """
    raise NotImplementedError


def average_data(data, dim, index, N, wgts=None, cov=None, truncate=False):
    """
    Average tensor data along a dimension using torch.index_add.

    Parameters
    ----------
    data : tensor
        nd-tensor to average along a dimension.
    dim : int
        Dimension of data to average over.
    index : tensor
        Indexing tensor of data along dim, denoting
        the output index for each element in data to sum together.
        e.g. [0, 1, 0, 1, ...] -> [sum(data[[0, 2]]), sum(data[[1, 3]]), ...]
    N : int
        The number of output elements along dim, i.e., len(index.unique())
    wgts : tensor
        Data weights to use for averaging.
        If None use uniform weights.
    cov : tensor
        Covariance of data to use in weighting, and to propagate to averaged
        data. Currently only supports diagonal covariances. Must match
        shape of data tensor.
    truncate : bool, optional
        If True, remove the last element from all outputs along dim.
        This is needed when some elements in data aren't actually
        needed, but due to torch.index_add must be assigned
        an element in the output data, so we just truncate it.

    Returns
    -------
    avg_data : tensor
        Data averaged along dim.
    sum_wgts : tensor
        Sum of weights along dim.
    avg_cov : tensor
        Covariance of averaged data.

    Notes
    -----
    If we estimate a compressed (scalar) basis x from a vector basis y, such that
        y = A x
    where A is a 1's vector (column vector) representing the signal mapping of 
    scalar x -> vector y, then the generalized least-squares average of x is

        x_avg = G^-1 A^T W y

    where W is a diagonal matrix representing the data weights (i.e. inverse
    variance of the noise in y), and

        G = A^T W A = tr(W)
        C_x = G^-1 A^T W C_y W A G^-1

    which in the limit W = C_y^-1 reduces C_x to

        C_x = G^-1 A^T W A G^-1 = G^-1 = 1 / tr(W)
    """
    # make sure dim is negative
    dim = np.arange(-data.ndim, 0, 1)[dim]
    
    # setup weights
    if wgts is None:
        # uniform weights
        shape = torch.ones(data.ndim, dtype=torch.int64).tolist()
        shape[dim] = data.shape[dim]
        wgts = torch.ones(shape, device=data.device)

    # make sure wgts and data have same len along dim
    if wgts.shape[dim] != data.shape[dim]:
        # broadcast wgts.shape[dim] to data.shape[dim]
        shape = list(wgts.shape)
        shape[dim] = data.shape[dim]
        wgts = wgts.expand(shape)

    # setup output data and wgt tensors
    shape = list(data.shape)
    shape[dim] = N
    avg_data = torch.zeros(shape, dtype=data.dtype, device=data.device)
    shape = list(wgts.shape)
    shape[dim] = N
    sum_wgts = torch.zeros(shape, dtype=wgts.dtype, device=wgts.device)

    # take weighted sum of data
    sum_wgts.index_add_(dim, index, wgts)
    avg_data.index_add_(dim, index, data * wgts)
    avg_data /= sum_wgts.clip(1e-40)

    # update cov array
    if cov is not None:
        # ensure cov broadcasts with data
        assert cov.shape == data.shape[-cov.ndim:]

        # propagate covariance through weighted sum
        shape = list(cov.shape)
        shape[dim] = N
        avg_cov = torch.zeros(shape, dtype=cov.dtype, device=cov.device)
        avg_cov.index_add_(dim, index, cov * wgts.pow(2))
        avg_cov /= sum_wgts.clip(1e-40).pow(2)

    else:
        avg_cov = None

    if truncate:
        # remove last element from all outputs
        slices = [slice(None)] * data.ndim
        slices[dim] = slice(0, N - 1)
        avg_data = avg_data[slices]
        slices = slices[-wgts.ndim:]
        sum_wgts = sum_wgts[slices]
        if cov is not None:
            avg_cov = avg_cov[slices]

    return avg_data, sum_wgts, avg_cov


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
