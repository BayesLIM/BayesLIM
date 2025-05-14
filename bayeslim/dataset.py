"""
Module for visibility and map data formats, and a torch style data loader
"""
import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import os
import copy
import h5py

from . import version, utils


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

        elif icov is not None:
            # only icov provided, try to get cov_logdet
            if cov_axis is None:
                cov_logdet = torch.sum(-torch.log(icov))

        # set covariance
        if cov is not None: cov = cov.clone()
        if icov is not None: icov = icov.clone()
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

    def copy(self, deepcopy=False, detach=True):
        """
        Copy and return self. This is equivalent
        to a detach and clone. Detach is optional

        Parameters
        ----------
        deepcopy : bool, optional
            If True (default) also make a copy of metadata
            in addition to data.
        detach : bool, optional
            If True (default) detach self.data for new object
        """
        flags, cov, icov = self.flags, self.cov, self.icov
        history = self.history

        # clone data
        data = self.data
        if data is not None:
            if detach:
                data = data.detach()
            data = data.clone()

        if deepcopy:
            if flags is not None: flags = flags.clone()
            if cov is not None: cov = cov.clone()
            if icov is not None: icov = icov.clone()

        td = TensorData()
        td.setup_data(data=data, flags=flags, cov=cov,
                      cov_axis=self.cov_axis, icov=icov, history=history)

        return td

    def __add__(self, other):
        out = self.copy()
        if isinstance(other, (float, int, complex, torch.Tensor)):
            out.data += other
        else:
            out.data += other.data
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
        out = self.copy()
        if isinstance(other, (float, int, complex, torch.Tensor)):
            out.data -= other
        else:
            out.data -= other.data
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
        out = self.copy()
        if isinstance(other, (float, int, complex, torch.Tensor)):
            out.data *= other
        else:
            out.data *= other.data
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
        out = self.copy()
        if isinstance(other, (float, int, complex, torch.Tensor)):
            out.data /= other
        else:
            out.data /= other.data
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
        self.atol = 1e-10
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
        self.ants, self.antvecs = None, None
        if antpos is not None:
            self.ants = antpos.ants
            self.antvecs = antpos.antvecs

    def setup_data(self, bls, times, freqs, pol=None,
                   data=None, flags=None, cov=None, cov_axis=None,
                   icov=None, history=''):
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

    def _set_bls(self, bls):
        """
        Set the blnums tensor for all
        baselines in the data.

        Note:
        self.blnums is numpy on cpu
        self._blnums is pytorch on self.data.device.

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
            if not utils.check_devices(self._blnums.device, self.data.device):
                self._blnums = self._blnums.to(self.data.device)

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
            rk = dict(bls=self.bls)
            array = ArrayModel(self.antpos, self.freqs, redtol=redtol, red_kwargs=rk)
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
        vd = VisData()
        telescope, antpos = self.telescope, self.antpos
        times, freqs, blnums = self.times, self.freqs, self.blnums
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
            blnums = blnums.clone()
            if flags is not None: flags = flags.clone()
            if cov is not None: cov = cov.clone()
            if icov is not None: icov = icov.clone()

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
        return np.where(np.isclose(self.times, time, atol=self.atol, rtol=1e-12))[0].tolist()

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
            vd = self.copy(detach=True)
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

    def bl_average(self, reds=None, wgts=None, redtol=1.0, inplace=True):
        """
        Average baselines together, weighted by inverse covariance. Note
        this drops all baselines not present in reds from the object.

        Parameters
        ----------
        reds : list of baseline groups, optional
            List of baseline groups to average.
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
            reds = telescope_model.build_reds(self.antpos, bls=self.bls, redtol=redtol)[0]

        # iterate over reds: select, average, then append
        avg_groups = []
        for i, red in enumerate(reds):
            # down select bls from self
            obj = self.select(bl=red, inplace=False)

            # setup weights
            if wgts is None:
                if obj.icov is not None:
                    wgt = optim.cov_get_diag(obj.icov, obj.cov_axis, mode='vis')
                elif obj.cov is not None:
                    wgt = 1 / optim.cov_get_diag(obj.cov, obj.cov_axis, mode='vis').clip(1e-40)
                else:
                    wgt = torch.ones_like(obj.data)
            else:
                # select wgt given bl selection
                wgt = wgts[self.get_inds(bl=red)]
                
            assert wgt.shape[2] == obj.data.shape[2]
            wgt = wgt.real

            # get covariance
            if obj.cov is not None:
                cov = optim.cov_get_diag(obj.cov, obj.cov_axis, mode='vis')
            elif obj.icov is not None:
                cov = 1 / optim.cov_get_diag(obj.icov, obj.cov_axis, mode='vis').clip(1e-40)
            else:
                cov = None

            # take average along bl axis
            avg_data, wgt_norm, avg_cov = average_data(obj.data, 2, wgts=wgt, cov=cov, keepdims=True)

            # update cov array
            if obj.icov is not None:
                avg_icov = optim.compute_icov(avg_cov, None)
            else:
                avg_icov = None

            # get rid of cov if not present in obj
            if obj.cov is None:
                avg_cov = None

            # update flag array
            if obj.flags is not None:
                avg_flags = obj.flags.all(dim=2, keepdims=True)
            else:
                avg_flags = None

            # setup data
            obj.setup_data(red[:1], obj.times, obj.freqs, pol=obj.pol,
                           data=avg_data, flags=avg_flags, cov=avg_cov,
                           icov=avg_icov, cov_axis=None, history=obj.history)

            avg_groups.append(obj)

        # concatenate the averaged groups
        out = concat_VisData(avg_groups, 'bl')

        if inplace:
            # overwrite self.data and appropriate metadata
            self.setup_data(out.blnums, out.times, out.freqs, pol=out.pol,
                            data=out.data, flags=out.flags, cov=out.cov,
                            cov_axis=out.cov_axis, icov=out.icov,
                            history=out.history)

        else:
            return out

    def time_average(self, time_inds=None, wgts=None, atol=1e-5, rephase=False, inplace=True):
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
        atol : float, optional
            Tolerance for matching time stamps in self.times array [julian date].
        rephase : bool, optional
            If True, assume data are in drift-scan mode. Rephase the data to bin center in LST
            before averaging.
        inplace : bool, optional
            If True, edit arrays inplace, otherwise return a deepcopy
        """
        from bayeslim import optim

        # setup times
        if time_inds is None:
            time_inds = [torch.arange(self.Ntimes)]

        # iterate over times: select, average, then append
        avg_groups = []
        for i, times in enumerate(time_inds):
            # down select bls from self
            obj = self.select(time_inds=times, inplace=False)

            # setup weights
            if wgts is None:
                if obj.icov is not None:
                    wgt = optim.cov_get_diag(obj.icov, obj.cov_axis, mode='vis')
                elif obj.cov is not None:
                    wgt = 1 / optim.cov_get_diag(obj.cov, obj.cov_axis, mode='vis').clip(1e-40)
                else:
                    wgt = torch.ones_like(obj.data)
            else:
                # select wgt given bl selection
                wgt = wgts[self.get_inds(time_inds=times)]
                
            assert wgt.shape[2] == obj.data.shape[2]
            wgt = wgt.real

            # get covariance
            if obj.cov is not None:
                cov = optim.cov_get_diag(obj.cov, obj.cov_axis, mode='vis')
            elif obj.icov is not None:
                cov = 1 / optim.cov_get_diag(obj.icov, obj.cov_axis, mode='vis').clip(1e-40)
            else:
                cov = None

            # rephase the data
            if rephase:
                # get phasor
                from bayeslim import telescope_model
                loc = self.telescope.location
                lsts = telescope_model.JD2LST(self.times[times], loc[0]) # rad
                dlst = lsts[obj.Ntimes//2] - lsts
                phs = telescope_model.vis_rephase(dlst, loc[1], self.get_bl_vecs(self.bls), self.freqs)
                data = obj.data * phs

            else:
                data = obj.data

            # take average along bl axis
            avg_data, wgt_norm, avg_cov = average_data(data, 3, wgts=wgt, cov=cov, keepdims=True)

            # update flag array
            avg_flags = abs(wgt_norm) > 1e-40

            # update cov array
            if obj.icov is not None:
                avg_icov = optim.compute_icov(avg_cov, None)
            else:
                avg_icov = None

            # get rid of cov if not present in obj
            if obj.cov is not None:
                avg_cov = None

            # get new time
            new_time = torch.atleast_1d(obj.times[obj.Ntimes//2])

            # setup data
            obj.setup_data(obj.blnums, new_time, obj.freqs, pol=obj.pol,
                           data=avg_data, flags=avg_flags, cov=avg_cov,
                           icov=avg_icov, cov_axis=None, history=obj.history)

            avg_groups.append(obj)

        # concatenate the averaged groups
        out = concat_VisData(avg_groups, 'time')

        if inplace:
            # overwrite self.data and appropriate metadata
            self.setup_data(out.blnums, out.times, out.freqs, pol=out.pol,
                            data=out.data, flags=out.flags, cov=out.cov,
                            cov_axis=out.cov_axis, icov=out.icov,
                            history=out.history)

        else:
            return out

    def _inflate_by_redundancy(self, new_bls, old_bls):
        """
        Inflate data by redundancies and return a new object

        Parameters
        ----------
        new_bls : list, ndarray
            List of new baseline tuples for inflated data
            e.g. [(0, 1), (1, 2), (1, 3), ...]
            or blnums ndarray
        old_bls : list, ndarray
            List of redundant baseline tuple for each bl tuples in new_bls
            e.g. [(0, 1), (0, 1), (1, 3), ...]
            or blnums ndarray

        Returns
        -------
        VisData
        """
        # expand data across redundant baselines
        data = self.get_data(bl=old_bls, squeeze=False, try_view=False)
        flags = self.get_flags(bl=old_bls, squeeze=False, try_view=False)
        cov = self.get_cov(bl=old_bls, squeeze=False, try_view=False)
        icov = self.get_icov(bl=old_bls, squeeze=False, try_view=False)

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
            bl2red = build_reds(self.antpos, **kwargs)[2]

        # get all new baselines
        if bls is None:
            bls = list(bl2red.keys())

        new_bls, red_bls = utils.inflate_bls(self.bls, bl2red, bls)

        return self._inflate_by_redundancy(new_bls, red_bls)

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
                f.attrs['antvecs'] = self.antvecs
                f.attrs['obj'] = 'VisData'
                f.attrs['version'] = version.__version__
        else:
            print("{} exists, not overwriting...".format(fname))

    def read_hdf5(self, fname, read_data=True,
                  bl=None, times=None, freqs=None, pol=None,
                  time_inds=None, freq_inds=None,
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
        time_inds, freq_inds : read options. see self.select() for details
        suppress_nonessential : bool, optional
            If True, suppress reading-in flags and cov, as only data and icov
            are essential for inference.
        """
        import h5py
        from bayeslim import telescope_model
        with h5py.File(fname, 'r') as f:
            # load metadata
            assert str(f.attrs['obj']) == 'VisData', "not a VisData object"
            _blnums = f['blnums'][:]
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
                data = torch.as_tensor(data)
                if 'flags' in f and not suppress_nonessential:
                    flags = self.get_flags(bl=bl, times=times, freqs=freqs, pol=pol,
                                           time_inds=time_inds, freq_inds=freq_inds,
                                           squeeze=False, flags=f['flags'], try_view=True)
                    flags = torch.as_tensor(flags)
                else:
                    flags = None
                if 'cov' in f and not suppress_nonessential:
                    cov = self.get_cov(bl=bl, times=times, freqs=freqs, pol=pol,
                                       time_inds=time_inds, freq_inds=freq_inds,
                                       squeeze=False, cov=f['cov'], try_view=True)
                    cov = torch.as_tensor(cov)
                else:
                    cov = None
                if 'icov' in f:
                    icov = self.get_icov(bl=bl, times=times, freqs=freqs, pol=pol,
                                         time_inds=time_inds, freq_inds=freq_inds,
                                         squeeze=False, icov=f['icov'], try_view=True)
                    icov = torch.as_tensor(icov)
                else:
                    icov = None

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
        if self.data is not None:
            assert isinstance(self.data, torch.Tensor)
            assert self.data.shape[-3:] == (self.Nbls, self.Ntimes, self.Nfreqs)
        if self.flags is not None:
            assert isinstance(self.flags, torch.Tensor)
            assert self.data.shape == self.flags.shape
        if self.cov is not None:
            assert self.cov_axis != 'full', "full data-sized covariance not implemented"
            if self.cov_axis is None:
                assert self.cov.shape == self.data.shape
            elif self.cov_axis == 'bl':
                assert self.cov.shape == (self.Nbls, self.Nbls, self.Npol, self.Npol,
                                          self.Ntimes, self.Nfreqs)
            elif self.cov_axis == 'time':
                assert self.cov.shape == (self.Ntimes, self.Ntimes, self.Npol, self.Npol,
                                          self.Nbls, self.Nfreqs)
            elif self.cov_axis == 'freq':
                assert self.cov.shape == (self.Nfreqs, self.Nfreqs, self.Npol, self.Npol,
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
        self.ants, self.antvecs = None, None
        if antpos is not None:
            self.ants = antpos.ants
            self.antvecs = antpos.antvecs

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
        return np.where(np.isclose(self.times, time, atol=self.atol, rtol=1e-12))[0].tolist()

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


def average_data(data, dim, wgts=None, cov=None, keepdims=True):
    """
    Average tensor data along a dimension

    Parameters
    ----------
    data : tensor
        nd-tensor to average along one dimension
    dim : int
        Dimension of data to average over
    wgts : tensor
        Weights to use for averaging. If None and cov is None
        use uniform weights. If None but cov is not None, use inverse-cov.
        If wgts and cov are passed, use wgts for weighting.
    cov : tensor
        Covariance of data to use in weighting, and to propgate to averaged
        data. Currently only supports diagonal covariances. Must match
        shape of data tensor along last cov.ndim dimensions.
    keepdims : bool, optional
        If True, keep averaged dimension, otherwise squeeze it.

    Returns
    -------
    avg_data : tensor
        Data averaged along dim.
    wgt_norm : tensor
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
    # setup weights
    if wgts is None:
        if cov is not None:
            wgts = 1 / cov.clip(1e-40)
        else:
            wgts = torch.ones_like(data)

    wgts = wgts.real

    # make sure wgts have same shape as data
    if wgts.shape != data.shape[-wgts.ndim:]:
        wgts = torch.ones_like(data).real * wgts

    # get dim relative to last dimension
    dim = np.arange(-data.ndim, 0, 1)[dim]
    
    # average data
    wgt_norm = torch.sum(wgts, dim=dim, keepdims=keepdims)

    # TODO: do a true divide where wgt_norm == 0
    avg_data = torch.sum(data * wgts, dim=dim, keepdims=keepdims) / wgt_norm.clip(1e-40)

    # update cov array
    if cov is not None:
        # ensure cov broadcasts with data
        assert cov.shape == data.shape[-cov.ndim:]  # only diagonal
    else:
        cov = torch.ones_like(data).real

    # propagate covariance through weighted sum
    avg_cov = torch.sum(cov * wgts**2, dim=dim, keepdims=keepdims) / (wgt_norm**2).clip(1e-40)

    return avg_data, wgt_norm, avg_cov


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
