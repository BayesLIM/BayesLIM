"""
Module for visibility and map data formats, and a torch style data loader
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import copy
import h5py

from . import version


class VisData:
    """
    An object for holding visibility data of shape
    (Npol, Npol, Nbl, Ntimes, Nfreqs)
    """
    def __init__(self):
        self.data, self.flags = None, None
        self.icov, self.cov, self.cov_axis = None, None, None
        self.bls, self.times = None, None
        self.freqs, self.pol, self.history = None, None, ''
        self.telescope = None
        self.antpos, self.ants, self.antvec = None, None, None
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
            Antenna number int as key, position vector as key
        """
        self.telescope = telescope
        self.antpos = antpos
        if antpos is not None:
            self.ants = np.array(list(antpos.keys()))
            self.antvec = np.array(list(antpos.values()))

    def setup_data(self, bls, times, freqs, pol=None,
                   data=None, flags=None, cov=None, cov_axis=None,
                   icov=None, history='', run_check=True):
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
            If None, it is assumed Npol == 2
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
                cov_axis = 'bl': (Nfreqs, Nfreqs, Npol, Npol, Nbl, Ntimes)
                cov_axis = 'freq': (Nfreqs, Nfreqs, Npol, Npol, Nbl, Ntimes)
                cov_axis = 'time': (Ntimes, Ntimes, Npol, Npol, Nbl, Nfreq)
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
        self.bls = bls
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

    def copy(self):
        """
        Copy and return self. Data tensor
        is detached and cloned.
        """
        vd = VisData()
        vd.setup_meta(telescope=self.telescope, antpos=self.antpos)
        vd.setup_data(self.bls, self.times, self.freqs, pol=self.pol,
                      data=self.data.detach().clone(),
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
            bl_list, pol_list = [], []
            for b in bl:
                _bl, _pol = self._bl2uniq_blpol(b)
                bl_list.extend(_bl)
                pol_list.append(_pol)
            bl = sorted(set(bl_list))
            pol = list(set(pol_list))
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
            if pol.lower() != self.pol:
                raise ValueError("cannot index pol from 1pol {}".format(self.pol))
            return slice(0, 1)
        if pol.lower() == 'ee':
            return slice(0, 1)
        elif pol.lower() == 'nn':
            return slice(1, 2)
        else:
            raise ValueError("cannot index cross-pols")

    def get_inds(self, bl=None, times=None, freqs=None, pol=None):
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

        Returns
        -------
        list
            A 5-len list holding slices along axes.
        """
        if bl is not None:
            # special case for antpairpols
            bl, _pol = self._bl2uniq_blpol(bl)
            bl_inds = self._bl2ind(bl)
            if pol is not None:
                if _pol is not None:
                    assert _pol == pol
            pol = _pol
        else:
            bl_inds = slice(None)

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

        inds = [pol_ind, pol_ind, bl_inds, time_inds, freq_inds]
        inds = tuple([_list2slice(ind) for ind in inds])
        slice_num = sum([isinstance(ind, slice) for ind in inds])
        assert slice_num > 3, "cannot fancy index more than 1 axis"

        return inds

    def get_data(self, bl=None, times=None, freqs=None, pol=None,
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
        squeeze : bool, optional
            If True, squeeze array before return
        data : tensor, optional
            Tensor to index. default is self.data
        """
        data = self.data if data is None else data
        if data is None:
            return None

        # get indexing
        inds = self.get_inds(bl=bl, times=times, freqs=freqs, pol=pol)
        data = data[inds]

        # squeeze if desired
        if squeeze:
            data = data.squeeze()

        return data

    def get_flags(self, bl=None, times=None, freqs=None, pol=None,
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
        squeeze : bool, optional
            If True, squeeze array before return
        flags : tensor, optional
            flag array to index. default is self.flags
        """
        flags = self.flags if flags is None else flags
        if flags is None:
            return None

        # get indexing
        inds = self.get_inds(bl=bl, times=times, freqs=freqs, pol=pol)
        flags = flags[inds]

        # squeeze if desired
        if squeeze:
            flags = flags.squeeze()

        return flags

    def get_cov(self, bl=None, times=None, freqs=None, pol=None,
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
        squeeze : bool, optional
            If True, squeeze array before return
        cov : tensor, optional
            Cov tensor to index. default is self.cov
        """
        cov = self.cov if cov is None else cov
        if cov is None:
            return None

        # get indexing
        inds = self.get_inds(bl=bl, times=times, freqs=freqs, pol=pol)

        if self.cov_axis is None:
            # cov is same shape as data
            cov = cov[inds]
        else:
            # cov is not the same shape as data
            if bl is not None:
                if self.cov_axis == 'bl':
                    cov = cov[inds[2]][:, inds[2]]
                elif self.cov_axis in ['time', 'freq']:
                    cov = cov[:, :, :, :, inds[2]]
                elif self.cov_axis == 'full':
                    raise NotImplementedError
            elif times is not None:
                if self.cov_axis == 'time':
                    cov = cov[inds[3]][:, inds[3]]
                elif self.cov_axis == 'bl':
                    cov = cov[:, :, :, :, inds[3]]
                elif self.cov_axis == 'freq':
                    cov = cov[:, :, :, :, :, inds[3]]
                elif self.cov_axis == 'full':
                    raise NotImplementedError
            elif freqs is not None:
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
            Options are [None, 'bl', 'time', 'freq', 'full'].
            See optim.apply_icov() for details
        icov : tensor, optional
            pre-computed inverse covariance to set.
            Recommended to first set cov and then call compute_icov()
        """
        # set covariance
        self.cov = cov
        self.icov = icov
        self.cov_axis = cov_axis
        self.cov_ndim, self.cov_logdet = None, None

        if self.cov is not None:
            # compute covariance properties
            self.cov_ndim = sum(self.data.shape)
            if self.cov_axis is None:
                self.cov_logdet = torch.sum(torch.log(self.cov))
            elif self.cov_axis == 'full':
                self.cov_logdet = torch.slogdet(self.cov).logabsdet
            else:
                self.cov_logdet = 0
                for i in range(self.cov.shape[2]):
                    for j in range(self.cov.shape[3]):
                        for k in range(self.cov.shape[4]):
                            for l in range(self.cov.shape[5]):
                                self.cov_logdet += torch.slogdet(self.cov[:, :, i, j, k, l]).logabsdet

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

    def select(self, bl=None, times=None, freqs=None, pol=None):
        """
        Downselect on data tensor. Can only specify one axis at a time.
        Operates in place.

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
        atol : float, optional
            absolute tolerance for matching times or freqs
        """
        assert sum([bl is not None, times is not None, freqs is not None,
                    pol is not None]) < 2, "only one axis can be fed at a time"

        if bl is not None:
            data = self.get_data(bl, squeeze=False)
            cov = self.get_cov(bl, squeeze=False)
            flags = self.get_flags(bl, squeeze=False)
            self.setup_data(bl, self.times, self.freqs, pol=self.pol, 
                            data=data, flags=self.flags, cov=cov,
                            cov_axis=self.cov_axis, history=self.history,
                            run_check=True)

        elif times is not None:
            data = self.get_data(times=times, squeeze=False)
            cov = self.get_cov(times=times, squeeze=False)
            flags = self.get_flags(times=times, squeeze=False)
            self.setup_data(self.bls, times, self.freqs, pol=self.pol, 
                            data=data, flags=self.flags, cov=cov,
                            cov_axis=self.cov_axis, history=self.history,
                            run_check=True)

        elif freqs is not None:
            data = self.get_data(freqs=freqs, squeeze=False)
            cov = self.get_cov(freqs=freqs, squeeze=False)
            flags = self.get_flags(freqs=freqs, squeeze=False)
            self.setup_data(self.bls, self.times, freqs, pol=self.pol, 
                            data=data, flags=self.flags, cov=cov,
                            cov_axis=self.cov_axis, history=self.history,
                            run_check=True)

        elif pol is not None:
            data = self.get_data(pol=pol)
            flags = self.get_flags(pol=pol)
            cov = self.get_cov(pol=pol)
            self.setup_data(self.bls, self.times, self.freqs, pol=pol, 
                            data=data, flags=self.flags, cov=cov,
                            cov_axis=self.cov_axis, history=self.history,
                            run_check=True)

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
                            cov_axis=cov_axis, run_check=False)

            data, flags, cov, icov = None, None, None, None
            if read_data:
                data = self.get_data(bl=bl, times=times, freqs=freqs, pol=pol,
                                     squeeze=False, data=f['data'])
                data = torch.as_tensor(data)
                if 'flags' in f and not suppress_nonessential:
                    flags = self.get_flags(bl=bl, times=times, freqs=freqs, pol=pol,
                                          squeeze=False, flags=f['flags'])
                    flags = torch.as_tensor(flags)
                if 'cov' in f and not suppress_nonessential:
                    cov = self.get_cov(bl=bl, times=times, freqs=freqs, pol=pol,
                                       squeeze=False, cov=f['cov'])
                    cov = torch.as_tensor(cov)
                if 'icov' in f:
                    icov = self.get_icov(bl=bl, times=times, freqs=freqs, pol=pol,
                                         squeeze=False, icov=f['icov'])
                    icov = torch.as_tensor(icov)

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
        assert isinstance(self.telescope, telescope_model.TelescopeModel)
        assert isinstance(self.antpos, dict)
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


class MapData:
    """
    An object for holding image or map data of shape
    (Npol, Npol, Nfreqs, Npix)
    """
    def __init__(self):
        raise NotImplementedError 


class TensorData:
    """
    A shallow object for holding an arbitrary tensor
    """
    ## TODO: Make this base class for VisData and MapData
    def __init__(self):
        self.data, self.flags = None, None
        self.icov, self.cov, self.cov_axis = None, None, None

    def setup_data(self, data=None, flags=None, cov=None,
                   cov_axis=None, icov=None, history=''):
        """
        Setup data tensors.

        Parameters
        ----------
        data : tensor, optional
            Data tensor of arbitrary shape
        flags : tensor, optional
            Boolean tensor holding data flags
        cov : tensor, optional
            Covariance of data tensor
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
            Options are [None, 'full', 'bl', 'time', 'freq'].
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
                cov_logdet = 0
                if cov.ndim == 2:
                    # if cov.ndim == 2 then compute logdet and return
                    return torch.slogdet(cov).logabsdet
                else:
                    # otherwise iterate over last axis and get logdet
                    for i in range(cov.shape[-1]):
                        cov_logdet += self.set_cov(cov[..., i], cov_axis)

        # set covariance
        self.cov = cov
        self.icov = icov
        self.cov_axis = cov_axis
        self.cov_ndim = sum(self.data.shape)
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


class Dataset(Dataset):
    """
    Dataset iterator for VisData, MapData, or TensorData
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
        if isinstance(data, (str, VisData, MapData)):
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
    Nvd = len(vds)
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
                   cov_axis=vd.cov_axis, history=vd.history,
                   run_check=run_check)

    return out


def concat_MapData(mds, axis, run_check=True):
    """
    Concatenate MapData objects
    """
    raise NotImplementedError


def load_data(fname, concat_ax=None, copy=False, **kwargs):
    """
    Load a VisData or MapData HDF5 file(s)

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
    VisData or MapData object
    """
    if isinstance(fname, (list, tuple)):
        dlist = [load_data(fn, **kwargs) for fn in fname]
        if concat_ax is not None:
            if isinstance(dlist[0], VisData):
                data = concat_VisData(dlist, concat_ax)
            else:
                data = concat_MapData(dlist, concat_ax)
        else:
            data = dlist

    elif isinstance(fname, (VisData, MapData)):
        data = fname

    else:
        with h5py.File(fname) as f:
            obj = f.attrs['obj']

        if obj == 'VisData':
            data = VisData()
            data.read_hdf5(fname, **kwargs)
        else:
            data = MapData()
            data.read_hdf5(fname, **kwargs)

    if copy:
        data = copy.deepcopy(data)

    return data


def pass_data(fname, copy=False, **kwargs):
    """Dummy load function. Use this when storing data in memory,
    rather than performing dynamic IO"""
    if copy:
        return deepcopy(fname)
    else:
        return fname


def _list2slice(inds):
    """convert list indexing to slice if possible"""
    if isinstance(inds, list):
        diff = list(set(np.diff(inds)))
        if len(diff) == 1:
            return slice(inds[0], inds[-1]+diff[0], diff[0])
    return inds

