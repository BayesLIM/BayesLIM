"""
Module for visibility data and other formats
"""
import torch
import numpy as np
import os
import copy

from . import utils, telescope_model
from .utils import _float, _cfloat


class VisData:
    """
    An object for holding visibility data of shape
    (Npol, Npol, Nbl, Ntimes, Nfreqs)
    """
    def __init__(self):
        self.data, self.flags = None, None
        self.icov, self.cov, self.cov_axis = None, None, None
        self.bls, self.time, self.times = None, None, None
        self.freqs, self.pol, self.history = None, None, ''
        self.telescope, self.array = None, None
        self.atol = 1e-5

    def copy(self):
        """
        Copy and return self. Data tensor
        is detached and cloned.
        """
        vd = VisData()
        vd.setup_telescope(telescope=self.telescope, array=self.array)
        vd.setup_data(self.bls, self.times, self.freqs, pol=self.pol,
                      time=self.time, data=self.data.detach().clone(),
                      flags=self.flags, cov=self.cov, icov=self.icov,
                      cov_axis=self.cov_axis, history=self.history)
        return vd

    def setup_telescope(self, telescope=None, array=None):
        """
        Set the telescope and array models

        Parameters
        ----------
        telescope : TelescopeModel, optional
            Telescope location
        array : ArrayModel, optional
            Array layout model
        """
        self.telescope = telescope
        self.array = array

    def setup_data(self, bls, times, freqs, pol=None, time=None,
                   data=None, flags=None, cov=None, icov=None, cov_axis=None,
                   history='', run_check=True):
        """
        Setup metadata and optionally data tensors.

        Parameters
        ----------
        bls : list
            List of baseline tuples (antpairs) matching
            ordering of Nbl axis of data.
        times : tensor
            Julian date of unique times in the data,
            relative to time.
        freqs : tensor
            Frequency array [Hz] of the data
        pol : str, optional
            If data Npol == 1, this is the name
            of the dipole polarization, one of ['ee', 'nn'].
            If None, it is assumed Npol == 2
        time : float, optional
            Anchor point for times (reduces neccesary precision)
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
        icov : tensor, optional
            Inverse covariance. must match shape of cov.
        cov_axis : str, optional
            If cov represents on and off diagonal components, this is the
            axis over which off-diagonal is modeled.
            One of ['bl', 'time', 'freq'].
        history : str, optional
            data history string
        """
        self.bls = bls
        self.Nbls = len(bls)
        self.ant1 = [bl[0] for bl in bls]
        self.ant2 = [bl[1] for bl in bls]
        if time is None:
            # try to get it from times
            time = int(times[0])
            times = torch.tensor(times - time, dtype=_float())
        self.times = times
        self.time = time
        self.Ntimes = len(times)
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        self.pol = pol
        if isinstance(pol, str):
            assert pol.lower() in ['ee', 'nn'], "pol must be 'ee' or 'nn' for 1pol mode"
        self.Npol = 2 if self.pol is None else 1
        self.data = data
        self.flags = flags
        self.cov = cov
        self.icov = icov
        self.cov_axis = cov_axis
        self.history = history

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
        return self.bls.index(bl[:2])

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
        return np.where(np.isclose(self.times, time, atol=self.atol))[0].tolist()

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
        assert isinstance(pol, str)
        if self.pol is not None:
            if pol.lower() != self.pol:
                raise ValueError("cannot index pol from 1pol {}".format(self.pol))
            return 0
        if pol.lower() == 'ee':
            return 0
        elif pol.lower() == 'nn':
            return 1
        else:
            raise ValueError("cannot index cross-pols")

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
        if bl is not None:
            inds = self._bl2ind(bl)
            data = data[:, :, inds]
        elif times is not None:
            inds = self._time2ind(times)
            data = data[:, :, :, inds]
        elif freqs is not None:
            inds = self._freq2ind(freqs)
            data = data[:, :, :, :, inds]
        elif pol is not None:
            ind = self._pol2ind(pol)
            data = data[ind:ind+1, ind:ind+1]
        else:
            data = data[:]
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
        if bl is not None:
            inds = self._bl2ind(bl)
            flags = flags[:, :, inds]
        elif times is not None:
            inds = self._time2ind(times)
            flags = flags[:, :, :, inds]
        elif freqs is not None:
            inds = self._freq2ind(freqs)
            flags = flags[:, :, :, :, inds]
        elif pol is not None:
            ind = self._pol2ind(pol)
            flags = flags[ind:ind+1, ind:ind+1]
        else:
            flags = flags[:]
        if squeeze:
            flags = flags.squeeze()
        return flags

    def get_cov(self, bl=None, times=None, freqs=None, pol=None,
                squeeze=True, cov=None):
        """
        Slice into cov tensor and return values.
        Only one axis can be specified at a time.
        See optim.apply_cov() for details on shape.

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
        if bl is not None:
            inds = self._bl2ind(bl)
            if self.cov_axis == 'bl':
                cov = cov[inds][:, inds]
            elif self.cov_axis in ['time', 'freq']:
                cov = cov[:, :, :, :, inds]
            else:
                raise NotImplementedError
        elif times is not None:
            inds = self._time2ind(times)
            if self.cov_axis == 'time':
                cov = cov[inds][:, inds]
            elif self.cov_axis == 'bl':
                cov = cov[:, :, :, :, inds]
            elif self.cov_axis == 'freq':
                cov = cov[:, :, :, :, :, inds]
            else:
                raise NotImplementedError
        elif freqs is not None:
            inds = self._freq2ind(freqs)
            if self.cov_axis == 'freq':
                cov = cov[inds][:, inds]
            elif self.cov_axis in ['bl', 'time']:
                cov = cov[:, :, :, :, :, inds]
            else:
                raise NotImplementedError
        elif pol is not None:
            ind = self._pol2ind(pol)
            cov = cov[:, :, ind:ind+1, ind:ind+1]
        else:
            cov = cov[:]
        if squeeze:
            cov = cov.squeeze()
        return cov

    def set_icov(self, icov=None, pinv=True, rcond=1e-15):
        """
        Set inverse covariance as self.icov.
        See optim.apply_cov() for details on shape.

        Parameters
        ----------
        icov : tensor, optional
            Pre-computed inverse covariance to set. Must
            match shape of self.cov. Default is to compute
            icov given self.cov
        pinv : bool, optional
            Use pseudo-inverse to compute covariance,
            otherwise use direct inversion
        rcond : float, optional
            rcond kwarg for pinverse
        """
        if icov is None:
            icov = optim.compute_icov(self.cov, self.cov_axis, pinv=pinv, rcond=rcond)
        self.icov = icov

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

    def select(self, bls=None, times=None, freqs=None, pol=None):
        """
        Downselect on data tensor. Can only specify one axis at a time.
        Operates in place.

        Parameters
        ----------
        bls : list, optional
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
        assert sum([bls is not None, times is not None, freqs is not None,
                    pol is not None]) < 2, "only one axis can be fed at a time"

        if bls is not None:
            data = self.get_data(bls, squeeze=False)
            cov = self.get_cov(bls, squeeze=False)
            flags = self.get_flags(bls, squeeze=False)
            self.setup_data(bls, self.times, self.freqs, pol=self.pol, 
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
                f.create_dataset('bls', data=self.bls)
                f.create_dataset('times', data=self.times)
                f.create_dataset('freqs', data=self.freqs)
                if self.pol is not None:
                    f.attrs['pol'] = self.pol
                f.attrs['history'] = self.history
                # write telescope and array objects
                f.attrs['tloc'] = self.telescope.location
                f.attrs['ants'] = self.array.ants
                f.attrs['antpos'] = self.array.antpos
                f.attrs['time'] = self.time

    def read_hdf5(self, fname, read_data=True, bls=None, times=None, freqs=None, pol=None):
        """
        Read HDF5

        Parameters
        ----------
        fname : str
            File to read
        read_data : bool, optional
            If True, read data arrays as well as metadata
        bls, times, freqs, pol : read options. see select for details
        """
        import h5py
        with h5py.File(fname, 'r') as f:
            # load metadata
            _bls = [tuple(bl) for bl in f['bls'][:]]
            _times = f['times'][:]
            _freqs = f['freqs'][:]
            _pol = f.attrs['pol'] if 'pol' in f.attrs else None
            time = f.attrs['time']
            cov_axis = f.attrs['cov_axis'] if 'cov_axis' in f.attrs else None
            history = f.attrs['history'] if 'history' in f.attrs else ''


            # setup just full-size metadata
            self.setup_data(_bls, _times, _freqs, pol=_pol, time=time,
                            cov_axis=cov_axis, run_check=False)

            data, flags, cov = None, None, None
            if read_data:
                data = self.get_data(bl=bls, times=times, freqs=freqs, pol=pol,
                                     squeeze=False, data=f['data'])
                if 'flags' in f:
                    flags = self.get_flags(bl=bls, times=times, freqs=freqs, pol=pol,
                                          squeeze=False, flags=f['flags'])
                if 'cov' in f:
                    cov = self.get_cov(bl=bls, times=times, freqs=freqs, pol=pol,
                                       squeeze=False, cov=f['cov'])

            # downselect metadata to selection
            self.select(bls=bls, times=times, freqs=freqs, pol=pol)

            # setup downselected metadata and data
            ants = f.attrs['ants'].tolist()
            antpos = f.attrs['antpos']
            antpos_d = dict(zip(ants, antpos))
            tloc = f.attrs['tloc']
            telescope = telescope_model.TelescopeModel(tloc)
            array = telescope_model.ArrayModel(antpos_d, self.freqs)
            self.setup_telescope(telescope, array)
            self.setup_data(self.bls, self.times, self.freqs, pol=self.pol,
                            data=data, flags=flags, cov=cov, time=time,
                            cov_axis=cov_axis, history=history)

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
        # load uvdata
        if isinstance(fname, str):
            uvd = UVData()
            uvd.read_uvh5(fname, **kwargs)
        elif isinstance(fname, UVData):
            uvd = fname
        # extract data and metadata
        bls = uvd.get_antpairs()
        Nbls = uvd.Nbls
        times = np.unique(uvd.time_array)
        time = int(times[0])
        times = torch.tensor(times - time, dtype=_float())
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
        self.setup_data(bls, times, freqs, pol=pol, time=time,
                        data=data, flags=flags, history=uvd.history)
        # configure telescope data
        antpos, ants = uvd.get_ENU_antpos()
        antpos = dict(zip(ants, antpos))
        loc = uvd.telescope_location_lat_lon_alt_degrees
        telescope = telescope_model.TelescopeModel((loc[1], loc[0], loc[2]))
        arr = telescope_model.ArrayModel(antpos, freqs)
        self.setup_telescope(telescope, arr)

        if run_check:
            self.check()

    def check(self):
        """
        Run checks on data
        """
        assert isinstance(self.telescope, telescope_model.TelescopeModel)
        assert isinstance(self.array, telescope_model.ArrayModel)
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
            assert (ant1 in self.array.ants) and (ant2 in self.array.ants)


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
        times = torch.cat([o.times for o in vds])
        freqs = vd.freqs
        pol = vd.pol
        bls = vd.bls
        if vd.cov is not None:
            raise NotImplementedError

    elif axis == 'freq':
        dim = 4
        times = vd.times
        freqs = torch.cat([o.freqs for o in vds])
        pol = vd.pol
        bls = vd.bls
        if vd.cov is not None:
            raise NotImplementedError

    data = torch.cat([o.data for o in vds], dim=dim)
    if vd.flags is not None:
        flags = torch.cat([o.flags for o in vds], dim=dim)

    out.setup_telescope(vd.telescope, vd.array)
    out.setup_data(bls, times, freqs, pol=pol,
                   data=data, flags=flags, cov=cov,
                   cov_axis=vd.cov_axis, history=vd.history,
                   run_check=run_check)

    return out


class MapData:
    """
    An object for holding image or map data of shape
    (Npol, Npol, Npix, Nfreqs)
    """
    def __init__(self):
        raise NotImplementedError 
