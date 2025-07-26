"""
Visibility imaging module
"""
import torch
import numpy as np

from . import telescope_model, utils
from .utils import _float, _cfloat
from .dataset import VisData, MapData


class VisMapper:
	"""
	A class for producing images from interferometric
	visibilities in VisData format.

	The complex visibilities (y) are related
	to the pixelized sky (x) via the linear model

	y = A x

	The dirty map is produced by

	m = D A^T W y

	where W are visibility weights and
	D is a diagonal normalization.

	Deconvolution is performed as

	dm = P^-1 m

	where P can take multiple values but is in the general case

	P = (A^T W A)

	Notes
	-----
	- Currently only supports single-pol imaging
	"""
	def __init__(self, vis, ra, dec, beam=None, fov=180, dtype=None, cache_A=False, **kwargs):
		"""
		Parameters
		----------
		vis : VisData object
			Contains metadata for self.build_A(), and visibility
			data for self.build_v()
		ra : array
			Right ascension [deg] of map pixels (Npix,)
		dec : array
			Declination [deg] of map pixels (Npix,)
		beam : PixelBeam object, optional
			Include beam in A matrix when mapping
		fov : int, optional
			fov parameter if beam is None, otherwise
			use beam.fov value
		dtype : torch.dtype
			Use torch.float32 or torch.float64 when building
			imaging matrices. Default (None) is to use
			torch.get_default_dtype().
		cache_A : bool, optional
			If True, save the A matrices in self.A
		kwargs : additional kwargs for ArrayModel()
		"""
		## TODO: add on-the-fly vis loading
		self.vis = vis
		self.telescope = vis.telescope
		self.array = telescope_model.ArrayModel(
			vis.antpos,
			vis.freqs,
			device=vis.data.device,
			skip_reds=True,
			**kwargs
		)
		self.ra = ra
		self.dec = dec
		self.Npix = len(ra)

		self.device = vis.data.device
		self.dtype = dtype

		self.beam = beam
		self.fov = beam.fov if beam is not None else fov

		# set freq indices
		self._freqs = vis.freqs
		self.set_freq_inds()

		# set time indices
		self._times = np.asarray(vis.times.cpu())  # this must be numpy array for caching purposes
		self.set_time_inds()

		# set bl indices
		self._blnums = vis.blnums
		self.set_bl_inds()

		self.cache_A = cache_A
		self.clear_cache()

		self.set_normalization()

	def clear_cache(self):
		self.A = {}
		self.D = None
 
	def set_freq_inds(self, freq_inds=None, freqs=None):
		"""
		Set frequency indexing

		Parameters
		----------
		freq_inds : tensor
			Freq indices of observations to image
		freqs : tensor
			Frequency bins [Hz] of observations to image
		"""
		assert not ((freqs is not None) and (freq_inds is not None))

		# get freq_inds
		fidx = lambda f: torch.where(torch.isclose(self._freqs, f, atol=1e-10))[0]
		if freqs is not None:
			iterable = False
			if isinstance(freqs, (list, np.ndarray)):
				iterable = True
			elif isinstance(freqs, torch.Tensor):
				if freqs.ndim == 1:
					iterable = True
			if iterable:
				freq_inds = torch.stack([fidx(f) for f in freqs]).tolist()
			else:
				freq_inds = fidx(freqs).tolist()

		if freq_inds is None:
			freq_inds = slice(None)

		self.freq_inds = utils._list2slice(freq_inds)
		self.freqs = self._freqs[self.freq_inds]
		self.Nfreqs = len(self.freqs)
		self.clear_cache()

	def set_time_inds(self, time_inds=None, times=None):
		"""
		Set times indexing. This works slightly different
		than freq_inds or bl_inds, b/c there are two sets
		of time_inds. self.time_inds tracks the indices
		of self.vis.times that we want to image. Whereas
		the "time_ind" in self.build_A() and self.build_v()
		and in the cache self.A tracks the indices of
		self.vis.times[self.time_inds].

		Parameters
		----------
		time_inds : tensor
			Time indices of observations to image
		times : tensor
			Observation times [Julian Date] of observations to image
		"""
		assert not ((times is not None) and (time_inds is not None))

		# get time_inds
		tidx = lambda t: np.where(np.isclose(self._times, t, atol=1e-10, rtol=1e-13))[0]
		if times is not None:
			iterable = False
			if isinstance(times, list):
				iterable = True
			elif isinstance(times, (torch.Tensor, np.ndarray)):
				if times.ndim == 1:
					iterable = True
			if iterable:
				time_inds = np.concatenate([tidx(t) for t in times]).tolist()
			else:
				time_inds = tidx(times).tolist()
			assert len(time_inds) == len(times)

		# make sure time_inds is a list
		if time_inds is None:
			time_inds = list(range(len(self._times)))
		elif isinstance(time_inds, slice):
			time_inds = utils._slice2tensor(time_inds).tolist()
		elif isinstance(time_inds, (np.ndarray, torch.Tensor)):
			if time_inds.ndim == 0:
				time_inds = [time_inds.tolist()]
			else:
				time_inds = time_inds.tolist()
		elif isinstance(time_inds, (int, np.integer)):
			time_inds = [time_inds]

		self.time_inds = time_inds
		self.times = self._times[time_inds]
		self.Ntimes = len(self.times)
		self.clear_cache()

	def set_bl_inds(self, bl_inds=None, blnums=None):
		"""
		Set baseline indexing

		Parameters
		----------
		bl_inds : tensor
			Baseline indices of observations to image
		blnums : tensor
			Baseline blnums to image
		"""
		assert not ((blnums is not None) and (bl_inds is not None))

		# get time_inds
		blidx = lambda bl: np.where(self._blnums == bl)[0]
		if blnums is not None:
			iterable = False
			if isinstance(blnums, list):
				iterable = True
			elif isinstance(times, (torch.Tensor, np.ndarray)):
				if blnums.ndim == 1:
					iterable = True
			if iterable:
				bl_inds = np.concatenate([blidx(bl) for bl in blnums]).tolist()
			else:
				bl_inds = blidx(blnums).tolist()

		if bl_inds is None:
			bl_inds = slice(None)

		self.bl_inds = utils._list2slice(bl_inds)
		self.blnums = self._blnums[self.bl_inds]
		self.Nbls = len(self.blnums)
		self.blvecs = self.array.get_blvecs(utils.blnum2ants(self.blnums))
		self.clear_cache()

	def set_normalization(self, method='A2w', icov=None, clip=1e-8):
		"""
		Set the normalization method.

		Parameters
		----------
		method : str, optional
			Choice of dirty map, diagonal normalization:
				'w'   : D = 1 / 1 @ w
				'Aw'  : D = 1 / w @ |A|
				'A2w' : D = 1 / w @ |A|^2]
			Default ('A2w') is standard least squares.
		icov : tensor, optional
			Visibility weights to use instead of self.vis.icov,
			must match self.vis.icov shape.
		"""
		assert method in ['w', 'Aw', 'A2w']
		self.method = method
		self.icov = icov
		self.D = None
		self.clip = clip

	@torch.no_grad()
	def build_A(self, time):
		"""
		Build the A matrix for a single observing time.

		Parameters
		----------
		time : float
			Julian date observing time.
			Must be a float or numpy float for caching
			purposes.

		Returns
		-------
		A : tensor
			Imaging matrix (conjugated RIME matrix)
			of shape (Nbls, Nfreqs, Npix)
		cut : tensor
			Indexing tensor along Npix dimension
			of beam.fov or self.fov.
		"""
		# A = (N_bls_times, Nfreqs, Npix)
		# get zen, az
		zen, az = self.telescope.eq2top(time, self.ra, self.dec, store=True)

		# get beam and cut
		if self.beam is not None:
			beam, cut, zen, az = self.beam.gen_beam(zen, az)
			beam = beam[:, :, :, self.freq_inds].to(self.device)
			# only single pol imaging with antenna-independent beam for now
			beam = beam[0, 0, :1]
			if not self.beam.powerbeam:
				beam = beam**2
		else:
			beam = None
			cut = torch.where(zen <= self.fov/2)[0]
			zen, az = zen[cut], az[cut]

		# get conjugate of fringe (for vis simulation we use fr, for mapping we use fr.conj)
		self.array.set_freq_index(self.freq_inds)
		A = self.array.gen_fringe(self.blvecs, zen, az, conj=True)

		# multiply in beam
		if beam is not None:
			A *= beam

		return A, cut

	@torch.no_grad()
	def build_v(self, time_ind, vis=None):
		"""
		Parameters
		----------
		time_ind : int
			Index of self.times to build visibility vector
		vis : VisData or list
			VisData to use instead of self.vis

		Returns
		-------
		v : tensor
			Visibility vector of shape (..., Nbls, Nfreqs)
		"""
		vis = self.vis if vis is None else vis
		time_ind = self.time_inds[time_ind]
		if isinstance(vis, list):
			# (Nvis, Nbls, Nfreqs)
			v = torch.stack([
			_vis.get_data(
				bl_inds=self.bl_inds,
				time_inds=time_ind,
				freq_inds=self.freq_inds,
				squeeze=False,
				try_view=True
			)[0, 0, :, 0] for _vis in vis
			])
		else:
			# (Nbls, Nfreqs)
			v = vis.get_data(
				bl_inds=self.bl_inds,
				time_inds=time_ind,
				freq_inds=self.freq_inds,
				squeeze=False,
				try_view=True
			)[0, 0, :, 0]

		return v

	def build_w(self, time_ind):
		"""
		Build weight tensor. First check self.icov,
		if None check self.vis.icov, if None use 1.

		Parameters
		----------
		time_ind : int
			Index of self.times to build visibility vector

		Returns
		-------
		w : tensor
			Visibility weights of shape (Nbls, Nfreqs)
		"""
		icov = self.icov if self.icov is not None else self.vis.icov
		time_ind = self.time_inds[time_ind]

		# get weights
		if icov is not None:
			w = self.vis.get_icov(
				bl_inds=self.bl_inds,
				time_inds=time_ind,
				icov=icov,
				freq_inds=self.freq_inds,
				squeeze=False
			)[0, 0, :, 0]
		else:
			w = torch.ones(self.Nbls, 1, device=self.device)

		return w

	def make_map(self, vis=None, return_P=True, diag=True):
		"""
		Make maps for each time integration, sum them, and normalize them.

		Parameters
		----------
		vis : VisData or list
			Visibility data to image instead of self.vis.
			Must match self.vis shape and metadata.
			Can also pass a list of VisData to image (but
			with identical weights)
		return_P : bool, optional
			If True, compute and return PSF matrix.
		diag : bool, optional
			If return_P, only compute the diagonal
			component.

		Returns
		-------
		maps : tensor
			Dirty maps of shape (..., Nfreqs, Npix)
		P : PSF matrix
			Only if compute_P = True, otherwise None
		"""
		assert self.method is not None, "First run set_normalization()"
		vis = self.vis if vis is None else vis
		Nmaps = 1
		if isinstance(vis, list):
			# feeding list of visdata for multiple maps
			Nmaps = len(vis)

		# init maps
		maps = torch.zeros(Nmaps, self.Nfreqs, self.Npix, device=self.device)
		if isinstance(vis, VisData):
			# get rid of Nmaps dim
			maps = maps[0]

		# init weights
		if self.method == 'w':
			Aw = torch.zeros(self.Nfreqs, 1, device=self.device)
		elif self.method in ['Aw', 'A2w']:
			Aw = torch.zeros(self.Nfreqs, self.Npix, device=self.device)

		# init the P matrix
		P = None
		if return_P:
			if diag:
				P = torch.zeros(self.Nfreqs, self.Npix, device=self.device)
			else:
				P = torch.zeros(self.Nfreqs, self.Npix, self.Npix, device=self.device)

		# iterate over times
		for i, time in enumerate(self.times):
			# build A
			if i in self.A:
				A, cut = self.A[i]
			else:
				A, cut = self.build_A(time)
				if self.cache_A:
					self.A[i] = (A, cut)

			# build v, w
			v = self.build_v(i, vis=vis)
			w = self.build_w(i)

			# make map
			m = make_map(v, w, A)

			if return_P:
				# get P for this patch of sky
				_P = compute_P(A, w, diag=diag)

				# insert into P
				if diag:
					P[:, cut] += _P
				else:
					P[:, cut[:, None], cut[None, :]] += _P

			# sum with tensors
			if self.method == 'w':
				Aw += w.sum(0)[:, None]
			elif self.method == 'Aw':
				Aw[..., cut] += (w[:, :, None] * A.abs()).sum(0)
			elif self.method == 'A2w':
				Aw[..., cut] += (w[:, :, None] * A.pow(2).real).sum(0)

			maps[..., cut] += m

		# get normalization
		self.D = 1 / Aw.clip(self.clip)

		# apply normalization
		maps *= self.D

		if return_P:
			# apply normalization
			if diag:
				# (Nfreqs, Npix) * (Nfreqs, Npix)
				P *= self.D

			else:
				# (Nfreqs, Npix, Npix) * (Nfreqs, Npix, 1)
				P *= self.D[:, :, None]

		return maps, P

	def compute_Pm(self, maps, D=None):
		"""
		Compute the matrix vector product of the
		PSF matrix and a set of maps.

		Parameters
		----------
		maps : tensor, VisData, list
			If provided compute the matrix-vector product P @ maps
			instead of the full P matrix.
		D : tensor, optional
			Pre-computed diagonal normalization tensor
			of shape (Nfreqs, Npix).

		Returns
		-------
		Pm : tensor
			Set of PSF-convolved maps of shape
			(Nmaps, Nfreqs, Npix)
		"""
		# get map data
		map2ten = lambda m: m.get_data() if isinstance(m, MapData) else m
		if isinstance(maps, list):
			maps = torch.stack([map2ten(_m) for _m in maps])
		elif isinstance(maps, VisData):
			maps = map2ten(maps)

		# init output tensor
		if maps.ndim == 3:
			Nmaps = len(maps)
			shape = (Nmaps, self.Nfreqs, self.Npix)
		else:
			Nmaps = 1
			shape = (self.Nfreqs, self.Npix)

		Pm = torch.zeros(shape, device=self.device)

		# init summed weights
		if D is None:
			# init weights
			if self.method == 'w':
				Aw = torch.zeros(self.Nfreqs, 1, device=self.device)
			elif self.method in ['Aw', 'A2w']:
				Aw = torch.zeros(self.Nfreqs, self.Npix, device=self.device)

		# iterate over time integrations
		for i, time in enumerate(self.times):
			# build A
			if i in self.A:
				A, cut = self.A[i]
			else:
				A, cut = self.build_A(time)
				if self.cache_A:
					self.A[i] = (A, cut)	                	

			# build w
			w = self.build_w(i)

			# get Pm for this patch of sky
			m = maps[..., cut]
			_Pm = compute_Pm(A, w, m)

			# insert into Pm
			Pm[..., cut] += _Pm

			if D is None:
				# sum with weights
				if self.method == 'w':
					Aw += w.sum(0)[:, None]
				elif self.method == 'Aw':
					Aw[..., cut] += (w[:, :, None] * A.abs()).sum(0)
				elif self.method == 'A2w':
					Aw[..., cut] += (w[:, :, None] * A.abs().pow(2)).sum(0)

		if D is None:
			# get normalization
			D = 1 / Aw.clip(self.clip)

		# apply normalization
		# (Nmaps, Nfreqs, Npix) * (Nfreqs, Npix)
		Pm *= D

		return Pm

	def compute_P(self, D=None, diag=True):
		"""
		Compute the full P matrix across all sky pixels, iterating
		over time integrations and summing patch P matrices.

		Parameters
		----------
		D : tensor, optional
			Pre-computed diagonal normalization tensor
			of shape (Nfreqs, Npix).
		diag : bool, optional
			If True, only compute the diagonal of P.

		Returns
		-------
		P : tensor
			PSF matrix of shape (Nfreqs, Npix, [Npix]) 
		"""
		# init the P matrix
		if diag:
			P = torch.zeros(self.Nfreqs, self.Npix, device=self.device)
		else:
			P = torch.zeros(self.Nfreqs, self.Npix, self.Npix, device=self.device)
		
		# init summed weights
		if D is None:
			# init weights
			if self.method == 'w':
				Aw = torch.zeros(self.Nfreqs, 1, device=self.device)
			elif self.method in ['Aw', 'A2w']:
				Aw = torch.zeros(self.Nfreqs, self.Npix, device=self.device)

		# iterate over time integrations
		for i, time in enumerate(self.times):
			# build A
			if i in self.A:
				A, cut = self.A[i]
			else:
				A, cut = self.build_A(time)
				if self.cache_A:
					self.A[i] = (A, cut)	                	

			# build w
			w = self.build_w(i)

			# get P for this patch of sky
			_P = compute_P(A, w, diag=diag)

			# insert into P
			if diag:
				P[:, cut] += _P
			else:
				P[:, cut[:, None], cut[None, :]] += _P

			if D is None:
				# sum with weights
				if self.method == 'w':
					Aw += w.sum(0)[:, None]
				elif self.method == 'Aw':
					Aw[..., cut] += (w[:, :, None] * A.abs()).sum(0)
				elif self.method == 'A2w':
					Aw[..., cut] += (w[:, :, None] * A.abs().pow(2)).sum(0)

		if D is None:
			# get normalization
			D = 1 / Aw.clip(self.clip)

		# apply normalization
		if diag:
			# (Nfreqs, Npix) * (Nfreqs, Npix)
			P *= D

		else:
			# (Nfreqs, Npix, Npix) * (Nfreqs, Npix, 1)
			P *= D[:, :, None]

		return P

	def push(self, device):
		"""
		Push objects, including Modules attached to self,
		to a new device or dtype
		"""
		dtype = isinstance(device, torch.dtype)
		if self.A is not None:
			for k, v in self.A.items():
				self.A[k] = (
					utils.push(v[0], device),
					utils.push(v[1], device)
					)
		if self.D is not None:
			self.D = utils.push(self.D, device)
		if not dtype:
			self.device = device
		if self.beam is not None:
			self.beam.push(device)
		self.array.push(device)
		self.vis.push(device)
		self.telescope.push(device)
		self.blvecs = utils.push(self.blvecs, device)


def make_map(v, w, A):
	"""
	Make a map given the visibility tensor, weight tensor, and A matrix

	Parameters
	----------
	v : tensor
		Visibility tensor of shape (..., Nbls, Nfreqs)
	w : tensor
		Visibility weight tensor of shape (Nbls, Nfreqs)
	A : tensor
		Imaging matrix of shape (Nbls, Nfreqs, Npix).
		See VisMapper.build_A()

	Returns
	-------
	m : tensor
		Map tensor of shape (..., Nfreqs, Npix)
	"""
	return torch.einsum('vfp,...vf->...fp', A, v * w).real


def deconvolve_map(m, P, pinv=True, rcond=1e-15, hermitian=True):
	"""
	Deconvolve a dirty map (currently experimental)
	"""
	### experimental
	if pinv:
		Pinv = torch.linalg.pinv(P, rcond=rcond, hermitian=hermitian)
	else:
		Pinv = torch.zeros_like(P)
		Pinv[..., range(P.shape[1]), range(P.shape[1])] = 1/torch.diagonal(P, dim1=1, dim2=2)

	dm = torch.einsum("ijk,ik->ij", Pinv, m)

	return dm


def compute_Pm(A, w, m, D=None):
	"""
	Compute the matrix vector product P @ m
	where P is the PSF matrix and m are set of maps.

		P m = D A^T w (A m)

	where D is a diagonal matrix.

	Parameters
	----------
	A : tensor
		Imaging A matrix of shape (Nbls, Nfreqs, Npix)
	w : tensor
		Visibility weight vector of shape (Nbls, Nfreqs)
	m : tensor
		Maps to use in P @ m product, shape (Nmaps, Nfreqs, Npix)
	D : tensor, optional
		Normalization tensor of shape (Nfreqs, Npix)

	Returns
	-------
	Pm : tensor
		The P @ m product of shape (Nmaps, Nfreqs, Npix)
	"""
	# compute matrix-vector product: P @ m
	# compute w * (A @ m): (Nmaps, Nbls, Nfreqs)
	if m.dtype != A.dtype:
		m = m.to(A.dtype)
	wAm = w * torch.einsum("vfp,...fp->...vf", A, m)

	# multiply by A^T: (Nmaps, Nfreqs, Npix)
	Pm = torch.einsum("vfp,...vf->...fp", A.conj(), wAm).real

	# normalize: (Nmaps, Nfreqs, Npix)
	if D is not None:
		Pm *= D

	return Pm.real


def compute_P(A, w, D=None, diag=True):
	"""
	Compute the PSF matrix given the A matrix,
	a weights vector and a pre-computed normalization.

		P = D A^T w A

	Warning: this tensor can get REALLY BIG.

	Parameters
	----------
	A : tensor
		Imaging A matrix of shape (Nbls, Nfreqs, Npix)
	w : tensor
		Visibility weight vector of shape (Nbls, Nfreqs)
	D : tensor, optional
		Normalization tensor of shape (Nfreqs, Npix)
	diag : bool, optional
		If True, only compute the diagonal of P.

	Returns
	-------
	P : tensor
		The PSF matrix of shape (Nfreqs, Npix, [Npix]),
	"""
	if diag:
		# compute diagonal
		P = (w[..., None] * A.abs().pow(2)).sum(0).real

	else:
		# compute full matrix
		P = torch.einsum("vfp,vfq->fpq", A.conj(), w[..., None] * A).real

	# normalize it
	if D is not None:
		if diag:
			P *= D
		else:
			P *= D[:, :, None]

	return P.real


def VisData2MapData(vd, data=None, angs=None, cov=None, icov=None,
					cov_axis=None, norm=None, df=None, name=None):
	"""
	Initialize a MapData
	object from a VisData object

	Parameters
	----------
	vd : VisData
		Visibility data that has been imaged.
		Assumes that all frequencies have been imaged.
	data : tensor, optional
		Image data to put into MapData object, of shape
		(Npols, 1, Nfreqs, Npix)
	angs : tensor or ndarray, optional
		RA & Dec positions of the pixelized sky, of shape
		(2, Npix) with (ra, dec) respectively.
	cov : tensor, optional
		Covariance of map pixels.
	icov : tensor, optional
		Inverse covariance of map pixels.
	cov_axis : str, optional
		Type of covariance, if provided.
	norm : tensor, optional
		Map normalization, shape (Nfreqs, Npix)
	df : tensor, optional
		Channel width of each frequency bin [Hz].
	name : str, optional
	"""
	# initialize
	md = MapData()
	md.setup_meta(name=name)

	# fill w/ metadata
	if vd.pol is None:
		pols = ['ee', 'nn']
	else:
		pols = [vd.pol]

	if angs is not None:
		Npix = angs.shape[1]
	else:
		Npix = 1

	# get flags
	flags = None
	if vd.flags is not None:
		# data are flagged if all baselines and times
		# are flagged for each freq channel
		flags = vd.flags.all(dim=(2,3))
		# (Npol, 1, Nfreqs, Npix)
		flags = flags.expand(flags.shape + (Npix,))

	md.setup_data(
		vd.freqs,
		df=df,
		data=data,
		pols=pols,
		angs=angs,
		flags=flags,
		cov=cov,
		icov=icov,
		cov_axis=cov_axis,
		norm=norm,
	)

	return md
