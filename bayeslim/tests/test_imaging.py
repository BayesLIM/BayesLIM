import numpy as np
import torch
import time
import healpy
from tempfile import TemporaryDirectory

torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

from test_dataset import setup_VisData

freqs = torch.linspace(120e6, 130e6, 2)


def setup_VisMapper(vd, fov=60, nside=32, cache_A=True):
	# setup HERA stripe mapping object
	theta, phi = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
	s = (abs(theta - (90+30.72148)*np.pi/180) < (20*np.pi/180)) & (phi < 110*np.pi/180)

	# airy beam
	R = ba.beam_model.AiryResponse(freq_ratio=1.0)
	p0 = torch.ones(1, 1, 1, len(freqs), 2) * torch.tensor([11., 11.])
	beam = ba.beam_model.PixelBeam(p0, freqs, ant2beam=None, R=R, pol='e', powerbeam=True, fov=fov, parameter=False)

	# init mapper
	angs = torch.as_tensor(np.array([phi[s]/ba.D2R - 15, 90 - theta[s] / ba.D2R]))
	VM = ba.imaging.VisMapper(vd, *angs, beam=beam, cache_A=cache_A)

	return VM


def test_imaging():
	# setup vd
	times = torch.as_tensor(np.linspace(2459861.41509122, 2459861.62089175, 20, endpoint=True))
	vd = setup_VisData(N=3, times=times, freqs=freqs)

	# setup VM
	VM = setup_VisMapper(vd)
	VM.set_normalization('A2w', clip=1e-8)

	# create maps (build cache)
	maps = VM.make_map()
	assert maps.shape == (vd.Nfreqs, VM.Npix)

	# assert cache is full
	assert len(VM.A) == len(VM.times)

	# compute full P
	P = VM.compute_P(diag=False)

	# assert that P is diagonally normalized
	assert torch.isclose(P.diagonal(dim1=1, dim2=2), torch.tensor(1.0), atol=1e-5, rtol=1e-5).all()

	# make maps into a point source
	idx = torch.argmin((VM.ra-40)**2 + (VM.dec--30.72)**2).item()
	maps = torch.zeros_like(maps)
	maps[:, idx] = 1.0

	# compute Pm and confirm that implicit P is diagonally normalized
	Pm = VM.compute_Pm(maps)
	assert torch.isclose(Pm[:, idx], maps[:, idx], atol=1e-5, rtol=1e-5).all()

	# get P@m
	Pam = torch.einsum('ijk,ik->ij', P, maps)

	# assert Pm and Pam are the same
	assert torch.isclose(Pm, Pam, atol=1e-5, rtol=1e-5).all()

	# test Pdiag vs P.diag()
	Pdiag = VM.compute_P(diag=True)
	assert torch.isclose(P.diagonal(dim1=1, dim2=2), Pdiag, atol=1e-5, rtol=1e-5).all()

	# test with a different normalization scheme
	VM.set_normalization('Aw', clip=1e-8)

	# get Pdiag and test against pre-computed expectation
	Pdiag = VM.compute_P(diag=True)
	assert torch.isclose(Pdiag.max(dim=1).values, torch.tensor(0.8), atol=1e-1).all()

	# test w/ icov...


def test_imaging_lazy():
	# write to temp file and then lazy_load
	with TemporaryDirectory() as tmp:
		if isinstance(tmp, str):
			tmpfile = tmp + "/test.h5"
		else:
			tmpfile = tmp.name + "/test.h5"

		# setup vd
		times = torch.as_tensor(np.linspace(2459861.41509122, 2459861.62089175, 20, endpoint=True))
		vd = setup_VisData(N=3, times=times, freqs=freqs)
		vd.write_hdf5(tmpfile)

		# setup mapper with in-memory data
		VM = setup_VisMapper(vd, cache_A=False)
		VM.set_normalization('A2w', clip=1e-8)
		maps1 = VM.make_map()

		# setup mapper with out-of-memory data
		vd.read_hdf5(tmpfile, lazy_load=True)
		VM = setup_VisMapper(vd, cache_A=False)
		VM.set_normalization('A2w', clip=1e-8)
		maps2 = VM.make_map()

	assert torch.isclose(maps1, maps2, atol=1e-8, rtol=1e-8).all()




