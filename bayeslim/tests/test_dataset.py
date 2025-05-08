import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

from test_telescope import setup_Array, setup_Telescope

freqs = torch.linspace(120e6, 130e6, 10)
times = torch.linspace(2458168.1, 2458168.3, 5)


def setup_VisData():

	# setup visdata
	telescope = setup_Telescope()
	array = setup_Array(N=3)
	antpos = array.to_antpos()
	bls = array.get_bls()

	vd = ba.VisData()
	vd.setup_meta(antpos=antpos, telescope=telescope)

	torch.manual_seed(0)
	data = torch.randn(1, 1, len(bls), len(times), len(freqs), dtype=ba._cfloat())
	cov = torch.ones(1, 1, len(bls), len(times), len(freqs))

	vd.setup_data(bls, times, freqs, data=data, cov=cov)

	vd.check()

	return vd


def test_visdata_select():

	vd = setup_VisData()

	# baseline select
	vds = vd.select(bl=vd.bls[:5], inplace=False)
	assert vds.data.shape[2] == 5
	assert vds.bls == vd.bls[:5]
	#assert vds.antpairs.shape == (2, 5)
	#assert vds._antpairs.shape == (2, 5)

	vds = vd.select(bl_inds=range(5), inplace=False)
	assert vds.data.shape[2] == 5
	assert vds.bls == vd.bls[:5]
	#assert vds.antpairs.shape == (2, 5)
	#assert vds._antpairs.shape == (2, 5)

	# time select
	vds = vd.select(times=vd.times[:2], inplace=False)
	assert vds.data.shape[3] == 2
	assert (vds.times == vd.times[:2]).all()

	vds = vd.select(time_inds=range(2), inplace=False)
	assert vds.data.shape[3] == 2
	assert (vds.times == vd.times[:2]).all()

	# freq select
	vds = vd.select(freqs=vd.freqs[:3], inplace=False)
	assert vds.data.shape[4] == 3
	assert (vds.freqs == vd.freqs[:3]).all()

	vds = vd.select(freq_inds=range(3), inplace=False)
	assert vds.data.shape[4] == 3
	assert (vds.freqs == vd.freqs[:3]).all()

	# multi-dim index
	vds = vd.select(
		bl=vd.bls[:10:2], freqs=vd.freqs[:6], times=vd.times[:3], inplace=False,
	)
	assert vds.data.shape == (1, 1, 5, 3, 6)
	assert vds.bls == vd.bls[:10:2]



