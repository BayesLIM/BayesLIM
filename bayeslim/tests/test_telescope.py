import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

freqs = torch.linspace(120e6, 130e6, 10)


def setup_Telescope():
	telescope = ba.telescope_model.TelescopeModel((21.42827, -30.72148))

	return telescope

def setup_Array(N=5, freqs=None):
	antnums, antvecs = ba.utils._make_hex(N, D=15)
	antpos_d = ba.utils.AntposDict(antnums, antvecs)
	array = ba.telescope_model.ArrayModel(
		antpos_d, freqs=freqs, cache_s=False, redtol=1.0,
	)

	return array


def test_Telescope():
	telescope = setup_Telescope()

	time = 2459861.5
	ra, dec = torch.tensor([0.0]), torch.tensor([0.0])  # deg

	# test eq2top conversion and caching
	angs = telescope.eq2top(time, ra, dec, store=True)

	# check for caching
	key = telescope.hash(time, ra)
	assert key in telescope.conv_cache


def test_Array():
	array = setup_Array(N=3, freqs=freqs)

	# test redundancy calculations
	assert len(array.ants) == 19  # hera19
	assert len(array.reds) == 31  # correct reds

	# test correct baseline vector retreival
	bl_vec = array.get_antpos(1) - array.get_antpos(0)
	assert (bl_vec - torch.tensor([15,0,0])).norm().abs() < 1e-10

	# test fringe generation
	az, zen = torch.meshgrid(
		torch.arange(0, 360, 10.0),
		torch.arange(0, 90, 2.5),
		indexing='ij',
	)
	az, zen = az.ravel(), zen.ravel()
	array.cache_s = True

	bls = [(0, 1), (1, 2), (0, 2)]
	blvecs = array.get_blvecs(bls)
	fringe1 = array.gen_fringe(blvecs, zen, az, conj=False)
	fringe2 = array.gen_fringe(blvecs[:1], zen, az, conj=False)
	fringe3 = array.gen_fringe(blvecs[:1], zen, az, conj=True)

	# test shape and dtype
	assert fringe1.shape == (len(bls), len(freqs), len(zen))
	assert fringe1.dtype == torch.complex128
	# test fringe computation is the same for singe-bl or multi-bl
	assert (fringe1[:1] - fringe2).norm().abs() < 1e-10
	# check conjugation
	assert (fringe2 - fringe3.conj()).norm().abs() < 1e-10
	# check phase center at zenith
	assert fringe1[:, :, 0].isclose(torch.tensor(1 + 0j)).all()
	# check unit amplitude
	assert (fringe1.abs() <= 1).all()

	# test get_bls
	sim_bls = array.get_bls(uniq_bls=True, keep_autos=True, min_len=1, max_len=29)
	assert (0, 0) not in sim_bls  # 0-m baseline
	assert (0, 2) not in sim_bls  # 30-m baseline
	assert (1, 2) not in sim_bls  # non-uniq baseline



