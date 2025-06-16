import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

freqs = torch.linspace(120e6, 130e6, 10)


def setup_Telescope():
	telescope = ba.telescope_model.TelescopeModel((21.42827, -30.72148))

	return telescope

def setup_Array(N=3, freqs=None, D=15):
	antnums, antvecs = ba.utils._make_hex(N, D=D)
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


def test_build_reds():
	"""
	      16    17    18


	   12    13    14    15


	07    08    09    10    11


	   03    04    05    06


	      00    01    02
	"""
	antpos = dict(zip(*ba.utils._make_hex(3)))
	Nants = len(antpos)

	# test fcluster True/False
	red_info1 = ba.telescope_model.build_reds(antpos, fcluster=True)
	red_info2 = ba.telescope_model.build_reds(antpos, fcluster=False)
	Nreds = len(red_info1[0])
	assert len(red_info1) == len(red_info2)
	for r1, r2 in zip(red_info1[0], red_info2[0]):
		assert red_info1[0] == red_info2[0]

	# assert number of bls is correct
	assert len(red_info1[3]) == (Nants * (Nants - 1) / 2 + Nants)
	# assert 1-unit EW group is correct based on hex layout
	assert all([bl[1] == bl[0]+1 for bl in red_info1[0][1]])
	# assert bl_lens are monotically increasing
	assert all(np.diff(red_info1[4]) >= -1e-14)
	# assert all bls are accounted for in reds
	assert len(ba.utils.flatten(red_info1[0])) == len(red_info1[3])

	# test other options
	red_info = ba.telescope_model.build_reds(antpos, red_bls=[(0, 1)])
	assert len(red_info[0]) == 1
	assert red_info[0][0] == red_info1[0][1]

	red_info = ba.telescope_model.build_reds(antpos, norm_vec=True)
	assert len(red_info[0]) == 9
	assert red_info[0][0] == red_info1[0][0]
	assert red_info[0][1] == sorted(ba.utils.flatten(red_info1[0][1:4]))

	red_info = ba.telescope_model.build_reds(antpos, min_len=16, max_len=40)
	assert min(red_info[4]) >= 16
	assert min(red_info[4]) <= 40

	red_info = ba.telescope_model.build_reds(antpos, min_EW_len=16)
	assert min(red_info[1][:, 0].abs()) >= 16

	red_info = ba.telescope_model.build_reds(antpos, exclude_reds=[(0, 1), (0, 2)])
	assert ((0, 1) not in red_info[2]) and ((0, 2) not in red_info[2])
	assert len(red_info[0]) == (Nreds - 2)

	red_info = ba.telescope_model.build_reds(antpos, use_blnums=True)
	assert isinstance(red_info[3][0], np.integer)
	assert ba.utils.blnum2ants(red_info[3]) == red_info1[3]

	red_info2 = ba.telescope_model.build_reds(antpos, red_info=red_info)
	assert (red_info[3] == red_info2[3]).all()

