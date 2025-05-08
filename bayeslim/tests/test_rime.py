import numpy as np
import torch
import time

torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

from test_beam import setup_PixBeam_Interp
from test_sky import setup_PixSky_Noise
from test_telescope import setup_Telescope, setup_Array

freqs = torch.linspace(120e6, 130e6, 10)


def setup_RIME(times, freqs, array, telescope, nside=32):
	beam = setup_PixBeam_Interp(freqs)
	sky = setup_PixSky_Noise(freqs, nside=nside)
	sim_bls = array.get_bls(uniq_bls=True,  keep_autos=False)

	rime = ba.rime_model.RIME(
		sky, telescope, beam, array, sim_bls, times, freqs
	)

	return rime


def test_RIME():
	times = torch.linspace(2459861, 2459862, 5)
	telescope = setup_Telescope()
	array = setup_Array(N=3, freqs=freqs)
	sim_bls = array.get_bls(uniq_bls=True,  keep_autos=False)

	rime = setup_RIME(times, freqs, array, telescope)

	# test forward pass
	with torch.no_grad():
		vis = rime()
	assert vis.data.shape == (1, 1, len(sim_bls), len(times), len(freqs))

	# test batching
	time_groups = ba.utils.split_into_groups(times, Nelem=2)
	rime.setup_sim_times(time_groups)
	assert rime.Nbatch == np.ceil(len(times) / 2)

	with torch.no_grad():
		vis = rime.run_batches()
	assert vis.data.shape == (1, 1, len(sim_bls), len(times), len(freqs))
	assert (vis.times == times).all()


def RIME_performance():
	cuda = torch.cuda.is_available()

	freqs = torch.linspace(120e6, 130e6, 32)
	times = torch.linspace(2459861, 2459862, 16)
	telescope = setup_Telescope()
	array = setup_Array(N=4, freqs=freqs)
	sim_bls = array.get_bls(uniq_bls=True,  keep_autos=False)

	rime = setup_RIME(times, freqs, array, telescope, nside=32)
	rime.array.cache_s = True

	# test forward pass
	start = time.time()
	with torch.no_grad():
		vis = rime()
		if cuda:
			torch.cuda.synchronize()

	elapsed = time.time() - start
	print(f"{elapsed:.2f} seconds")



