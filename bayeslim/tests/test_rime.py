import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

from test_beam import setup_PixBeam_Interp
from test_sky import setup_PixSky_Noise
from test_telescope import setup_Telescope, setup_Array

freqs = torch.linspace(120e6, 130e6, 10)


def setup_RIME():
	telescope = setup_Telescope()
	times = torch.linspace(2459861, 2459862, 10)
	array = setup_Array(N=4, freqs=freqs)
	sim_bls = array.get_bls(uniq_bls=True,  keep_autos=False)

	beam = setup_PixBeam_Interp(freqs)
	sky = setup_PixSky_Noise(freqs)

	rime = ba.rime_model.RIME(
		sky, telescope, beam, array, sim_bls, times, freqs
	)

	# test forward pass
	with torch.no_grad():
		vis = rime()
	assert vis.data.shape == (1, 1, len(sim_bls), len(times), len(freqs))
