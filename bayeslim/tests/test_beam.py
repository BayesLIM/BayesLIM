import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH


freqs = torch.linspace(120e6, 130e6, 10)


def setup_PixBeam_Interp(freqs, interp_mode='linear'):
	# setup an Airy-Interpolated PixelBeam
	theta = torch.arange(0, 90.1, 1.0) * ba.D2R
	phi = torch.arange(0, 360, 1.0) * ba.D2R

	b_phi, b_theta = torch.meshgrid(phi, theta, indexing='xy')
	b_phi, b_theta = b_phi.ravel(), b_theta.ravel()
	airy = ba.beam_model.airy_disk(b_theta, b_phi, 10.0, freqs, square=True)

	R = ba.beam_model.PixelResponse(freqs, 'rect', interp_mode=interp_mode, 
		theta=b_theta/ba.D2R, phi=b_phi/ba.D2R,
		theta_grid=theta/ba.D2R, phi_grid=phi/ba.D2R,
		freq_mode='channel', powerbeam=True, realbeam=True,
		log=False)
	p = torch.as_tensor(airy[None, None, None, :, :])
	beam = ba.beam_model.PixelBeam(p, freqs, ant2beam=None, R=R, pol='e',
		powerbeam=True, fov=180, parameter=False
	)

	return beam
 

def setup_PixBeam_Airy(freqs, D=10.):
	# setup an Airy PixelBeam
	R = ba.beam_model.AiryResponse(powerbeam=True)
	params = torch.ones(1, 1, 1, 1, 1) * D
	beam = ba.beam_model.PixelBeam(params, freqs, R=R, pol='e',
		powerbeam=True, fov=180, parameter=False
	)

	return beam


def test_pixbeam_interpolation():
	# setup beams
	beam_interp = setup_PixBeam_Interp(freqs, interp_mode='linear')
	beam_airy = setup_PixBeam_Airy(freqs)

	# test beam interpolation
	az, zen = torch.meshgrid(
		torch.arange(0, 360, 10.0),
		torch.arange(0, 90, 2.5),
		indexing='ij',
	)
	az, zen = az.ravel(), zen.ravel()

	out1 = beam_interp.gen_beam(zen, az)[0]
	out2 = beam_airy.gen_beam(zen, az)[0]

	# test interpolation against ground truth
	assert (out1 - out2).std() < 1e-3


