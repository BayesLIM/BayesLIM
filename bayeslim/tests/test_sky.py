import numpy as np
import healpy

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

freqs = torch.linspace(120e6, 130e6, 10)



def setup_PixSky_Noise(freqs):
	# create healpix noise-like sky model
	nside = 32
	pxarea = healpy.nside2pixarea(nside)
	hpix_colat, hpix_ra = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
	hpix_dec = ba.utils.colat2lat(hpix_colat, deg=False)
	cut = hpix_dec < 59.27852 * np.pi / 180
	angs = torch.as_tensor([hpix_ra[cut], hpix_dec[cut]]) / ba.utils.D2R
	R = ba.sky_model.PixelSkyResponse(freqs)
	params = torch.randn(1, 1, len(freqs), angs.shape[1])
	sky = ba.sky_model.PixelSky(params, angs, pxarea, R=R, parameter=False)

	return sky


def setup_PointSky(freqs):
	# setup power-law point source model at HERA's zenith pointings
	R = ba.sky_model.PointSkyResponse(freqs, freq_mode='powerlaw', f0=freqs[0])
	params = torch.ones(1, 1, 2, 10)
	params[..., 0, :] = 1.0
	params[..., 1, :] = -2.2

	# degrees
	angs = torch.stack([torch.arange(10) * 5, torch.ones(10) * -30.7])

	sky = ba.sky_model.PointSky(params, angs, R=R, parameter=False)

	return sky


def test_point_sky():
	sky = setup_PointSky(freqs)
	with torch.no_grad():
		data = sky().data

	assert data.shape == (1, 1, len(freqs), 10)
	assert data.isclose((freqs[:, None]/freqs[0])**-2.2).all()

