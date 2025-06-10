import numpy as np
from tempfile import TemporaryDirectory

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

from test_telescope import setup_Array, setup_Telescope
from test_sky import setup_PointSky
from test_beam import setup_PixBeam_Airy


freqs = torch.linspace(120e6, 130e6, 10)
times = torch.linspace(2458168.1, 2458168.3, 5)


def setup_VisData(N=3, times=times, freqs=freqs):
	# setup HERA-like array and some random data

	vd = ba.VisData()

	# setup visdata
	telescope = setup_Telescope()
	array = setup_Array(N=N)
	antpos = array.to_antpos()
	bls = array.get_bls()

	vd.setup_meta(antpos=antpos, telescope=telescope)

	torch.manual_seed(0)
	data = torch.randn(1, 1, len(bls), len(times), len(freqs), dtype=ba._cfloat())
	cov = torch.ones(1, 1, len(bls), len(times), len(freqs))

	vd.setup_data(bls, times, freqs, data=data, cov=cov)

	vd.check()

	return vd


def test_visdata_get(vd=None):

	if vd is None:
		vd = setup_VisData()
		
	# get data
	data = vd.get_data()
	assert data.shape == (vd.Nbls, vd.Ntimes, vd.Nfreqs)

	data = vd.get_data(squeeze=False)
	assert data.shape == vd.data.shape

	data = vd.get_data(time_inds=range(3), freq_inds=range(4))
	assert data.shape == (vd.Nbls, 3, 4)

	# copy
	vdc = vd.copy()
	assert vd.data.shape == vdc.data.shape


def test_visdata_get_lazy():
	# write to temp file and then lazy_load
	with TemporaryDirectory() as tmp:
		if isinstance(tmp, str):
			tmpfile = tmp + "/test.h5"
		else:
			tmpfile = tmp.name + "/test.h5"

		vd = setup_VisData()
		vd.write_hdf5(tmpfile)
		vd.read_hdf5(tmpfile, lazy_load=True)

		test_visdata_get(vd)


def test_visdata_select():
	# setup visdata
	vd = setup_VisData()

	# baseline select
	vds = vd.select(bl=vd.bls[:5], inplace=False)
	assert vds.data.shape[2] == 5
	assert vds.bls == vd.bls[:5]
	assert (vds.blnums == ba.utils.ants2blnum(vds.bls)).all()
	assert (vds._blnums.numpy() == ba.utils.ants2blnum(vds.bls)).all()

	vds = vd.select(bl_inds=range(5), inplace=False)
	assert vds.data.shape[2] == 5
	assert vds.bls == vd.bls[:5]
	assert (vds.blnums == ba.utils.ants2blnum(vds.bls)).all()
	assert (vds._blnums.numpy() == ba.utils.ants2blnum(vds.bls)).all()

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


def test_visdata_bl_average():
	vd = setup_VisData()
	reds = ba.telescope_model.ArrayModel(vd.antpos).reds
	Navgs = torch.as_tensor([len(red) for red in reds])

	# test averaged noise and covariance
	torch.manual_seed(0)
	Ntest = 30
	vds = [setup_VisData() for i in range(Ntest)]
	for vd in vds: vd.bl_average(inplace=True)
	# get variance per baseline group
	var = torch.stack([vd.data[0,0].var(dim=(-1,-2)) for vd in vds]).mean(0)
	# check it matches with 1/Navgs to within 2sigma of Ntest
	assert ((var - 1 / Navgs).abs() < 1/np.sqrt(Ntest) * 2).all()
	# assert propagated covariance is correct
	assert torch.isclose(vds[0].cov[0,0,:,0,0], 1 / Navgs, atol=1e-5, rtol=1e-5).all()

	# test bl_average with missing bls in reds
	vd = setup_VisData()
	vd.bl_average(reds=reds[1:], inplace=True)
	assert vd.Nbls == (len(reds) - 1)

	# test w/ blnums as reds
	blnum_reds = [ba.utils.ants2blnum(red) for red in reds]
	vd = setup_VisData()
	vd.bl_average(reds=blnum_reds[1:], inplace=True)
	assert vd.Nbls == (len(reds) - 1)

	# test w/ icov instead of cov, and with flags
	vd = setup_VisData()
	vd.icov = 1 / vd.cov
	vd.cov = None
	vd.flags = torch.zeros_like(vd.data, dtype=bool)
	vd.set(reds[0], True, arr='flags')
	vd.bl_average(reds=reds, inplace=True)
	# assert flags have been correctly propagated
	assert vd.get_flags(reds[0][0]).all()
	assert not vd.get_flags([red[0] for red in reds[1:]]).any()
	# assert icov correctly propagated
	assert torch.isclose(vd.icov[0,0,:,0,0], Navgs*1.0, atol=1e-5, rtol=1e-5).all()


def test_visdata_time_average():
	## test uniform average and propagated covariance
	Ntimes = 10
	times = torch.linspace(2458168.1, 2458168.3, Ntimes)
	torch.manual_seed(0)
	vd = setup_VisData(times=times)
	Ntest = 30
	vds = [setup_VisData(times=times) for i in range(Ntest)]
	for _vd in vds: _vd.time_average(inplace=True)
	# check averaged shape
	assert vds[0].data.shape == vd.data.shape[:3] + (1,) + vd.data.shape[-1:]
	# get variance
	var = torch.stack([vd.data.var()for vd in vds]).mean(0)
	# check it matches with 1/Navgs to within 2sigma of Ntest
	assert ((var - 1 / Ntimes).abs() < (1/Ntest * 2)).all()
	# assert propagated covariance is correct
	assert torch.isclose(1/vds[0].cov, torch.tensor(float(Ntimes)), atol=1e-5).all()

	## test multi-bin average with missing chunks, out of place
	vd = setup_VisData(times=times)
	time_inds = [range(0, 3), range(3, 6), range(6, 9)]
	vda = vd.time_average(time_inds=time_inds, inplace=False)
	# assert shape is correct
	assert vda.data.shape == vd.data.shape[:3] + (3,) + vd.data.shape[-1:]
	# assert covariance is correct
	assert torch.isclose(1/vda.cov, torch.tensor(3.), atol=1e-5).all()
	# assert avg_times are correct
	assert torch.isclose(vda.times, vd.times[1::3], atol=1e-10, rtol=1e-13).all()

	## test rephasing
	vd = setup_VisData(times=times)
	time_inds = [range(0, 3), range(3, 6), range(6, 9)]
	vda = vd.time_average(time_inds, rephase=True, inplace=False)
	# assert shape is correct
	assert vda.data.shape == vd.data.shape[:3] + (3,) + vd.data.shape[-1:]
	# assert covariance is correct
	assert torch.isclose(1/vda.cov, torch.tensor(3.), atol=1e-5).all()
	# assert avg_times are correct
	assert torch.isclose(vda.times, vd.times[1::3], atol=1e-10, rtol=1e-13).all()


def test_vis_rephase():
	# point source sim
	freqs = torch.linspace(100e6, 200e6, 16)
	times = torch.linspace(2458168.02, 2458168.04, 10)  # times centered at ra=0 deg

	sky = setup_PointSky(freqs, Nsource=1)
	beam = setup_PixBeam_Airy(freqs)

	telescope = setup_Telescope()
	lsts = ba.telescope_model.JD2LST(times, telescope.location[0]) * 180 / np.pi
	array = setup_Array(N=3, freqs=freqs, D=30)

	sim_bls = array.get_bls(uniq_bls=True,  keep_autos=False)

	rime = ba.rime_model.RIME(
		sky, telescope, beam, array, sim_bls, times, freqs
	)

	with torch.no_grad():
		vd = rime()

	# rephase!
	vd_phs = vd.lst_rephase(vd.times[vd.Ntimes//2] - vd.times, inplace=False)

	# get phase drift from middle integration
	dphs = (vd_phs.data / vd_phs.data[:, :, :, vd.Ntimes//2:vd.Ntimes//2+1]).angle().squeeze()

	# the result should be more stable visibility phase wrt time
	assert dphs.abs().max() < 1.0

	# test time_nn_interp()
	new_lsts = lsts[:-1] + np.diff(lsts)[0] / 4
	vd_int = vd.time_nn_interp(new_lsts*np.pi/180, inplace=False)

	assert vd_int.data.shape == (1, 1, 30, 9, 16)
	assert (vd.data.abs()[:,:,:,:-1] - vd_int.data.abs()).max() < 1e-10


def test_visdata_inflate():
	vd = setup_VisData()
	reds = ba.telescope_model.ArrayModel(vd.antpos).reds
	bl2red = {}
	for i, red in enumerate(reds):
		for bl in red:
			bl2red[bl] = i

	# test correct average
	for i, red in enumerate(reds):
		vd.set(red, i, arr='data')
	vdr = vd.bl_average(reds=reds, inplace=False)
	assert torch.isclose(vdr.data[0,0,:,0,0].real, torch.arange(float(len(reds)))).all()

	# test RedVisAvg
	RVG = ba.dataset.RedVisAvg(reds, inplace=False)
	vdr2 = RVG(vd)
	assert torch.isclose(vdr.data, vdr2.data).all()

	# inflate by redundancy
	vdi = vdr.inflate_by_redundancy()

	# test shapes and values
	assert vdi.data.shape == vd.data.shape
	assert torch.isclose(vd.data, vdi.data).all()

	# test RedVisInflate
	new_bls, red_inds = ba.utils.inflate_bls(vdr.bls, bl2red, vd.bls)
	RVG = ba.dataset.RedVisInflate(new_bls, red_inds)
	vdi2 = RVG(vdr)
	assert torch.isclose(vdi2.data, vdi.data).all()


