import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

from test_telescope import setup_Array, setup_Telescope
from test_dataset import setup_VisData

try:
	import symengine as sympy
	import_sympy = True
except ImportError:
	import_sympy = False


freqs = torch.linspace(120e6, 130e6, 10)
times = torch.linspace(2458168.1, 2458168.3, 5)


def setup_Coupling(freqs=freqs, times=times):
	# setup HERA-7 array
	ants, antvecs = ba.utils._make_hex(2)
	antpos = dict(zip(ants, torch.as_tensor(antvecs)))
	array = ba.telescope_model.ArrayModel(antpos)

	# set metadata
	bls = array.get_bls(uniq_bls=True)  # red_bls
	data_bls = array.get_bls(uniq_bls=False)  # all_bls

	# setup RedVisCoupling object
	R = ba.calibration.VisModelResponse(
	    freq_kwargs={'freqs': freqs},
	    time_kwargs={'times': times},
	    time_dim=-2,
	    freq_dim=-1
	)
	coupling_terms, coupling_idx = ba.calibration.gen_coupling_terms(
		antpos,
		no_auto_coupling=False,
		compress_to_red=True
	)
	torch.manual_seed(1)
	params = torch.randn(1, 1, len(coupling_terms), 1, len(freqs), dtype=ba._cfloat())
	rvis_cpl = ba.calibration.RedVisCoupling(
		params,
		freqs,
		antpos,
		coupling_terms,
		bls,
		data_bls,
		coupling_idx=coupling_idx,
		R=R
	)

	return rvis_cpl


def test_Coupling_sympy():
	if not import_sympy:
		return

	# can only test with 1 freq and 1 time due to sympy constraints
	freqs = torch.linspace(120e6, 130e6, 1)
	times = torch.linspace(2458168.1, 2458168.3, 1)

	# get RedVisCoupling
	rvis_cpl = setup_Coupling(freqs, times)
	ants = list(rvis_cpl.antpos.keys())

	rvis_cpl.params.data[:, :, 2:] = 0

	# simulate mock redundant bls data
	torch.manual_seed(0)
	vd = ba.VisData()
	vd.setup_meta(antpos=rvis_cpl.antpos)
	data = torch.randn(1, 1, len(rvis_cpl.bls_in), len(times), len(freqs), dtype=ba._cfloat())
	vd.setup_data(rvis_cpl.bls_in, times, freqs, data=data)
	vd[(0, 0)] = vd[(0, 0)].abs() # fix autocorr to abs

	# get red_info
	reds, _, bl2red_idx, _, _, _, _ = ba.telescope_model.build_reds(
		rvis_cpl.antpos,
		bls=sorted(rvis_cpl.bls_out),
		redtol=1.,
	)
	bl2red = {}
	for k in bl2red_idx:
		bl2red[k] = reds[bl2red_idx[k]][0]
		bl2red[k[::-1]] = reds[bl2red_idx[k]][0][::-1]

	# create redundant visibility matrix in sympy
	Vr = []
	Vdata = {}
	for i in range(len(ants)):
	    _V = []
	    for j in range(len(ants)):
	        bl = (i, j)
	        if j >= i:
	            _V.append("V_{}".format(bl2red_idx[bl]))
	            Vdata["V_{}".format(bl2red_idx[bl])] = vd[bl2red[bl]].item()
	        else:
	            _V.append(sympy.conjugate("V_{}".format(bl2red_idx[bl[::-1]])))
	    Vr.append(_V)
	Vr = sympy.Matrix(Vr)

	# create redundant coupling matrix in sympy
	Er = []
	red_vecs = torch.zeros(0, 3)
	red_num = 0
	for i in range(len(ants)):
		_E = []
		for j in range(len(ants)):
			bl_vec = rvis_cpl.antpos[j] - rvis_cpl.antpos[i]
			diff_norm = (red_vecs - bl_vec).norm(dim=-1).isclose(torch.tensor(0.), atol=1.0)
			if diff_norm.any():
				# found a match in red_vecs
				_E.append("e_{}".format(diff_norm.argwhere()[0,0]))
			else:
				# no match in red_vecs, new red_bl
				red_vecs = torch.cat([red_vecs, bl_vec[None, :]], dim=0)
				_E.append("e_{}".format(red_num))
				red_num += 1
		Er.append(_E)

	for i in range(len(ants)):        
		Er[i][i] = '1 + {}'.format(Er[i][i])

	Er = sympy.Matrix(Er)

	# perform coupling operation
	Vc = Er @ Vr @ Er.conjugate().T

	# substitute in data and params
	Vc = np.array(Vc.subs(dict(
	    list(Vdata.items()) + \
	    list({f'e_{i}': rvis_cpl.params[0,0,i,0,0].item() for i in range(len(rvis_cpl.coupling_terms))}.items())
		))).astype(np.complex128)

	# setup coupling indexing arrays
	rvis_cpl.setup_coupling(copydata=True, use_reds=True, include_second_order=True)

	# take forward pass of RedVisCoupling
	with torch.no_grad():
		vout = rvis_cpl(vd)

	# compare RedVisCoupling against analytic result
	r = vout[[bl for bl in vout.bls]].numpy() / np.array([Vc[bl[0], bl[1]] for bl in vout.bls])
	assert np.isclose(r, 1 + 0j, atol=1e-10).all()

	#### export to VisCoupling and test it ####
	CI = ba.calibration.CouplingInflate(
		-vd.get_bl_vecs(rvis_cpl.coupling_terms),
		vd.antpos,
	)
	vis_cpl = ba.calibration.VisCoupling(
		CI(rvis_cpl.params),
		freqs,
		vd.antpos,
		rvis_cpl.bls_out,
		R=rvis_cpl.R,
	)
	vis_cpl.setup_coupling()

	# take forward pass of VisCoupling
	with torch.no_grad():
		vout = vis_cpl(vd.inflate_by_redundancy(rvis_cpl.bls_out))

	# compare VisCoupling against analytic result
	r = vout[[bl for bl in vout.bls]].numpy() / np.array([Vc[bl[0], bl[1]] for bl in vout.bls])
	assert np.isclose(r, 1 + 0j, atol=1e-10).all()

	#### test just first-order coupling ####
	Er = []
	red_vecs = torch.zeros(0, 3)
	red_num = 0
	for i in range(len(ants)):
		_E = []
		for j in range(len(ants)):
			bl_vec = rvis_cpl.antpos[j] - rvis_cpl.antpos[i]
			diff_norm = (red_vecs - bl_vec).norm(dim=-1).isclose(torch.tensor(0.), atol=1.0)
			if diff_norm.any():
				# found a match in red_vecs
				_E.append("e_{}".format(diff_norm.argwhere()[0,0]))
			else:
				# no match in red_vecs, new red_bl
				red_vecs = torch.cat([red_vecs, bl_vec[None, :]], dim=0)
				_E.append("e_{}".format(red_num))
				red_num += 1
		Er.append(_E)
	Er = sympy.Matrix(Er)

	# perform coupling operation
	Vc = Vr + Er @ Vr + Vr @ Er.conjugate().T

	# substitute in data and params
	Vc = np.array(Vc.subs(dict(
	    list(Vdata.items()) + \
	    list({f'e_{i}': rvis_cpl.params[0,0,i,0,0].item() for i in range(len(rvis_cpl.coupling_terms))}.items())
		))).astype(np.complex128)

	# setup coupling indexing arrays
	rvis_cpl.setup_coupling(use_reds=True, include_second_order=False)

	# take forward pass of RedVisCoupling
	with torch.no_grad():
		vout = rvis_cpl(vd)

	# compare RedVisCoupling against analytic result
	r = vout[[bl for bl in vout.bls]].numpy() / np.array([Vc[bl[0], bl[1]] for bl in vout.bls])
	assert np.isclose(r, 1 + 0j, atol=1e-10).all()


def test_Coupling_sympy_double_path():
	if not import_sympy:
		return

	# can only test with 1 freq and 1 time due to sympy constraints
	freqs = torch.linspace(120e6, 130e6, 1)
	times = torch.linspace(2458168.1, 2458168.3, 1)

	# get RedVisCoupling
	rvis_cpl = setup_Coupling(freqs, times)
	ants = list(rvis_cpl.antpos.keys())

	# simulate mock redundant bls data
	torch.manual_seed(0)
	vd = ba.VisData()
	vd.setup_meta(antpos=rvis_cpl.antpos)
	data = torch.randn(1, 1, len(rvis_cpl.bls_in), len(times), len(freqs), dtype=ba._cfloat())
	vd.setup_data(rvis_cpl.bls_in, times, freqs, data=data)
	vd[(0, 0)] = vd[(0, 0)].abs() # fix autocorr to abs

	# get red_info
	reds, _, bl2red_idx, _, _, _, _ = ba.telescope_model.build_reds(
		rvis_cpl.antpos,
		bls=sorted(rvis_cpl.bls_out),
		redtol=1.,
	)
	bl2red = {}
	for k in bl2red_idx:
		bl2red[k] = reds[bl2red_idx[k]][0]
		bl2red[k[::-1]] = reds[bl2red_idx[k]][0][::-1]

	# create redundant visibility matrix in sympy
	Vr = []
	Vdata = {}
	for i in range(len(ants)):
	    _V = []
	    for j in range(len(ants)):
	        bl = (i, j)
	        if j >= i:
	            _V.append("V_{}".format(bl2red_idx[bl]))
	            Vdata["V_{}".format(bl2red_idx[bl])] = vd[bl2red[bl]].item()
	        else:
	            _V.append(sympy.conjugate("V_{}".format(bl2red_idx[bl[::-1]])))
	    Vr.append(_V)
	Vr = sympy.Matrix(Vr)

	# create redundant coupling matrix in sympy
	Er = []
	red_vecs = torch.zeros(0, 3)
	red_num = 0
	for i in range(len(ants)):
		_E = []
		for j in range(len(ants)):
			bl_vec = rvis_cpl.antpos[j] - rvis_cpl.antpos[i]
			diff_norm = (red_vecs - bl_vec).norm(dim=-1).isclose(torch.tensor(0.), atol=1.0)
			if diff_norm.any():
				# found a match in red_vecs
				_E.append("e_{}".format(diff_norm.argwhere()[0,0]))
			else:
				# no match in red_vecs, new red_bl
				red_vecs = torch.cat([red_vecs, bl_vec[None, :]], dim=0)
				_E.append("e_{}".format(red_num))
				red_num += 1
		Er.append(_E)

	Er = sympy.Matrix(Er)
	I = sympy.Matrix([['1' if i == j else '0' for j in range(len(ants))] for i in range(len(ants))])

	# get I + X + XX
	Er = I + Er + Er @ Er

	# perform coupling operation
	Vc = Er @ Vr @ Er.conjugate().T

	# substitute in data and params
	Vc = np.array(Vc.subs(dict(
	    list(Vdata.items()) + \
	    list({f'e_{i}': rvis_cpl.params[0,0,i,0,0].item() for i in range(len(rvis_cpl.coupling_terms))}.items())
		))).astype(np.complex128)

	# export RedVisCoupling to VisCoupling that has double-path capability
	CI = ba.calibration.CouplingInflate(
		-vd.get_bl_vecs(rvis_cpl.coupling_terms),
		vd.antpos,
	)
	vis_cpl = ba.calibration.VisCoupling(
		CI(rvis_cpl.params),
		freqs,
		vd.antpos,
		rvis_cpl.bls_out,
		R=rvis_cpl.R,
		double=True,
	)
	vis_cpl.setup_coupling()

	# take forward pass of VisCoupling
	with torch.no_grad():
		vout = vis_cpl(vd.inflate_by_redundancy(), add_I=True, double=True)

	# compare output against analytic result
	r = vout[[bl for bl in vout.bls]].numpy() / np.array([Vc[bl[0], bl[1]] for bl in vout.bls])
	assert np.isclose(r, 1 + 0j, atol=1e-10).all()


def test_VisCoupling():

	freqs = torch.linspace(120e6, 130e6, 8)
	times = torch.linspace(2458168.1, 2458168.3, 4)
	# setup redviscoupling
	rvis_cpl = setup_Coupling(freqs, times)

	# simulate mock redundant bls data
	torch.manual_seed(0)
	vd = ba.VisData()
	vd.setup_meta(antpos=rvis_cpl.antpos)
	data = torch.randn(1, 1, len(rvis_cpl.bls_in), len(times), len(freqs), dtype=ba._cfloat())
	vd.setup_data(rvis_cpl.bls_in, times, freqs, data=data)
	vd[(0, 0)] = vd[(0, 0)].abs() # fix autocorr to abs
	vd = vd.inflate_by_redundancy()

	# export ot viscoupling
	CI = ba.calibration.CouplingInflate(
		-vd.get_bl_vecs(rvis_cpl.coupling_terms),
		rvis_cpl.antpos,
	)
	vis_cpl = ba.calibration.VisCoupling(
		CI(rvis_cpl.params),
		freqs,
		rvis_cpl.antpos,
		rvis_cpl.bls_out,
		R=rvis_cpl.R,
	)
	vis_cpl.setup_coupling()

	# test forward pass
	with torch.no_grad():
		vout = vis_cpl(vd)
	assert vout.data.shape == vd.data.shape

	# test forward pass with double reflections
	with torch.no_grad():
		vout2 = vis_cpl(vd, double=True)
	assert vout.data.shape == vd.data.shape


def test_VisModel():
	vd = setup_VisData()
	vd.data[:] = 0
	bls = vd.get_bls()
	blnums = ba.utils.ants2blnum(bls, tensor=True)

	params = torch.randn(1, 1, len(bls), len(times), len(freqs), dtype=ba._cfloat())
	R = ba.calibration.VisModelResponse(
		freq_kwargs=dict(freqs=freqs),
		time_kwargs=dict(times=times)
	)

	vis_mdl = ba.calibration.VisModel(params, R=R, parameter=False, blnums=blnums)

	# take forward pass and assert vout == params
	vout = vis_mdl(vd)
	assert torch.isclose(vout.data, params, atol=1e-10).all()

	# now try time minibatching
	vd2 = vd.select(time_inds=range(3), inplace=False)
	vis_mdl.clear_cache()
	vout = vis_mdl(vd2)
	assert vout.Ntimes == 3

	# assert vd2.times picks up a hash
	# important for high-perf indexing
	assert hasattr(vd2.times, '_arr_hash')
	assert vd2.times._arr_hash in vis_mdl.cache_tidx

	# now try bl minibatching
	vd2 = vd.select(bl_inds=range(50), inplace=False)
	vis_mdl.clear_cache()
	vout = vis_mdl(vd2)
	assert vout.Nbls == 50

	# assert vd2._blnums picks up a hash
	assert hasattr(vd2._blnums, '_arr_hash')
	assert vd2._blnums._arr_hash in vis_mdl.cache_bidx


def test_PartialRedVisInflate():
	# setup data
	vd = setup_VisData()
	red_info = ba.telescope_model.build_reds(vd.antpos, bls=vd.bls)
	vd_red = vd.bl_average(red_info[0], inplace=False)
	vd = vd_red.inflate_by_redundancy() # make sure the data are actually redundant

	## create purely redundant mapping
	model = ba.calibration.PartialRedVisInflate(red_info[2], vd.bls, parameter=False)
	A = model._buildA(model.params)
	vd_inf = model(vd_red)

	assert A.sum(1).isclose(torch.tensor(1.0)).all()
	assert vd.bls == vd_inf.bls
	assert vd.data.shape == vd_inf.data.shape
	assert (vd.data - vd_inf.data).abs().max() < 1e-10

	## create partial redundant mapping (2 red bls per red group)
	vd = setup_VisData()
	vd_red = vd.bl_average(red_info[0], inplace=False)
	vd = vd_red.inflate_by_redundancy() # make sure the data are actually redundant

	# built new bl2red mapping
	bl2red = {}
	k = 0
	reds = []
	for i, red in enumerate(red_info[0]):
		reds.append([red[0]])
		if len(red) > 1:
			reds.append([red[1]])
		for bl in red:
			bl2red[bl] = np.arange(k, k+len(red[:2]))
		k += len(red[:2])

	# get red data and its inflated data
	vd_red = vd.bl_average(reds, inplace=False)
	vd = vd_red.inflate_by_redundancy()

	model = ba.calibration.PartialRedVisInflate(bl2red, vd.bls, parameter=False)
	A = model._buildA(model.params)
	vd_inf = model(vd_red)

	assert A.sum(1).isclose(torch.tensor(1.0)).all()
	assert vd.bls == vd_inf.bls
	assert vd.data.shape == vd_inf.data.shape
	assert (vd.data - vd_inf.data).abs().max() < 1e-10

