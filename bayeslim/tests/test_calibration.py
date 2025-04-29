import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

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
	    time_dim=4,
	    freq_dim=5
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


def test_RedVisCoupling_sympy():
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
	rvis_cpl.setup_coupling(use_reds=True, include_second_order=True)

	# take forward pass of RedVisModel
	with torch.no_grad():
		vout = rvis_cpl(vd)

	# compare RedVisModel against analytic result
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

	# take forward pass of RedVisModel
	with torch.no_grad():
		vout = rvis_cpl(vd)

	# compare RedVisModel against analytic result
	r = vout[[bl for bl in vout.bls]].numpy() / np.array([Vc[bl[0], bl[1]] for bl in vout.bls])
	assert np.isclose(r, 1 + 0j, atol=1e-10).all()
