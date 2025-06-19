import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH

from test_dataset import setup_VisData



def test_GPFilter():
	freqs = torch.linspace(120e6, 130e6, 64)
	times = torch.linspace(2458168.1, 2458168.3, 5)

	vd = setup_VisData(N=3, times=times, freqs=freqs)

	# enact a high-pass filter
	Cs = ba.filt.rbf_cov(freqs, 2e6)  # 500 ns
	Cn = torch.eye(len(freqs)) * 1e-8

	F = ba.filt.GPFilter(Cs, Cn, dim=-1, residual=True, hermitian=True, dtype=ba._cfloat())

	dfilt = F(vd)

	assert dfilt.data.shape == vd.data.shape
	assert dfilt.data.std() < vd.data.std()
	assert dfilt.data.mean(-1).abs().mean() < 1e-5 * vd.data.mean(-1).abs().mean()

	_dfilt = F(vd.data)
	assert (dfilt.data - _dfilt).abs().max() < 1e-15


	# enact low-delay inpainting
	vd = setup_VisData(N=3, times=times, freqs=freqs)
	flags = torch.zeros(len(freqs), dtype=torch.bool)
	flags[::3] = True
	vd.data[..., flags] = 0.0

	Cs_cross = ba.filt.rbf_cov(freqs, 2e6, x2=freqs[flags])
	Cs_pred = ba.filt.rbf_cov(freqs[flags], 2e6)

	F = ba.filt.GPFilter(Cs, Cn, Cs_cross=Cs_cross, Cs_pred=Cs_pred,
		input_idx=flags, dim=-1, residual=False, hermitian=True, dtype=ba._cfloat()
	)

	output = F(vd)

	# assert non-flagged channels are unchanged
	assert (output.data[..., ~flags] - vd.data[..., ~flags]).abs().max() < 1e-15

	# assert flagged channels are updated
	assert (output.data[..., flags].abs() > 0).all()




