import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH


def test_LM():
	# test basic pass through
	freqs = torch.linspace(100e6, 200e6, 128)

	for dim in [-1, -2, 5, 4]:
		# test basic forward pass and inverse pass
		A = ba.linear_model.gen_poly_A(freqs, 32, basis='legendre', whiten=True)

		yshape = [1, 1, 20, 5, 10, 10]
		yshape[dim] = 128 
		yshape = torch.Size(yshape)
		xshape = list(yshape)
		xshape[dim] = 32
		xshape = torch.Size(xshape)

		x = torch.randn(xshape)

		LM = ba.linear_model.LinearModel('custom', A=A, dim=dim)

		y = LM(x)
		assert y.shape == yshape

		xhat = LM.least_squares(y)[0]
		assert xhat.shape == xshape

		# test reshaping output dimension and a multi-dim A matrix
		yshape = yshape[:2] + (100,) + yshape[4:]
		Ashape = list(xshape)[3:dim] + [128, 32]
		out_shape = list(xshape)
		out_shape[dim] = 128

		for i in range(len(Ashape) - A.ndim):
			A = A[None]
		A = A.expand(Ashape).clone()
		A[0] = 0

		LM = ba.linear_model.LinearModel(
			'custom',
			A=A, 
			dim=dim,
			out_reshape=yshape,
			out_shape=out_shape,
		)

		y = LM(x)
		assert y.shape == yshape
		# check A[0] entry is zero-d out
		assert torch.isclose(y.index_select(-3, torch.tensor(0)), torch.tensor(0.)).all()

		# try inverse
		xhat = LM.least_squares(y)[0]
		assert xhat.shape == x.shape
		assert torch.isclose(xhat[:, :, :, 0], torch.tensor(0.)).all()
		assert torch.isclose(xhat[:, :, :, 1:], x[:, :, :, 1:]).all()

		# try inverse with diag noise cov
		Ninv = torch.ones(128)
		xhat = LM.least_squares(y, Ninv=Ninv)[0]
		assert torch.isclose(xhat[:, :, :, 1:], x[:, :, :, 1:]).all()

		# try inverse with diag norm (this won't be equal to x)
		xhat = LM.least_squares(y, norm='diag')[0]
		assert xhat.shape == x.shape
		assert torch.isclose(xhat[:, :, :, 0], torch.tensor(0.)).all()

		# try inverse with diag norm, full diag Ninv
		Ninv = torch.ones_like(y)
		xhat = LM.least_squares(y, Ninv=Ninv, norm='diag')[0]
		assert torch.isclose(xhat[:, :, :, 0], torch.tensor(0.)).all()

