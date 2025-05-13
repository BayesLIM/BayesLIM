import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import bayeslim as ba
from bayeslim.data import DATA_PATH


class Normal(ba.utils.Module):
    def __init__(self, shape, LM=None):
        super().__init__()
        self.params = torch.nn.Parameter(torch.zeros(shape, dtype=ba._float()))
        self.LM = LM
        
    def forward(self, params=None, *args, **kwargs):
        params = params if params is not None else self.params
        if self.LM is not None:
            params = self.LM(params)
        return params


def setup_NormalProb(N=100, seed=0, scale=1, cond=1.5, dtype=None):
	# test a multi-variate gaussian
	model = Normal(N)

	raw = ba.dataset.TensorData()
	raw.setup_data(data=torch.zeros(N, dtype=dtype))

	torch.manual_seed(seed)
	a = torch.randn(N, int(cond * N))  # cond inv prop. to condition number
	cov = (a @ a.T) / (cond * N)

	if isinstance(scale, (int, float)):
		cov *= scale
	else:
		cov = scale.sqrt()[:, None] * cov * scale.sqrt()[None, :]

	raw.set_cov(cov.to(raw.data.dtype), 'full')
	raw.compute_icov()

	target = ba.dataset.Dataset([raw])

	prob = ba.optim.LogProb(model, target, complex_circular=False)

	return prob


def test_normal_bfgs():
	# get multivariate normal
	N = 50
	prob = setup_NormalProb(N, cond=2)

	# perturb params
	torch.manual_seed(100)
	prob.model.params.data[:] = torch.randn(N) * prob.target[0].cov.diag().sqrt()

	# setup bfgs
	opt = ba.bfgs.BFGS((prob.model.params,), H0=torch.tensor(1.0), max_iter=1)

	# perform 40 iterations
	for i in range(40):
		opt.zero_grad()
		opt.step(prob.closure)

	# assert rms is below 1e-7 (tested May 2025)
	assert prob.model.params.data.std() < 1e-7

	# assert approx cov diagonal is roughly accurate (within factor of 2)
	assert ((opt.H / prob.target[0].cov).diag() - 1).abs().mean() < 0.5


def test_normal_lbfgs():
	# get multivariate normal
	N = 50
	prob = setup_NormalProb(N, cond=2)

	# perturb params
	torch.manual_seed(100)
	prob.model.params.data[:] = torch.randn(N) * prob.target[0].cov.diag().sqrt()

	# setup bfgs
	opt = ba.bfgs.LBFGS((prob.model.params,), max_iter=1, lr=3)

	# perform 40 iterations
	for i in range(40):
		opt.zero_grad()
		opt.step(prob.closure)

	# assert rms is below 1e-7 (tested May 2025)
	assert prob.model.params.data.std() < 1e-7

	# assert approx cov diagonal is roughly accurate (within factor of 2)
	assert ((opt.H.diag / prob.target[0].cov.diag()) - 1).abs().mean() < 0.5


def test_scaled_normal():
	# get multivariate normal
	N = 50
	scale = torch.ones(N) * 0.1
	prob = setup_NormalProb(N, cond=1, scale=scale)

	# perturb params
	torch.manual_seed(100)
	prob.model.params.data[:] = torch.randn(N) * prob.target[0].cov.diag().sqrt()

	# setup bfgs
	opt = ba.bfgs.LBFGS((prob.model.params,), max_iter=1, lr=1.0, update_Hdiag=True)

	# perform 40 iterations
	for i in range(40):
		opt.zero_grad()
		opt.step(prob.closure)

	# assert rms is below 1e-7 (tested May 2025)
	assert prob.model.params.data.std() < 1e-7

	# ensure that the estimated Hdiag is between eigenvalues of covariance
	evals = torch.linalg.eigh(prob.target[0].cov)[0]
	assert (opt._Hdiag[0] > evals.min()) & (opt._Hdiag[0] < evals.max())


def test_multi_scaled_normal():
	# get multivariate normal
	N = 50
	scale = torch.ones(N) * 0.1
	scale[:N//2] *= 0.01
	prob = setup_NormalProb(N, cond=1, scale=scale)

	# perturb params
	torch.manual_seed(100)
	prob.model.params.data[:] = torch.randn(N) * prob.target[0].cov.diag().sqrt()

	# setup bfgs
	opt = ba.bfgs.LBFGS((prob.model.params,), max_iter=1, lr=1.0, update_Hdiag=True)

	# perform 40 iterations
	for i in range(40):
		opt.zero_grad()
		opt.step(prob.closure)

	# assert rms is below 1e-2 (tested May 2025)
	assert prob.model.params.data.std() < 1e-2

	# ensure that the estimated Hdiag is between eigenvalues of covariance
	evals = torch.linalg.eigh(prob.target[0].cov)[0]
	assert (opt._Hdiag[0] > evals.min()) & (opt._Hdiag[0] < evals.max())

	# repeat with better H0 guess (but still with an overall scale difference)
	torch.manual_seed(100)
	prob.model.params.data[:] = torch.randn(N) * prob.target[0].cov.diag().sqrt()

	H0 = ba.hmat.DiagMat(scale.clone() * 10)
	opt = ba.bfgs.LBFGS((prob.model.params,), max_iter=1, lr=1.0, update_Hdiag=True, H0=H0)

	# perform 40 iterations
	for i in range(40):
		opt.zero_grad()
		opt.step(prob.closure)

	# assert rms is below 1e-7 (tested May 2025)
	assert prob.model.params.data.std() < 1e-7

	# ensure that the estimated Hdiag is between eigenvalues of covariance
	evals = torch.linalg.eigh(prob.target[0].cov)[0]
	assert (opt._Hdiag[0] > evals.min()) & (opt._Hdiag[0] < evals.max())

