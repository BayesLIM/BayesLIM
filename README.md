# BayesLIM

### A Framework for End-to-End, Bayesian Line Intensity Mapping Analysis

BayesLIM is a tool for performing end-to-end analysis of line intensity mapping (LIM) datasets in a differentiable, Bayesian framework.
It is built on PyTorch for its automatic differentiation engine and to easily enable GPU portability.
Currently, it is tuned for 21 cm intensity mapping, but future versions will support multi-line analyses.

Separately, BayesLIM is a 

* fast and accurate forward model visibility simulator
* generalized direction-dependent and direction-independent calibration solver
* interferometric sky imager
* signal parameterization and modeling tool
* posterior density estimator

Together, these functionalities enable BayesLIM to constrain the joint posterior of a cosmological LIM signal in addition to the complex and often poorly constrained foregrounds and instrumental response.

![flowchart](https://github.com/nkern/bayescal/blob/main/docs/source/_static/img/flowchart.png)

# Install

Clone this repo and

```bash
cd BayesLIM
pip install .
```

# Dependencies
See the `pyproject.toml` file for dependencies, listed under `[project.optional-dependences]`.
I place them here because I don't like the fact that they are automatically installed when placed in
`[project] dependencies=[]`, which often doesn't place nicely when other packages have installed pinned
versions of common Python packages.
If you'd like `pip` to automatically install dependecies anyways, then you can make this happen by installing the `dev` version,
```bash
pip install .[dev]
```

PyTorch: there is currently not a great way to install different CPU/GPU versions of PyTorch from within a `pyproject.toml`, but I suspect this will change relatively soon. In the meantime, you should install `pytorch>=2.0.0` (and optionally CUDA) on your own (<https://pytorch.org/get-started/locally/>).


