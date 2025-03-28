<p align="center">
<img src="docs/source/_static/img/icon_dark.jpg" width=50% height=50%>
</p>

# BayesLIM: Differentiable Bayesian Forward Models for Line Intensity Mapping

BayesLIM is a toolbox for performing end-to-end analysis of line intensity mapping (LIM) datasets in a differentiable, Bayesian forward mdoel framework.
It is built on PyTorch for its automatic differentiation engine and for easily GPU portability.
Currently, it is tuned for 21 cm intensity mapping, but future versions will support multi-spectral line analyses.

Separately, BayesLIM is a 

* fast and accurate telescope forward model
* generalized telescope calibration solver
* interferometric sky imager
* signal parameterization and modeling tool
* posterior density estimator

Together, these functionalities enable BayesLIM to constrain the joint posterior of a cosmological LIM signal in addition to the complex and often poorly constrained foregrounds and instrumental response.
The flowchart below summarizes the BayesLIM forward modeling process for a 21 cm intensity mapping experiment.

![flowchart](https://github.com/nkern/bayescal/blob/main/docs/source/_static/img/flowchart.png)

In addition to solving the LIM inverse problem of constraining a 3D cosmological field given an experiment's time-ordered dataset, BayesLIM can also be used for inverse design for experimental hardware or observational strategies.
In other words, one can begin to answer the question, "How tight should my prior model on the instrumental beam pattern be to enable a 10\% constraint on the power spectrum?"
Furthermore, one can use BayesLIM to seamlessly incorporate constraints from multiple experiments in different locations on Earth, taking data at different times, with different instrumental prior models.
Enabling the wide-range of applications for high-redshift LIM science made possible by the BayesLIM framework is ongoing work.

# Install

Clone this repo and

```bash
cd BayesLIM
pip install .
```

If installed properly, you should be able import it in Python as:
```python
import bayeslim as ba
```

# Dependencies
See the `pyproject.toml` file for dependencies, listed under `[project.optional-dependences]`.
I place them here because I don't like the fact that they are automatically installed when placed in
`[project]`, which often doesn't place nicely when other packages have installed pinned
versions of common Python packages.
If you'd like `pip` to automatically install dependecies anyways, then you can make this happen by installing the `dev` version:
```bash
pip install .[dev]
```

**PyTorch**: there is currently not a great way to install different CPU/GPU versions of PyTorch from within a `pyproject.toml`, but I suspect this will change relatively soon. In the meantime, you should install `pytorch>=2.0.0` (and optionally CUDA) on your own (<https://pytorch.org/get-started/locally/>), before installing `BayesLIM.

# Getting Started
See the `notebooks/getting_started.ipynb` to get acquinted with model building, optimization, and inference with `BayesLIM`.
Note that the core API is still under development and may undergo changes.

