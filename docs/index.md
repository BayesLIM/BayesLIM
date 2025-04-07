
# BayesLIM
<h3>Differentiable Bayesian Forward Modeling for LIM Cosmology</h3>

BayesLIM is a toolbox for performing end-to-end analysis of line intensity mapping (LIM) datasets in a differentiable, Bayesian forward model framework.
It is built on the widely used PyTorch library, which provides the autodiff engine and easy access to GPU acceleration.
Currently, it is tuned for 21 cm intensity mapping, but future versions will support multi-spectral line analyses.

Separately, BayesLIM is a 

* fast and accurate LIM telescope forward model
* generalized calibration solver
* interferometric sky imager
* signal parameterization and modeling tool
* posterior density estimator

Together, these functionalities enable BayesLIM to constrain the joint posterior of a cosmological LIM signal in addition to the complex and often poorly constrained foreground and instrumental response.
The flowchart below, for example, summarizes the BayesLIM forward modeling process for a 21 cm intensity mapping experiment.

<p align="center">
<img src="_static/img/flowchart.png" width=100%>
</p>

## Installation

See the [installation instructions](install.md) to get running.

## Table of Contents

```{toctree}
:maxdepth: 2

install
introduction
tutorials
```

## Authors

Nicholas Kern

## Acknowledgements

Kern 2025
<br>
Reionization simulation graphic: Alvarez et al. 2009 ApJ 703L.167A
