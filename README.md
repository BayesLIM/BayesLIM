# BayesLIM

### Bayesian Line Intensity Mapping: Calibration, Imaging, and Signal Separation

BayesLIM is a tool for performing end-to-end analysis of line intensity mapping (LIM) datasets in a Bayesian framework.
Separately, BayesLIM is a 

* fast and accurate forward model visibility simulator
* generalized direction-dependent and direction-independent calibration solver
* interferometric sky imager
* sky signal parameterization and modeling tool
* constrained optimizer and MCMC sampler

Together, these functionalities enable BayesLIM to constrain the joint posterior of a cosmological LIM signal in addition to a complex and often unknown foreground and instrumental response.

![flowchart](https://github.com/nkern/bayescal/blob/main/docs/source/_static/img/flowchart.png)

# Install

Clone this repo and

```bash
cd BayesLIM
pip install .
```

# Dependencies

### Primary Dependencies 
* pytorch >= 1.7.0
* numpy
* scipy
* astropy
* h5py

### Optional Dependencies
* sklearn
* mpmath
* healpy
* pyuvdata
