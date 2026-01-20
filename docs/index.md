
# BayesLIM
<h3>Differentiable Bayesian Forward Modeling for LIM Cosmology</h3>

BayesLIM is a toolbox for performing end-to-end analysis of line intensity mapping (LIM) datasets in a differentiable, Bayesian forward model framework.
It is built on the widely used PyTorch library, which provides the autodiff engine and easy access to GPU acceleration.
Currently, it is tuned for 21 cm intensity mapping, but future versions will support multi-tracer line intensity mapping analyses.

Separately, BayesLIM is a 

* fast and accurate LIM telescope forward model
* generalized calibration solver
* interferometric sky imager
* signal parameterization and modeling tool
* posterior density estimator

Together, these functionalities enable BayesLIM to constrain the joint posterior of a cosmological LIM signal in addition to the complex and often poorly constrained foreground signal and instrumental response.
The flowchart below, for example, summarizes the BayesLIM forward modeling process for a 21 cm intensity mapping experiment.

<p align="center">
<img src="_static/img/flowchart.png" width=100%>
</p>

BayesLIM seeks to unlock the statistical power of next-generation LIM cosmology and astrophysics by unifying traditionally disparate analysis steps into a single, end-to-end analysis chain.


## Installation

See the [installation instructions](install.md) to get running. The codebase can be found on [github](https://github.com/BayesLIM/BayesLIM).


## Table of Contents

```{toctree}
:maxdepth: 2

install
introduction
tutorials
```

## Authors

Nicholas Kern



## Citation

[Kern 2025, MNRAS 541 687K (arxiv:2504.07090)](https://arxiv.org/abs/2504.07090)
```
@ARTICLE{Kern2025,
       author = {{Kern}, Nicholas},
        title = "{A differentiable, end-to-end forward model for 21 cm cosmology: estimating the foreground, instrument, and signal joint posterior}",
      journal = {\mnras},
     keywords = {methods: data analysis, techniques: interferometric, (cosmology:) dark ages, reionization, first stars, Cosmology and Nongalactic Astrophysics, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = aug,
       volume = {541},
       number = {2},
        pages = {687-713},
          doi = {10.1093/mnras/staf1007},
archivePrefix = {arXiv},
       eprint = {2504.07090},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.541..687K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


## Acknowledgements

Reionization simulation graphic: Alvarez et al. 2009 ApJ 703L.167A



## Support

Support for BayesLIM has come from:

* The MIT Pappalardo Fellowship in Physics
* The NASA Hubble Fellowship grant #HST-HF2-51533.001-A awarded by the Space Telescope Science Institute.
