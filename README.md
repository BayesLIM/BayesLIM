<p align="center">
<img src="docs/source/_static/img/icon_dark.jpg" width=50% height=50%>
</p>

# BayesLIM

<h3>Differentiable Bayesian Forward Modeling for LIM Cosmology</h3>

BayesLIM is a toolbox for performing end-to-end analysis of line intensity mapping (LIM) datasets in a differentiable, Bayesian forward model framework.
It is built on the widely used PyTorch library, which provides easy access to GPU acceleration.
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
<img src="docs/source/_static/img/flowchart.png" width=100% height=100%>
</p>

## Documentation

See the documentation at <https://bayeslim.readthedocs.io> for more details.

## Installation

For installation, see <https://bayeslim.readthedocs.io/en/latest/install.html>.

## Citation

[Kern 2025](https://arxiv.org/abs/2504.07090)

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
