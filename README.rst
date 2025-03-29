.. raw:: html

	<h3 align="center">
	<a href="your href link"><img width="400" src="docs/source/_static/img/icon_dark.jpg" alt="text"></a>
	</h3>

	<h1 align="left">BayesLIM
	</h1>
	<h3>Differentiable, End-to-End Bayesian Forward Models for LIM Cosmology</h3>

BayesLIM is a toolbox for performing end-to-end analysis of line intensity mapping (LIM) datasets in a differentiable, Bayesian forward model framework.
It is built on PyTorch for its automatic differentiation engine and for easy GPU portability.
Currently, it is tuned for 21 cm intensity mapping, but future versions will support multi-spectral line analyses.

Separately, BayesLIM is a 

* fast and accurate telescope forward model
* generalized telescope calibration solver
* interferometric sky imager
* signal parameterization and modeling tool
* posterior density estimator

Together, these functionalities enable BayesLIM to constrain the joint posterior of a cosmological LIM signal in addition to the complex and often poorly constrained foregrounds and instrumental response.
The flowchart below summarizes the BayesLIM forward modeling process for a 21 cm intensity mapping experiment.

.. raw:: html

	<h3 align="center">
	<a href="your href link"><img width="800" src="docs/source/_static/img/flowchart.png" alt="text"></a>
	</h3>

In addition to solving the LIM inverse problem of constraining a 3D cosmological field given an experiment's time-ordered dataset, BayesLIM can also be used for inverse design for experimental hardware or observational strategies.
In other words, one can begin to answer the question, "How tight should my prior model on the instrumental beam sidelobes be to enable a 10\% constraint on the 21 cm power spectrum?"
Furthermore, one can use BayesLIM to seamlessly incorporate constraints from multiple experiments in different locations on Earth, taking data at different times, with different instrumental prior models.
Enabling the wide-range of applications for high-redshift LIM science made possible by the BayesLIM framework is ongoing work.

See the documentation at `https://bayeslim.readthedocs.io <https://bayeslim.readthedocs.io>`_ for more details.

Install
-------

Clone this repo and

.. code-block:: bash

	cd BayesLIM
	pip install .

If installed properly, you should be able to import it in Python as:

.. code-block:: python

	import bayeslim as ba

Dependencies
------------

See the ``pyproject.toml`` file for dependencies, listed under ``[project.optional-dependences]``.
I place them here so that they are not automatically installed during ``pip install .``.
If you'd like pip to automatically install dependencies anyways, you can make this happen by installing the ``dev`` version:

.. code-block:: bash

	pip install .[dev]

**PyTorch**: there is currently not a great way to install different CPU/GPU versions of PyTorch from within a ``pyproject.toml``, but I suspect this will change relatively soon. In the meantime, you should install ``pytorch>=2.0.0`` (and optionally CUDA) on your own (`https://pytorch.org/get-started/locally/ <https://pytorch.org/get-started/locally/>`_), before installing BayesLIM.

Getting Started
---------------
See the ``notebooks/getting_started.ipynb`` to get acquinted with model building, optimization, and inference with BayesLIM.
Note that the core API is still under development and may undergo changes.

Authors
-------
Nicholas Kern, University of Michigan, MIT, NASA

Acknowledgements
-----------------
Kern 2025 in prep.

Reonization simulation graphic: Alvarez et al. 2009 ApJ 703L.167A
