(introduction)=

# Introduction

Here we will introduce some core concepts for building a differentiable Bayesian forward model in `BayesLIM`.

At a high-level, the basic API pattern for a component in a multi-component `BayesLIM` forward model is the "Module" (or the component) and it's associated "Response" function. You can think of the "Response" function as any linear or non-linear mapping that maps the parameters of that component from its native space (e.g. spherical harmonic coefficients) to the space required by the forward model (e.g. a pixelized sky distribution).
This holds for any component because any component, be it the sky distribution, calibration gains as a function of frequency and time, or the antenna beam pattern with respect to angle and frequency, can have a native parameterization that is defined in a compressed space. If it isn't, it can just take an identity responese.

Therefore, every component of the forward model will at some point do this:

```python
output = Response(parameters)
```

and in `BayesLIM` (currently) the user is given lots of flexibility on how to create their chosen `Response` function given the application at hand.

We can then chain together multiple of components to create our forward model, which is then simulated and compared against "real" data to compute a loss, which we then backpropagate against to derive gradients. Therefore, conceptually, a simple forward model might look something like this:

```python
# get the sky brightness distribution
sky_pixels = SkyResponse(sky_params)

# get the primary beam distribution
beam_pixels = BeamResponse(beam_params)

# integrate over the sky to get visibilities
simulated_visibilities = RIME(sky_pixels, beam_pixels)

# compute a loss against real data given data weights (however you want)
loss = MSELoss(simulated_visibilities, real_visibilities, weights)

# backpropagate
loss.backward()

# get gradients, populated by the loss.backward() call
grads = (sky_params.grad, beam_params.grad)
```

Ultimately, the goal is to compute the gradients of the loss function with respect to the input parameters, which is where the autodiff engine comes into play. Where does the Bayesian part of the Bayesian forward model come in? Well, that comes in by optimizing a posterior distribution as our loss, as opposed to any old loss function. To create a posterior distribution, we need to specify a likelihood and **priors**. Currently, only Gaussian likelihoods are supported.

## Specifying Priors

In every `BayesLIM` component we have the option to specify a prior distribution. Because the computational graph always links back to the top-level parameters, we can place priors on the parameters themselves or on their intermediate representations (the outputs of `Response(params)`), either of which might be better motivated depending on the application. For example, within the `sky_model.PointSky()` class we can set priors on the model like

```python
import bayeslim as ba
import torch

# initialize parameters
sky_params = torch.randn(...)

# instantiate model with a chosen response function
response = ba.sky_model.SkyResponse(...)
model = ba.sky_model.PointSky(sky_params, R=response, ...)

# set priors on the model
mean, cov = ...
sky_prior = ba.optim.LogGaussPrior(mean, cov)  # a Gaussian prior model
model.set_priors(sky_prior)
```
One can also easily create their own prior model by subclassing the base class, `ba.optim.BaseLogPrior`.

If we wrap this component into a *log-posterior*, then the evaluation of the likelihood and prior is taken care of for us automatically. This is found in the `ba.optim.LogProb` object, which defines a strictly Gaussian likelihood along with whatever prior model one has attached to their models. It looks something like this,

```python
# specify targets "y" for the likelihood (this is just the real data)
target = ba.dataset.Dataset(real_visibilities)  # subclass of torch.utils.data.Dataset
prob = ba.optim.LogProb(model, target, compute='post')

# take a forward pass of the model and compute: log-posterior = log-likelihood + log-prior
logpost = prob()

# now backpropagate against this
logpost.backward()

# now collect gradients
grads = (model.params.grad,)
```

<div class="alert alert-block alert-info"><b>Note: </b>The parameters of an instantiated model are <b>always</b> stored as <code>model.params</code></div> 
