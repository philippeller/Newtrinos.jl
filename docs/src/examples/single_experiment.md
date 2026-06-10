# Single Experiment

This tutorial shows how to configure a single experiment and explore its likelihood.

## Configuring an Experiment

Each experiment has a `configure()` function that returns a struct with everything needed for analysis. Using defaults:

```julia
using Newtrinos
using DensityInterface

exp = Newtrinos.dayabay.configure()
```

The returned struct contains:
- `physics` — configured physics modules (oscillations, etc.)
- `params` — nominal parameter values (NamedTuple)
- `priors` — prior distributions (NamedTuple)
- `assets` — data, MC, interpolations, etc.
- `forward_model` — callable that maps parameters to a predicted distribution
- `plot` — visualization function

## Extracting Parameters and Priors

Wrap the experiment in a NamedTuple and use the accessor functions:

```julia
experiments = (; dayabay = exp)
params = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
```

`get_params` and `get_priors` recursively merge parameters from both the experiment and its physics modules, checking for conflicts.

## Evaluating the Likelihood

```julia
likelihood = Newtrinos.generate_likelihood(experiments)
llh = logdensityof(likelihood, params)
```

## Computing Gradients

All code is ForwardDiff-compatible:

```julia
using ForwardDiff

f(p) = logdensityof(likelihood, p)
grad = ForwardDiff.gradient(f, params)
```

## Plotting

If the experiment provides a `plot` function:

```julia
img = exp.plot(params)
```

## Running a Scan

```julia
using DataStructures

# Fix some parameters, scan over others
conditional_vars = Dict(:θ₁₂ => params.θ₁₂)
priors = Newtrinos.condition(priors, conditional_vars, params)

vars_to_scan = OrderedDict(:θ₁₃ => 21)
result = Newtrinos.scan(likelihood, priors, vars_to_scan, params)
```
