# Data Analysis Tutorial

This tutorial shows how to go from a configured experiment to analysis results — both Frequentist (profile likelihood, grid scan, MLE) and Bayesian (adaptive importance sampling). For experiment configuration, see [Configuring Experiments](configure_experiment.md).

---

## Setup

The code below is shared by all subsequent examples. It configures two experiments with default physics, merges their parameters and priors, and builds the joint likelihood.

```julia
using Newtrinos
using DensityInterface
using DataStructures
using Accessors

experiments = (
    deepcore = Newtrinos.deepcore.configure(),
    dayabay  = Newtrinos.dayabay.configure(),
)

params = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)
```

`likelihood` is a callable that accepts a `NamedTuple` of parameters and via `logdensityof(likelihood, params)` we get the log-likelihood value.

---

## Frequentist Analysis

### Finding the Maximum Likelihood Estimate

Before scanning a grid it is useful to locate the global likelihood maximum. `find_mle` uses L-BFGS optimization starting from the nominal parameters. It requires a prior *distribution* (not a NamedTuple), which you build with `distprod`:

```julia
using BAT

prior_dist = distprod(; priors...)
llh, log_posterior, best_params = Newtrinos.find_mle(likelihood, prior_dist, params)

# best_params is a NamedTuple of optimized parameter values
println("Best-fit θ₂₃ = ", best_params.θ₂₃)
println("Best-fit log-likelihood = ", llh)
```

!!! tip
    Use `best_params` as the starting point for subsequent scans and profiles — grid points are placed around the prior range, but `params` provides the nuisance starting values for optimization.

### Conditioning: Fixing Parameters

A full joint analysis can have dozens of free parameters. To scan only one or two physics parameters, you can **fix** the others to their best-fit or nominal values using `condition`. This replaces their priors with a fixed constant, so the optimizer ignores them.

```julia
# Fix solar parameters and δCP; only θ₂₃ and Δm²₃₁ will be free
priors_cond = Newtrinos.condition(
    priors,
    Dict(:θ₁₂ => params.θ₁₂, :Δm²₂₁ => params.Δm²₂₁, :δCP => params.δCP),
    params,
)
```

You can also change the range of a prior for a scanned parameter using `@reset`:

```julia
# Restrict the θ₂₃ scan range to the upper octant
using Distributions
@reset priors_cond.θ₂₃ = Uniform(pi/4, pi/2)
```

### Grid Scan

A grid scan evaluates the likelihood on a fixed Cartesian grid of parameter values **without** optimizing the remaining parameters. It is fast and useful for a quick overview, but it does not marginalize over nuisance parameters — every non-scanned parameter is held at the value in `params`.

```julia
# Define the scan grid: parameter name => number of evenly spaced points
vars_to_scan = OrderedDict(:θ₂₃ => 21, :Δm²₃₁ => 21)

result = Newtrinos.scan(likelihood, priors_cond, vars_to_scan, params)
```

### Profile Likelihood

A profile likelihood scan is statistically more rigorous. At each grid point, it minimizes the negative log-likelihood over all parameters that are **not** being scanned. The call looks identical to `scan`, but uses `profile`:

```julia
result = Newtrinos.profile(
    likelihood, priors_cond, vars_to_scan, params,
    cache_dir = "my_profile_cache",   # save progress after each row
)
```

The `cache_dir` option writes a checkpoint file per grid row so that a long profile run can be resumed if interrupted.

!!! note
    Profile likelihood is significantly slower than a grid scan because it runs an L-BFGS minimization at every grid point. For an 11×11 grid with 20 nuisance parameters, expect minutes to hours depending on the experiment. Start with a coarse grid (e.g. 11×11) and refine once you know where the likelihood peak is.

### Interpreting Results

Both `scan` and `profile` return a `NewtrinosResult`:

```julia
# Grid axes (one vector per scanned parameter)
result.axes[:θ₂₃]     # vector of θ₂₃ grid values
result.axes[:Δm²₃₁]   # vector of Δm²₃₁ grid values

# Likelihood values (shape: n_θ₂₃ × n_Δm²₃₁)
result.values.llh

# Compute Δllh relative to the maximum (for confidence regions)
Δllh = result.values.llh .- maximum(result.values.llh)

# For profile results: the optimized nuisance parameter value at each grid point
result.values.θ₁₃       # best-fit θ₁₃ at each profile point
result.values.deepcore_ice_absorption  # optimized ice absorption
```

The best-fit grid point across the full scan or profile:

```julia
bf = Newtrinos.bestfit(result)
println("Best-fit θ₂₃ = ", bf.θ₂₃)
println("Best-fit Δm²₃₁ = ", bf.Δm²₃₁)
println("Best-fit log-posterior = ", bf.log_posterior)
```

Confidence regions are defined by `Δllh` thresholds: $\Delta(-2\log\mathcal{L}) < 1$ for 1-D 68% CL, and $< 2.30$ for 2-D 68% CL (Wilks' theorem).

### Saving and Loading Results

```julia
# Save to a JLD2 file (produces "my_result.jld2")
Newtrinos.save_result(result, "my_result")

# Load it back
using FileIO
result_loaded = FileIO.load("my_result.jld2", "result")
```

---

## Bayesian Analysis

Bayesian inference treats parameters as random variables and returns a full posterior distribution. Newtrinos.jl implements adaptive importance sampling via the **whack-a-mole** algorithm: it iteratively fits local Gaussian approximations at high-weight sample points, building up a mixture proposal that closely tracks the posterior shape.

### Setting up the BAT Posterior

The Bayesian sampler uses [BAT.jl](https://github.com/bat/BAT.jl). Build a prior distribution with `distprod` and wrap it with the likelihood into a `PosteriorMeasure`:

```julia
using BAT

prior_dist = distprod(; priors...)
posterior  = PosteriorMeasure(likelihood, prior_dist)
```

### Initializing the Sampler

Before running the full sampler, initialize an importance sampling approximation. `make_init_samples` finds approximate posterior modes via L-BFGS (starting from Sobol quasi-random points), fits a local Gaussian at each mode, and draws initial weighted samples from the resulting mixture:

```julia
init = Newtrinos.make_init_samples(
    posterior,
    5,       # nseeds: number of mode-finding restarts
    1_000,   # nsamples: initial importance sample count
)
```

Increase `nseeds` if the posterior is known to be multimodal (e.g. mass ordering degeneracy). Increase `nsamples` for a better initial coverage.

### Adaptive Importance Sampling

`whack_many_moles` refines the approximation iteratively. At each step it identifies the highest-weight samples, fits new Gaussian components there, and adds them to the mixture. It stops when any of the convergence criteria is met:

```julia
samples = Newtrinos.whack_many_moles(
    posterior,
    init,
    target_samplesize = 5_000,   # stop when Kish ESS exceeds this
    target_efficiency = 0.1,     # stop when ESS/n_samples exceeds this
    maxiter           = 50,      # hard iteration cap
    cache_dir         = "whack_cache",  # save checkpoint after each iteration
)
```

The Kish Effective Sample Size (ESS) measures how many independent samples the weighted collection is worth. An efficiency of 0.1 means the weighted samples are equivalent to 10% of the raw sample count — a reasonable target for most analyses.

### Working with the Posterior Samples

The result is a `NamedTuple` with three fields:

```julia
samples.approx_dist    # the final Gaussian mixture proposal
samples.samples_p      # DensitySampleVector in the transformed (normal) space
samples.samples_user   # DensitySampleVector in the original parameter space
```

For most purposes you will work with `samples_user`. It is a BAT `DensitySampleVector` where each entry carries a parameter NamedTuple and an importance weight:

```julia
# Extract all θ₂₃ values as a plain vector
theta23_vals = [s.θ₂₃ for s in samples.samples_user.v]

# The importance weights (normalized so the maximum is 1)
weights = samples.samples_user.weight

# Weighted mean and standard deviation of θ₂₃
using StatsBase
theta23_mean = mean(theta23_vals, Weights(weights))
theta23_std  = std(theta23_vals, Weights(weights))
```

You can also use BAT.jl utilities directly on the sample vector:

```julia
# Kish Effective Sample Size
ess = bat_eff_sample_size(samples.samples_user, KishESS()).result
println("ESS = ", ess, " out of ", length(samples.samples_user), " samples")

# Marginal mode of θ₂₃
mode_result = bat_marginalmode(samples.samples_user)
```

!!! tip
    If `target_samplesize` is not reached after `maxiter` iterations, the samples are still valid — just with a lower ESS. You can continue sampling by passing the returned `samples` as `init_samples` to another `whack_many_moles` call.

---

## Choosing Between Frequentist and Bayesian

| | Frequentist (scan/profile) | Bayesian (whack_many_moles) |
|---|---|---|
| **Output** | Likelihood on a grid | Weighted posterior samples |
| **Marginalization** | Profile (minimize) or none (scan) | Bayesian (integrate) |
| **Speed** | Fast (scan) to slow (profile) | Moderate to slow |
| **Best for** | Confidence contours, δ results | Posterior distributions, multimodal cases |
| **Parallelism** | `--workers` via distributed profile | `n_parallel` threads |

For the analysis API reference, see the [Analysis API Reference](../api/analysis.md).


Now you're ready to work with the Newtrinos.jl package by yourself. If you want to look at some further examples you can find them at the examples section. Have fun and play around to explore all the functionalities! 

If you want to contribute to the package functionality or have some proposals for improvement, have a look at the [contribution guidelines](../contribution_guidelines.md).