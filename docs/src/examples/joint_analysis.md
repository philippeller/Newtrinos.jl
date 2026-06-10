# Joint Analysis

This tutorial shows how to combine multiple experiments into a joint likelihood and run profile/scan analyses.

## Combining Experiments

```julia
using Newtrinos
using DensityInterface
using DataStructures

experiments = (
    deepcore = Newtrinos.deepcore.configure(),
    dayabay  = Newtrinos.dayabay.configure(),
    kamland  = Newtrinos.kamland.configure(),
    minos    = Newtrinos.minos.configure(),
)
```

Parameters and priors are automatically merged across all experiments and their physics modules:

```julia
params = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)
```

## Conditioning (Fixing Parameters)

Fix parameters to specific values before scanning:

```julia
conditional_vars = Dict(
    :θ₁₂  => params.θ₁₂,
    :δCP   => -1.89,
    :Δm²₂₁ => params.Δm²₂₁,
)
priors = Newtrinos.condition(priors, conditional_vars, params)
```

## Likelihood Scan

A scan evaluates the likelihood on a grid without optimization:

```julia
using Distributions, Accessors

@reset priors.θ₂₃ = Uniform(pi/4 - 0.2, pi/4 + 0.2)
@reset priors.Δm²₃₁ = Uniform(0.0018, 0.0028)

vars_to_scan = OrderedDict(:θ₂₃ => 31, :Δm²₃₁ => 31)
result = Newtrinos.scan(likelihood, priors, vars_to_scan, params)
```

## Profile Likelihood

A profile scan optimizes over nuisance parameters at each grid point:

```julia
vars_to_scan = OrderedDict(:θ₂₃ => 11, :Δm²₃₁ => 11)
result = Newtrinos.profile(likelihood, priors, vars_to_scan, params; cache_dir="my_profile")
```

Results are cached to disk, so interrupted runs can be resumed.

## Plotting Results

```julia
using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="θ₂₃", ylabel="Δm²₃₁")
plot!(ax, result)
save("contours.png", fig)
```

## Extracting Best Fit

```julia
bf = Newtrinos.bestfit(result)
```
