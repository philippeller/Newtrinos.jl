# Analysis API

The analysis layer provides inference tools that treat experiments as black boxes.
It is composed of three source files:

- **`analysis_tools.jl`** — core types, parameter utilities, likelihood generation,
  scanning, profiling, and data generation. All symbols are part of the `Newtrinos`
  module.
- **`molewhacker.jl`** — adaptive importance sampling via MGVI Gaussian mixture
  refinement. All symbols are part of the `Newtrinos` module.
- **`cli_common.jl`** — helper functions for CLI scripts. These are standalone
  utilities loaded via `include` and are **not** exported from the `Newtrinos` module.

---

## Parameter Utilities

These functions collect and merge parameters and priors across physics and experiment
modules. `condition` and `correlated_priors_vars` allow fixing or correlating
individual parameters before constructing a posterior.

```@docs
Newtrinos.get_params
Newtrinos.get_priors
Newtrinos.condition
Newtrinos.safe_merge
Newtrinos.correlated_priors_vars
```

---

## Data Generation

```@docs
Newtrinos.generate_asimov_data
Newtrinos.generate_toy_data
```

---

## Adaptive Importance Sampling

The `molewhacker` tools implement an adaptive importance sampling strategy. Starting
from an initial Gaussian mixture proposal, each iteration fits a new local MGVI
Gaussian component at the highest-weight sample point and adds it to the mixture. 
This "whack-a-mole" strategy adaptively targets undersampled regions of the posterior.
`whack_many_moles` is the recommended entry point for production runs. 

```@docs
Newtrinos.make_prior_samples
Newtrinos.make_init_samples
Newtrinos.whack_a_mole
Newtrinos.whack_many_moles
Newtrinos.importance_sampling
Newtrinos.local_MGVI_approx
```

---

## Likelihood Construction

```@docs
Newtrinos.generate_likelihood
Newtrinos.get_observed
Newtrinos.get_fwd_model
```

---

## Scanning and Profiling

`scan` evaluates the joint likelihood on a fixed grid without optimization.
`profile` minimizes over nuisance parameters at each grid point.
`find_mle` / `find_mle_cached` locate the global maximum-likelihood estimate.

```@docs
Newtrinos.find_mle
Newtrinos.find_mle_cached
Newtrinos.profile
Newtrinos.scan
Newtrinos.bestfit
Newtrinos.generate_scanpoints
Newtrinos.assemble_profile_results
```

---

## Utilities

```@docs
Newtrinos.add_meta!
```

---

## Internal API

Low-level helpers used internally by the scanning and profiling pipeline.

```@docs
Newtrinos.sort_nt
Newtrinos._generate_grid
Newtrinos._profile
```

---

## CLI Usage

### `analysis.jl`

`src/analysis/analysis.jl` is the main command-line entry point for running analyses.
It is invoked with `julia --project`:

```bash
julia --project src/analysis/analysis.jl \
  --experiments deepcore dayabay \
  --name myrun \
  --task scan
```

#### Command-line Arguments

| Argument | Type | Required | Description |
|---|---|---|---|
| `--experiments` | `String...` | Yes | Space-separated experiment names (e.g. `deepcore dayabay`) |
| `--name` | `String` | Yes | Base name for output files |
| `--task` | `String` | Yes | One of `Scan`, `Profile`, `ImportanceSampling`, `NestedSampling` |
| `--plot` | flag | No | If set, render and save a PNG plot of the result |
| `--workers` | `Int` | No | Number of distributed workers (default: `1`) |
| `--threads` | `Int` | No | Threads per worker when `--workers > 1` (default: `1`) |

#### Task Modes

- **`scan`** — evaluates the joint likelihood on a fixed parameter grid defined in the
  script body. No nuisance-parameter optimization. Fast.
- **`profile`** — minimizes over nuisance parameters at each grid point via L-BFGS.
  Supports distributed execution via `--workers`. Results are checkpointed per row.
- **`importancesampling`** — runs [`whack_many_moles`](@ref) to produce
  importance-weighted posterior samples from a Gaussian mixture approximation.
- **`nestedsampling`** — runs `ReactiveNestedSampling` from UltraNest.

#### Output Files

| File | Contents |
|---|---|
| `<name>.jld2` | `NewtrinosResult` (scan/profile) or sample dict (sampling) |
| `<name>.png` | Plot of the result (only with `--plot`) |
| `<name>/` | Per-iteration checkpoint directory (profile and importance sampling) |

#### Physics Configuration

To use non-default oscillation models (inverted ordering, sterile neutrinos, custom
flux, etc.), uncomment and edit the override block near the top of the script.

#### Helper Functions (`cli_common.jl`)

The following functions are defined in `src/analysis/cli_common.jl` and loaded by CLI
scripts via `include`. They are **not** exported from the `Newtrinos` module.

---

**`configure_experiments(experiment_list)`**

Configure a list of experiments using each experiment's built-in defaults. Looks up
each name in the `Newtrinos` module (case-insensitive), calls its `configure()` method,
and returns a `NamedTuple` keyed by lowercased experiment name.

```julia
experiments = configure_experiments(["deepcore", "dayabay"])
# (deepcore = ..., dayabay = ...)
```

---

**`configure_experiments(experiment_list, physics)`**

Like the single-argument form but passes a shared `physics` object to each
experiment's `configure(physics)` method, allowing a custom oscillation or
cross-section configuration to be shared across all experiments.

```julia
experiments = configure_experiments(["deepcore", "dayabay"], physics)
```

---

**`save_result(result, name)`**

Save a `NewtrinosResult` to `<name>.jld2` in the current working directory using JLD2.

```julia
save_result(result, "myrun")   # writes myrun.jld2
```

---

**`plot_result(result, name, vars_to_scan; title=nothing)`**

Render a scan or profile result and save it as `<name>.png`. Requires a Makie backend
(e.g. `using CairoMakie`) to be loaded in the caller's scope before calling this
function. The x-axis is labelled with the first scanned parameter; the y-axis shows
`-2ΔLLH` (1-D) or the second scanned parameter (2-D).

```julia
using CairoMakie
plot_result(result, "myrun", vars_to_scan; title="DeepCore scan")
```

#### Examples

```bash
# 2-D scan over θ₂₃ and Δm²₃₁ with DeepCore + Daya Bay, save PNG
julia --project src/analysis/analysis.jl \
  --experiments deepcore dayabay \
  --name joint_scan \
  --task scan \
  --plot

# Profile likelihood using 4 distributed workers, 2 threads each
julia --project src/analysis/distributed_profile.jl \
  --experiments deepcore \
  --name dc_profile \
  --workers 4 \
  --threads 2

# Importance sampling posterior
julia --project src/analysis/analysis.jl \
  --experiments deepcore \
  --name dc_posterior \
  --task importancesampling
```
