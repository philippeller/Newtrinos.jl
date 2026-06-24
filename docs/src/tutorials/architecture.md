# Architecture
We start with understanding the structure and architecture of the package in order to see how everything works together and then will later focus on specific use cases.

## Three-Layer Design

Newtrinos.jl is organized into three orthogonal layers that enforce separation of concerns. Each layer treats the other layers as black boxes, allowing flexible composition for statistical data analysis of neutrino data.

- **Physics layer**: configure the physics theory predictions and physical parameters and priors
- **Experiment layer**: configure the different experiments and their parameters, priors, and forward models
- **Analysis layer**: inference tools for both Frequentist (profile likelihood) and Bayesian (nested sampling, importance sampling) methods

The layers communicate through two well-defined interfaces: `NamedTuple`s of parameters and priors, and callable functions stored in structs. Physics modules expose callables that experiments use inside their forward models. Experiments expose a `forward_model` callable that the analysis layer uses to build the joint likelihood — without ever inspecting the internals of either.

| Layer | Exposes | Consumed by |
|---|---|---|
| Physics | `params`, `priors`, callable functions (e.g. `osc_prob`, `compute_paths`) | Experiments |
| Experiments | `params`, `priors`, `forward_model(params)` | Analysis |
| Analysis | `NewtrinosResult`, scan/profile grids, posterior samples | User |

In the following we will take a detailed view on each layer.

---

### Physics Layer (`src/physics/`)

Theory predictions with no experiment knowledge. Each module returns a struct `<: Newtrinos.Physics` with at minimum `params::NamedTuple` and `priors::NamedTuple`, plus module-specific callable functions that experiments call inside their forward models.

- **`osc.jl`** — Neutrino oscillation probability engine. Exposes `osc_prob(E, paths, params)`, which returns oscillation probabilities for a given energy array and set of matter-layer traversal paths.
- **`earth_layers.jl`** — PREM Earth density model. Exposes `compute_layers(params)` (returns concentric density shells) and `compute_paths(cz, layers)` (returns per-baseline layer traversals for a cosine-zenith array).
- **`atm_flux.jl`** — Atmospheric neutrino fluxes (HKKM with Barr systematics). Exposes `nominal_flux(params)` (unoscillated flux on an energy–angle grid) and `sys_flux(params)` (flux modified by systematic parameters).
- **`xsec.jl`** — Neutrino cross-section models. Exposes `scale(params)`, a scale factor applied to the interaction cross-section.
- **`cevns_xsec.jl`**, **`sns_flux.jl`** — COHERENT-specific physics (CEvNS cross-section and SNS neutrino flux). Self-contained; used only by the COHERENT experiments.

All physics modules are configured via `configure(cfg)` (or `configure()` for defaults). The oscillation module supports a rich set of sub-models selected through `OscillationConfig`:

| Axis | Options |
|---|---|
| Flavour | `ThreeFlavour`, `Sterile`, `ADD`, `Darkdim_*` |
| Interaction | `Vacuum`, `SI`, `NSI` |
| Propagation | `Basic`, `Decoherent`, `Damping` |

See the [Physics API Reference](../api/physics.md) for the full list of types and functions.

---

### Experiment Layer (`src/experiments/`)

Each experiment module has `configure(physics=default_physics())` returning a struct `<: Newtrinos.Experiment` with the following fields:

| Field | Type | Description |
|---|---|---|
| `physics` | `NamedTuple` | The configured physics module structs passed in at configure time |
| `params` | `NamedTuple` | Nominal values of all experiment-specific parameters |
| `priors` | `NamedTuple` | Prior distributions for all experiment-specific parameters |
| `assets` | `NamedTuple` | Read-only data (MC templates, observed counts, response matrices, etc.) |
| `forward_model` | `Function` | `forward_model(params) -> prediction` — maps a full parameter vector to event-rate predictions |
| `plot` | `Function` | Visualize observed data vs. model prediction |

Calling `configure()` without arguments uses a `default_physics()` defined inside each experiment, with the oscillation model and other physics chosen to match the experiment's analysis.

Experiments are grouped by their physics requirements:

| Group | Experiments | Physics required |
|---|---|---|
| Atmospheric | `deepcore`, `ic_upgrade`, `super_k`, `orca` | `osc`, `atm_flux`, `earth_layers`, `xsec` |
| Reactor | `dayabay`, `kamland`, `juno`, `tao` | `osc` |
| Accelerator | `minos` | `osc`, `xsec` |
| COHERENT | `coherent_csi`, `coherent_lAr` | self-contained (no external physics input) |

---

### Analysis Layer (`src/analysis/`)

Inference tools treat experiments as black boxes. The standard **Frequentist** workflow is:

1. **Collect parameters and priors** across all configured experiments and their physics modules:
   ```julia
   params = Newtrinos.get_params(experiments)
   priors = Newtrinos.get_priors(experiments)
   ```

2. **Build the joint likelihood** — a callable `params -> log_likelihood`:
   ```julia
   likelihood = Newtrinos.generate_likelihood(experiments)
   ```

3. **Evaluate** — choose between a fast grid scan or a full profile likelihood:
   - `scan(likelihood, priors, vars_to_scan, params)` — evaluates the likelihood at fixed grid points without optimizing nuisance parameters. Fast.
   - `profile(likelihood, priors, vars_to_scan, params)` — minimizes over nuisance parameters at each grid point via L-BFGS. Slower but gives the true profile likelihood.

For **Bayesian inference**, the molewhacker tools build an adaptive Gaussian mixture approximation to the posterior:

```julia
init = Newtrinos.make_init_samples(posterior, nseeds=10)
samples = Newtrinos.whack_many_moles(posterior, init, target_samplesize=10_000)
```

The `Wrapper` type allows aliasing parameter names across experiments (e.g. sharing a single oscillation parameter between two experiments that use different internal names).

Key source files:

- **`analysis_tools.jl`** — `NewtrinosResult`, `find_mle`, `profile`, `scan`, `generate_likelihood`, `get_params`, `get_priors`, `Wrapper`
- **`molewhacker.jl`** — adaptive importance sampling (`make_init_samples`, `whack_a_mole`, `whack_many_moles`)
- **`cli_common.jl`** — shared CLI utilities (`configure_experiments`, `save_result`, `plot_result`)

See the [Analysis API Reference](../api/analysis.md) and [Types Reference](../api/types.md) for the full list of functions and types.

---

## Parameter Flow

Parameters flow as `NamedTuple`s throughout the codebase. `get_params`/`get_priors` merge across all physics and experiment modules using `safe_merge` (which checks for conflicts). Use `@reset` from Accessors.jl to modify individual fields. Parameters are always passed by value — no mutation occurs anywhere in the stack, which keeps the code fully compatible with ForwardDiff dual-number propagation.

## ForwardDiff Compatibility

The oscillation code runs in the inner loop of gradient-based optimization. All code avoids `Float64` literals that would strip Dual numbers; uses `zero(T)`, `one(T)`, `promote_type` instead. This means gradients of the full joint likelihood with respect to any parameter are available automatically via `ForwardDiff.gradient`.
