# Architecture

Newtrinos.jl is organized into three orthogonal layers that enforce separation of concerns.

## Three-Layer Design

```
┌─────────────────────────────────────────────────┐
│                  Analysis Layer                  │
│   scan, profile, find_mle, nested sampling      │
│   Treats experiments as black boxes              │
├─────────────────────────────────────────────────┤
│               Experiment Layer                   │
│   deepcore, dayabay, kamland, minos, ...         │
│   Each defines forward_model, params, priors     │
├─────────────────────────────────────────────────┤
│                 Physics Layer                    │
│   osc, atm_flux, earth_layers, xsec             │
│   Theory predictions, no experiment knowledge    │
└─────────────────────────────────────────────────┘
```

### Physics Layer (`src/physics/`)

Theory predictions with no experiment knowledge. Each module returns a struct `<: Newtrinos.Physics` with `params`, `priors`, and callable functions.

- **`osc.jl`** — Neutrino oscillation probability engine
- **`earth_layers.jl`** — PREM Earth density model
- **`atm_flux.jl`** — Atmospheric neutrino fluxes (HKKM with Barr systematics)
- **`xsec.jl`** — Cross-section models
- **`cevns_xsec.jl`**, **`sns_flux.jl`** — COHERENT-specific physics

### Experiment Layer (`src/experiments/`)

Each experiment module has `configure(physics=default_physics())` returning a struct `<: Newtrinos.Experiment` with fields: `physics`, `params`, `priors`, `assets`, `forward_model`, `plot`.

### Analysis Layer (`src/analysis/`)

Inference tools treating experiments as black boxes:

- **`analysis_tools.jl`** — `NewtrinosResult`, `find_mle`, `profile`, `scan`, `generate_likelihood`
- **`molewhacker.jl`** — Adaptive importance sampling
- **`cli_common.jl`** — Shared CLI utilities

## Parameter Flow

Parameters flow as `NamedTuple`s throughout the codebase. `get_params`/`get_priors` merge across all physics and experiment modules using `safe_merge` (which checks for conflicts). Use `@reset` from Accessors.jl to modify individual fields.

## ForwardDiff Compatibility

The oscillation code runs in the inner loop of gradient-based optimization. All code avoids `Float64` literals that would strip Dual numbers; uses `zero(T)`, `one(T)`, `promote_type` instead.
