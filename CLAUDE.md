# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run unit tests (~230 tests, ~20s without Mooncake, ~40min with Mooncake on threads)
julia --project -e 'using Pkg; Pkg.test()'

# Run just the Mooncake tests (parallel rule building requires threads)
julia --project --threads=auto -e 'using Newtrinos, Test; include("test/test_mooncake.jl")'

# Run benchmarks
julia --project benchmark/bench_osc.jl
julia --project benchmark/bench_likelihood.jl --experiments dayabay
julia --project benchmark/bench_likelihood.jl --experiments deepcore super_k

# AD backend comparison benchmark (ForwardDiff vs PolyesterForwardDiff vs Mooncake)
julia --project --threads=auto benchmark/bench_ad_comparison.jl

# Run an experiment's validation script (from its directory)
cd src/experiments/icecube/deepcore_9y_verification_sample && julia --project=../../../.. test.jl

# Analysis CLI
julia --project src/analysis/analysis.jl --experiments deepcore dayabay --name myrun --task scan
julia --project src/analysis/analysis.jl --experiments deepcore orca super_k --name myrun --task profile --ad mooncake
julia --project src/analysis/analysis.jl --experiments deepcore --name myrun --task profile --workers 4
```

## Architecture

Newtrinos.jl is a neutrino physics global analysis framework with three orthogonal layers:

### Physics (`src/physics/`)
Theory predictions with no experiment knowledge. Each module returns a struct `<: Newtrinos.Physics` with `params`, `priors`, and callable functions.

- **`osc.jl`** — Core oscillation probability engine. Configurable via `OscillationConfig` with flavour models (`ThreeFlavour`, `Sterile`, `ADD`, `Darkdim_*`), interaction models (`Vacuum`, `SI`, `NSI`), and propagation models (`Basic`, `Decoherent`, `Damping`). Performance-critical: uses `SMatrix`/`SVector` for 3-flavour, `eigen` for matter effects.
- **`earth_layers.jl`** — PREM Earth density model. `compute_layers()` → `compute_paths(coszen, layers)`.
- **`atm_flux.jl`** — HKKM atmospheric neutrino fluxes with Barr systematics. Site-specific flux files in `src/physics/*.d`. The `nominal_flux` function creates a meshgrid and evaluates cubic spline interpolations; `sys_flux` applies systematic flux modifications.
- **`xsec.jl`** — Cross-section models: `SimpleScaling` or `Differential_H2O` (for Super-K). The `Differential_H2O` model uses energy-dependent CC channel ratios with MA_QE, MA_Res, and FSI systematics.
- **`cevns_xsec.jl`**, **`sns_flux.jl`** — COHERENT-specific physics.
- **`mooncake_eigen_rule.jl`** — Custom Mooncake reverse-mode AD rule for Hermitian eigen decomposition (LAPACK). Required for differentiating through matter-effect oscillation probabilities.

### Experiments (`src/experiments/`)
Each experiment module has `configure(physics=default_physics())` returning a struct `<: Newtrinos.Experiment` with fields: `physics`, `params`, `priors`, `assets`, `forward_model`, `plot`. Each experiment defines its own `default_physics()` with appropriate oscillation config, flux files, and cross-section models.

Experiment groups and their physics requirements:
- **Atmospheric** (deepcore, ic_upgrade, super_k, orca): `osc` (SI), `atm_flux`, `earth_layers`, `xsec`
- **Reactor** (dayabay, kamland, juno, tao): `osc` (Vacuum)
- **Accelerator** (minos): `osc`, `xsec`
- **COHERENT** (coherent_csi, coherent_lAr): self-contained, no physics input

Future experiments (juno, tao, ic_upgrade) have no observed data — use `generate_asimov_data` to create unfluctuated expectations, then pass to `generate_likelihood(experiments, observed)`.

### Analysis (`src/analysis/`)
Inference tools treating experiments as black boxes.

- **`analysis_tools.jl`** — `NewtrinosResult` type, `find_mle`, `profile`, `scan`, `generate_likelihood`, `get_params`/`get_priors`, `condition`, `generate_asimov_data`, `generate_toy_data`, `Wrapper` for parameter aliasing. Also contains AD backend selection via `set_ad_backend`/`select_ad`.
- **`molewhacker.jl`** — Adaptive importance sampling (`whack_a_mole`, `whack_many_moles`).
- **`cli_common.jl`** — Shared `configure_experiments()` for CLI scripts.
- **`analysis.jl`** — Main CLI entry point. Accepts `--ad auto|forwarddiff|polyester|mooncake` to select the AD backend.

## Key Patterns

**Combining experiments into a joint likelihood:**
```julia
experiments = (
    deepcore = Newtrinos.deepcore.configure(),       # uses defaults
    dayabay = Newtrinos.dayabay.configure(physics),   # custom physics override
)
params = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)
```

**Future experiments (no observed data):**
```julia
exp = Newtrinos.juno.configure()
params = Newtrinos.get_params((juno=exp,))
asimov = Newtrinos.generate_asimov_data(exp, params)
likelihood = Newtrinos.generate_likelihood((juno=exp,), (juno=asimov,))
```

**Parameters flow as NamedTuples** throughout the codebase. `get_params`/`get_priors` merge across all physics and experiment modules using `safe_merge` (checks for conflicts). Use `@reset` from Accessors.jl to modify individual fields.

**AD backend selection:**
```julia
Newtrinos.set_ad_backend(:mooncake)  # or :forwarddiff, :polyester, :auto
adsel = Newtrinos.select_ad(length(params))  # returns ADTypes backend
```

## AD Compatibility

Three AD backends are supported: **ForwardDiff**, **PolyesterForwardDiff**, and **Mooncake**.

**ForwardDiff/PolyesterForwardDiff compatibility** is critical. The oscillation code runs in the inner loop of gradient-based optimization. Avoid `Float64` literals that would strip Dual numbers; use `zero(T)`, `one(T)`, `promote_type`. Never convert computed values to concrete float types.

**Mooncake compatibility** — all 11 experiments support Mooncake reverse-mode AD. Key anti-patterns to avoid:

1. **`NamedTuple(key => f(key) for key in keys(nt))`** — Mooncake cannot differentiate through dynamic NamedTuple construction from generators. Use `map(f, nt)` or explicit static construction instead.
2. **Array splatting in vcat: `[((matrix...)...)]`** — Use `vec(matrix)` instead.
3. **File I/O or heavy object construction in the forward model** — Precompute at configuration time and store in `assets`. The `nominal_flux` function reads files and builds interpolation splines; atmospheric experiments (ORCA, IC Upgrade) precompute `flux_nominal` in `get_assets`.
4. **`sum([x[k] for k in keys(x)])`** — Creates `Vector{Any}` with problematic vcat. Use `reduce(+, values(x))` instead.
5. **`FlexTable`/`setproperty!` during differentiation** — Use pre-built typed Tables from assets.

The custom eigen rule in `mooncake_eigen_rule.jl` handles Hermitian eigendecomposition (needed for matter-effect oscillation). It wraps `_eigen_hermitian` as a Mooncake primitive with hand-written adjoint.

## Performance Notes

- `osc_prob` is the hot path: uses `SMatrix`/`SVector` for zero-allocation vacuum oscillations. Matter effects require `eigen` which allocates (~19 allocs/call).
- Response matrix contractions (Super-K) use `contract_R` with pre-flattened Float64 matrices for BLAS-accelerated matrix-vector multiply, avoiding Dual number broadcast over large arrays.
- ForwardDiff chunk size is 12 by default; with N params, gradient costs `ceil(N/12)` passes.
- Mooncake has higher startup cost (~25-35s for `build_rrule`) but constant gradient overhead (~11-14× likelihood eval) regardless of parameter count, and uses 5-7× less memory than ForwardDiff for ≥20 params.
- For small models (≤12 params): ForwardDiff is fastest (single chunk pass).
- For large models (≥20 params): Mooncake is fastest and most memory-efficient.
- Mooncake rules can be built in parallel via `Threads.@spawn` (used in test suite).
