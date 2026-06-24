# Contributing to Newtrinos.jl

Thank you for your interest in contributing! Newtrinos.jl is a open-source neutrino physics analysis framework, and contributions of all sizes are welcome — whether you are a physicist adding a new physics model, an experimentalist integrating a new dataset, a software engineer improving performance, or a writer fixing a typo in the documentation.
We aim to make the codebase a welcoming place for researchers and developers alike.

---

## Ways to Contribute

### Core Physics (`src/physics/`)

Theoretical models that produce predictions with no knowledge of any specific experiment. Examples of valuable contributions:

- New oscillation models (new `FlavourModel`, `InteractionModel`, or `PropagationModel` subtypes)
- Updated oscillation parameters or priors from global fits
- New cross-section models (subtype `Newtrinos.Physics`, follow the `xsec.jl` pattern)
- Atmospheric or beam flux models beyond HKKM

### Experimental Likelihoods (`src/experiments/`)

Data analysis modules that wrap real experimental results. Each experiment lives in its own subdirectory. Examples of valuable contributions:

- New experiments (DUNE, NOvA, T2K, SNO+, Borexino, …)
- Updated datasets for existing experiments (new run periods, improved systematics)
- Bug fixes or improvements to existing forward models

### Analysis & Statistics (`src/analysis/`)

Inference tooling that treats experiments as black boxes. Useful directions:

- Improved minimisers or MLE solvers for `find_mle`
- New profile / scan strategies
- Bayesian samplers or importance-sampling improvements (`molewhacker.jl`)
- Distributed / parallel analysis utilities

### Documentation & Examples

Good documentation is as valuable as good code:

- Fixing typos or unclear explanations anywhere under `docs/src/`
- Writing new tutorials or Jupyter / Pluto notebooks
- Adding docstrings to exported functions that currently lack them
- Improving the API reference pages under `docs/src/api/`

---

## Architectural & Design Rules

Newtrinos.jl is built on strict decoupling between physics theory, experimental likelihoods, and statistical analysis. Please respect these rules when contributing.

### Abstract Type Interfaces

Every new physics module **must** subtype `Newtrinos.Physics` and expose:

```julia
@kwdef struct MyModel <: Newtrinos.Physics
    params::NamedTuple   # nominal parameter values
    priors::NamedTuple   # Distributions.jl distributions, one per parameter
    # ... callable closure fields
end
```

Every new experiment module **must** subtype `Newtrinos.Experiment` and expose all six fields:

```julia
@kwdef struct MyExperiment <: Newtrinos.Experiment
    physics::NamedTuple       # physics modules consumed by this experiment
    params::NamedTuple        # experiment-specific nuisance parameter values
    priors::NamedTuple        # priors for experiment-specific nuisance parameters
    assets::NamedTuple        # preloaded data, binning, MC — computed once at configure time
    forward_model::Function   # params -> Distribution (a Distributions.jl object)
    plot::Function            # (params[, data]) -> Figure
end
```

The canonical entry point is always `configure(physics=default_physics()) -> MyExperiment`. Always define `default_physics()` so the experiment works standalone.

### Experimental Independence

Experimental modules must treat the physics layer as an **interchangeable black box**. An experiment module must **not**:

- Import or depend on a specific physics struct by name (e.g., do not reference `Osc` directly)
- Hard-code oscillation parameter values or mass orderings
- Bypass `params` / `priors` by using module-level global constants for fit parameters

The physics object is passed in through `configure(physics)` and accessed only via its documented callable closures (`physics.osc.osc_prob(...)`, etc.).

### Parameter Naming

All parameters flow as `NamedTuple`s throughout the framework. `get_params` / `get_priors` merge across all physics and experiment modules using `safe_merge`, which **errors on key conflicts**. To avoid clashes:

- Prefix experiment-specific nuisance parameters with the experiment name (e.g., `deepcore_ice_absorption`, not just `ice_absorption`)
- Use `@reset` from `Accessors.jl` to modify individual NamedTuple fields non-destructively

### Performance & Type Stability

The oscillation forward model runs inside the inner loop of gradient-based optimisation. ForwardDiff propagates `Dual` numbers through the entire call chain; **any accidental cast to `Float64` silently breaks gradients**.

Rules for performance-critical code:

- Use `zero(T)`, `one(T)`, `promote_type` instead of `0.0`, `1.0` literals
- Never call `convert(Float64, x)` or `Float64(x)` on values that flow through `forward_model`
- Check type stability with `@code_warntype` or `Test.@inferred` before opening a PR
- Prefer `SMatrix` / `SVector` for small fixed-size linear algebra (used throughout `osc.jl`)
- Run the benchmarks after any change to the oscillation or likelihood hot path:

```bash
julia --project benchmark/bench_osc.jl
julia --project benchmark/bench_likelihood.jl --experiments deepcore dayabay
```

---

## Step-by-Step Contribution Workflow

### 1. Discuss First (for Non-Trivial Changes)

Before writing significant code, **open a GitHub issue** to describe what you want to add or change. This avoids duplicate effort, surfaces design concerns early, and lets maintainers give feedback on the proposed interface before you invest time implementing it.

For small fixes (typos, docstrings, one-line bug fixes) you can skip this step and go straight to a pull request.

### 2. Fork & Clone

```bash
git clone https://github.com/philippeller/Newtrinos.jl.git
cd Newtrinos.jl
```

### 3. Set Up the Environment

Activate the local package environment and install all dependencies:

```julia
using Pkg
Pkg.activate(".")          # or Pkg.activate("/path/to/Newtrinos.jl")
Pkg.instantiate()
```

Or equivalently from the shell:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### 4. Create a Branch

Use a descriptive branch name that summarises the change:

```bash
git checkout -b feat/add-nova-experiment
git checkout -b fix/dayabay-covariance-sign
git checkout -b docs/improve-osc-tutorial
```

### 5. Implement Your Changes

Follow the architectural rules above. A few practical tips:

- Copy the structure of the nearest existing module rather than starting from scratch (e.g., copy `src/experiments/daya_bay/` as a template for a new reactor experiment)
- Keep physics and experiment logic separated — if you find yourself duplicating physics code inside an experiment module, it belongs in `src/physics/` instead
- Add docstrings to every exported function, including **physical units** explicitly (see the Style section below)

### 6. Style & Formatting

This project uses **[JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl)**. Before committing, format your code:

```julia
using JuliaFormatter
format("src/")
format("test/")
```

Docstring conventions:

- Every exported function and type must have a docstring
- State physical units explicitly (e.g., `energy` in GeV, `baseline` in km, `coszen` dimensionless)
- For physics functions, include a brief statement of the model or equation implemented

Example:

```julia
"""
    osc_prob(E, paths, layers, params; anti=false) -> Array{T,4}

Compute oscillation probabilities P(να → νβ) for neutrino energies `E` (GeV) and
Earth paths defined by `paths` and `layers` (output of `compute_paths`).

Returns an array of shape `(n_E, n_L, n_flav, n_flav)` where index `[i, j, β, α]`
gives P(να → νβ) at energy `E[i]` along path `j`. Set `anti=true` for antineutrinos.
"""
```

### 7. Write Tests

- Add tests for any new physics function in the appropriate file under `test/`
- For new experiments, add at minimum a smoke test that calls `configure()` and evaluates `forward_model` at the default parameters
- If your change intentionally alters any likelihood value, update `test/test_regression.jl` with the new reference values and explain the change in your PR description

Run the full test suite locally before opening a PR:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

### 8. Push & Open a Pull Request

```bash
git push origin feat/add-nova-experiment
```

Then open a pull request on GitHub. The PR description should include:

- A summary of what the change does and why
- For physics changes: a reference to the paper or derivation, or a brief mathematical summary, to help maintainers cross-verify the implementation (see the Citation Policy below)
- For experiment updates: the data release citation and any changed likelihood values

CI will automatically run the test suite on Julia stable. All tests must pass before a PR can be merged.

---

## Scientific Data & Citation Policy

Neutrino physics depends on carefully curated experimental data. If your contribution adds or updates experimental assets (observed spectra, Monte Carlo samples, response matrices, efficiency tables):

1. **Cite the official collaboration publication** in the PR description. Preferred format: DOI or arXiv identifier (e.g., `arXiv:2311.XXXXX` or `https://doi.org/10.1103/PhysRevLett.XXX`).

2. **State the data release version or run period** (e.g., "Daya Bay 3158-day dataset, PRD 95 (2017) 072006").

3. **Include the data licence** if the collaboration has released data under a specific open-data licence. If the data is proprietary or requires collaboration membership to access, note this clearly so users know the experiment will not be reproducible from the public repository alone.

4. **Physics cross-verification**: For new forward models or new theoretical predictions, include either a reference to a published validation or a brief derivation in the PR description so maintainers can verify the physics before merging.

---

## Code of Conduct

This project follows the [Julia Community Standards](https://julialang.org/community/standards/). All contributors are expected to treat each other with respect, regardless of background, experience level, or institutional affiliation. Please be constructive in code review and welcoming to newcomers. If you experience or witness behaviour that violates these standards, you can report it via the channels described on the Julia community page.

---

## Questions?

If you are unsure about anything — the right place to put a new file, whether an interface change is in scope, how to handle a tricky physics edge case — please open an issue or start a discussion on GitHub. We would rather answer a question early than review a large PR that goes in the wrong direction.

Happy contributing!
