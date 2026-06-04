# Physics Model Tutorial

This page shows how to configure, inspect, and use the physics modules in Newtrinos.jl directly — without attaching any experiment. For the underlying theory, see [Neutrino Oscillation Physics](neutrino_physics.md).

---

## Configuring the Oscillation Module

### Default Configuration

The simplest way to get a fully configured oscillation module is to call `configure()` with no arguments. This uses standard three-flavour vacuum oscillations with basic propagation, all oscillating states, and best-fit parameter values as defaults.

```julia
using Newtrinos

osc = Newtrinos.osc.configure()
```

The returned `Osc` struct contains the nominal parameter values and prior distributions:

```julia
osc.params   # NamedTuple: θ₁₂, θ₁₃, θ₂₃, δCP, Δm²₂₁, Δm²₃₁, ...
osc.priors   # NamedTuple: prior distributions for each parameter
```

### Custom Configuration

Pass an `OscillationConfig` to choose a different physics model. The config accepts four independent axes: flavour model, interaction model, propagation model, and eigendecomposition algorithm.

```julia
osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour     = Newtrinos.osc.ThreeFlavour(ordering = :IO), # inverted mass ordering
    interaction = Newtrinos.osc.SI(),                         # standard MSW matter effects
    propagation = Newtrinos.osc.Basic(),                      # coherent propagation
)

osc = Newtrinos.osc.configure(osc_cfg)
```

Available sub-models per axis:

| Axis | Options |
|---|---|
| `flavour` | `ThreeFlavour`, `Sterile`, `ADD`, `Darkdim_*` |
| `interaction` | `Vacuum`, `SI`, `NSI` |
| `propagation` | `Basic`, `Decoherent`, `Damping` |

---

## Assembling the Full Physics Module

Atmospheric-neutrino experiments require not just oscillations but also a flux model, an Earth density model, and a cross-section model. Configure each independently and assemble them into a `NamedTuple`:

```julia
osc          = Newtrinos.osc.configure()
atm_flux     = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec         = Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec)
```

To collect all parameters and priors across the full physics module:

```julia
params = Newtrinos.get_params(physics)
priors = Newtrinos.get_priors(physics)
```

`get_params` merges the parameter `NamedTuple`s from every sub-module using `safe_merge`, which errors if any parameter name appears twice (preventing silent conflicts).

---

## Modifying Parameters

All parameter collections are immutable `NamedTuple`s. Use `@reset` from [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl) to create a modified copy without mutating the original:

```julia
using Accessors

p = Newtrinos.get_params(physics)

# Set θ₂₃ to maximal mixing
@reset p.θ₂₃ = pi/4

# Switch to CP conserving model by setting δCP = 0
@reset p.δCP = 0 

# Scale the atmospheric flux normalisation
@reset p.atm_flux_nuenumu_sigma = 0.5
```

Each `@reset` returns a new `NamedTuple`; chain them or assign to a new variable.

---

## Computing Oscillation Probabilities Directly

### Vacuum Oscillations

Define energy and baseline grids, then call `osc.osc_prob`:

```julia
E = 10 .^ range(-1, 2, 50)   # 50 log-spaced energies from 0.1 to 100 GeV
L = [295.0, 810.0, 1300.0]   # baselines in km (T2K, NOvA, DUNE)

probs = osc.osc_prob(E, L, params)
# probs has shape (n_E, n_L, n_flav, n_flav)
# probs[i, j, β, α] = P(να → νβ) with flavour index 1=e, 2=μ, 3=τ
```

Extracting specific channels (flavour index: 1 = e, 2 = μ, 3 = τ):

```julia
P_mumu     = probs[:, :, 2, 2]  # νμ survival, shape (n_E, n_L)
P_mue      = probs[:, :, 1, 2]  # νμ → νe appearance, shape (n_E, n_L)
P_mutau    = probs[:, :, 3, 2]  # νμ → ντ appearance, shape (n_E, n_L)
```

Each column of `P_mumu` gives the survival probability vs. energy for one baseline.

### Matter Effects (Earth Crossing)

For matter effects, compute the layer structure and cosine-zenith paths from the Earth density model, then pass them to `osc_prob`:

```julia
layers = earth_layers.compute_layers()               # StructVector{Layer}
cz     = collect(range(-1.0, 0.0, 30))              # 30 upward-going zenith angles
paths  = earth_layers.compute_paths(cz, layers)      # VectorOfVectors{Path}

# Use matter-effect variant (interaction = SI() required in osc_cfg)
probs_matter = osc.osc_prob(E, paths, layers, params)
# shape (n_E, n_cz, n_flav, n_flav)
```

Matter effects are most significant for upward-going ($\cos\theta_z \approx -1$) GeV-scale neutrinos traversing the full Earth diameter.

---

## Antineutrinos

Pass `anti=true` to `osc_prob` to compute antineutrino oscillation probabilities. This internally conjugates the PMNS matrix ($U \to U^*$), which flips the sign of the CP-violating term and reverses the MSW resonance condition:

```julia
probs_anti = osc.osc_prob(E, L, params; anti=true)

# Compare νμ survival vs ν̄μ survival
P_mumu_nu   = probs[:, :, 2, 2]
P_mumu_anti = probs_anti[:, :, 2, 2]
```

The difference `P_mumu_nu - P_mumu_anti` is the CP asymmetry, which is non-zero when
$\delta_{CP} \neq 0, \pi$.

---

## Accessing the PMNS Matrix

### From Parameters Directly

`get_PMNS` constructs the full $3\times 3$ PMNS mixing matrix from a parameter `NamedTuple` containing the mixing angles and CP phase:

```julia
U = Newtrinos.osc.get_PMNS(params)
# U is a 3×3 complex SMatrix; rows = flavor (e,μ,τ), columns = mass (1,2,3)

U[1, 3]   # Ue3 = sin(θ₁₃) e^{-iδ}
abs2(U[2, 3])  # |Uμ3|² ≈ sin²θ₂₃ cos²θ₁₃
```

### From the Configured `Osc` Struct

The `matrices` function stored in `osc` gives the in-matter-corrected mixing matrix and mass-squared eigenvalues (useful when computing matter-modified oscillations manually):

```julia
U_eff, h = osc.matrices(params)
# U_eff : 3×3 unitary mixing matrix (vacuum: same as get_PMNS)
# h     : vector of mass-squared eigenvalues [eV²]
```

---

## Accessing Absolute Neutrino Masses

`get_abs_masses` reconstructs the three absolute neutrino masses from the mass splittings and the lightest-neutrino mass $m_0$ (set to 0 by `configure()` by default):

```julia
m1, m2, m3 = Newtrinos.osc.get_abs_masses(params)
# returns masses in eV

# Lightest mass is m₀ = params.m₀; change it with @reset:
@reset params.m₀ = 0.05   # eV
m1, m2, m3 = Newtrinos.osc.get_abs_masses(params)
```

Note: oscillation probabilities depend only on the mass splittings $\Delta m^2_{ij}$, not on the absolute masses. Absolute masses are relevant for e.g. neutrinoless double beta decay or cosmological constraints.

---

## Accessing the Atmospheric Flux Directly

The `AtmFlux` module provides direct access to the HKKM atmospheric neutrino flux on an energy–zenith-angle grid, with and without systematic shifts.

### Nominal Flux

```julia
E_grid  = 10 .^ range(-1, 2, 50)    # 50 log-spaced energies [GeV]
cz_grid = collect(range(-1, 1, 20)) # 20 cosine-zenith bins

flux = atm_flux.nominal_flux(E_grid, cz_grid)
# flux is a Table with columns:
#   flux.nue, flux.numu, flux.nuebar, flux.numubar
#   flux.true_energy, flux.true_coszen
```

Each flux column is a 2-D array of shape `(n_E, n_cz)` giving the unoscillated differential flux $d\Phi/dE\,d\Omega$ at each grid point.

### Systematic Flux Modifications

To apply Barr systematic parameters (spectral tilt, ν/ν̄ ratio, up/horizontal ratio):

```julia
flux_mod = atm_flux.sys_flux(flux, params)
# Returns NamedTuple(:nue, :numu, :nuebar, :numubar) of shifted flux arrays
```

The systematic parameters (e.g. `atm_flux_delta_spectral_index`) are included in `params` automatically when `get_params` is called on a physics tuple containing `atm_flux`.

---

## Cross-Section Scale Factor

The `xsec` module provides a multiplicative scale factor applied to the neutrino interaction cross-section inside experiment forward models:

```julia
# SimpleScaling model (default):
factor = xsec.scale(:numu, :cc, params)   # :cc or :nc; flavour as Symbol
```

This is mostly used internally by experiments. Direct use is mainly relevant if you want to study cross-section systematic effects in isolation.

---

## Summary

| What | How | Returns |
|---|---|---|
| Custom Configuration | `Newtrinos.osc.OscillationConfig(flavour, interaction, propagation, states, eigen_method)` | `OscillationConfig` struct |
| Oscillation module | `Newtrinos.osc.configure(osc_cfg)` | `Osc` struct shape `(cfg, params, priors, matrices, osc_prob)`|
| Oscillation probabilities (vacuum) | `osc.osc_prob(E, L, params)` | `Array` shape `(n_E, n_L, 3, 3)` |
| Oscillation probabilities (matter) | `osc.osc_prob(E, paths, layers, params)` | `Array` shape `(n_E, n_cz, 3, 3)` |
| Antineutrino probabilities | `osc.osc_prob(E, L, params; anti=true)` | same shape |
| PMNS matrix | `Newtrinos.osc.get_PMNS(params)` | `SMatrix{3,3,Complex}` |
| Matter mixing matrix + eigenvalues | `osc.matrices(params)` | `(U, h)` |
| Absolute neutrino masses | `Newtrinos.osc.get_abs_masses(params)` | `(m1, m2, m3)` in eV |
| Earth layer structure | `earth_layers.compute_layers()` | `StructVector{Layer}` |
| Neutrino paths through Earth | `earth_layers.compute_paths(cz, layers)` | `VectorOfVectors{Path}` |
| Nominal atmospheric flux | `atm_flux.nominal_flux(E_grid, cz_grid)` | `Table` with flux columns |
| Systematic flux shifts | `atm_flux.sys_flux(flux, params)` | `NamedTuple` of flux arrays |
| Cross-section scale | `xsec.scale(flav, interaction, params)` | scalar |

See the [Physics API Reference](../api/physics.md) for the full list of types, functions, and configuration options.
