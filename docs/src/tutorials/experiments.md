# Experiments

Newtrinos.jl includes likelihoods for 11 neutrino experiments across four categories.

## Atmospheric Neutrinos

These experiments require oscillation physics with matter effects (`SI`), atmospheric fluxes, Earth density profiles, and cross-sections.

| Experiment | Module | Description |
|:-----------|:-------|:------------|
| IceCube DeepCore | `Newtrinos.deepcore` | 9-year verification sample |
| Super-Kamiokande | `Newtrinos.super_k` | Atmospheric 2023 analysis |
| ORCA | `Newtrinos.orca` | KM3NeT/ORCA 6-line, 433 kton-years |
| IceCube Upgrade | `Newtrinos.ic_upgrade` | Simulated upgrade detector |

## Reactor Neutrinos

Reactor experiments use vacuum oscillations only.

| Experiment | Module | Description |
|:-----------|:-------|:------------|
| Daya Bay | `Newtrinos.dayabay` | 3158-day dataset |
| KamLAND | `Newtrinos.kamland` | 7-year dataset |
| JUNO | `Newtrinos.juno` | Simulated (no observed data) |
| TAO | `Newtrinos.tao` | Simulated (no observed data) |

## Accelerator Neutrinos

| Experiment | Module | Description |
|:-----------|:-------|:------------|
| MINOS | `Newtrinos.minos` | Sterile search, 16e20 POT |

## Coherent Elastic Neutrino-Nucleus Scattering

Self-contained experiments with their own flux and cross-section models.

| Experiment | Module | Description |
|:-----------|:-------|:------------|
| COHERENT CsI | `Newtrinos.coherent_csi` | CsI detector |
| COHERENT LAr | `Newtrinos.coherent_lAr` | Liquid Argon detector |

## Usage

Each experiment is configured via its `configure()` function:

```julia
# With default physics
exp = Newtrinos.dayabay.configure()

# With custom physics
exp = Newtrinos.deepcore.configure(physics)
```

Each experiment's subdirectory under `src/experiments/` contains a `test.jl` script that reproduces official results.
