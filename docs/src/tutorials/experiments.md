# Experiments

Newtrinos.jl includes likelihoods for 11 neutrino experiments across four physics categories. Every experiment exposes the same interface — `configure()`, `forward_model`, `params`, `priors`, `assets`, and `plot` — so analysis code written for one experiment transfers directly to any other. See [Configuring Experiments](configure_experiment.md) for a hands-on walkthrough.

!!! tip "Good starting points"
    - **Daya Bay** (`Newtrinos.dayabay`) for reactor neutrinos — it has no detector systematics, so the only free parameters are the oscillation angles. Great for a first likelihood evaluation.
    - **IceCube DeepCore** (`Newtrinos.deepcore`) for atmospheric neutrinos — the most complete example of the full physics stack (oscillations, flux, Earth density, cross-sections).

---

## Atmospheric Neutrinos

Atmospheric neutrino experiments detect neutrinos produced by cosmic-ray interactions in the upper atmosphere. Because the neutrinos travel through varying amounts of the Earth depending on their arrival direction, these experiments are simultaneously sensitive to $\theta_{23}$, $\Delta m^2_{31}$, and Earth-matter effects. They require the full physics stack: oscillations with matter effects (`SI`), the HKKM atmospheric flux, the PREM Earth density model, and a neutrino cross-section model.

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| IceCube DeepCore | `Newtrinos.deepcore` | 9-year verification sample |
| Super-Kamiokande | `Newtrinos.super_k` | Atmospheric 2023 analysis |
| KM3NeT/ORCA | `Newtrinos.orca` | 6-line, 433 kton-years |
| IceCube Upgrade | `Newtrinos.ic_upgrade` | Simulated upgrade detector |

Required physics modules: `osc` (with `interaction = SI()`), `atm_flux`, `earth_layers`, `xsec`.

---

## Reactor Neutrinos

Reactor experiments detect electron antineutrinos ($\bar{\nu}_e$) produced by
$\beta$-decay of fission products in nuclear power plants. At the MeV energies and
km-scale baselines involved, matter effects are negligible and the oscillation
probability is well described by vacuum oscillations alone — making these the cleanest
measurements of $\theta_{13}$ and $\Delta m^2_{21}$. Reactor experiments in Newtrinos.jl
require only the oscillation module.

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| Daya Bay | `Newtrinos.dayabay` | 3158-day near/far comparison |
| KamLAND | `Newtrinos.kamland` | 7-year reactor disappearance |
| JUNO | `Newtrinos.juno` | Simulated (design sensitivity) |
| TAO | `Newtrinos.tao` | Simulated (design sensitivity) |

!!! note
    JUNO and TAO are simulated experiments with no observed data. Their `forward_model` generates expected event spectra against an Asimov dataset, making them useful for sensitivity studies and future projections.

Required physics modules: `osc` (with `interaction = Vacuum()`).

---

## Accelerator Neutrinos

Accelerator-based experiments use a controlled $\nu_\mu$ beam produced from pion decays at a proton target. The known beam composition and energy spectrum allow precise measurements of $\theta_{23}$, $\Delta m^2_{31}$, and sterile-neutrino mixing.

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| MINOS | `Newtrinos.minos` | Sterile neutrino search, $16 \times 10^{20}$ POT |

Required physics modules: `osc` (sterile models supported), `xsec`.

---

## Coherent Elastic Neutrino-Nucleus Scattering (CEvNS)

The COHERENT experiments at the Spallation Neutron Source (SNS) at Oak Ridge measure Coherent Elastic neutrino-Nucleus Scattering — a Standard Model process first observed in 2017. These experiments are self-contained: they include their own SNS flux model and CEvNS cross-section model, and do not depend on any oscillation physics. They are primarily used to constrain the weak mixing angle, nuclear form factors, and BSM
couplings such as NSI.

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| COHERENT CsI | `Newtrinos.coherent_csi` | CsI[Na] scintillator detector |
| COHERENT LAr | `Newtrinos.coherent_lAr` | Liquid argon detector |

Required physics modules: none (self-contained).

---

## Validation Scripts

Each experiment's subdirectory under `src/experiments/` contains a `test.jl` script
that runs the forward model and reproduces plots from the corresponding published
analysis. These scripts serve as the ground-truth reference for each likelihood
implementation and are a good place to look when working with an unfamiliar experiment:

```bash
# Run from the Newtrinos.jl root directory
cd src/experiments/icecube/deepcore_9y_verification_sample
julia --project=../../../.. test.jl
```
