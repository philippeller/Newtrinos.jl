# Physics API

## Oscillations

The oscillation module is accessed via `Newtrinos.osc`. Use [`Newtrinos.osc.configure`](@ref) to
create an [`Newtrinos.osc.Osc`](@ref) physics module that can be passed to any experiment.

### Configuration

```@docs
Newtrinos.osc.OscillationConfig
Newtrinos.osc.Osc
Newtrinos.osc.configure
```

### Data Types

```@docs
Newtrinos.osc.ftype
Newtrinos.osc.Layer
Newtrinos.osc.Path
```

### Core Functions

```@docs
Newtrinos.osc.get_osc_prob
Newtrinos.osc.get_matrices
Newtrinos.osc.get_PMNS
Newtrinos.osc.get_abs_masses
```

### Flavour Models

```@docs
Newtrinos.osc.FlavourModel
Newtrinos.osc.ThreeFlavour
Newtrinos.osc.ThreeFlavourXYCP
Newtrinos.osc.Sterile
Newtrinos.osc.ADD
```

### Parameters and Priors

```@docs
Newtrinos.osc.get_params
Newtrinos.osc.get_priors
```

### Interaction Models

```@docs
Newtrinos.osc.InteractionModel
Newtrinos.osc.Vacuum
Newtrinos.osc.SI
Newtrinos.osc.NSI
```

### Propagation Models

```@docs
Newtrinos.osc.PropagationModel
Newtrinos.osc.Basic
Newtrinos.osc.Decoherent
Newtrinos.osc.Damping
```

### State Selection

```@docs
Newtrinos.osc.StateSelector
Newtrinos.osc.All
Newtrinos.osc.Cut
```

### Eigendecomposition

```@docs
Newtrinos.osc.EigenMethod
Newtrinos.osc.DefaultEigen
Newtrinos.osc.decompose
Newtrinos.BargerEigen
```

### Analytic Eigendecomposition

Closed-form 3×3 Hermitian eigendecomposition underlying [`Newtrinos.BargerEigen`](@ref).
These functions are defined in `src/physics/eigen_hermitian_3x3.jl` and live in the
top-level `Newtrinos` namespace.

```@docs
Newtrinos.fast_eigen
Newtrinos._fast_eigen_efzero
Newtrinos._fast_eigen_edzero
Newtrinos._fast_eigen_dfzero
Newtrinos._sorted_eigen
```

### Internal Functions

These functions are not part of the public API but are documented for developers working
on the oscillation engine.

```@docs
Newtrinos.osc.osc_kernel
Newtrinos.osc.compute_matter_matrices
Newtrinos.osc.osc_reduce
Newtrinos.osc.matter_osc_per_e
Newtrinos.osc.select
Newtrinos.osc.propagate
```

## Earth Layers

The Earth density module is accessed via `Newtrinos.earth_layers`. Use
[`Newtrinos.earth_layers.configure`](@ref) to create an [`Newtrinos.earth_layers.EarthLayers`](@ref)
physics module providing layer structures and path computations for matter-effect oscillations.

### Density Models

```@docs
Newtrinos.earth_layers.DensityModel
Newtrinos.earth_layers.PREM
Newtrinos.earth_layers.EarthLayers
```

### Configuration and Functions

```@docs
Newtrinos.earth_layers.configure
Newtrinos.earth_layers.get_compute_layers
Newtrinos.earth_layers.compute_paths
Newtrinos.earth_layers.ray_circle_path_length
```

## Atmospheric Flux

The atmospheric flux module is accessed via `Newtrinos.atm_flux`. Use
[`Newtrinos.atm_flux.configure`](@ref) to create an [`Newtrinos.atm_flux.AtmFlux`](@ref)
physics module providing nominal HKKM fluxes and Barr systematic modifications.

### Flux Models

```@docs
Newtrinos.atm_flux.NominalFluxModel
Newtrinos.atm_flux.HKKM
Newtrinos.atm_flux.FluxSystematicsModel
Newtrinos.atm_flux.Barr
Newtrinos.atm_flux.AtmFluxConfig
Newtrinos.atm_flux.AtmFlux
```

### Configuration and Functions

```@docs
Newtrinos.atm_flux.configure
Newtrinos.atm_flux.get_params
Newtrinos.atm_flux.get_priors
Newtrinos.atm_flux.get_nominal_flux
Newtrinos.atm_flux.get_sys_flux
```

### Internal Functions

```@docs
Newtrinos.atm_flux.get_hkkm_flux
Newtrinos.atm_flux.scale_flux
Newtrinos.atm_flux.uphorizontal
Newtrinos.atm_flux.updown
```

## SNS Flux

The Spallation Neutron Source flux module is accessed via `Newtrinos.sns_flux`. Use
[`Newtrinos.sns_flux.configure`](@ref) to create an [`Newtrinos.sns_flux.SNSFlux`](@ref)
physics module for COHERENT-style experiments.

### Configuration

```@docs
Newtrinos.sns_flux.SNSFlux
Newtrinos.sns_flux.configure
Newtrinos.sns_flux.get_params
Newtrinos.sns_flux.get_priors
```

### Internal Functions

```@docs
Newtrinos.sns_flux.get_assets
Newtrinos.sns_flux.get_flux
Newtrinos.sns_flux.flux_nu_mu
Newtrinos.sns_flux.flux_nu_e
Newtrinos.sns_flux.flux_nu_mu_bar
```

## CEvNS Cross-Section

The CEvNS cross-section module is accessed via `Newtrinos.cevns_xsec`. Use
[`Newtrinos.cevns_xsec.configure`](@ref) to create a
[`Newtrinos.cevns_xsec.CevnsXsec`](@ref) physics module for COHERENT-style experiments.

### Configuration

```@docs
Newtrinos.cevns_xsec.CevnsXsec
Newtrinos.cevns_xsec.configure
Newtrinos.cevns_xsec.build_params_and_priors
```

### Internal Functions

```@docs
Newtrinos.cevns_xsec.get_assets
Newtrinos.cevns_xsec.ffsq
Newtrinos.cevns_xsec.ds
Newtrinos.cevns_xsec.get_diff_xsec
Newtrinos.cevns_xsec.get_diff_xsec_lar
Newtrinos.cevns_xsec.get_diff_xsec_csi
```

## Cross-Sections

The cross-section module is accessed via `Newtrinos.xsec`. Use [`Newtrinos.xsec.configure`](@ref)
to create a [`Newtrinos.xsec.Xsec`](@ref) physics module.

### Models

```@docs
Newtrinos.xsec.XsecModel
Newtrinos.xsec.SimpleScaling
Newtrinos.xsec.Differential_H2O
Newtrinos.xsec.Xsec
```

### Configuration and Functions

```@docs
Newtrinos.xsec.configure
Newtrinos.xsec.get_params
Newtrinos.xsec.get_priors
Newtrinos.xsec.get_scale
```

## Utility Functions

```@docs
Newtrinos.Helpers.bin
Newtrinos.Helpers.rebin
```
