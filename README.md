# Newtrinos.jl

**Newtrinos.jl** is a Julia package for the **global analysis of neutrino data**.

## Overview

The package is built to support flexible and modular analysis of neutrino physics, combining experimental data with theoretical models and inference tools.

## Code Structure

Newtrinos.jl is organized into three core components:

- **Experimental Likelihoods** (`src/experiments`):  
  Modules for various neutrino experiments and datasets, each encapsulating experiment-specific behavior.

- **Physics Modules** (`src/physics`):  
  Functions and tools for computing neutrino oscillation probabilities, atmospheric fluxes, and other theoretical predictions.

- **Analysis Tools** (`src/analysis`):  
  Interfaces for running inference — both **Frequentist** and **Bayesian** — using experimental and theoretical models.

## Design Philosophy

The codebase follows a **modular** and **orthogonal** architecture:

- **Experiments** only depend on their specific setup and data; they do **not** contain any theory or inference logic.
- **Physics** modules focus solely on theoretical modeling; they are unaware of experiments or statistical methods.
- **Inference** tools treat experiments and theory modules as interchangeable black boxes — allowing flexible composition.

This separation is enforced through consistent interfaces and data structures.

## Module Conventions

To ensure interoperability, each module (experimental or theoretical) should follow these conventions:

- Define all model parameters and priors using **`NamedTuple`s**.
- Experimental modules should implement the following functions:

  ```julia
  configure(config::NamedTuple)     # Configure the experiment with physics modules
  setup()                            # Initialize experiment internals
  get_forward_model()               # Return a callable model for likelihood evaluation
  plot(params::NamedTuple)          # (Optional) Visualize data or model output
  ```

Here, `config` is a `NamedTuple` containing any required physics module dependencies.

## Example Lieklihood

This section shows an example how to set up a joint likelihood.


```julia
using Newtrinos
```

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mPrecompiling Newtrinos [5b289081-bab5-45e8-97fc-86872f1653a0] (cache misses: include_dependency fsize change (4), incompatible header (6), dep missing source (2))
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mSetting new default BAT context BATContext{Float64}(Random123.Philox4x{UInt64, 10}(0xbc1844051d6d0a09, 0x762d33499ab30525, 0x66b94b0200af4152, 0x7b6b62da974bf682, 0x9d072c38dbf87fcf, 0x55a45c9058d5f8ed, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0), HeterogeneousComputing.CPUnit(), BAT._NoADSelected())


Here we specify three physics modules in our config, where we decided on a module for standard 3-flavour oscillations, and modules for computing atmospheric fluxes and Earth density profiles:


```julia
config = (
    osc = Newtrinos.osc.standard,
    atm_flux = Newtrinos.atm_flux,
    earth_layers = Newtrinos.earth_layers
)
```




    (osc = Newtrinos.osc.standard, atm_flux = Newtrinos.atm_flux, earth_layers = Newtrinos.earth_layers)



Here we choose four experimental likelihoods:


```julia
exp_modules = (Newtrinos.deepcore, Newtrinos.minos, Newtrinos.dayabay, Newtrinos.kamland)
```




    (Newtrinos.deepcore, Newtrinos.minos, Newtrinos.dayabay, Newtrinos.kamland)




```julia
[m.configure(;config...) for m in exp_modules]
```




    4-element Vector{Bool}:
     1
     1
     1
     1



The setup function should be called once for each module, which will cause the module to load its data into `module.assets`


```julia
[m.setup() for m in exp_modules]
```




    4-element Vector{Bool}:
     1
     1
     1
     1



This is enough to generate a joint likelihood for everything:


```julia
likelihood = Newtrinos.generate_likelihood(exp_modules);
```

Let's evaluate the likelihood! For this we also need parameter values. The following function goeas through both. experimental and theory modules and collects all parameters:


```julia
p = Newtrinos.get_params(exp_modules)
```




    (deepcore_lifetime = 2.5, deepcore_atm_muon_scale = 1.0, deepcore_ice_absorption = 1.0, deepcore_ice_scattering = 1.0, deepcore_opt_eff_overall = 1.0, deepcore_opt_eff_lateral = 0.0, deepcore_opt_eff_headon = 0.0, nc_norm = 1.0, nutau_cc_norm = 1.0, θ₁₂ = 0.5872523687443223, θ₁₃ = 0.1454258194533693, θ₂₃ = 0.8556288707523761, δCP = 1.0, Δm²₂₁ = 7.53e-5, Δm²₃₁ = 0.0024752999999999997, atm_flux_nunubar_sigma = 0.0, atm_flux_nuenumu_sigma = 0.0, atm_flux_delta_spectral_index = 0.0, atm_flux_uphorizonzal_sigma = 0.0, kamland_energy_scale = 0.0, kamland_geonu_scale = 0.0, kamland_flux_scale = 0.0)




```julia
using DensityInterface
```


```julia
@time logdensityof(likelihood, p)
```

      6.621959 seconds (21.56 M allocations: 1.153 GiB, 2.47% gc time, 98.63% compilation time)





    -1080.6543117055216



## Automatic Differentiation

For efficient inference, it is important to have access to gradients. therefore all code is fully differentiable via auto-diff, using the ForwardDiff package:


```julia
using ForwardDiff
```


```julia
ForwardDiff.gradient(p -> logdensityof(likelihood, p), p)
```




    (deepcore_lifetime = -191.94268496042022, deepcore_atm_muon_scale = -1.1612509099067818, deepcore_ice_absorption = 23.215534562299336, deepcore_ice_scattering = 309.74179980457177, deepcore_opt_eff_overall = -257.1048344886879, deepcore_opt_eff_lateral = 40.41140595498514, deepcore_opt_eff_headon = -33.947486872831675, nc_norm = -25.46398416438826, nutau_cc_norm = -59.46669650099511, θ₁₂ = -19.102454876163787, θ₁₃ = 420.73886060499507, θ₂₃ = -170.30277734032478, δCP = -0.3588894919058898, Δm²₂₁ = 687419.0875551306, Δm²₃₁ = 15034.795584326293, atm_flux_nunubar_sigma = -3.7454697695197274, atm_flux_nuenumu_sigma = -0.6494110485544047, atm_flux_delta_spectral_index = 262.8120902081794, atm_flux_uphorizonzal_sigma = 1.5904384931010436, kamland_energy_scale = -1.3322233658320863, kamland_geonu_scale = 1.3338116809640645, kamland_flux_scale = 1.1102273782788554)



## Inference

Let's run a likelihood analysis to construct confidence contours in the (θ₂₃, Δm²₃₁) parameter space.
Here we use a conditional likelihood for illusatration. More realistically, you may want to run `Newtrinos.profile` instead for a full profile likelihood.
Examples on Bayesian Inference will follow.


```julia
result = Newtrinos.scan(likelihood, Newtrinos.get_priors(exp_modules), (θ₂₃=11, Δm²₃₁=11), p)
```




    NewtrinosResult((θ₂₃ = [0.5235987755982988, 0.5759586531581287, 0.6283185307179586, 0.6806784082777885, 0.7330382858376183, 0.7853981633974483, 0.837758040957278, 0.890117918517108, 0.9424777960769379, 0.9948376736367678, 1.0471975511965976], Δm²₃₁ = [0.002, 0.0021, 0.0022, 0.0023, 0.0024000000000000002, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029000000000000002, 0.003]), (deepcore_lifetime = [2.5 2.5 … 2.5 2.5; 2.5 2.5 … 2.5 2.5; … ; 2.5 2.5 … 2.5 2.5; 2.5 2.5 … 2.5 2.5], deepcore_atm_muon_scale = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], deepcore_ice_absorption = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], deepcore_ice_scattering = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], deepcore_opt_eff_overall = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], deepcore_opt_eff_lateral = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], deepcore_opt_eff_headon = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], nc_norm = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], nutau_cc_norm = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], θ₁₂ = [0.5872523687443223 0.5872523687443223 … 0.5872523687443223 0.5872523687443223; 0.5872523687443223 0.5872523687443223 … 0.5872523687443223 0.5872523687443223; … ; 0.5872523687443223 0.5872523687443223 … 0.5872523687443223 0.5872523687443223; 0.5872523687443223 0.5872523687443223 … 0.5872523687443223 0.5872523687443223], θ₁₃ = [0.1454258194533693 0.1454258194533693 … 0.1454258194533693 0.1454258194533693; 0.1454258194533693 0.1454258194533693 … 0.1454258194533693 0.1454258194533693; … ; 0.1454258194533693 0.1454258194533693 … 0.1454258194533693 0.1454258194533693; 0.1454258194533693 0.1454258194533693 … 0.1454258194533693 0.1454258194533693], δCP = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], Δm²₂₁ = [7.53e-5 7.53e-5 … 7.53e-5 7.53e-5; 7.53e-5 7.53e-5 … 7.53e-5 7.53e-5; … ; 7.53e-5 7.53e-5 … 7.53e-5 7.53e-5; 7.53e-5 7.53e-5 … 7.53e-5 7.53e-5], atm_flux_nunubar_sigma = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], atm_flux_nuenumu_sigma = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], atm_flux_delta_spectral_index = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], atm_flux_uphorizonzal_sigma = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], kamland_energy_scale = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], kamland_geonu_scale = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], kamland_flux_scale = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], llh = Any[-1348.6158199397485 -1321.1743703380853 … -1265.7114020615265 -1280.5279671478045; -1262.0831327830065 -1234.2690832084893 … -1193.1275643094436 -1212.4738549775045; … ; -1253.8083295194608 -1225.8670964372093 … -1190.18484175515 -1211.3971688909617; -1337.5934227086098 -1309.9484138401403 … -1258.2566706564346 -1274.5095917508493], log_posterior = Any[-1348.6158199397485 -1321.1743703380853 … -1265.7114020615265 -1280.5279671478045; -1262.0831327830065 -1234.2690832084893 … -1193.1275643094436 -1212.4738549775045; … ; -1253.8083295194608 -1225.8670964372093 … -1190.18484175515 -1211.3971688909617; -1337.5934227086098 -1309.9484138401403 … -1258.2566706564346 -1274.5095917508493]))




```julia
using CairoMakie
```


```julia
img = plot(result)
display("image/png", img)
```


    
![png](README_files/README_22_0.png)
    



```julia

```
