# Newtrinos.jl

**Newtrinos.jl** is a Julia package for the **global analysis of neutrino data**, fully open source and free to use under the MIT license

## Overview

The package is built to support flexible and modular analysis of neutrino physics, combining experimental data with physics models and inference tools.

## Code Structure

Newtrinos.jl is organized into three core components:

- **Experimental Likelihoods** (`src/experiments`):  
  Modules for various neutrino experiments and datasets, each encapsulating experiment-specific behavior.

- **Physics Modules** (`src/physics`):  
  Functions and tools for computing physics, such as neutrino oscillation probabilities, atmospheric fluxes, and other theoretical predictions.

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

- Physics Modules should upon configuration return a struct of abstract type Newtrinos.Physics that contains at least the follwoing:
```julia
params::NamedTuple     # Nominal values of the parameters concerning the module
priors::NamedTuple     # Priors (Distributions) for the parameters of the module
```
And in addition provided some functionality to be used by experiments, for instance some functions.
  
- Experiments should return a struct of abstract type Newtrinos.Experiment that contains the follwoing:
```julia
physics::NamedTuple     # The configured physics module structs for that module
params::NamedTuple      # Nominal values of the parameters concerning the module
priors::NamedTuple      # Priors (Distributions) for the parameters of the module
assets::NamedTuple      # all (meta)data the module needs, such as MC and other data.
                        # This NamedTuple is also expected to have a field `observed` that contains the observed data
forward_model::Function # A callable model for likelihood evaluation
plot::Function          # (Optional) Visualize data or model output
```

## Example Lieklihood

This section shows an example how to set up a joint likelihood.


```julia
using Newtrinos
```

    [ Info: Precompiling Newtrinos [5b289081-bab5-45e8-97fc-86872f1653a0] (cache misses: include_dependency fsize change (4), incompatible header (12))
    [ Info: Setting new default BAT context BATContext{Float64}(Random123.Philox4x{UInt64, 10}(0xc71940e4338f89b2, 0xaa127cd6c08b4f7f, 0x1619f45f6b1c7b1a, 0xeb9be264c54b83fd, 0x6d27ffc7012eba73, 0xd648d7d68f0df426, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0), HeterogeneousComputing.CPUnit(), BAT._NoADSelected())
    [ Info: (cuinit = HeterogeneousComputing.CPUnit(), precision = Float64, rng = Random123.Philox4x{UInt64, 10}(0xc71940e4338f89b2, 0xaa127cd6c08b4f7f, 0x1619f45f6b1c7b1a, 0xeb9be264c54b83fd, 0x6d27ffc7012eba73, 0xd648d7d68f0df426, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0), ad = ADTypes.AutoForwardDiff())
    WARNING: Method definition flat_params(Distributions.ProductDistribution{N, M, D, S, T} where T where S<:Distributions.ValueSupport where D where M where N) in module MGVI at /home/peller/.julia/packages/MGVI/Appxw/src/shapes.jl:52 overwritten in module Newtrinos at /mnt/c/Users/peller/work/Newtrinos/src/analysis/molewhacker.jl:31.
    ERROR: Method overwriting is not permitted during Module precompilation. Use `__precompile__(false)` to opt-out of precompilation.
    ┌ Info: Skipping precompilation due to precompilable error. Importing Newtrinos [5b289081-bab5-45e8-97fc-86872f1653a0].
    └   exception = Error when precompiling module, potentially caused by a __precompile__(false) declaration in the module.
    [ Info: Setting new default BAT context BATContext{Float64}(Random123.Philox4x{UInt64, 10}(0x512eaf914e0adbcf, 0x46fcd693d4f09955, 0x3034e380c1fa6459, 0x7fd3578f4e4d8dcb, 0xa10b2020a6d67384, 0x426f46282821c86c, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0), HeterogeneousComputing.CPUnit(), BAT._NoADSelected())
    [ Info: (cuinit = HeterogeneousComputing.CPUnit(), precision = Float64, rng = Random123.Philox4x{UInt64, 10}(0x512eaf914e0adbcf, 0x46fcd693d4f09955, 0x3034e380c1fa6459, 0x7fd3578f4e4d8dcb, 0xa10b2020a6d67384, 0x426f46282821c86c, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0), ad = ADTypes.AutoForwardDiff())


We need to specify the physics we want to use.
Here we decided on a module for standard 3-flavour oscillations with basic propagation, all states used for oscillations, and standard interactions (matter effects).
Furthermore we use modules for computing atmospheric fluxes, Earth density profiles, and cross sections.

The Earth density model can be configured as fixed (`PREM`, the default) or variable (`VariableDensity`), which adds a `matter_density_scale` nuisance parameter with a 6.8% uncertainty on the Earth's electron density.


```julia
osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.ThreeFlavour(),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg)
atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()  # fixed PREM densities
# earth_layers = Newtrinos.earth_layers.configure(Newtrinos.earth_layers.VariableDensity())  # parameterized densities
xsec = Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec);
```

Here we choose four experimental likelihoods. The experimental likelihoods need to be configured with the physics. here we use the same physics for all modules, and each module grabs whatever it needs.


```julia
experiments = (
    deepcore = Newtrinos.deepcore.configure(physics),
    dayabay = Newtrinos.dayabay.configure(physics),
    kamland = Newtrinos.kamland.configure(physics),
    minos = Newtrinos.minos.configure(physics),
    orca = Newtrinos.orca.configure(physics),
);
```

    [ Info: Loading deepcore data
    [ Info: Loading dayabay data
    [ Info: Loading kamland data
    [ Info: Loading minos data


This is enough to generate a joint likelihood for everything:


```julia
likelihood = Newtrinos.generate_likelihood(experiments);
```

Let's evaluate the likelihood! For this we also need parameter values. The following function goes through both. all experimts and all their theory modules and collects all parameters:


```julia
p = Newtrinos.get_params(experiments)
```




    (atm_flux_delta_spectral_index = 0.0, atm_flux_nuenumu_sigma = 0.0, atm_flux_nunubar_sigma = 0.0, atm_flux_uphorizonzal_sigma = 0.0, deepcore_atm_muon_scale = 1.0, deepcore_ice_absorption = 1.0, deepcore_ice_scattering = 1.0, deepcore_lifetime = 2.5, deepcore_opt_eff_headon = 0.0, deepcore_opt_eff_lateral = 0.0, deepcore_opt_eff_overall = 1.0, kamland_energy_scale = 0.0, kamland_flux_scale = 0.0, kamland_geonu_scale = 0.0, nc_norm = 1.0, nutau_cc_norm = 1.0, orca_energy_scale = 1.0, orca_norm_all = 1.0, orca_norm_he = 1.0, orca_norm_hpt = 1.0, orca_norm_muons = 1.0, orca_norm_showers = 1.0, Δm²₂₁ = 7.53e-5, Δm²₃₁ = 0.0024752999999999997, δCP = 1.0, θ₁₂ = 0.5872523687443223, θ₁₃ = 0.1454258194533693, θ₂₃ = 0.8556288707523761)




```julia
using DensityInterface
@time logdensityof(likelihood, p)
```

      0.411396 seconds (1.21 M allocations: 159.695 MiB, 12.85% gc time, 46.79% compilation time)





    -2243.85131243232



If the experiment provides a plotting function, we can make convenient plots:


```julia
img = experiments.minos.plot(p)
display("image/png", img)
```


    
![png](README_files/README_13_0.png)
    


## Automatic Differentiation

For efficient inference, it is important to have access to gradients. therefore all code is fully differentiable via auto-diff, using the ForwardDiff package:


```julia
using ForwardDiff
```


```julia
f(p) = logdensityof(likelihood, p)
```




    f (generic function with 1 method)




```julia
@time ForwardDiff.gradient(f, p)
```

      2.119425 seconds (3.67 M allocations: 3.105 GiB, 28.51% gc time)





    (atm_flux_delta_spectral_index = 767.3930553910443, atm_flux_nuenumu_sigma = 0.2932231207919994, atm_flux_nunubar_sigma = 1.4273368970672733, atm_flux_uphorizonzal_sigma = 7.580889446841681, deepcore_atm_muon_scale = -1.554139984910445, deepcore_ice_absorption = 21.90125067470847, deepcore_ice_scattering = 317.00690255688073, deepcore_lifetime = -198.82833941678624, deepcore_opt_eff_headon = -34.218501767097806, deepcore_opt_eff_lateral = 40.71226525108279, deepcore_opt_eff_overall = -273.662684969215, kamland_energy_scale = -1.3322233658320863, kamland_flux_scale = 1.1102273782788554, kamland_geonu_scale = 1.3338116809640645, nc_norm = -34.68743333444963, nutau_cc_norm = -61.93074228378721, orca_energy_scale = -656.9413726516052, orca_norm_all = 236.28296294297638, orca_norm_he = 65.95711884913455, orca_norm_hpt = 61.38394666379235, orca_norm_muons = 5.543985312339062, orca_norm_showers = 15.986288015598543, Δm²₂₁ = 702846.212096108, Δm²₃₁ = -10564.304287452665, δCP = -0.514728745795177, θ₁₂ = -21.444471819004114, θ₁₃ = 443.7825545005947, θ₂₃ = -172.4771736700132)



## Inference

Let's run a likelihood analysis to construct confidence contours in the (θ₂₃, Δm²₃₁) parameter space.
Here we use a conditional likelihood for illusatration. More realistically, you may want to run `Newtrinos.profile` instead for a full profile likelihood.
For more realistic examples, browse the GitHub directories of the different experiments under ´src/experiments´, which contain GitHub-style ´README.md´ pages that show realistic reproductions of official results.


```julia
priors = Newtrinos.get_priors(experiments)
```




    (atm_flux_delta_spectral_index = Truncated(Distributions.Normal{Float64}(μ=0.0, σ=0.1); lower=-0.3, upper=0.3), atm_flux_nuenumu_sigma = Truncated(Distributions.Normal{Float64}(μ=0.0, σ=1.0); lower=-3.0, upper=3.0), atm_flux_nunubar_sigma = Truncated(Distributions.Normal{Float64}(μ=0.0, σ=1.0); lower=-3.0, upper=3.0), atm_flux_uphorizonzal_sigma = Truncated(Distributions.Normal{Float64}(μ=0.0, σ=1.0); lower=-3.0, upper=3.0), deepcore_atm_muon_scale = Distributions.Uniform{Float64}(a=0.0, b=2.0), deepcore_ice_absorption = Truncated(Distributions.Normal{Float64}(μ=1.0, σ=0.1); lower=0.85, upper=1.15), deepcore_ice_scattering = Truncated(Distributions.Normal{Float64}(μ=1.0, σ=0.1); lower=0.85, upper=1.15), deepcore_lifetime = Distributions.Uniform{Float64}(a=2.0, b=4.0), deepcore_opt_eff_headon = Distributions.Uniform{Float64}(a=-5.0, b=2.0), deepcore_opt_eff_lateral = Truncated(Distributions.Normal{Float64}(μ=0.0, σ=1.0); lower=-2.0, upper=2.0), deepcore_opt_eff_overall = Truncated(Distributions.Normal{Float64}(μ=1.0, σ=0.1); lower=0.8, upper=1.2), kamland_energy_scale = Truncated(Distributions.Normal{Float64}(μ=0.0, σ=1.0); lower=-3.0, upper=3.0), kamland_flux_scale = Truncated(Distributions.Normal{Float64}(μ=0.0, σ=1.0); lower=-3.0, upper=3.0), kamland_geonu_scale = Distributions.Uniform{Float64}(a=-0.5, b=0.5), nc_norm = Truncated(Distributions.Normal{Float64}(μ=1.0, σ=0.2); lower=0.4, upper=1.6), nutau_cc_norm = Truncated(Distributions.Normal{Float64}(μ=1.0, σ=0.2); lower=0.4, upper=1.6), orca_energy_scale = Truncated(Distributions.Normal{Float64}(μ=1.0, σ=0.09); lower=0.7, upper=1.3), orca_norm_all = Distributions.Uniform{Float64}(a=0.5, b=1.5), orca_norm_he = Truncated(Distributions.Normal{Float64}(μ=1.0, σ=0.5); lower=0.0, upper=3.0), orca_norm_hpt = Distributions.Uniform{Float64}(a=0.5, b=1.5), orca_norm_muons = Distributions.Uniform{Float64}(a=0.0, b=2.0), orca_norm_showers = Distributions.Uniform{Float64}(a=0.5, b=1.5), Δm²₂₁ = Distributions.Uniform{Float64}(a=6.5e-5, b=9.0e-5), Δm²₃₁ = Distributions.Uniform{Float64}(a=0.002, b=0.003), δCP = Distributions.Uniform{Float64}(a=0.0, b=6.283185307179586), θ₁₂ = Distributions.Uniform{Float64}(a=0.4205343352839651, b=0.7853981633974483), θ₁₃ = Distributions.Uniform{Float64}(a=0.1, b=0.2), θ₂₃ = Distributions.Uniform{Float64}(a=0.5235987755982988, b=1.0471975511965976))




```julia
result = Newtrinos.scan(likelihood, priors, (θ₂₃=31, Δm²₃₁=31), p)
```

    Progress: 100%|█████████████████████████████████████████| Time: 0:00:29





    NewtrinosResult((θ₂₃ = [0.5235987755982988, 0.5410520681182421, 0.5585053606381853, 0.5759586531581287, 0.593411945678072, 0.6108652381980153, 0.6283185307179586, 0.6457718232379018, 0.6632251157578452, 0.6806784082777885  …  0.890117918517108, 0.9075712110370513, 0.9250245035569946, 0.9424777960769379, 0.9599310885968813, 0.9773843811168245, 0.9948376736367678, 1.012290966156711, 1.0297442586766543, 1.0471975511965976], Δm²₃₁ = [0.002, 0.002033333333333333, 0.0020666666666666667, 0.0021, 0.0021333333333333334, 0.0021666666666666666, 0.0022, 0.0022333333333333333, 0.002266666666666667, 0.0023  …  0.0027, 0.0027333333333333333, 0.002766666666666667, 0.0028, 0.0028333333333333335, 0.0028666666666666667, 0.0029000000000000002, 0.0029333333333333334, 0.002966666666666667, 0.003]), (atm_flux_delta_spectral_index = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], atm_flux_nuenumu_sigma = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], atm_flux_nunubar_sigma = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], atm_flux_uphorizonzal_sigma = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], deepcore_atm_muon_scale = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], deepcore_ice_absorption = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], deepcore_ice_scattering = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], deepcore_lifetime = [2.5 2.5 … 2.5 2.5; 2.5 2.5 … 2.5 2.5; … ; 2.5 2.5 … 2.5 2.5; 2.5 2.5 … 2.5 2.5], deepcore_opt_eff_headon = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], deepcore_opt_eff_lateral = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], deepcore_opt_eff_overall = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], kamland_energy_scale = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], kamland_flux_scale = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], kamland_geonu_scale = [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], nc_norm = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], nutau_cc_norm = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], orca_energy_scale = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], orca_norm_all = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], orca_norm_he = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], orca_norm_hpt = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], orca_norm_muons = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], orca_norm_showers = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], Δm²₂₁ = [7.53e-5 7.53e-5 … 7.53e-5 7.53e-5; 7.53e-5 7.53e-5 … 7.53e-5 7.53e-5; … ; 7.53e-5 7.53e-5 … 7.53e-5 7.53e-5; 7.53e-5 7.53e-5 … 7.53e-5 7.53e-5], δCP = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], θ₁₂ = [0.5872523687443223 0.5872523687443223 … 0.5872523687443223 0.5872523687443223; 0.5872523687443223 0.5872523687443223 … 0.5872523687443223 0.5872523687443223; … ; 0.5872523687443223 0.5872523687443223 … 0.5872523687443223 0.5872523687443223; 0.5872523687443223 0.5872523687443223 … 0.5872523687443223 0.5872523687443223], θ₁₃ = [0.1454258194533693 0.1454258194533693 … 0.1454258194533693 0.1454258194533693; 0.1454258194533693 0.1454258194533693 … 0.1454258194533693 0.1454258194533693; … ; 0.1454258194533693 0.1454258194533693 … 0.1454258194533693 0.1454258194533693; 0.1454258194533693 0.1454258194533693 … 0.1454258194533693 0.1454258194533693], llh = Any[-2531.4797803458123 -2521.8096768626456 … -2465.8215957828675 -2471.6684194523004; -2494.5863970630285 -2484.87117986747 … -2434.845192581074 -2441.2200026652495; … ; -2482.1341723664837 -2472.2799416447842 … -2425.622654453538 -2432.4969016045643; -2517.9197269736997 -2508.0949741271925 … -2454.52242998112 -2460.8053775948447], log_posterior = Any[-2531.4797803458123 -2521.8096768626456 … -2465.8215957828675 -2471.6684194523004; -2494.5863970630285 -2484.87117986747 … -2434.845192581074 -2441.2200026652495; … ; -2482.1341723664837 -2472.2799416447842 … -2425.622654453538 -2432.4969016045643; -2517.9197269736997 -2508.0949741271925 … -2454.52242998112 -2460.8053775948447]))




```julia
using CairoMakie
img = plot(result, levels=[0, 0.68, 0.9, 0.99])
display("image/png", img)
```


    
![png](README_files/README_21_0.png)
    


## Further Reading / Examples

Please consult the subdirectories of the various experimental datasets under `src/experiments/x/y` to find more examples. Each of those subdirectory contains a julia script `test.jl` that is aimed at reproducing official results.

# References

Newtrinos has been used to produce the results presented in:
* [Testing the number of neutrino species with a global fit of neutrino data](https://arxiv.org/abs/2402.00490) Published in: Phys.Rev.D 109 (2024) 9, 095016
* [Constraints on non-unitary neutrino mixing in light of atmospheric and reactor neutrino data](https://arxiv.org/abs/2407.20388) Published in: JHEP 05 (2025) 130
* [A neutrino data analysis of extra-dimensional theories with massive bulk fields](https://arxiv.org/abs/2508.04274) Published in: Phys.Rev.D 112 (2025) 5, 055009

