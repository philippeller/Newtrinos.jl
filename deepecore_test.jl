# %%
using LinearAlgebra
using Distributions
import JLD2

# %%
using DataFrames


# %%
using Revise
using Newtrinos
using Newtrinos.osc


# %%
osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NND(),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )


# %%
osc = Newtrinos.osc.configure(osc_cfg)

# %%

atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec = Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec);

# %%
experiments = (

 deepcore = Newtrinos.deepcore.configure(physics),
 
);

# %%
p = Newtrinos.get_params(experiments)

# %%
using CairoMakie

# %%

all_priors = Newtrinos.get_priors(experiments)


vars_to_scan = (r=31, N=31)

modified_priors = (
    N =all_priors.N, 
    m₀= all_priors.m₀,
    r = all_priors.r,
    


    Δm²₂₁ = p.Δm²₂₁,  
    Δm²₃₁ =all_priors.Δm²₃₁ , 
    δCP = p.δCP,    
    θ₁₂ = p.θ₁₂,    
    θ₁₃= p.θ₁₃,       
    θ₂₃ = all_priors.θ₂₃
)


# %%
likelihood = Newtrinos.generate_likelihood(experiments);


# %%
result = Newtrinos.scan(likelihood, modified_priors, vars_to_scan, p)


# %%
JLD2.@save "scan_deepcore_rN_NND.jld2" result

