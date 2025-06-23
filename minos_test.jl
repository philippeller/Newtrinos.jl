
# %%
using LinearAlgebra
using Distributions
#using Plots
using LaTeXStrings
using Printf
using FileIO
import JLD2

# %%
using DataFrames
#using StatsPlots

# %%
#using Revise
using Newtrinos
using Newtrinos.osc

# %%
#using Pkg
#Pkg.status() 

# %%
osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NND(),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg)

# %%

atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec=Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec);

# %%
experiments = (
 
   minos= Newtrinos.minos.configure(physics),
);

# %%
p = Newtrinos.get_params(experiments)

# %%
img = experiments.minos.plot(p)
#display("image/png", img)
#save("/home/sofialon/Newtrinos.jl/natural plot/minos/minos_data_NND_20.png", img)

# %%

all_priors = Newtrinos.get_priors(experiments)


vars_to_scan = (r=31,N=31)  

modified_priors = (
    N = all_priors.N, 
    m₀= p.m₀,
    r =all_priors.r,
    
   
  

    Δm²₂₁ = p.Δm²₂₁,  
    Δm²₃₁ = p.Δm²₃₁ , 
    δCP = p.δCP,    
    θ₁₂ = p.θ₁₂,    
    θ₁₃= p.θ₁₃,       
    θ₂₃ = p.θ₂₃   
    

)


# %%

likelihood = Newtrinos.generate_likelihood(experiments);


# %%

result = Newtrinos.scan(likelihood, modified_priors, vars_to_scan, p)

# %%
#likelihood = Newtrinos.generate_likelihood(experiments);
#result = Newtrinos.scan(likelihood, Newtrinos.get_priors(experiments), (θ₁₃=31, Δm²₃₁=31), p)
#result = Newtrinos.profile(likelihood,  Newtrinos.get_priors(experiments), (r=31, m₀=31), p; gradient_map=false)

# %%
JLD2.@save "scan_minos_rN_NND.jld2" result

# %%
using CairoMakie

# %%
img = CairoMakie.plot(result)
#display("image/png", img)
save("/home/sofialon/Newtrinos.jl/natural plot/minos/minos_rN_NND.png", img)


