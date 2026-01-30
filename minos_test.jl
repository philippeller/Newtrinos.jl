
using LinearAlgebra
using Distributions
using LaTeXStrings
using Printf
using FileIO
import JLD2

using DataFrames

using Newtrinos
using Newtrinos.osc



osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NNM(),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg)


atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec=Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec);


experiments = (
 
   dayabay= Newtrinos.dayabay.configure(physics),
);

p = Newtrinos.get_params(experiments)



all_priors = Newtrinos.get_priors(experiments)


vars_to_scan = (r=31,N=31)  

modified_priors = (
    N = DiscreteUniform(2, 200),
    m₀= p.m₀,
    r =all_priors.r,
    
   
  

    Δm²₂₁ = p.Δm²₂₁,  
    Δm²₃₁ = p.Δm²₃₁ , 
    δCP = p.δCP,    
    θ₁₂ = p.θ₁₂,    
    θ₁₃= p.θ₁₃,       
    θ₂₃ = p.θ₂₃   
    

)



likelihood = Newtrinos.generate_likelihood(experiments);


result = Newtrinos.scan(likelihood, modified_priors, vars_to_scan, p)

JLD2.@save "/home/sofialon/Newtrinos.jl/plots_fix/dayabay/scan_dayabay_Nr_m0=0.01_NNM_NO.jld2" result

