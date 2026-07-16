# NNATURALNESS
using LinearAlgebra
using Distributions
using FileIO
import JLD2
using DataFrames
using CSV
using Revise
using Newtrinos

osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NNM(three_flavour=Newtrinos.osc.ThreeFlavour(ordering=:NO)),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI(),
    output=Newtrinos.osc.EeOnly()
    )

osc = Newtrinos.osc.configure(osc_cfg)

physics = (; osc);


experiments = (

   gerda= Newtrinos.gerda.configure(physics),
);

par= Newtrinos.get_params(experiments)

all_priors = Newtrinos.get_priors(experiments)

vars_to_scan = (r=11, N=11)  

    modified_priors = merge(all_priors,(
        N = all_priors.N,
        m₀ =all_priors.m₀,
        r = all_priors.r,
        
    
    

        Δm²₂₁ = par.Δm²₂₁,
        Δm²₃₁ = all_priors.Δm²₃₁,
        δCP = par.δCP,
        θ₁₂ = par.θ₁₂,
        θ₁₃ = all_priors.θ₁₃,
        θ₂₃ = par.θ₂₃
    ),)
        

    likelihood_NN = Newtrinos.generate_likelihood(experiments);

    result = Newtrinos.scan(likelihood_NN, modified_priors, vars_to_scan, par)


    JLD2.@save "/home/sofialon/Newtrinos.jl/scan_plot_paper/scan/gerda_0.35_rN_NNM_NO_m0=$m0.jld2" result
    