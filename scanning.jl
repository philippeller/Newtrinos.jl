using LinearAlgebra
using Distributions
using LaTeXStrings
using Printf
using FileIO
import JLD2
using DataFrames


using Revise
using Newtrinos
using CairoMakie


osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NNM(),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg)

physics = (; osc);


experiments = (

   katrin= Newtrinos.katrin.configure(physics),
);


par= Newtrinos.get_params(experiments)


all_priors = Newtrinos.get_priors(experiments)

m0_values=[1e-1,1e-2,1e-3,1e-4]

N_treshold=[1,5, 50, 500]


for i in 1:length(m0_values)

    m0 =m0_values[i]
    par= merge(par, (m₀ =m0,))
    
    vars_to_scan = (r=31, N=31)  

    modified_priors = (
        N = Uniform((N_treshold[i]),(600)) ,
        m₀ =all_priors.m₀,
        r = all_priors.r,
        
    
    

        Δm²₂₁ = par.Δm²₂₁,
        Δm²₃₁ = all_priors.Δm²₃₁,
        δCP = par.δCP,
        θ₁₂ = par.θ₁₂,
        θ₁₃ = all_priors.θ₁₃,
        θ₂₃ = par.θ₂₃
    )
        

    likelihood_NN = Newtrinos.generate_likelihood(experiments);

    result = Newtrinos.scan(likelihood_NN, modified_priors, vars_to_scan, par)


    JLD2.@save "/home/sofialon/Newtrinos.jl/plot_final/scans_m04_file/katrin_rN_NNM_NO_m0=$m0.jld2" result
    

    img = CairoMakie.plot(result; title="Katrin - LogLikelihood r vs N, mo=$m0 NNM NO", log=0, mass=0)
   
    save("/home/sofialon/Newtrinos.jl/plot_final/scans_m04/katrin_rN_NNM_NO_m0=$m0.png", img)

end    
