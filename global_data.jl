using LinearAlgebra
using Distributions
using LaTeXStrings
using Printf
using FileIO
import JLD2
using DataFrames
using CSV


using Revise
using Newtrinos
using CairoMakie


osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NNM(three_flavour=Newtrinos.osc.ThreeFlavour(ordering=:NO)),
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
 
 gerda=Newtrinos.gerda.configure(physics),
 katrin=Newtrinos.katrin.configure(physics),
 dayabay=Newtrinos.dayabay.configure(physics),

);


p = Newtrinos.get_params(experiments)

all_priors = Newtrinos.get_priors(experiments)

vars_to_scan = (r=31, N=31)  


modified_priors = merge(all_priors,(N = DiscreteUniform(2,200),r =LogUniform(1e-8,1),))

likelihood = Newtrinos.generate_likelihood(experiments);

result = Newtrinos.scan(likelihood, modified_priors, vars_to_scan, p )

JLD2.@save "/dss/dsshome1/08/go67jac2/julia/my_env/plots_fix/scan_new/scan_global_data_rN_m0=0.01_NNM_NO.jld2" result

