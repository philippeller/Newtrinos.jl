
using Base.Threads
println("Threads available: ", nthreads())


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
using Dates


# NNM configuration
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

# TAO only
experiments = (
   tao=Newtrinos.tao.configure(physics),
);

p = Newtrinos.get_params(experiments)


all_priors = Newtrinos.get_priors(experiments)



# Standard Model configuration for toy data generation
osc_cfg_SM = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.ThreeFlavour(),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg_SM)


atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec=Newtrinos.xsec.configure()

physics_SM = (; osc, atm_flux, earth_layers, xsec);


# TAO only for SM
experiments_SM = (
   tao=Newtrinos.tao.configure(physics_SM),
);


p_SM = Newtrinos.get_params(experiments_SM)



# Create parameters with TAO fields
p_complete = merge(p, (
        # TAO only systematics
        tao_detection_epsilon = 1.0,
        tao_res_a = 0.015,
        tao_res_b = 0.0,
        tao_res_c = 0.0,
        
        # TAO specific backgrounds
        tao_accidental_norm = 1.0,
        tao_fast_neutron_norm = 1.0,
        tao_lihe_norm = 1.0,
))


# Same for SM
p_complete_SM = merge(p_SM, (
        # TAO only systematics
        tao_detection_epsilon = 1.0,
        tao_res_a = 0.015,
        tao_res_b = 0.0,
        tao_res_c = 0.0,
        
        # TAO specific backgrounds
        tao_accidental_norm = 1.0,
        tao_fast_neutron_norm = 1.0,
        tao_lihe_norm = 1.0,
))

# Generate toy data with SM parameters
toy_data_t = Newtrinos.generate_asimov_data(experiments_SM.tao, p_complete_SM)  
toy_data_t = Float64.(toy_data_t)

# Create the final NamedTuple with TAO only
toy_data = (tao = toy_data_t,)



m0 = 0.01
p_complete_new = merge(p_complete, (m₀ = m0,))
    
# Scan parameters
vars_to_scan = (r=31, N=31)

modified_priors = all_priors
    


likelihood = Newtrinos.generate_likelihood(experiments, toy_data);


result = Newtrinos.scan(likelihood, modified_priors, vars_to_scan, p_complete_new)


#JLD2.@save "/home/sofialon/Newtrinos.jl/plots_fix/tao_alone/tao_scan_rN_NNM_NO_m0=$m0.jld2" result


img = CairoMakie.plot(result; title="TAO alone NNM NO - LogLikelihood r vs N, m₀=$m0", log=0, mass=0)

save("/home/sofialon/Newtrinos.jl/plots_fix/tao_scan_rN_NNM_NO_m0=$m0.png", img)
