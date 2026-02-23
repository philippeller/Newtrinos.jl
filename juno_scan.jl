
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

# JUNO only
experiments = (
   juno=Newtrinos.juno.configure(physics),
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


# JUNO only for SM
experiments_SM = (
   juno=Newtrinos.juno.configure(physics_SM),
);


p_SM = Newtrinos.get_params(experiments_SM)



# Create parameters with JUNO fields
p_complete = merge(p, (
        # JUNO only systematics
        juno_detection_epsilon = 1.0,
        
        juno_res_a = 0.0261,
        juno_res_b = 0.0082 ,
        juno_res_c = 0.0123,
    
        juno_geo_shape_eps = 0.0,
        
        juno_geo_rate_norm = 1.0,
        juno_accidental_norm = 1.0,
        juno_world_reactor_norm = 1.0,
        juno_lihe_norm = 1.0,
        juno_co_norm = 1.0,
        juno_atmnc_norm = 1.0,
        juno_fast_neutron_norm = 1.0,
))


# Same for SM
p_complete_SM = merge(p_SM, (
        # JUNO only systematics
        juno_detection_epsilon = 1.0,
        
        juno_res_a = 0.0261,
        juno_res_b = 0.0082 ,
        juno_res_c = 0.0123,
    
        juno_geo_shape_eps = 0.0,
        
        juno_geo_rate_norm = 1.0,
        juno_accidental_norm = 1.0,
        juno_world_reactor_norm = 1.0,
        juno_lihe_norm = 1.0,
        juno_co_norm = 1.0,
        juno_atmnc_norm = 1.0,
        juno_fast_neutron_norm = 1.0,
))

# Generate toy data with SM parameters
toy_data_j = Newtrinos.generate_asimov_data(experiments_SM.juno, p_complete_SM)  
toy_data_j = Float64.(toy_data_j)

# Create the final NamedTuple with JUNO only
toy_data = (juno = toy_data_j,)



m0 = 0.01
p_complete_new = merge(p_complete, (m₀ = m0,))
    
# Scan parameters
vars_to_scan = (r=31, N=31)

modified_priors = all_priors
    


likelihood = Newtrinos.generate_likelihood(experiments, toy_data);


result = Newtrinos.scan(likelihood, modified_priors, vars_to_scan, p_complete_new)


#JLD2.@save "/home/sofialon/Newtrinos.jl/plots_fix/juno_alone/juno_scan_rN_NNM_NO_m0=$m0.jld2" result


img = CairoMakie.plot(result; title="JUNO alone NNM NO - LogLikelihood r vs N, m₀=$m0", log=0, mass=0)

save("/home/sofialon/Newtrinos.jl/plots_fix/juno_scan_rN_NNM_NO_res_a_m0=$m0.png", img)
