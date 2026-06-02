using LinearAlgebra
using Distributions
using LaTeXStrings
using Printf
using FileIO
import JLD2
using DataFrames
using Setfield

using Accessors

using Revise
using Newtrinos
using CairoMakie



using CSV

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
 
   juno= Newtrinos.juno.configure(physics;livetime_years = 6.0),
   tao=Newtrinos.tao.configure(physics;livetime_years = 6.0),
);

p = Newtrinos.get_params(experiments)


all_priors = Newtrinos.get_priors(experiments)

# First, create parameters that include both JUNO and TAO fields
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


osc_cfg_SM = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.ThreeFlavour(ordering=:NO),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg_SM)

atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec=Newtrinos.xsec.configure()

physics_SM = (; osc, atm_flux, earth_layers, xsec);


experiments_SM = (
 
   juno= Newtrinos.juno.configure(physics_SM;livetime_years = 6.0),
   tao=Newtrinos.tao.configure(physics_SM;livetime_years = 6.0),
);


p_SM = Newtrinos.get_params(experiments_SM)

all_priors = Newtrinos.get_priors(experiments_SM)


# First, create parameters that include both JUNO and TAO fields
#=p_complete = merge(p, (

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
=#

# First, create parameters that include both JUNO and TAO fields
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

# Now generate toy data with the complete parameters
toy_data_j = Newtrinos.generate_asimov_data(experiments_SM.juno, p_complete_SM)
toy_data_j = Int64.(toy_data_j)

toy_data_t = Newtrinos.generate_asimov_data(experiments_SM.tao, p_complete_SM)  
toy_data_t = Int64.(toy_data_t)

energies_j = experiments_SM.juno.assets.E_bins_visible
energies_t=experiments_SM.tao.assets.E_bins_visible


dfj = DataFrame(
   E_vis_MeV = energies_j,
    Counts = toy_data_j
)

dft = DataFrame(
   E_vis_MeV = energies_t,
    Counts = toy_data_t
)


CSV.write("juno_NO_s.csv", dfj)
CSV.write("tao_NO_s.csv", dft)
# Create the final NamedTuple


m0_values=[1e-2]


for i in 1:length(m0_values)

    m0 =m0_values[i]
    p_complete_new= merge(p_complete, (m₀=m0,))
      
    vars_to_scan = (r=5, N=5)  

    #modified_priors = merge(all_priors,(N=DiscreteUniform(2,200),r =LogUniform(1e-8,1),))
    #merge(p_complete_new, (η=Uniform(1.001,1.1),))
    modified_priors = merge(p_complete_new,(N=DiscreteUniform(2,200), r =LogUniform(1e-8,1), 
        junotao_flux_scale = Truncated(Normal(1.0, 0.02), 0.0, Inf), 
        junotao_energy_scale = Truncated(Normal(1.0, 0.005), 0.0, Inf),
        juno_detection_epsilon = Truncated(Normal(1.0, 0.01), 0.0, Inf),

        juno_res_a = Truncated(Normal(0.0261, 0.0002), 0.0, Inf),
        juno_res_b = Truncated(Normal(0.0082, 0.0001), 0.0, Inf),
        juno_res_c = Truncated(Normal(0.0123, 0.0004), 0.0, Inf),
        
        junotao_shape_eps = Normal(0,1),
        juno_geo_shape_eps = Normal(0,1),
        
        juno_geo_rate_norm = Truncated(Normal(1.0, 0.30), 0.0, Inf),
        juno_accidental_norm = Truncated(Normal(1.0, 0.01), 0.0, Inf),     
        juno_world_reactor_norm = Truncated(Normal(1.0, 0.02), 0.0, Inf),  
        juno_lihe_norm = Truncated(Normal(1.0, 0.20), 0.0, Inf),      
        juno_co_norm = Truncated(Normal(1.0, 0.50), 0.0, Inf),         
        juno_atmnc_norm = Truncated(Normal(1.0, 0.50), 0.0, Inf),   
        juno_fast_neutron_norm = Truncated(Normal(1.0, 1.0), 0.0, Inf), 
        

        tao_detection_epsilon = Truncated(Normal(1.0, 0.005), 0.0, Inf),
        tao_res_a = Truncated(Normal(0.015, 0.015 * 0.05), 0.0, Inf),
        tao_res_b = Truncated(Normal(0.0, 0.001), 0.0, Inf),
        tao_res_c = Truncated(Normal(0.0, 0.001), 0.0, Inf),
        
        tao_accidental_norm = Truncated(Normal(1.0, 0.20), 0.0, Inf),
        tao_fast_neutron_norm = Truncated(Normal(1.0, 0.30), 0.0, Inf),
        tao_lihe_norm = Truncated(Normal(1.0, 0.30), 0.0, Inf),  )
        
    )
        

    
    toy_data = (juno=toy_data_j, tao=toy_data_t)
    likelihood = Newtrinos.generate_likelihood(experiments, toy_data)
        
    cache_dir = "cache"
    mkpath(cache_dir)

    result = Newtrinos.profile(likelihood, modified_priors, vars_to_scan, p_complete_new; cache_dir=cache_dir)


    JLD2.@save "./plots_17_05/junotao_full6_rN_NNM_NO_prof_log.jld2" result
    

    img = CairoMakie.plot(result; title="Juno-Tao NNM NO - LogLikelihood r vs N, mo=$m0, η=1+1/N profiled", log=0, mass=0)
   
    save("./plots_17_05/junotao_full6_rN_NNM_NO_prof_log.png", img)

end
