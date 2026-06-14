#!/usr/bin/env julia
# Cluster worker script: computes ONE profile likelihood point
# Usage: julia single_point.jl <POINT_INDEX> <TOTAL_POINTS>

using LinearAlgebra
using Distributions
using Printf
using FileIO
import JLD2
using Setfield
using Accessors
#using Revise
using Newtrinos

# Parse command line arguments
const POINT_INDEX = parse(Int, ARGS[1])
const TOTAL_POINTS = parse(Int, ARGS[2])

println("Processing point $POINT_INDEX / $TOTAL_POINTS")

# === PHYSICS CONFIGURATION (Standard Model for toy data) ===
osc_cfg_SM = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.ThreeFlavour(ordering=:NO),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
)

osc_SM = Newtrinos.osc.configure(osc_cfg_SM)
atm_flux_SM = Newtrinos.atm_flux.configure()
earth_layers_SM = Newtrinos.earth_layers.configure()
xsec_SM = Newtrinos.xsec.configure()

physics_SM = (; osc=osc_SM, atm_flux=atm_flux_SM, earth_layers=earth_layers_SM, xsec=xsec_SM)

experiments_SM = (
    juno=Newtrinos.juno.configure(physics_SM; livetime_years=6.0),
    tao=Newtrinos.tao.configure(physics_SM; livetime_years=6.0),
)

p_SM = Newtrinos.get_params(experiments_SM)

# Complete parameters including TAO-specific systematics
p_complete_SM = merge(p_SM, (
    tao_detection_epsilon=1.0,
    tao_res_a=0.015,
    tao_res_b=0.0,
    tao_res_c=0.0,
    tao_accidental_norm=1.0,
    tao_fast_neutron_norm=1.0,
    tao_lihe_norm=1.0,
))

# === PHYSICS CONFIGURATION (NNM for fitting) ===
osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NNM(three_flavour=Newtrinos.osc.ThreeFlavour(ordering=:NO)),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
)

osc = Newtrinos.osc.configure(osc_cfg)
atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec = Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec)

experiments = (
    juno=Newtrinos.juno.configure(physics; livetime_years=6.0),
    tao=Newtrinos.tao.configure(physics; livetime_years=6.0),
)

p = Newtrinos.get_params(experiments)

# Complete parameters including TAO-specific systematics
p_complete = merge(p, (
    tao_detection_epsilon=1.0,
    tao_res_a=0.015,
    tao_res_b=0.0,
    tao_res_c=0.0,
    tao_accidental_norm=1.0,
    tao_fast_neutron_norm=1.0,
    tao_lihe_norm=1.0,
))

# === SET m0 VALUE ===
m0 = 1e-2
p_complete_new = merge(p_complete, (m₀=m0,))

# === PRIORS ===
modified_priors = merge(p_complete_new, (
    N=DiscreteUniform(2, 200),
    r=LogUniform(1e-8, 1),
    junotao_flux_scale=Truncated(Normal(1.0, 0.02), 0.0, Inf),
    junotao_energy_scale=Truncated(Normal(1.0, 0.005), 0.0, Inf),
    juno_detection_epsilon=Truncated(Normal(1.0, 0.01), 0.0, Inf),
    juno_res_a=Truncated(Normal(0.0261, 0.0002), 0.0, Inf),
    juno_res_b=Truncated(Normal(0.0082, 0.0001), 0.0, Inf),
    juno_res_c=Truncated(Normal(0.0123, 0.0004), 0.0, Inf),
    junotao_shape_eps=Normal(0, 1),
    juno_geo_shape_eps=Normal(0, 1),
    juno_geo_rate_norm=Truncated(Normal(1.0, 0.30), 0.0, Inf),
    juno_accidental_norm=Truncated(Normal(1.0, 0.01), 0.0, Inf),
    juno_world_reactor_norm=Truncated(Normal(1.0, 0.02), 0.0, Inf),
    juno_lihe_norm=Truncated(Normal(1.0, 0.20), 0.0, Inf),
    juno_co_norm=Truncated(Normal(1.0, 0.50), 0.0, Inf),
    juno_atmnc_norm=Truncated(Normal(1.0, 0.50), 0.0, Inf),
    juno_fast_neutron_norm=Truncated(Normal(1.0, 1.0), 0.0, Inf),
    tao_detection_epsilon=Truncated(Normal(1.0, 0.005), 0.0, Inf),
    tao_res_a=Truncated(Normal(0.015, 0.015 * 0.05), 0.0, Inf),
    tao_res_b=Truncated(Normal(0.0, 0.001), 0.0, Inf),
    tao_res_c=Truncated(Normal(0.0, 0.001), 0.0, Inf),
    tao_accidental_norm=Truncated(Normal(1.0, 0.20), 0.0, Inf),
    tao_fast_neutron_norm=Truncated(Normal(1.0, 0.30), 0.0, Inf),
    tao_lihe_norm=Truncated(Normal(1.0, 0.30), 0.0, Inf),
))

# === GENERATE TOY DATA (from SM) ===
toy_data_j = Newtrinos.generate_asimov_data(experiments_SM.juno, p_complete_SM)
toy_data_t = Newtrinos.generate_asimov_data(experiments_SM.tao, p_complete_SM)
toy_data = (juno=Int64.(toy_data_j), tao=Int64.(toy_data_t))

# === GENERATE LIKELIHOOD (with NNM) ===
likelihood = Newtrinos.generate_likelihood(experiments, toy_data)

# === GENERATE SCAN POINTS ===
vars_to_scan = (r=5, N=5)
values, scanpoints = Newtrinos.generate_scanpoints(vars_to_scan, modified_priors)

# === PROCESS THIS POINT ===
@assert 1 <= POINT_INDEX <= length(scanpoints) "Invalid point index"
scanpoint = scanpoints[POINT_INDEX]

cache_dir = "cache"
mkpath(cache_dir)

println("Computing point $POINT_INDEX: N=\$(scanpoint.N.val), r=\$(round(scanpoint.r.val, digits=6))")

opt_result = Newtrinos.find_mle_cached(likelihood, scanpoint, p_complete_new, cache_dir)

# Save result
output_file = "cluster/results/point_$(POINT_INDEX).jld2"
JLD2.@save output_file opt_result

println("Saved result to $output_file")
