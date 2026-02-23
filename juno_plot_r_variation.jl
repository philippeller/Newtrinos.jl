using Base.Threads
println("Threads available: ", nthreads())

using LinearAlgebra
using Distributions
using Printf
using FileIO
import JLD2
using DataFrames

using Revise

using Newtrinos
using CairoMakie
using Dates

# Configure NNM physics (model) for plotting
osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour = Newtrinos.osc.NND(three_flavour = Newtrinos.osc.ThreeFlavour(ordering = :NO)),
    propagation = Newtrinos.osc.Basic(),
    states = Newtrinos.osc.All(),
    interaction = Newtrinos.osc.SI()
)
osc = Newtrinos.osc.configure(osc_cfg)

atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec = Newtrinos.xsec.configure()
physics = (; osc, atm_flux, earth_layers, xsec)

# JUNO experiment configured with default binning
experiments = (juno = Newtrinos.juno.configure(physics),)

# Base parameters and priors
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# For toy data, use SM oscillations (ThreeFlavour)
osc_cfg_SM = Newtrinos.osc.OscillationConfig(
    flavour = Newtrinos.osc.ThreeFlavour(),
    propagation = Newtrinos.osc.Basic(),
    states = Newtrinos.osc.All(),
    interaction = Newtrinos.osc.SI()
)
osc_SM = Newtrinos.osc.configure(osc_cfg_SM)
physics_SM = (; osc = osc_SM, atm_flux, earth_layers, xsec)
experiments_SM = (juno = Newtrinos.juno.configure(physics_SM),)

p_SM = Newtrinos.get_params(experiments_SM)

# Build complete parameter set for toy-data generation (use JUNO defaults)
p_complete_SM = merge(p_SM, (
    juno_detection_epsilon = 1.0,
    juno_res_a = 0.0261,
    juno_res_b = 0.0082,
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

# Generate toy (asimov) data using SM
toy_data_j = Newtrinos.generate_asimov_data(experiments_SM.juno, p_complete_SM)
toy_data_j = Float64.(toy_data_j)

# Prepare plotting function from the JUNO experiment
plot_fn = experiments.juno.plot

# Choose r values to compare and fixed N
r_values = [1e-8, 1e-4, 1e-2, 1.0]
N_fixed =80
m0 = 0.01

# Base params for the model (NNM)
p_complete_model = merge(p, (
    juno_detection_epsilon = 1.0,
    juno_res_a = 0.0261,
    juno_res_b = 0.0082,
    juno_res_c = 0.0123,
    juno_geo_shape_eps = 0.0,
    juno_geo_rate_norm = 1.0,
    juno_accidental_norm = 1.0,
    juno_world_reactor_norm = 1.0,
    juno_lihe_norm = 1.0,
    juno_co_norm = 1.0,
    juno_atmnc_norm = 1.0,
    juno_fast_neutron_norm = 1.0,
    m₀ = m0,
    N = N_fixed,
))

outdir = "/home/sofialon/Newtrinos.jl/plots_fix/juno_r_variation"
if !isdir(outdir)
    mkpath(outdir)
end

for r in r_values
    params_r = merge(p_complete_model, (r = r,))
    fig = plot_fn(params_r; data_to_plot = toy_data_j, title_suffix = " r = $(r) ")
    fname = joinpath(outdir, @sprintf("juno_osc_r=%.3f_N=%d_m0=%.4f.png", r, N_fixed, m0))
    save(fname, fig)
    println("Saved plot: ", fname)
end

println("Done: generated $(length(r_values)) plots in $outdir")
