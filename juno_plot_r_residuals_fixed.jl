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
    flavour = Newtrinos.osc.NNM(three_flavour = Newtrinos.osc.ThreeFlavour(ordering = :NO)),
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

# Base parameters
p = Newtrinos.get_params(experiments)

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

# Build complete parameter set for toy-data generation
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

# Get assets to extract energy bins
juno_exp = experiments.juno
E_vis = copy(juno_exp.assets.E_bins_visible)

# Check dimensions
n_energy_bins = length(E_vis)
println("Energy bins: ", n_energy_bins)
println("Toy data length: ", length(toy_data_j))

# Choose r values to compare and fixed N
r_values = [1e-8, 1e-4, 1e-2, 1.0]
N_fixed = 80
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

# Import get_expected from juno module
import Newtrinos.juno: get_expected

# Define colors for different r values
colors = [:blue, :red, :green, :purple, :orange, :brown]

# Create single figure
fig = Figure(resolution = (1000, 700), fontsize = 12)
ax = Axis(fig[1, 1],
    title = @sprintf("JUNO: (Data - Model) / √Model for Different r (N = %d, m₀ = %.4f)", N_fixed, m0),
    xlabel = "E_vis [MeV]",
    ylabel = "(Data - Model) / √Model"
)

# Plot residuals for all r values on the same axis
for (idx, r) in enumerate(r_values)
    params_r = merge(p_complete_model, (r = r,))
    
    try
        # Get full model prediction using the proper forward model from juno
        model = get_expected(params_r, physics, juno_exp.assets)
        
        # Safety check: trim to matching length if needed
        min_len = min(length(model), length(toy_data_j), length(E_vis))
        model_trim = model[1:min_len]
        data_trim = toy_data_j[1:min_len]
        E_vis_trim = E_vis[1:min_len]
        
        # Compute residuals: (data - model) / sqrt(model + small_value)
        residuals = (data_trim .- model_trim) ./ sqrt.(model_trim .+ 1e-9)
        
        # Plot with different color
        color = colors[mod(idx - 1, length(colors)) + 1]
        scatter!(ax, E_vis_trim, residuals,
            color = color, markersize = 6, alpha = 0.7,
            label = @sprintf("r = %.2e", r))
        
    catch e
        println("Error processing r = $r: ", e)
    end
end

# Add reference line at zero
hlines!(ax, 0.0, color = :black, linestyle = :dash, linewidth = 2, label = "Zero")
ylims!(ax, -3, 3)
axislegend(ax, position = :rt)

outdir = "/home/sofialon/Newtrinos.jl/plots_fix/juno_r_variation"
if !isdir(outdir)
    mkpath(outdir)
end

fname = joinpath(outdir, @sprintf("juno_residuals_1fig_r_variation_N=%d_m0=%.4f.png", N_fixed, m0))
save(fname, fig)
println("Saved combined residuals plot: ", fname)
