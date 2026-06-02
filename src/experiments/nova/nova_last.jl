module nova

using LinearAlgebra
using Distributions
using UnROOT
using BAT
using DataStructures
using CairoMakie
using Logging
using Printf
using Statistics
import ..Newtrinos

@kwdef struct Nova <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function configure(physics)
    physics = (;physics.osc, physics.xsec)
    assets = get_assets(physics)
    return Nova(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)
    @info "Loading NOvA data"
    
    # Load data from ROOT files
    data_file = ROOTFile(joinpath(datadir, "NOvA_2020_data_histograms.root"))
    mc_file = ROOTFile(joinpath(datadir, "NOvA_2020_data_release_predictions_with_systs_all_hists.root"))
    
    # Energy binning for analysis
    energy_edges = collect(range(0.5, stop=4.5, length=9))
    
    # Load electron neutrino data
    nue_data = load_nue_data(data_file, mc_file, energy_edges)
    nuebar_data = load_nuebar_data(data_file, mc_file, energy_edges)
    
    # Load muon neutrino data by quartiles
    numu_data, numubar_data = load_numu_data(data_file, mc_file)
    
    # Flatten observed data
    observed_flat = Vector{Float64}()
    
    # NUE segments
    append!(observed_flat, nue_data.observed.segment1)
    append!(observed_flat, nue_data.observed.segment2)
    append!(observed_flat, nue_data.observed.segment3)
    
    # NUEBAR segments
    append!(observed_flat, nuebar_data.observed.segment1)
    append!(observed_flat, nuebar_data.observed.segment2)
    append!(observed_flat, nuebar_data.observed.segment3)
    
    # NUMU quartiles
    for i in 1:4
        append!(observed_flat, numu_data.quartiles[i].observed)
    end
    
    # NUMUBAR quartiles
    for i in 1:4
        append!(observed_flat, numubar_data.quartiles[i].observed)
    end
    
    observed = observed_flat
    
    # NOvA baseline and matter density
    L = 810.0  # km
    density = 2.84 * 0.5  # g/cm³ * Z/A ratio
    
    assets = (
        L = L,
        density = density,
        energy_edges = energy_edges,
        nue_data = nue_data,
        nuebar_data = nuebar_data,
        numu_data = numu_data,
        numubar_data = numubar_data,
        # Smearing parameters for each quartile
        numu_smearing = [0.078, 0.092, 0.104, 0.115],
        numubar_smearing = [0.085, 0.089, 0.097, 0.102],
        # Energy scale and bias parameters
        numu_e_scale = 1.05,
        numu_e_bias = 0.0,
        nue_e_scale = 0.65,
        nue_e_bias = 0.02,
        observed = observed
    )
    
    # Files are automatically closed by UnROOT when going out of scope
    return assets
end

# ===================================================
# Data loading functions
# ===================================================

function load_nue_data(data_file, mc_file, energy_edges)
    """Load electron neutrino data with 3 segments"""
    
    # Get neutrino data histogram
    data_hist = data_file["neutrino_mode_nue"]
    data_values = UnROOT.array(data_hist)
    
    # Extract segments (matching Python logic)
    observed1 = data_values[2:9]      # 8 elements
    observed2 = data_values[11:18]    # 8 elements  
    observed3 = vcat(data_values[19:21], zeros(5))  # 8 elements: 3 data + 5 zeros
    
    # Load Monte Carlo components for FHC
    mc_components = Dict{String, Vector{Float64}}()
    
    if haskey(mc_file, "prediction_components_nue_fhc")
        mc_dir = mc_file["prediction_components_nue_fhc"]
        
        for key in keys(mc_dir)
            component_hist = mc_dir[key]
            mc_values = UnROOT.array(component_hist)
            
            component_name = string(key)
            
            if length(mc_values) >= 21
                mc_components[component_name * "1"] = mc_values[1:8]
                mc_components[component_name * "2"] = mc_values[9:16]
                mc_components[component_name * "3"] = vcat(mc_values[17:21], zeros(3))
            end
        end
    end
    
    return (
        observed = (segment1 = observed1, segment2 = observed2, segment3 = observed3),
        mc_components = mc_components,
        energy_edges = energy_edges
    )
end

function load_nuebar_data(data_file, mc_file, energy_edges)
    """Load antineutrino data with 3 segments"""
    
    # Get antineutrino data histogram
    data_hist = data_file["antineutrino_mode_nue;1"]
    data_values = UnROOT.array(data_hist)
    
    observed1 = data_values[2:9]      # 8 elements
    observed2 = data_values[11:18]    # 8 elements  
    observed3 = vcat(data_values[19:21], zeros(5))  # 8 elements: 3 data + 5 zeros
    
    # Load Monte Carlo components for RHC
    mc_components = Dict{String, Vector{Float64}}()
    
    if haskey(mc_file, "prediction_components_nue_rhc")
        mc_dir = mc_file["prediction_components_nue_rhc"]
        
        for key in keys(mc_dir)
            component_hist = mc_dir[key]
            mc_values = UnROOT.array(component_hist)
            
            component_name = string(key)
            
            if length(mc_values) >= 21
                mc_components[component_name * "1"] = mc_values[1:8]
                mc_components[component_name * "2"] = mc_values[9:16]
                mc_components[component_name * "3"] = vcat(mc_values[17:21], zeros(3))
            end
        end
    end
    
    return (
        observed = (segment1 = observed1, segment2 = observed2, segment3 = observed3),
        mc_components = mc_components,
        energy_edges = energy_edges
    )
end

function load_numu_data(data_file, mc_file)
    """Load muon neutrino data by quartiles"""
    
    # Get energy binning from first quartile
    first_hist = data_file["neutrino_mode_numu_quartile1"]
    energy_edges = collect(UnROOT.array(first_hist.axis.edges))
    n_bins = length(energy_edges) - 1
    
    # Initialize storage
    numu_quartiles = []
    numubar_quartiles = []
    
    # Process each quartile
    for q in 1:4
        # Load neutrino data
        neutrino_quartile = load_quartile_data(data_file, mc_file, q, "neutrino", energy_edges)
        push!(numu_quartiles, neutrino_quartile)
        
        # Load antineutrino data
        antineutrino_quartile = load_quartile_data(data_file, mc_file, q, "antineutrino", energy_edges)
        push!(numubar_quartiles, antineutrino_quartile)
    end
    
    numu_data = (
        quartiles = numu_quartiles,
        energy_edges = energy_edges
    )
    
    numubar_data = (
        quartiles = numubar_quartiles,
        energy_edges = energy_edges
    )
    
    return numu_data, numubar_data
end

function load_quartile_data(data_file, mc_file, quartile, mode, energy_edges)
    """Load observed and MC data for a single quartile"""
    
    mode_prefix = mode == "neutrino" ? "neutrino_mode" : "antineutrino_mode"
    beam_mode = mode == "neutrino" ? "fhc" : "rhc"
    
    # Load observed data
    obs_hist_key = "$(mode_prefix)_numu_quartile$(quartile)"
    obs_hist = data_file[obs_hist_key]
    observed = UnROOT.array(obs_hist)[2:end-1]  # Skip underflow/overflow
    
    # Load MC components
    mc_key = "prediction_components_numu_$(beam_mode)_Quartile$(quartile)"
    quartile_data = Dict{String, Vector{Float64}}()
    quartile_data["observed"] = observed
    
    if haskey(mc_file, mc_key)
        mc_dir = mc_file[mc_key]
        
        for key in keys(mc_dir)
            component_hist = mc_dir[key]
            component_name = string(key)
            mc_values = UnROOT.array(component_hist)[2:end-1]  # Skip underflow/overflow
            
            quartile_data[component_name] = mc_values
        end
    end
    
    return quartile_data
end

# ===================================================
# Smearing and rebinning functions
# ===================================================

function smearnorm(energies, probabilities, percent, width=10, e_scale=1.0, e_bias=0.0)
    """
    Apply energy resolution smearing to oscillation probabilities.
    Convolves probabilities with a boxcar function.
    """
    n = length(probabilities)
    out = zeros(n)
    
    for i in 1:n
        norm = 0.0
        for j in max(1, i - width):min(n, i + width)
            coeff = 1.0
            norm += coeff
            out[i] += coeff * probabilities[j]
        end
        out[i] /= norm
    end
    
    return out
end

function calculate_energy_edges(energy_centers)
    """Calculate bin edges from bin centers for logarithmic binning"""
    
    if length(energy_centers) < 2
        throw(ArgumentError("Need at least 2 energy centers"))
    end
    if any(energy_centers .<= 0)
        throw(ArgumentError("Energy centers must be positive"))
    end
    
    edges = zeros(length(energy_centers) + 1)
    
    # Interior edges as geometric mean
    edges[2:end-1] = sqrt.(energy_centers[2:end] .* energy_centers[1:end-1])
    
    # First and last edges maintaining logarithmic spacing
    log_spacing = log(energy_centers[2] / energy_centers[1])
    edges[1] = energy_centers[1] / exp(log_spacing/2)
    edges[end] = energy_centers[end] * exp(log_spacing/2)
    
    return edges
end

function rebin_energy_spectrum(input_data, edges, e_min=0.5, e_max=4.5, num_bins=8)
    """
    Rebin spectrum from irregular energy bins to regular bins.
    """
    data = replace(input_data, missing => 0.0, NaN => 0.0)
    edge_values = isa(edges, AbstractArray) ? edges : collect(edges)
    
    # Create new equally spaced bin edges
    new_edges = range(e_min, e_max, length=num_bins + 1)
    new_counts = zeros(num_bins)
    
    # For each input bin
    for i in 1:(length(edge_values)-1)
        old_e_low = edge_values[i]
        old_e_high = edge_values[i+1]
        old_width = old_e_high - old_e_low
        
        # Skip bins outside range
        if old_e_high < e_min || old_e_low > e_max
            continue
        end
        
        # For each new bin
        for j in 1:num_bins
            new_e_low = new_edges[j]
            new_e_high = new_edges[j+1]
            
            # Calculate overlap
            overlap_low = max(old_e_low, new_e_low)
            overlap_high = min(old_e_high, new_e_high)
            
            if overlap_high > overlap_low
                overlap = overlap_high - overlap_low
                fraction = overlap / old_width
                new_counts[j] += data[i] * fraction
            end
        end
    end
    
    return new_counts
end

function fast_predictions_new(signal, backgrounds, norm_factor=1; condense_to_bin3=false)
    """
    Efficiently combine signal and background components with normalization.
    """
    total = copy(signal)
    
    # Add backgrounds
    for bg in backgrounds
        total .+= bg
    end
    
    # Apply normalization
    total .*= norm_factor
    
    # Condense to bin 3 if requested
    if condense_to_bin3 && length(total) > 2
        condensed_sum = sum(total)
        fill!(total, 0.0)
        total[3] = condensed_sum
    end
    
    return total
end

# ===================================================
# Prediction functions
# ===================================================

function make_numu_predictions(params, physics, assets)
    """Calculate muon neutrino disappearance predictions for all quartiles"""
    
    L = [assets.L]
    density = [assets.density]
    
    # Energy grid for oscillation calculation
    energy_grid = exp.(range(log(0.1), log(10.0), length=100))
    
    # Calculate oscillation probabilities for neutrinos
    p_nu = physics.osc.osc_prob(energy_grid * assets.numu_e_scale .+ assets.numu_e_bias, 
                               L, params; anti=false)
    p_nu_survival = p_nu[:, 1, 2, 2]  # νμ → νμ survival probability
    
    # Calculate for antineutrinos
    p_nubar = physics.osc.osc_prob(energy_grid * assets.numu_e_scale .+ assets.numu_e_bias, 
                                  L, params; anti=true)
    p_nubar_survival = p_nubar[:, 1, 2, 2]  # ν̄μ → ν̄μ survival probability
    
    # Apply smearing and make predictions for each quartile
    predictions_numu = Vector{Float64}[]
    predictions_numubar = Vector{Float64}[]
    
    for i in 1:4
        # Neutrino quartile
        p_smeared = smearnorm(energy_grid, p_nu_survival, assets.numu_smearing[i], 4, 
                            assets.numu_e_scale, assets.numu_e_bias)
        
        # Rebin to detector energy bins
        quartile_data = assets.numu_data.quartiles[i]
        n_bins = length(quartile_data["observed"])
        p_rebinned = rebin_energy_spectrum(p_smeared, energy_grid, 0.5, 10.0, n_bins)
        p_rebinned ./= sum(p_rebinned)
        
        # Calculate prediction
        prediction = (quartile_data["NoOscillations_Signal"] .* p_rebinned .+
                     quartile_data["NoOscillations_Total_beam_bkg"] .+
                     quartile_data["Cosmic_bkg"])
        push!(predictions_numu, prediction)
        
        # Antineutrino quartile
        p_smeared_bar = smearnorm(energy_grid, p_nubar_survival, assets.numubar_smearing[i], 4,
                                assets.numu_e_scale, assets.numu_e_bias)
        
        quartile_data_bar = assets.numubar_data.quartiles[i]
        p_rebinned_bar = rebin_energy_spectrum(p_smeared_bar, energy_grid, 0.5, 10.0, n_bins)
        p_rebinned_bar ./= sum(p_rebinned_bar)
        
        prediction_bar = (quartile_data_bar["NoOscillations_Signal"] .* p_rebinned_bar .+
                         quartile_data_bar["NoOscillations_Total_beam_bkg"] .+
                         quartile_data_bar["Cosmic_bkg"])
        push!(predictions_numubar, prediction_bar)
    end
    
    return (numu = predictions_numu, numubar = predictions_numubar)
end

function make_nue_predictions(params, physics, assets)
    """Calculate electron neutrino appearance predictions"""
    
    L = [assets.L]
    density = [assets.density]
    
    # Energy grid for oscillation calculation
    energy_grid = range(0.5, 4.5, length=100)
    energy_edges = calculate_energy_edges(energy_grid)
    
    # Calculate νμ → νe oscillation probabilities
    p_nu = physics.osc.osc_prob(energy_grid * assets.nue_e_scale .+ assets.nue_e_bias,
                               L, params; anti=false)
    p_nu_appearance = p_nu[:, 1, 2, 1]  # νμ → νe probability
    
    p_nubar = physics.osc.osc_prob(energy_grid * assets.nue_e_scale .+ assets.nue_e_bias,
                                  L, params; anti=true)
    p_nubar_appearance = p_nubar[:, 1, 2, 1]  # ν̄μ → ν̄e probability
    
    # Apply smearing
    p_nu_smeared = smearnorm(energy_grid, p_nu_appearance, 1.0, 5, 
                           assets.nue_e_scale, assets.nue_e_bias)
    p_nubar_smeared = smearnorm(energy_grid, p_nubar_appearance, 1.0, 5,
                              assets.nue_e_scale, assets.nue_e_bias)
    
    # Calculate signal from muon neutrino flux
    numu_total_bins = length(assets.numu_data.quartiles[1]["NoOscillations_Signal"])
    signal_nu = rebin_energy_spectrum(p_nu_smeared, energy_edges, 0.5, 4.5, numu_total_bins)
    signal_nu ./= sum(signal_nu)
    signal_nu .*= assets.numu_data.quartiles[1]["NoOscillations_Signal"]
    
    signal_nubar = rebin_energy_spectrum(p_nubar_smeared, energy_edges, 0.5, 4.5, numu_total_bins)
    signal_nubar ./= sum(signal_nubar)
    signal_nubar .*= assets.numubar_data.quartiles[1]["NoOscillations_Signal"]
    
    # Rebin to electron neutrino analysis bins
    signal_nue = rebin_energy_spectrum(signal_nu, assets.numu_data.energy_edges, 0.5, 4.5, 8)
    signal_nuebar = rebin_energy_spectrum(signal_nubar, assets.numubar_data.energy_edges, 0.5, 4.5, 8)
    
    # Make predictions for each segment
    predictions_nue = Dict{String, Vector{Float64}}()
    predictions_nuebar = Dict{String, Vector{Float64}}()
    
    for segment in 1:3
        # Neutrino segment
        backgrounds_nu = [
            assets.nue_data.mc_components["Wrong_sign_bkg$(segment)"],
            assets.nue_data.mc_components["Beam_nue_bkg$(segment)"],
            assets.nue_data.mc_components["Cosmic_bkg$(segment)"]
        ]
        
        condense_mode = (segment == 3)
        prediction_nu = fast_predictions_new(signal_nue, backgrounds_nu,
                                           condense_to_bin3=condense_mode)
        
        predictions_nue["segment$(segment)"] = prediction_nu
        
        # Antineutrino segment
        backgrounds_nubar = [
            assets.nuebar_data.mc_components["Wrong_sign_bkg$(segment)"],
            assets.nuebar_data.mc_components["Beam_nue_bkg$(segment)"],
            assets.nuebar_data.mc_components["Cosmic_bkg$(segment)"]
        ]
        
        prediction_nubar = fast_predictions_new(signal_nuebar, backgrounds_nubar, 
                                              condense_to_bin3=condense_mode)
        
        predictions_nuebar["segment$(segment)"] = prediction_nubar
    end
    
    return (nue = predictions_nue, nuebar = predictions_nuebar)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        # Get predictions
        numu_predictions = make_numu_predictions(params, physics, assets)
        nue_predictions = make_nue_predictions(params, physics, assets)
        
        # Combine everything into a single vector
        all_predictions = Vector{Float64}()
        
        # NUE segments
        append!(all_predictions, nue_predictions.nue["segment1"])
        append!(all_predictions, nue_predictions.nue["segment2"])
        append!(all_predictions, nue_predictions.nue["segment3"])
        
        # NUEBAR segments
        append!(all_predictions, nue_predictions.nuebar["segment1"])
        append!(all_predictions, nue_predictions.nuebar["segment2"])
        append!(all_predictions, nue_predictions.nuebar["segment3"])
        
        # NUMU quartiles
        for i in 1:4
            append!(all_predictions, numu_predictions.numu[i])
        end
        
        # NUMUBAR quartiles
        for i in 1:4
            append!(all_predictions, numu_predictions.numubar[i])
        end
        
        # Return vector of Poisson distributions
        return Poisson.(max.(all_predictions, 1e-10))
    end
    return forward_model
end

function get_plot(physics, assets)
    function plot(params)
        f = Figure(resolution=(1400, 1200))
        
        # Get predictions
        numu_predictions = make_numu_predictions(params, physics, assets)
        nue_predictions = make_nue_predictions(params, physics, assets)
        
        # Plot muon neutrino disappearance by quartile
        for i in 1:4
            ax = Axis(f[1, i], title="νμ Quartile $(i)")
            
            observed = assets.numu_data.quartiles[i]["observed"]
            predicted = numu_predictions.numu[i]
            predicted_errors = sqrt.(max.(predicted, 0))
            energy_centers = (assets.numu_data.energy_edges[1:end-1] .+ 
                            assets.numu_data.energy_edges[2:end]) ./ 2
            
            scatter!(ax, energy_centers, observed, color=:black, label="Observed", markersize=6)
            errorbars!(ax, energy_centers, predicted, predicted_errors, color=:red, linewidth=2)
            lines!(ax, energy_centers, predicted, color=:red, linewidth=2, label="Predicted")
            
            ax.xlabel = "Energy (GeV)"
            ax.ylabel = "Events"
            axislegend(ax, position=:rt, labelsize=6)
        end
        
        # Plot muon antineutrino disappearance by quartile
        for i in 1:4
            ax_bar = Axis(f[2, i], title="ν̄μ Quartile $(i)")
            
            observed = assets.numubar_data.quartiles[i]["observed"]
            predicted = numu_predictions.numubar[i]
            predicted_errors = sqrt.(max.(predicted, 0))
            energy_centers = (assets.numubar_data.energy_edges[1:end-1] .+ 
                            assets.numubar_data.energy_edges[2:end]) ./ 2
            
            scatter!(ax_bar, energy_centers, observed, color=:black, label="Observed", markersize=6)
            errorbars!(ax_bar, energy_centers, predicted, predicted_errors, color=:red, linewidth=2)
            lines!(ax_bar, energy_centers, predicted, color=:red, linewidth=2, label="Predicted")
            
            ax_bar.xlabel = "Energy (GeV)"
            ax_bar.ylabel = "Events"
            axislegend(ax_bar, position=:rt, labelsize=6)
        end
        
        # Plot electron neutrino appearance
        for seg in 1:3
            ax_nue = Axis(f[3, seg], title="νμ → νe Segment $seg")
            
            observed_nue = assets.nue_data.observed["segment$(seg)"]
            predicted_nue = nue_predictions.nue["segment$(seg)"]
            predicted_nue_errors = sqrt.(max.(predicted_nue, 0))
            energy_centers = (assets.nue_data.energy_edges[1:end-1] .+ assets.nue_data.energy_edges[2:end]) ./ 2
            
            scatter!(ax_nue, energy_centers, observed_nue, color=:black, label="Observed", markersize=6)
            errorbars!(ax_nue, energy_centers, predicted_nue, predicted_nue_errors, color=:blue, linewidth=2)
            lines!(ax_nue, energy_centers, predicted_nue, color=:blue, linewidth=2, label="Predicted")
            
            ax_nue.xlabel = "Energy (GeV)"
            ax_nue.ylabel = "Events"
            axislegend(ax_nue, position=:rt, labelsize=6)
        end
        
        # Plot antineutrino appearance
        for seg in 1:3
            ax_nuebar = Axis(f[4, seg], title="ν̄μ → ν̄e Segment $seg")
            
            observed_nuebar = assets.nuebar_data.observed["segment$(seg)"]
            predicted_nuebar = nue_predictions.nuebar["segment$(seg)"]
            predicted_nuebar_errors = sqrt.(max.(predicted_nuebar, 0))
            energy_centers = (assets.nuebar_data.energy_edges[1:end-1] .+ assets.nuebar_data.energy_edges[2:end]) ./ 2
            
            scatter!(ax_nuebar, energy_centers, observed_nuebar, color=:black, label="Observed", markersize=6)
            errorbars!(ax_nuebar, energy_centers, predicted_nuebar, predicted_nuebar_errors, color=:green, linewidth=2)
            lines!(ax_nuebar, energy_centers, predicted_nuebar, color=:green, linewidth=2, label="Predicted")
            
            ax_nuebar.xlabel = "Energy (GeV)"
            ax_nuebar.ylabel = "Events"
            axislegend(ax_nuebar, position=:rt, labelsize=6)
        end
        
        return f
    end
    return plot
end

# Cleanup on module unload
function __init__()
    # Initialize data files if needed
end

end # module nova
