module nova

using LinearAlgebra
using Distributions
using UnROOT
using BAT
using DataStructures
using CairoMakie
using Logging
using Statistics
import ..Newtrinos

@kwdef struct novaExperiment <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

# Helper function to safely extract data from ROOT histograms
function extract_histogram_data(hist)
    """Safely extract bin contents from ROOT histogram"""
    if hasfield(typeof(hist), :fArray)
        if hasfield(typeof(hist), :fN) && hist.fN > 0
            return hist.fArray[1:hist.fN]
        else
            return hist.fArray
        end
    else
        # Alternative access method for different ROOT histogram types
        return hist.weights
    end
end

function extract_histogram_edges(hist)
    """Extract bin edges from ROOT histogram axis"""
    if hasfield(typeof(hist), :fXaxis)
        axis = hist.fXaxis
        if hasfield(typeof(axis), :fXbins) && !isempty(axis.fXbins.fArray)
            return axis.fXbins.fArray
        else
            # Uniform binning case
            nbins = axis.fNbins
            xmin = axis.fXmin
            xmax = axis.fXmax
            return range(xmin, xmax, length=nbins+1)
        end
    else
        throw(ArgumentError("Cannot extract edges from histogram"))
    end
end

function configure(physics)
    physics = (;physics.osc, physics.xsec)
    assets = get_assets(physics)
    return novaExperiment(
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
    
    # Load data from ROOT files (as in original Python code)
    data_file = ROOTFile(joinpath(datadir, "NOvA_2020_data_histograms.root"))
    mc_file = ROOTFile(joinpath(datadir, "NOvA_2020_data_release_predictions_with_systs_all_hists.root"))
    
    # Energy binning for electron neutrino analysis
    energy_edges = range(0.5, 4.5, length=9)
    
    # Load electron neutrino data
    nue_data = load_nue_data(data_file, mc_file, energy_edges)
    nuebar_data = load_nuebar_data(data_file, mc_file, energy_edges)
    
    # Load muon neutrino data by quartiles
    numu_data, numubar_data = load_numu_data(data_file, mc_file)
    
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
        nue_e_bias = 0.02
    )
    
    # Files are automatically closed by UnROOT when going out of scope
    return assets
end

function load_nue_data(data_file, mc_file, energy_edges)
    """Load electron neutrino data with 3 segments"""
    
    # Read observed data from ROOT histogram
    data_hist = data_file["neutrino_mode_nue"]
    data_values = data_hist.fN > 0 ? data_hist.fArray : data_hist.fArray[1:data_hist.fN]
    
    observed1 = data_values[2:9]  # Julia 1-indexed
    observed2 = data_values[11:18]
    observed3 = vcat(data_values[19:21], zeros(5))  # Pad with zeros
    
    # Load Monte Carlo components for each segment
    mc_components = Dict{String, Vector{Vector{Float64}}}()
    
    # Get all keys from MC file for nue FHC predictions
    for key in keys(mc_file)
        if startswith(string(key), "prediction_components_nue_fhc")
            component_name = replace(string(key), r"_\d+$" => "")  # Remove trailing numbers
            mc_hist = mc_file[key]
            mc_values = mc_hist.fN > 0 ? mc_hist.fArray : mc_hist.fArray[1:mc_hist.fN]
            
            mc_components[component_name * "1"] = mc_values[1:8]
            mc_components[component_name * "2"] = mc_values[9:16]
            mc_components[component_name * "3"] = vcat(mc_values[17:21], zeros(3))
        end
    end
    
    return (
        observed = (segment1 = observed1, segment2 = observed2, segment3 = observed3),
        mc_components = mc_components,
        energy_edges = energy_edges
    )
end

function load_nuebar_data(data_file, mc_file, energy_edges)
    """Load electron antineutrino data with 3 segments"""
    
    # Read observed data from ROOT histogram
    data_hist = data_file["antineutrino_mode_nue;1"]  # Note the ;1 from Python code
    data_values = data_hist.fN > 0 ? data_hist.fArray : data_hist.fArray[1:data_hist.fN]
    
    observed1 = data_values[2:9]
    observed2 = data_values[11:18]
    observed3 = vcat(data_values[19:21], zeros(5))
    
    # Load Monte Carlo components
    mc_components = Dict{String, Vector{Vector{Float64}}}()
    
    # Get all keys from MC file for nue RHC predictions
    for key in keys(mc_file)
        if startswith(string(key), "prediction_components_nue_rhc")
            component_name = replace(string(key), r"_\d+$" => "")
            mc_hist = mc_file[key]
            mc_values = mc_hist.fN > 0 ? mc_hist.fArray : mc_hist.fArray[1:mc_hist.fN]
            
            mc_components[component_name * "1"] = mc_values[1:8]
            mc_components[component_name * "2"] = mc_values[9:16]
            mc_components[component_name * "3"] = vcat(mc_values[17:21], zeros(3))
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
    
    numu_quartiles = []
    numubar_quartiles = []
    
    # Initialize totals
    numu_total = Dict{String, Vector{Float64}}()
    numubar_total = Dict{String, Vector{Float64}}()
    
    # Get energy edges from first quartile
    first_quartile_hist = data_file["neutrino_mode_numu_quartile1"]
    # Extract bin edges from ROOT histogram
    energy_edges = first_quartile_hist.fXaxis.fXbins.fArray
    if isempty(energy_edges)
        # If fXbins is empty, use fXmin, fXmax, fNbins to construct edges
        nbins = first_quartile_hist.fXaxis.fNbins
        xmin = first_quartile_hist.fXaxis.fXmin
        xmax = first_quartile_hist.fXaxis.fXmax
        energy_edges = range(xmin, xmax, length=nbins+1)
    end
    
    # Initialize total arrays
    n_bins = length(energy_edges) - 1
    numu_total["NoOscillations_Signal"] = zeros(n_bins)
    numu_total["Oscillated_Signal"] = zeros(n_bins)
    numu_total["NoOscillations_Total_beam_bkg"] = zeros(n_bins)
    numu_total["Cosmic_bkg"] = zeros(n_bins)
    
    numubar_total["NoOscillations_Signal"] = zeros(n_bins)
    numubar_total["Oscillated_Signal"] = zeros(n_bins)
    numubar_total["NoOscillations_Total_beam_bkg"] = zeros(n_bins)
    numubar_total["Cosmic_bkg"] = zeros(n_bins)
    
    for q in 1:4
        # Load neutrino quartile
        quartile_data = Dict{String, Any}()
        
        # Load observed data for this quartile
        quartile_hist = data_file["neutrino_mode_numu_quartile$(q)"]
        quartile_data["observed"] = quartile_hist.fN > 0 ? quartile_hist.fArray : quartile_hist.fArray[1:quartile_hist.fN]
        quartile_data["energy_edges"] = energy_edges
        
        # Load MC components for this quartile
        for key in keys(mc_file)
            if startswith(string(key), "prediction_components_numu_fhc_Quartile$(q)")
                component_name = replace(string(key), r"_\d+$" => "")
                mc_hist = mc_file[key]
                mc_values = mc_hist.fN > 0 ? mc_hist.fArray : mc_hist.fArray[1:mc_hist.fN]
                quartile_data[component_name] = mc_values
            end
        end
        
        push!(numu_quartiles, quartile_data)
        
        # Accumulate totals
        numu_total["NoOscillations_Signal"] += quartile_data["NoOscillations_Signal"]
        numu_total["Oscillated_Signal"] += quartile_data["Oscillated_Signal"]
        numu_total["NoOscillations_Total_beam_bkg"] += quartile_data["NoOscillations_Total_beam_bkg"]
        numu_total["Cosmic_bkg"] += quartile_data["Cosmic_bkg"]
        
        # Load antineutrino quartile
        quartile_data_bar = Dict{String, Any}()
        
        # Load observed antineutrino data for this quartile
        quartile_hist_bar = data_file["antineutrino_mode_numu_quartile$(q)"]
        quartile_data_bar["observed"] = quartile_hist_bar.fN > 0 ? quartile_hist_bar.fArray : quartile_hist_bar.fArray[1:quartile_hist_bar.fN]
        quartile_data_bar["energy_edges"] = energy_edges
        
        # Load MC components for antineutrino quartile
        for key in keys(mc_file)
            if startswith(string(key), "prediction_components_numu_rhc_Quartile$(q)")
                component_name = replace(string(key), r"_\d+$" => "")
                mc_hist = mc_file[key]
                mc_values = mc_hist.fN > 0 ? mc_hist.fArray : mc_hist.fArray[1:mc_hist.fN]
                quartile_data_bar[component_name] = mc_values
            end
        end
        
        push!(numubar_quartiles, quartile_data_bar)
        
        # Accumulate antineutrino totals
        numubar_total["NoOscillations_Signal"] += quartile_data_bar["NoOscillations_Signal"]
        numubar_total["Oscillated_Signal"] += quartile_data_bar["Oscillated_Signal"]
        numubar_total["NoOscillations_Total_beam_bkg"] += quartile_data_bar["NoOscillations_Total_beam_bkg"]
        numubar_total["Cosmic_bkg"] += quartile_data_bar["Cosmic_bkg"]
    end
    
    numu_data = (
        quartiles = numu_quartiles,
        total = numu_total,
        energy_edges = energy_edges
    )
    
    numubar_data = (
        quartiles = numubar_quartiles,
        total = numubar_total,
        energy_edges = energy_edges
    )
    
    return numu_data, numubar_data
end

function smearnorm(energies, probabilities, percent, width=10, e_scale=1.0, e_bias=0.0)
    """
    Apply energy resolution smearing to oscillation probabilities.
    Convolves probabilities with a boxcar function to model detector resolution.
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
    Handles partial bin overlaps properly.
    """
    # Handle masked arrays or missing data
    data = replace(input_data, missing => 0.0, NaN => 0.0)
    
    # Convert edges to array if needed
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

function fast_predictions_new(signal, backgrounds, norm_factor; condense_to_bin3=false)
    """
    Efficiently combine signal and background components with normalization.
    Option to condense all values into bin 3 for systematic uncertainties.
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

function make_numu_predictions(params, physics, assets)
    """Calculate muon neutrino disappearance predictions for all quartiles"""
    
    L = [assets.L]
    density = [assets.density]
    
    # Energy grid for oscillation calculation
    energy_grid = exp.(range(log(0.1), log(10.0), length=100))
    
    # Calculate oscillation probabilities for neutrinos
    p_nu = physics.osc.osc_prob(energy_grid * assets.numu_e_scale .+ assets.numu_e_bias, 
                               L, params, false, density)
    p_nu_survival = p_nu[:, 1, 2, 2]  # νμ → νμ survival probability
    
    # Calculate for antineutrinos
    p_nubar = physics.osc.osc_prob(energy_grid * assets.numu_e_scale .+ assets.numu_e_bias, 
                                  L, params, true, density)
    p_nubar_survival = p_nubar[:, 1, 2, 2]  # ν̄μ → ν̄μ survival probability
    
    # Apply smearing and make predictions for each quartile
    predictions = Dict{String, Vector{Vector{Float64}}}()
    predictions["numu"] = []
    predictions["numubar"] = []
    
    for i in 1:4
        # Neutrino quartile
        p_smeared = smearnorm(energy_grid, p_nu_survival, assets.numu_smearing[i], 4, 
                            assets.numu_e_scale, assets.numu_e_bias)
        
        # Rebin to detector energy bins
        quartile_data = assets.numu_data.quartiles[i]
        p_rebinned = rebin_energy_spectrum(p_smeared, energy_grid, 0.5, 10.0, 
                                         length(quartile_data["observed"]))
        p_rebinned ./= sum(p_rebinned)  # Normalize
        
        # Calculate prediction
        prediction = (quartile_data["NoOscillations_Signal"] .* p_rebinned .+
                     quartile_data["NoOscillations_Total_beam_bkg"] .+
                     quartile_data["Cosmic_bkg"]) .* params["nova_norm"]
        
        push!(predictions["numu"], prediction)
        
        # Antineutrino quartile
        p_smeared_bar = smearnorm(energy_grid, p_nubar_survival, assets.numubar_smearing[i], 4,
                                assets.numu_e_scale, assets.numu_e_bias)
        
        quartile_data_bar = assets.numubar_data.quartiles[i]
        p_rebinned_bar = rebin_energy_spectrum(p_smeared_bar, energy_grid, 0.5, 10.0,
                                             length(quartile_data_bar["observed"]))
        p_rebinned_bar ./= sum(p_rebinned_bar)
        
        prediction_bar = (quartile_data_bar["NoOscillations_Signal"] .* p_rebinned_bar .+
                         quartile_data_bar["NoOscillations_Total_beam_bkg"] .+
                         quartile_data_bar["Cosmic_bkg"]) .* params["nova_norm"]
        
        push!(predictions["numubar"], prediction_bar)
    end
    
    return predictions
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
                               L, params, false, density)
    p_nu_appearance = p_nu[:, 1, 2, 1]  # νμ → νe probability
    
    p_nubar = physics.osc.osc_prob(energy_grid * assets.nue_e_scale .+ assets.nue_e_bias,
                                  L, params, true, density)
    p_nubar_appearance = p_nubar[:, 1, 2, 1]  # ν̄μ → ν̄e probability
    
    # Apply smearing
    p_nu_smeared = smearnorm(energy_grid, p_nu_appearance, 1.0, 5, 
                           assets.nue_e_scale, assets.nue_e_bias)
    p_nubar_smeared = smearnorm(energy_grid, p_nubar_appearance, 1.0, 5,
                              assets.nue_e_scale, assets.nue_e_bias)
    
    # Calculate signal from muon neutrino flux
    numu_total_bins = length(assets.numu_data.total["NoOscillations_Signal"])
    signal_nu = rebin_energy_spectrum(p_nu_smeared, energy_edges, 0.5, 4.5, numu_total_bins)
    signal_nu ./= sum(signal_nu)
    signal_nu .*= assets.numu_data.total["NoOscillations_Signal"]
    
    signal_nubar = rebin_energy_spectrum(p_nubar_smeared, energy_edges, 0.5, 4.5, numu_total_bins)
    signal_nubar ./= sum(signal_nubar)
    signal_nubar .*= assets.numubar_data.total["NoOscillations_Signal"]
    
    # Rebin to electron neutrino analysis bins
    signal_nue = rebin_energy_spectrum(signal_nu, assets.numu_data.energy_edges, 0.5, 4.5, 8)
    signal_nuebar = rebin_energy_spectrum(signal_nubar, assets.numubar_data.energy_edges, 0.5, 4.5, 8)
    
    # Make predictions for each segment
    predictions = Dict{String, Dict{String, Vector{Float64}}}()
    predictions["nue"] = Dict{String, Vector{Float64}}()
    predictions["nuebar"] = Dict{String, Vector{Float64}}()
    
    for segment in 1:3
        # Neutrino segment
        backgrounds_nu = [
            assets.nue_data.mc_components["Wrong_sign_bkg$(segment)"],
            assets.nue_data.mc_components["Beam_nue_bkg$(segment)"],
            assets.nue_data.mc_components["Cosmic_bkg$(segment)"]
        ]
        
        condense_mode = (segment == 3)
        prediction_nu = fast_predictions_new(signal_nue, backgrounds_nu, params["nova_norm"],
                                           condense_to_bin3=condense_mode)
        
        predictions["nue"]["segment$(segment)"] = prediction_nu
        
        # Antineutrino segment
        backgrounds_nubar = [
            assets.nuebar_data.mc_components["Wrong_sign_bkg$(segment)"],
            assets.nuebar_data.mc_components["Beam_nue_bkg$(segment)"],
            assets.nuebar_data.mc_components["Cosmic_bkg$(segment)"]
        ]
        
        prediction_nubar = fast_predictions_new(signal_nuebar, backgrounds_nubar, params["nova_norm"],
                                              condense_to_bin3=condense_mode)
        
        predictions["nuebar"]["segment$(segment)"] = prediction_nubar
    end
    
    return predictions
end

function get_forward_model(physics, assets)

    function forward_model(params)
        exp_events_numu = make_numu_predictions(params, physics, assets)
        exp_events_nue = make_nue_predictions(params, physics, assets)
        
       
        return (
            numu = Poisson.(exp_events_numu),
            nue = Poisson.(exp_events_nue)
        )
    end
    return forward_model
end



function get_plot(physics, assets)
    """Create plotting function for NOvA data and predictions"""
    
    function plot(params)
        f = Figure(resolution=(1200, 800))
        
        # Get predictions
        numu_predictions = make_numu_predictions(params, physics, assets)
        nue_predictions = make_nue_predictions(params, physics, assets)
        
        # Plot muon neutrino disappearance by quartile
        for i in 1:4
            ax = Axis(f[1, i], title="νμ Quartile $(i)")
            
            observed = assets.numu_data.quartiles[i]["observed"]
            predicted = numu_predictions["numu"][i]
            energy_centers = (assets.numu_data.energy_edges[1:end-1] .+ 
                            assets.numu_data.energy_edges[2:end]) ./ 2
            
            scatter!(ax, energy_centers, observed, color=:black, label="Observed")
            lines!(ax, energy_centers, predicted, color=:red, label="Predicted")
            
            ax.xlabel = "Energy (GeV)"
            ax.ylabel = "Events"
            axislegend(ax)
        end
        
        # Plot electron neutrino appearance
        ax_nue = Axis(f[2, 1:2], title="νμ → νe Appearance")
        
        # Combine all segments
        all_observed_nue = vcat(assets.nue_data.observed.segment1,
                               assets.nue_data.observed.segment2,
                               assets.nue_data.observed.segment3)
        all_predicted_nue = vcat(nue_predictions["nue"]["segment1"],
                                nue_predictions["nue"]["segment2"],
                                nue_predictions["nue"]["segment3"])
        
        energy_bins = 1:length(all_observed_nue)
        scatter!(ax_nue, energy_bins, all_observed_nue, color=:black, label="Observed")
        lines!(ax_nue, energy_bins, all_predicted_nue, color=:blue, label="Predicted")
        
        ax_nue.xlabel = "Bin Number"
        ax_nue.ylabel = "Events"
        axislegend(ax_nue)
        
        # Plot antineutrino appearance
        ax_nuebar = Axis(f[2, 3:4], title="ν̄μ → ν̄e Appearance")
        
        all_observed_nuebar = vcat(assets.nuebar_data.observed.segment1,
                                  assets.nuebar_data.observed.segment2,
                                  assets.nuebar_data.observed.segment3)
        all_predicted_nuebar = vcat(nue_predictions["nuebar"]["segment1"],
                                   nue_predictions["nuebar"]["segment2"],
                                   nue_predictions["nuebar"]["segment3"])
        
        scatter!(ax_nuebar, energy_bins, all_observed_nuebar, color=:black, label="Observed")
        lines!(ax_nuebar, energy_bins, all_predicted_nuebar, color=:green, label="Predicted")
        
        ax_nuebar.xlabel = "Bin Number"
        ax_nuebar.ylabel = "Events"
        axislegend(ax_nuebar)
        
        return f
    end
    
    return plot
end

end  # module Nova


#outside the module

function calculate_nllh_numu(params, physics, assets)
    
    
    predictions = make_numu_predictions(params, physics, assets)
    
    nllh_total = 0.0
    epsilon = 1e-10
    
    # Process each quartile
    for i in 1:4
        # Neutrino quartile
        observed = assets.numu_data.quartiles[i]["observed"]
        predicted = predictions["numu"][i] .+ epsilon
        
        mask = .!((predicted .== epsilon) .& (observed .== 0))
        nllh_total += sum(predicted[mask] .- observed[mask] .* log.(predicted[mask]))
        
        # Antineutrino quartile
        observed_bar = assets.numubar_data.quartiles[i]["observed"]
        predicted_bar = predictions["numubar"][i] .+ epsilon
        
        mask_bar = .!((predicted_bar .== epsilon) .& (observed_bar .== 0))
        nllh_total += sum(predicted_bar[mask_bar] .- observed_bar[mask_bar] .* log.(predicted_bar[mask_bar]))
    end
    
    return nllh_total
end

function calculate_nllh_nue(params, physics, assets)
    """Calculate negative log-likelihood for electron neutrino appearance"""
    
    predictions = make_nue_predictions(params, physics, assets)
    
    nllh_total = 0.0
    epsilon = 1e-10
    
    # Process each segment
    for segment in 1:3
        # Neutrino segment
        observed_nu = getfield(assets.nue_data.observed, Symbol("segment$(segment)"))
        predicted_nu = predictions["nue"]["segment$(segment)"] .* sum(observed_nu) .+ epsilon
        
        mask_nu = .!((predicted_nu .== epsilon) .& (observed_nu .== 0))
        nllh_total += sum(predicted_nu[mask_nu] .- observed_nu[mask_nu] .* log.(predicted_nu[mask_nu]))
        
        # Antineutrino segment
        observed_nubar = getfield(assets.nuebar_data.observed, Symbol("segment$(segment)"))
        predicted_nubar = predictions["nuebar"]["segment$(segment)"] .* sum(observed_nubar) .+ epsilon
        
        mask_nubar = .!((predicted_nubar .== epsilon) .& (observed_nubar .== 0))
        nllh_total += sum(predicted_nubar[mask_nubar] .- observed_nubar[mask_nubar] .* log.(predicted_nubar[mask_nubar]))
    end
    
    return nllh_total
end

function get_forward_model(physics, assets)
    """Create the forward model function combining all channels"""
    
    function forward_model(params)
        # Calculate likelihoods for both channels
        nllh_numu = calculate_nllh_numu(params, physics, assets)
        nllh_nue = calculate_nllh_nue(params, physics, assets)
        
        # Return as exponential of negative log-likelihood
        total_nllh = nllh_numu + nllh_nue
        
        return Distributions.product_distribution([
            Distributions.Exponential(1/total_nllh)
        ])
    end
    
    return forward_model
end


function forward_model_numu(params, physics, assets)
    """Forward model for νμ disappearance channel"""
    # Get predictions
    expected_numu = get_expected_numu(params, physics, assets)
    expected_numubar = get_expected_numubar(params, physics, assets)
    
    # Combine neutrino and antineutrino data
    expected_total = vcat(expected_numu, expected_numubar)
    observed_total = vcat(
        get_observed_data(assets.numu_data, :neutrino),
        get_observed_data(assets.numubar_data, :antineutrino)
    )
    
    # Build covariance matrix
    covariance = build_covariance_numu(expected_total, assets)
    
    return Distributions.MvNormal(expected_total, covariance)
end

function forward_model_nue(params, physics, assets)
    """Forward model for νe appearance channel"""
    # Get predictions  
    expected_nue = get_expected_nue(params, physics, assets)
    expected_nuebar = get_expected_nuebar(params, physics, assets)
    
    # Combine neutrino and antineutrino data
    expected_total = vcat(expected_nue, expected_nuebar)
    observed_total = vcat(
        get_observed_data(assets.nue_data, :neutrino),
        get_observed_data(assets.nuebar_data, :antineutrino)
    )
    
    # Build covariance matrix
    covariance = build_covariance_nue(expected_total, assets)
    
    return Distributions.MvNormal(expected_total, covariance)
end



function get_forward_model(physics, assets)
    """Create the forward model combining all NOvA channels"""
    
    function forward_model(params)
        # Create distributions for each channel
        numu_dist = forward_model_numu(params, physics, assets)
        nue_dist = forward_model_nue(params, physics, assets)
        
        # Return product distribution for joint analysis
        return Distributions.StructArray((
            numu = numu_dist,
            nue = nue_dist
        ))
    end
    
    return forward_model
end

function build_covariance_numu(expected_events, assets)
    """Build covariance matrix for νμ channel"""
    n_bins = length(expected_events)
    
    # Statistical uncertainty (Poisson)
    stat_cov = Diagonal(expected_events)
    
    # Systematic uncertainties
    sys_cov = zeros(n_bins, n_bins)
    
    # Energy scale uncertainties (correlated across quartiles)
    energy_scale_frac = assets.systematics.energy_scale_uncertainty
    for i in 1:n_bins, j in 1:n_bins
        quartile_i = get_quartile_index(i)
        quartile_j = get_quartile_index(j)
        correlation = assets.systematics.energy_scale_correlation[quartile_i, quartile_j]
        sys_cov[i,j] += (energy_scale_frac * expected_events[i]) * 
                        (energy_scale_frac * expected_events[j]) * correlation
    end
    
    # Flux uncertainties (fully correlated)
    flux_frac = assets.systematics.flux_uncertainty
    flux_vector = flux_frac .* expected_events
    sys_cov += flux_vector * flux_vector'
    
    # Cross-section uncertainties
    for reaction in assets.systematics.xsec_reactions
        frac = assets.systematics.xsec_uncertainty[reaction]
        weights = get_xsec_weights(expected_events, reaction, assets)
        sys_cov += (frac^2) * (weights * weights')
    end
    
    total_cov = stat_cov + sys_cov
    return Symmetric(total_cov)
end