
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

# ===================================================
# Configuration
# ===================================================

function configure(physics)
    physics = (;physics.osc, physics.xsec)
    assets = get_assets(physics)
    return Nova(
        physics = physics,
        params = (nova_norm = 1.0,),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

# ===================================================
# Asset loading
# ===================================================

function get_assets(physics; datadir = @__DIR__)
    @info "Loading NOvA data"

    # Load data from ROOT files
    data_file = ROOTFile(joinpath(datadir, "NOvA_2020_data_histograms.root"))
    mc_file = ROOTFile(joinpath(datadir, "NOvA_2020_data_release_predictions_with_systs_all_hists.root"))

    # Energy binning for electron neutrino analysis
    energy_edges = collect(range(0.5, stop=4.5, length=9))

    # Load electron neutrino data
    nue_data = load_nue_data(data_file, mc_file, energy_edges)
    nuebar_data = load_nuebar_data(data_file, mc_file, energy_edges)

    # Load muon neutrino data by quartiles
    numu_data, numubar_data = load_numu_data(data_file, mc_file)

    # Compute total NoOscillations_Signal across all quartiles (for NUE signal)
    # CRITICAL: nova_new.jl uses TOTAL flux, not just quartile 1
    numu_total_signal = sum(q["NoOscillations_Signal"] for q in numu_data.quartiles)
    numubar_total_signal = sum(q["NoOscillations_Signal"] for q in numubar_data.quartiles)

    # Flatten observed data — order MUST match forward model assembly
    observed_flat = Vector{Float64}()

    # NUE segments (3 × 8 bins)
    append!(observed_flat, nue_data.observed.segment1)
    append!(observed_flat, nue_data.observed.segment2)
    append!(observed_flat, nue_data.observed.segment3)

    # NUEBAR segments (3 × 8 bins)
    append!(observed_flat, nuebar_data.observed.segment1)
    append!(observed_flat, nuebar_data.observed.segment2)
    append!(observed_flat, nuebar_data.observed.segment3)

    # NUMU quartiles
    for i in 1:4
        append!(observed_flat, numu_data.quartiles[i]["observed"])
    end

    # NUMUBAR quartiles
    for i in 1:4
        append!(observed_flat, numubar_data.quartiles[i]["observed"])
    end

    observed = observed_flat

    # NOvA baseline and matter density
    L = 810.0  # km
    density = 2.84 * 0.5  # g/cm³ * Z/A ratio (not passed to osc_prob; handled by framework)

    assets = (
        L = L,
        density = density,
        energy_edges = energy_edges,
        nue_data = nue_data,
        nuebar_data = nuebar_data,
        numu_data = numu_data,
        numubar_data = numubar_data,
        numu_total_signal = numu_total_signal,
        numubar_total_signal = numubar_total_signal,
        # Smearing parameters for each quartile (from nova_new.jl)
        numu_smearing = [0.078, 0.092, 0.104, 0.115],
        numubar_smearing = [0.085, 0.089, 0.097, 0.102],
        # Energy scale and bias parameters (from nova_new.jl)
        numu_e_scale = 1.05,
        numu_e_bias = 0.0,
        nue_e_scale = 0.65,
        nue_e_bias = 0.02,
        observed = observed
    )

    return assets
end

# ===================================================
# Data loading functions
# ===================================================

function load_nue_data(data_file, mc_file, energy_edges)
    """Load electron neutrino data with 3 segments — nova_new.jl indexing"""

    println("=== LOADING NUE DATA ===")

    # Get neutrino data histogram — access via Symbol key (:fN) for bin contents
    data_hist = data_file["neutrino_mode_nue"]
    data_values = haskey(data_hist, :fN) ? data_hist[:fN] : error("Cannot find :fN in neutrino data")

    # Extract segments (nova_new.jl indexing: Python → Julia +1)
    # Python: data[1:9], data[10:18], pad(data[18:21], (0,5))
    observed1 = Float64.(data_values[2:9])       # 8 elements: bins 1-8 (Python [1:9])
    observed2 = Float64.(data_values[11:18])     # 8 elements: bins 10-17 (Python [10:18])
    observed3 = vcat(Float64.(data_values[19:21]), zeros(5))  # 8 elements: 3 data + 5 zeros (Python pad[18:21])

    # Load Monte Carlo components for FHC
    mc_components = Dict{String, Vector{Float64}}()

    if haskey(mc_file, "prediction_components_nue_fhc")
        mc_dir = mc_file["prediction_components_nue_fhc"]

        for key in keys(mc_dir)
            component_hist = mc_dir[key]
            mc_values = haskey(component_hist, :fN) ? component_hist[:fN] : continue
            component_name = string(key)

            if length(mc_values) >= 21
                mc_components[component_name * "1"] = Float64.(mc_values[1:8])     # Python [0:8]  → Julia [1:8]
                mc_components[component_name * "2"] = Float64.(mc_values[9:16])    # Python [8:16] → Julia [9:16]
                # FIXED: Use MC values for segment 3 (nova_new.jl had bug using data_values)
                mc_components[component_name * "3"] = vcat(Float64.(mc_values[17:21]), zeros(3))  # Python [16:21] → Julia [17:21]
            end
        end
    else
        println("Warning: 'prediction_components_nue_fhc' not found in MC file")
    end

    return (
        observed = (segment1 = observed1, segment2 = observed2, segment3 = observed3),
        mc_components = mc_components,
        energy_edges = energy_edges
    )
end

function load_nuebar_data(data_file, mc_file, energy_edges)
    """Load antineutrino data with 3 segments — nova_new.jl indexing"""

    println("=== LOADING NUEBAR DATA ===")

    # Get antineutrino data histogram
    data_hist = data_file["antineutrino_mode_nue"]
    data_values = haskey(data_hist, :fN) ? data_hist[:fN] : error("Cannot find :fN in antineutrino data")

    # Extract segments (nova_new.jl indexing)
    observed1 = Float64.(data_values[2:9])
    observed2 = Float64.(data_values[11:18])
    observed3 = vcat(Float64.(data_values[19:21]), zeros(5))

    # Load Monte Carlo components for RHC
    mc_components = Dict{String, Vector{Float64}}()

    if haskey(mc_file, "prediction_components_nue_rhc")
        mc_dir = mc_file["prediction_components_nue_rhc"]

        for key in keys(mc_dir)
            component_hist = mc_dir[key]
            mc_values = haskey(component_hist, :fN) ? component_hist[:fN] : continue
            component_name = string(key)

            if length(mc_values) >= 21
                mc_components[component_name * "1"] = Float64.(mc_values[1:8])
                mc_components[component_name * "2"] = Float64.(mc_values[9:16])
                # FIXED: Use MC values for segment 3
                mc_components[component_name * "3"] = vcat(Float64.(mc_values[17:21]), zeros(3))
            end
        end
    else
        println("Warning: 'prediction_components_nue_rhc' not found in MC file")
    end

    return (
        observed = (segment1 = observed1, segment2 = observed2, segment3 = observed3),
        mc_components = mc_components,
        energy_edges = energy_edges
    )
end

function load_numu_data(data_file, mc_file)
    """Load muon neutrino data by quartiles"""

    println("=== LOADING NUMU DATA ===")

    # Get energy binning from first quartile
    first_hist = data_file["neutrino_mode_numu_quartile1"]
    energy_edges = extract_energy_edges(first_hist)
    n_bins = length(energy_edges) - 1

    # Initialize storage
    numu_quartiles = []
    numubar_quartiles = []

    # Process each quartile
    for q in 1:4
        # Load neutrino data
        neutrino_quartile = load_quartile_data(data_file, mc_file, q, "neutrino")
        push!(numu_quartiles, neutrino_quartile)

        # Load antineutrino data
        antineutrino_quartile = load_quartile_data(data_file, mc_file, q, "antineutrino")
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

function extract_energy_edges(histogram)
    """Extract energy bin edges from ROOT histogram Dict"""
    if haskey(histogram, :fXaxis_fXbins)
        edges = histogram[:fXaxis_fXbins]
        return Float64.(collect(edges))
    else
        error("Cannot extract energy edges from histogram")
    end
end

function load_quartile_data(data_file, mc_file, quartile, mode)
    """Load observed and MC data for a single quartile — strips underflow/overflow"""

    mode_prefix = mode == "neutrino" ? "neutrino_mode" : "antineutrino_mode"
    beam_mode = mode == "neutrino" ? "fhc" : "rhc"

    # Load observed data
    obs_hist_key = "$(mode_prefix)_numu_quartile$(quartile)"
    obs_hist = data_file[obs_hist_key]
    obs_raw = haskey(obs_hist, :fN) ? obs_hist[:fN] : error("Cannot find :fN in $obs_hist_key")
    observed = Float64.(obs_raw[2:end-1])  # Strip underflow/overflow

    # Load MC components
    mc_key = "prediction_components_numu_$(beam_mode)_Quartile$(quartile)"
    quartile_data = Dict{String, Vector{Float64}}()
    quartile_data["observed"] = observed

    if haskey(mc_file, mc_key)
        mc_dir = mc_file[mc_key]

        for key in keys(mc_dir)
            component_hist = mc_dir[key]
            component_name = string(key)
            mc_raw = haskey(component_hist, :fN) ? component_hist[:fN] : continue
            mc_values = Float64.(mc_raw[2:end-1])  # Strip underflow/overflow

            quartile_data[component_name] = mc_values
        end
    end

    return quartile_data
end

# ===================================================
# Physics: Rebinning functions
# ===================================================


function digitize_and_bin(values::Vector{Float64}, fine_centers::Vector{Float64},
                          coarse_edges::Vector{Float64}; method::Symbol=:mean)
    n_bins = length(coarse_edges) - 1
    result = zeros(Float64, n_bins)
    bin_sums = zeros(Float64, n_bins)
    bin_counts = zeros(Int, n_bins)

    for i in 1:length(fine_centers)
        center = fine_centers[i]
        # searchsortedlast: returns k s.t. edges[k] <= center < edges[k+1]
        bin_idx = searchsortedlast(coarse_edges, center)

        # Handle rightmost-edge-inclusive (matching dama)
        if bin_idx == length(coarse_edges) && center == coarse_edges[end]
            bin_idx = length(coarse_edges) - 1
        end

        # Skip underflow (0) and overflow (length(edges))
        if bin_idx >= 1 && bin_idx <= n_bins
            bin_sums[bin_idx] += values[i]
            bin_counts[bin_idx] += 1
        end
    end

    for b in 1:n_bins
        if bin_counts[b] > 0
            if method == :mean
                result[b] = bin_sums[b] / bin_counts[b]
            elseif method == :sum
                result[b] = bin_sums[b]
            end
        end
        # Bins with no fine points stay at 0.0 (safe for Poisson downstream)
    end

    return result
end

function rebin_energy_spectrum(input_data::Vector{Float64}, edges::Vector{Float64},
                               e_min::Float64=0.5, e_max::Float64=4.5, num_bins::Int=8)
    # Handle masked arrays or missing data
    data = replace(input_data, missing => 0.0, NaN => 0.0)
    edge_values = collect(edges)

    # Create new equally spaced bin edges
    new_edges = collect(range(e_min, stop=e_max, length=num_bins + 1))
    new_counts = zeros(Float64, num_bins)

    # For each input bin
    for i in 1:(length(edge_values)-1)
        old_e_low = edge_values[i]
        old_e_high = edge_values[i+1]
        old_width = old_e_high - old_e_low

        # Skip bins completely outside our range
        if old_e_high < e_min || old_e_low > e_max
            continue
        end

        # For each new bin
        for j in 1:num_bins
            new_e_low = new_edges[j]
            new_e_high = new_edges[j + 1]

            # Calculate overlap
            overlap_low = max(old_e_low, new_e_low)
            overlap_high = min(old_e_high, new_e_high)

            if overlap_high > overlap_low
                # Calculate fractional overlap
                overlap = overlap_high - overlap_low
                fraction = overlap / old_width

                # Add fractional counts to new bin
                new_counts[j] += data[i] * fraction
            end
        end
    end

    return new_counts
end

# ===================================================
# Physics: Energy edge calculation
# ===================================================

function calculate_energy_edges(energy_centers::Vector{Float64})
    if length(energy_centers) < 2
        error("Need at least 2 energy centers")
    end
    if any(energy_centers .<= 0)
        error("Energy centers must be positive")
    end

    n = length(energy_centers)
    edges = zeros(Float64, n + 1)

    # Detect log spacing (all positive and logarithmic)
    is_log = all(energy_centers .> 0) &&
             length(energy_centers) > 1 &&
             let d = diff(log.(energy_centers))
                 all(d .> 0) && isapprox(std(d), 0, atol=1e-10)
             end

    if is_log
        # Log-spaced: use geometric mean between adjacent centers (matching dama)
        edges[2:n] = sqrt.(energy_centers[2:end] .* energy_centers[1:end-1])

        # Boundary edges: use first/last diff of log (matching dama's edges_from_points)
        diff_log = diff(log.(energy_centers))
        edges[1] = energy_centers[1] / exp(diff_log[1] / 2)
        edges[end] = energy_centers[end] * exp(diff_log[end] / 2)
    else
        # Linear-spaced: use arithmetic mean between adjacent centers (matching dama)
        edges[2:n] = (energy_centers[2:end] .+ energy_centers[1:end-1]) ./ 2

        # Boundary edges: use first/last diff (matching dama's edges_from_points)
        diff_lin = diff(energy_centers)
        edges[1] = energy_centers[1] - diff_lin[1] / 2
        edges[end] = energy_centers[end] + diff_lin[end] / 2
    end

    return edges
end

# ===================================================
# Physics: Smearing functions
# ===================================================

function smearnorm(energies::Vector{Float64}, p::Vector{Float64}, percent::Float64,
                  width::Int=10, e_scale::Float64=1.0, e_bias::Float64=0.0)
    out = zeros(Float64, length(p))
    n = length(p)
    for i in 1:n
        norm_val = 0.0
        for j in max(1, i - width):min(n, i + width)
            coeff = 1.0
            norm_val += coeff
            out[i] += coeff * p[j]
        end
        if norm_val > 0
            out[i] /= norm_val
        end
    end
    return out
end

function smear_philipp(energies::Vector{Float64}, p::Vector{Float64}, percent::Float64,
                     width::Int=10, e_scale::Float64=1.0, e_bias::Float64=0.0)
    out = zeros(Float64, length(p))
    n = length(p)
    for i in 1:n
        e = energies[i] * e_scale + e_bias
        norm_val = 0.0
        for j in max(1, i - width):min(n, i + width)
            sigma = percent * energies[j]
            coeff = (1/sigma) * exp(-0.5 * ((e - energies[j]) / sigma)^2)
            norm_val += coeff
            out[i] += coeff * p[j]
        end
        if norm_val > 0
            out[i] /= norm_val
        end
    end
    return out
end

# ===================================================
# Physics: Signal + background combination
# ===================================================

function fast_predictions_new(signal::Vector{Float64}, backgrounds::Vector{Vector{Float64}},
                             norm_factor::Float64; condense_to_bin3::Bool=false)
    total = zeros(Float64, length(signal))

    # First calculate total sum including signal and all backgrounds
    total_sum = 0.0
    for i in eachindex(signal)
        total_sum += signal[i]
    end

    for bg in backgrounds
        for i in eachindex(bg)
            total_sum += bg[i]
        end
    end

    # Add signal components
    for i in eachindex(signal)
        total[i] = signal[i]
    end

    # Add normalized backgrounds (divided by total_sum)
    for bg in backgrounds
        for i in eachindex(bg)
            if total_sum > 0
                total[i] += (bg[i] / total_sum)
            end
        end
    end

    # Apply norm factor
    total .*= norm_factor

    # Final normalization
    total_sum = sum(total)

    if total_sum > 0
        for i in eachindex(total)
            total[i] = (total[i] / total_sum) * norm_factor
        end
    end

    # Condense all values into bin 3 if flag is True (segment 3 systematics)
    if condense_to_bin3 && length(total) > 2
        condensed_sum = sum(total)

        # Zero out all bins
        fill!(total, 0.0)

        # Put the entire sum in bin 3 (index 3 in Julia 1-based)
        total[3] = condensed_sum
    end

    return total
end

# ===================================================
# Prediction: Muon neutrino disappearance
# ===================================================


function make_numu_predictions(params, physics, assets)
    L = [assets.L]

    # Energy grid: logspace 0.1 to 10 GeV, 100 points (CENTERS)
    # From nova_new.jl: 10.0 .^ range(-1, stop=1, length=100)
    p_energy_centers = 10.0 .^ range(-1, stop=1, length=100)

    # Calculate neutrino oscillation probabilities
    # Using Newtrinos convention: osc_prob(E, L, params; anti=false), no explicit density
    p_probs = physics.osc.osc_prob(
        collect(p_energy_centers) .* assets.numu_e_scale .+ assets.numu_e_bias,
        L, params; anti=false)
    # νμ → νμ survival probability [:, 1, 2, 2]
    p_nu_survival = ndims(p_probs) >= 4 ? p_probs[:, 1, 2, 2] : vec(p_probs)

    # Calculate antineutrino oscillation probabilities
    p_probs_bar = physics.osc.osc_prob(
        collect(p_energy_centers) .* assets.numu_e_scale .+ assets.numu_e_bias,
        L, params; anti=true)
    p_nubar_survival = ndims(p_probs_bar) >= 4 ? p_probs_bar[:, 1, 2, 2] : vec(p_probs_bar)

    # Apply smearing and make predictions for each quartile
    predictions_numu = Vector{Float64}[]
    predictions_numubar = Vector{Float64}[]

    for i in 1:4
        # --- Neutrino quartile ---
        p_smeared = smearnorm(collect(p_energy_centers), collect(p_nu_survival),
                             assets.numu_smearing[i], 4,
                             assets.numu_e_scale, assets.numu_e_bias)

        # CRITICAL: digitize_and_bin (NOT rebin_energy_spectrum)
        # This is the key physics difference from nova_last.jl
        quartile_data = assets.numu_data.quartiles[i]
        q_edges = assets.numu_data.energy_edges
        p_binned = digitize_and_bin(p_smeared, collect(p_energy_centers), q_edges, method=:mean)
        # NOTE: p_binned is NOT normalized to sum=1 (matching nova_new.jl behavior)

        # Calculate prediction with nova_norm
        prediction = (quartile_data["NoOscillations_Signal"] .* p_binned .+
                     quartile_data["NoOscillations_Total_beam_bkg"] .+
                     quartile_data["Cosmic_bkg"]) .* params.nova_norm
        push!(predictions_numu, prediction)

        # --- Antineutrino quartile ---
        p_smeared_bar = smearnorm(collect(p_energy_centers), collect(p_nubar_survival),
                                 assets.numubar_smearing[i], 4,
                                 assets.numu_e_scale, assets.numu_e_bias)

        quartile_data_bar = assets.numubar_data.quartiles[i]
        p_binned_bar = digitize_and_bin(p_smeared_bar, collect(p_energy_centers),
                                        assets.numubar_data.energy_edges, method=:mean)

        prediction_bar = (quartile_data_bar["NoOscillations_Signal"] .* p_binned_bar .+
                         quartile_data_bar["NoOscillations_Total_beam_bkg"] .+
                         quartile_data_bar["Cosmic_bkg"]) .* params.nova_norm
        push!(predictions_numubar, prediction_bar)
    end

    return (numu = predictions_numu, numubar = predictions_numubar)
end

# ===================================================
# Prediction: Electron neutrino appearance
# ===================================================


function make_nue_predictions(params, physics, assets)
    L = [assets.L]

    # Energy scale and bias for electron neutrino analysis (from nova_new.jl)
    e_scale_nue = assets.nue_e_scale  # 0.65
    e_bias_nue = assets.nue_e_bias    # 0.02

    # Energy grid: linspace 0.5 to 4.5, 100 points (CENTERS)
    p_energy = collect(range(0.5, stop=4.5, length=100))
    energy_edges_calc = calculate_energy_edges(p_energy)

    # --- Neutrino appearance (νμ → νe) ---
    p_probs = physics.osc.osc_prob(
        p_energy .* e_scale_nue .+ e_bias_nue,
        L, params; anti=false)
    # νμ → νe probability [:, 1, 2, 1]
    p_nue_appearance = ndims(p_probs) >= 4 ? p_probs[:, 1, 2, 1] : vec(p_probs)

    # Smear
    p_nue_smeared = smearnorm(p_energy, collect(p_nue_appearance), 1.0, 5,
                              e_scale_nue, e_bias_nue)

    # Rebin probabilities to NUMU detector binning (fractional overlap)
    numu_nbins = length(assets.numu_data.energy_edges) - 1
    binned_means_tot = rebin_energy_spectrum(collect(p_nue_smeared), energy_edges_calc,
                                              0.5, 4.5, numu_nbins)
    binned_means_tot ./= sum(binned_means_tot)  # Normalize to probability shape

    # Signal from TOTAL muon neutrino flux (all quartiles summed)
    # CRITICAL: nova_new.jl uses sum of all 4 quartiles, not just quartile 1
    signal_numu = assets.numu_total_signal .* binned_means_tot

    # Rebin signal to electron neutrino analysis bins (8 bins)
    signal_nue = rebin_energy_spectrum(collect(signal_numu),
                                       assets.numu_data.energy_edges, 0.5, 4.5, 8)

    # --- Antineutrino appearance (ν̄μ → ν̄e) ---
    p_bar_probs = physics.osc.osc_prob(
        p_energy .* e_scale_nue .+ e_bias_nue,
        L, params; anti=true)
    p_nuebar_appearance = ndims(p_bar_probs) >= 4 ? p_bar_probs[:, 1, 2, 1] : vec(p_bar_probs)

    p_nuebar_smeared = smearnorm(p_energy, collect(p_nuebar_appearance), 1.0, 5,
                                 e_scale_nue, e_bias_nue)

    numubar_nbins = length(assets.numubar_data.energy_edges) - 1
    binned_means_tot_bar = rebin_energy_spectrum(collect(p_nuebar_smeared), energy_edges_calc,
                                                   0.5, 4.5, numubar_nbins)
    binned_means_tot_bar ./= sum(binned_means_tot_bar)

    signal_numubar = assets.numubar_total_signal .* binned_means_tot_bar

    signal_nuebar = rebin_energy_spectrum(collect(signal_numubar),
                                          assets.numubar_data.energy_edges, 0.5, 4.5, 8)

    # --- Make predictions for each segment ---
    predictions_nue = Dict{String, Vector{Float64}}()
    predictions_nuebar = Dict{String, Vector{Float64}}()

    for segment in 1:3
        # Neutrino segment
        backgrounds_nu = Vector{Float64}[]
        for bkg_type in ["Wrong_sign_bkg", "Beam_nue_bkg", "Cosmic_bkg"]
            bkg_name = bkg_type * string(segment)
            if haskey(assets.nue_data.mc_components, bkg_name)
                push!(backgrounds_nu, assets.nue_data.mc_components[bkg_name])
            end
        end

        condense_mode = (segment == 3)
        prediction_nu = fast_predictions_new(signal_nue, backgrounds_nu,
                                            params.nova_norm; condense_to_bin3=condense_mode)
        predictions_nue["segment$(segment)"] = prediction_nu

        # Antineutrino segment
        backgrounds_nubar = Vector{Float64}[]
        for bkg_type in ["Wrong_sign_bkg", "Beam_nue_bkg", "Cosmic_bkg"]
            bkg_name = bkg_type * string(segment)
            if haskey(assets.nuebar_data.mc_components, bkg_name)
                push!(backgrounds_nubar, assets.nuebar_data.mc_components[bkg_name])
            end
        end

        prediction_nubar = fast_predictions_new(signal_nuebar, backgrounds_nubar,
                                                params.nova_norm; condense_to_bin3=condense_mode)
        predictions_nuebar["segment$(segment)"] = prediction_nubar
    end

    return (nue = predictions_nue, nuebar = predictions_nuebar)
end

# ===================================================
# Forward model
# ===================================================

function get_forward_model(physics, assets)
    function forward_model(params)
        numu_predictions = make_numu_predictions(params, physics, assets)
        nue_predictions = make_nue_predictions(params, physics, assets)

        # Assemble predictions in same order as observed data
        all_predictions = Vector{Float64}()

        # NUE segments (3 × 8 bins)
        append!(all_predictions, nue_predictions.nue["segment1"])
        append!(all_predictions, nue_predictions.nue["segment2"])
        append!(all_predictions, nue_predictions.nue["segment3"])

        # NUEBAR segments (3 × 8 bins)
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

        return Poisson.(max.(all_predictions, 1e-10))
    end
    return forward_model
end

# ===================================================
# Plot function
# ===================================================

function get_plot(physics, assets)
    function plot(params)
        f = Figure(resolution=(1400, 1200))

        # Get predictions
        numu_predictions = make_numu_predictions(params, physics, assets)
        nue_predictions = make_nue_predictions(params, physics, assets)

        # Row 1: Muon neutrino disappearance by quartile
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

        # Row 2: Muon antineutrino disappearance by quartile
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

        # Row 3: Electron neutrino appearance (3 segments)
        for seg in 1:3
            ax_nue = Axis(f[3, seg], title="νμ → νe Segment $seg")

            if seg == 1
                observed_nue = assets.nue_data.observed.segment1
            elseif seg == 2
                observed_nue = assets.nue_data.observed.segment2
            else
                observed_nue = assets.nue_data.observed.segment3
            end

            predicted_nue = nue_predictions.nue["segment$(seg)"]
            predicted_nue_errors = sqrt.(max.(predicted_nue, 0))
            energy_centers = (assets.nue_data.energy_edges[1:end-1] .+
                            assets.nue_data.energy_edges[2:end]) ./ 2

            scatter!(ax_nue, energy_centers, observed_nue, color=:black, label="Observed", markersize=6)
            errorbars!(ax_nue, energy_centers, predicted_nue, predicted_nue_errors, color=:blue, linewidth=2)
            lines!(ax_nue, energy_centers, predicted_nue, color=:blue, linewidth=2, label="Predicted")

            ax_nue.xlabel = "Energy (GeV)"
            ax_nue.ylabel = "Events"
            axislegend(ax_nue, position=:rt, labelsize=6)
        end

        # Row 4: Electron antineutrino appearance (3 segments)
        for seg in 1:3
            ax_nuebar = Axis(f[4, seg], title="ν̄μ → ν̄e Segment $seg")

            if seg == 1
                observed_nuebar = assets.nuebar_data.observed.segment1
            elseif seg == 2
                observed_nuebar = assets.nuebar_data.observed.segment2
            else
                observed_nuebar = assets.nuebar_data.observed.segment3
            end

            predicted_nuebar = nue_predictions.nuebar["segment$(seg)"]
            predicted_nuebar_errors = sqrt.(max.(predicted_nuebar, 0))
            energy_centers = (assets.nuebar_data.energy_edges[1:end-1] .+
                            assets.nuebar_data.energy_edges[2:end]) ./ 2

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

end # module nova
