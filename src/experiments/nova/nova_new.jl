# ---------------------------------------------------
# This code analyzes neutrino oscillation data from the NOvA experiment.
# Julia translation of nova_last.py — reproduces exact same physics.
# It processes both neutrino and antineutrino data and calculates oscillation probabilities
# ---------------------------------------------------
module nova

using UnROOT
using Distributions
using LinearAlgebra
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
# Data structures to replace dama.GridData and dama.Edges
# ===================================================

mutable struct Edges
    edges::Vector{Float64}

    Edges(e::Vector{Float64}) = new(e)
end

function Base.getindex(e::Edges, i::Int)
    return e.edges[i]
end

function Base.length(e::Edges)
    return length(e.edges)
end

function squeezed_edges(e::Edges)
    return e.edges
end

mutable struct GridData
    energy::Edges
    grid::Union{Nothing, GridData}
    data::Dict{Symbol, Any}

    GridData(; energy::Edges=Edges(Float64[]), grid::Union{Nothing, GridData}=nothing) =
        new(energy, grid, Dict{Symbol, Any}())
end

# Helper constructor matching dama's GridData(energy=...)
GridData(energy::Edges) = GridData(energy=energy)

# Allow attribute-style access via getproperty/setproperty
function Base.getproperty(gd::GridData, name::Symbol)
    if name === :energy || name === :grid
        return getfield(gd, name)
    elseif name === :data
        return getfield(gd, :data)
    elseif haskey(gd.data, name)
        return gd.data[name]
    else
        error("GridData has no property :$name")
    end
end

function Base.setproperty!(gd::GridData, name::Symbol, value)
    if name === :energy || name === :grid
        setfield!(gd, name, value)
    elseif name === :data
        setfield!(gd, :data, value)
    else
        gd.data[name] = value
    end
end

function Base.getindex(gd::GridData, key::String)
    return gd.data[Symbol(key)]
end

function Base.setindex!(gd::GridData, value, key::String)
    gd.data[Symbol(key)] = value
end

function Base.haskey(gd::GridData, key::Union{String, Symbol})
    k = key isa Symbol ? key : Symbol(key)
    return haskey(gd.data, k)
end

# ===================================================
# digitize_and_bin — matches dama's .binwise(energy=edges).mean()
#
# In dama, .binwise(energy=coarse_edges).mean():
#   1. Takes fine-grid CENTERS (not edges) from the source GridData
#   2. Uses np.digitize to assign each center to ONE coarse bin
#   3. Computes the mean of values whose centers fall in each bin
#   4. Centers outside the coarse edges are excluded (handled via NaN)
#
# This is a SIMPLE digitize-and-average, NOT fractional-overlap rebinning.
# ===================================================

"""
    digitize_and_bin(values, fine_centers, coarse_edges; method=:mean)

Rebin values from a fine grid (defined by centers) to a coarse grid (defined by edges).
Matches dama's .binwise(energy=coarse_edges).mean() exactly.

- `values`: array of values at each fine center
- `fine_centers`: energy CENTERS of the fine grid
- `coarse_edges`: energy EDGES of the coarse grid (length N+1 for N bins)
- `method`: `:mean` (default) or `:sum`

Returns: array of length (length(coarse_edges) - 1) with rebinned values.
Bins with no fine points get NaN (matching dama's fill_value=NaN default).
"""
function digitize_and_bin(values::Vector{Float64}, fine_centers::Vector{Float64},
                          coarse_edges::Vector{Float64}; method::Symbol=:mean)
    n_bins = length(coarse_edges) - 1
    result = fill(NaN, n_bins)
    bin_sums = zeros(Float64, n_bins)
    bin_counts = zeros(Int, n_bins)

    for i in 1:length(fine_centers)
        center = fine_centers[i]
        # searchsortedlast: returns k s.t. edges[k] <= x < edges[k+1] (roughly)
        # For x < edges[1]: returns 0
        # For edges[k] <= x < edges[k+1]: returns k
        # For x >= edges[end]: returns length(edges)
        bin_idx = searchsortedlast(coarse_edges, center)

        # Handle rightmost-edge-inclusive (matching dama: idx[sample == bins[-1]] -= 1)
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
    end

    return result
end

# ===================================================
# Load NOvA experiment data files
# ===================================================

# Get the directory containing this file
const _datadir = dirname(@__FILE__)

# Load data files
const _data_file = joinpath(_datadir, "NOvA_2020_data_histograms.root")
const _mc_file = joinpath(_datadir, "NOvA_2020_data_release_predictions_with_systs_all_hists.root")

# Open ROOT files
_data_root = nothing
_mc_root = nothing

function init_data()
    global _data_root, _mc_root
    _data_root = UnROOT.RootFile(_data_file)
    _mc_root = UnROOT.RootFile(_mc_file)
end

function close_data()
    global _data_root, _mc_root
    if _data_root !== nothing
        close(_data_root)
        _data_root = nothing
    end
    if _mc_root !== nothing
        close(_mc_root)
        _mc_root = nothing
    end
end

# Initialize data on module load
init_data()

# Define energy binning for analysis
# Python: np.linspace(0.5, 4.5, 9) -> 8 bins with 9 edges
const _energy_edges = collect(range(0.5, stop=4.5, length=9))

# ===================================================
# Helper to get array from ROOT, handling underflow/overflow
# ===================================================

# In ROOT histograms:
# - Python uproot: values()[0] = underflow, values()[1:N] = bins, values()[N+1] = overflow
# - Julia UnROOT: array[1] = underflow, array[2:N+1] = bins, array[N+2] = overflow
# So Python index i corresponds to Julia index i+1
#
# Python values()[a:b] corresponds to Julia array[a+1:b]

function get_root_array(root_obj)
    arr = UnROOT.array(root_obj)
    return arr
end

# ===================================================
# Initialize electron neutrino data structures
# ===================================================

const nue = GridData(energy=Edges(collect(_energy_edges)))
const data_nue = get_root_array(_data_root["neutrino_mode_nue"])

# Extract observed data across different segments
# Python: data_values[1:9], data_values[10:18], np.pad(data_values[18:21], (0,5))
nue.observed1 = data_nue[2:9]        # Python [1:9]  -> Julia [2:9]   (8 elements: bins 1-8)
nue.observed2 = data_nue[11:18]      # Python [10:18] -> Julia [11:18] (8 elements: bins 10-17)
segment3 = vcat(data_nue[19:21], zeros(5))  # Python [18:21] -> Julia [19:21], pad to 8
nue.observed3 = segment3

# Load Monte Carlo prediction components for electron neutrinos
mc_nue_fhc = _mc_root["prediction_components_nue_fhc"]
for key in keys(mc_nue_fhc)
    component_name = string(key)[1:end-2]  # Remove last 2 chars like Python [:-2]
    mc_values = get_root_array(mc_nue_fhc[key])

    nue[component_name * "1"] = mc_values[1:8]     # Python [0:8]  -> Julia [1:8]
    nue[component_name * "2"] = mc_values[9:16]    # Python [8:16] -> Julia [9:16]
    # Note: segment 3 uses data_nue (from DATA file), matching Python behavior
    segment3 = vcat(data_nue[17:21], zeros(3))  # Python [16:21] -> Julia [17:21], pad to 8
    nue[component_name * "3"] = segment3
end

# ===================================================
# Initialize electron antineutrino data structures
# ===================================================

const nuebar = GridData(energy=Edges(collect(_energy_edges)))
const data_nuebar = get_root_array(_data_root["antineutrino_mode_nue;1"])
nuebar.observed1 = data_nuebar[2:9]
nuebar.observed2 = data_nuebar[11:18]
segment3 = vcat(data_nuebar[19:21], zeros(5))
nuebar.observed3 = segment3

# Load Monte Carlo prediction components for antineutrinos
mc_nue_rhc = _mc_root["prediction_components_nue_rhc"]
for key in keys(mc_nue_rhc)
    component_name = string(key)[1:end-2]
    mc_values = get_root_array(mc_nue_rhc[key])

    nuebar[component_name * "1"] = mc_values[1:8]
    nuebar[component_name * "2"] = mc_values[9:16]
    # Note: segment 3 uses data_nuebar (from DATA file), matching Python behavior
    segment3 = vcat(data_nuebar[17:21], zeros(3))
    nuebar[component_name * "3"] = segment3
end

# ===================================================
# Initialize muon neutrino data structures
# The data is divided into quartiles based on resolution
# ===================================================

const numu_q = GridData[]
const numubar_q = GridData[]

# Get quartile 1 edges for initialization
q1_numu_hist = _data_root["neutrino_mode_numu_quartile1"]
q1_numubar_hist = _data_root["antineutrino_mode_numu_quartile1"]

q1_numu_edges = collect(get_root_array(q1_numu_hist.axis.edges))
q1_numubar_edges = collect(get_root_array(q1_numubar_hist.axis.edges))

# Create total data containers for all quartiles combined
const numu_tot = GridData(energy=Edges(q1_numu_edges))
const numubar_tot = GridData(energy=Edges(q1_numubar_edges))

# Initialize signal and background components for total containers
q1_numu_values = get_root_array(q1_numu_hist)
q1_numubar_values = get_root_array(q1_numubar_hist)

numu_tot.NoOscillations_Signal = zeros(Float64, size(q1_numu_values))
numu_tot.Oscillated_Signal = zeros(Float64, size(q1_numu_values))
numubar_tot.NoOscillations_Signal = zeros(Float64, size(q1_numubar_values))
numubar_tot.Oscillated_Signal = zeros(Float64, size(q1_numubar_values))

numu_tot.NoOscillations_Total_beam_bkg = zeros(Float64, size(q1_numu_values))
numu_tot.Cosmic_bkg = zeros(Float64, size(q1_numu_values))
numubar_tot.NoOscillations_Total_beam_bkg = zeros(Float64, size(q1_numubar_values))
numubar_tot.Cosmic_bkg = zeros(Float64, size(q1_numubar_values))

# Load data for each quartile and accumulate into totals
for q in 1:4
    # Process neutrino data
    q_numu_key = "neutrino_mode_numu_quartile$q"
    q_numu_hist = _data_root[q_numu_key]
    q_edges = collect(get_root_array(q_numu_hist.axis.edges))

    push!(numu_q, GridData(energy=Edges(q_edges)))
    numu_q[end].observed = get_root_array(q_numu_hist)

    mc_quartile_key = "prediction_components_numu_fhc_Quartile$q"
    mc_quartile = _mc_root[mc_quartile_key]

    for key in keys(mc_quartile)
        key_str = string(key)
        component_name = key_str[1:end-2]
        numu_q[end][component_name] = get_root_array(mc_quartile[key])
    end

    # Accumulate signal and background components
    numu_tot.NoOscillations_Signal .+= numu_q[end].NoOscillations_Signal
    numu_tot.Oscillated_Signal .+= numu_q[end].Oscillated_Signal
    numu_tot.NoOscillations_Total_beam_bkg .+= numu_q[end].NoOscillations_Total_beam_bkg
    numu_tot.Cosmic_bkg .+= numu_q[end].Cosmic_bkg

    # Process antineutrino data
    q_numubar_key = "antineutrino_mode_numu_quartile$q"
    q_numubar_hist = _data_root[q_numubar_key]
    q_edges = collect(get_root_array(q_numubar_hist.axis.edges))

    push!(numubar_q, GridData(energy=Edges(q_edges)))
    numubar_q[end].observed = get_root_array(q_numubar_hist)

    mc_quartile_key = "prediction_components_numu_rhc_Quartile$q"
    mc_quartile = _mc_root[mc_quartile_key]

    for key in keys(mc_quartile)
        key_str = string(key)
        component_name = key_str[1:end-2]
        numubar_q[end][component_name] = get_root_array(mc_quartile[key])
    end

    # Accumulate signal and background components for antineutrinos
    numubar_tot.NoOscillations_Signal .+= numubar_q[end].NoOscillations_Signal
    numubar_tot.Oscillated_Signal .+= numubar_q[end].Oscillated_Signal
    numubar_tot.NoOscillations_Total_beam_bkg .+= numubar_q[end].NoOscillations_Total_beam_bkg
    numubar_tot.Cosmic_bkg .+= numubar_q[end].Cosmic_bkg
end

# ===================================================
# Smearing functions
# ===================================================

"""
Apply energy resolution smearing to oscillation probabilities.
Convolves the input probabilities with a boxcar function.
"""
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

"""
Energy smearing function with Gaussian weights.
"""
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

# Energy scale and bias parameters for systematic effects
const e_scale_global = 1.05
const e_bias_global = 0.0

# ===================================================
# Prediction functions for muon neutrinos
# ===================================================

"""
Calculate muon neutrino disappearance predictions for both
neutrino and antineutrino modes across all quartiles.
"""
function make_predictions(params::Dict{String, Float64}, osc_prob::Function,
                         numu_q::Vector{GridData}, numubar_q::Vector{GridData})
    # Baseline and matter density for NOvA experiment
    L = [810.0]  # Baseline in km
    density = [2.84]  # Matter density in g/cm³

    # Z/A ratio for matter effects
    zoa = 0.5
    density[1] *= zoa

    # Calculate neutrino oscillation probabilities
    # Python: dm.GridData(energy=np.logspace(-1,1,100))
    # The 100 energy points are CENTERS (stored as points in dama's Axis)
    p_energy_centers = 10.0 .^ range(-1, stop=1, length=100)

    # Calculate probabilities: osc_prob(energy, L, params, is_antineutrino, density)
    p_probs = osc_prob(collect(p_energy_centers) .* e_scale_global .+ e_bias_global,
                       L, params, false, density)

    # Smearing parameters for each quartile (different detector resolutions)
    s_numu = [0.078, 0.092, 0.104, 0.115]

    # Process each quartile for neutrinos
    for i in 1:4
        # Extract probability: Python uses [:, 0, 1, 1] = (all E, L=0, final=νμ, initial=νμ)
        # Julia equivalent: [:, 1, 2, 2] = (all E, L=1, final=νμ=2, initial=νμ=2)
        if ndims(p_probs) >= 4
            p_col = p_probs[:, 1, 2, 2]
        else
            p_col = vec(p_probs)
        end

        p_smeared = smearnorm(collect(p_energy_centers), collect(p_col),
                              s_numu[i], 4, e_scale_global, e_bias_global)

        # CRITICAL: dama's .binwise(energy=coarse_edges).mean() digitizes fine-grid
        # CENTERS into coarse bins. Use digitize_and_bin, NOT fractional-overlap rebinning.
        q_edges = squeezed_edges(numu_q[i].energy)
        p_binned = digitize_and_bin(p_smeared, collect(p_energy_centers), q_edges, method=:mean)
        numu_q[i].p = p_binned

        numu_q[i].my_predicted = (numu_q[i].NoOscillations_Signal .* numu_q[i].p .+
                                  numu_q[i].NoOscillations_Total_beam_bkg .+
                                  numu_q[i].Cosmic_bkg) .* params["nova_norm"]
    end

    # Calculate antineutrino oscillation probabilities
    p_probs_bar = osc_prob(collect(p_energy_centers) .* e_scale_global .+ e_bias_global,
                           L, params, true, density)

    # Smearing parameters for each quartile (antineutrinos)
    s_numubar = [0.085, 0.089, 0.097, 0.102]

    # Process each quartile for antineutrinos
    for i in 1:4
        if ndims(p_probs_bar) >= 4
            p_col = p_probs_bar[:, 1, 2, 2]
        else
            p_col = vec(p_probs_bar)
        end

        p_smeared = smearnorm(collect(p_energy_centers), collect(p_col),
                              s_numubar[i], 4, e_scale_global, e_bias_global)

        q_edges = squeezed_edges(numubar_q[i].energy)
        p_binned = digitize_and_bin(p_smeared, collect(p_energy_centers), q_edges, method=:mean)
        numubar_q[i].p = p_binned

        numubar_q[i].my_predicted = (numubar_q[i].NoOscillations_Signal .* numubar_q[i].p .+
                                     numubar_q[i].NoOscillations_Total_beam_bkg .+
                                     numubar_q[i].Cosmic_bkg) .* params["nova_norm"]
    end
end

"""
Extract expected and observed event counts and bin information for muon neutrino analysis.
"""
function get_expected_observed_bins(params::Dict{String, Float64}, osc_prob::Function)
    make_predictions(params, osc_prob, numu_q, numubar_q)
    expected = Vector{Float64}[]
    observed = Vector{Float64}[]
    bins = Vector{Float64}[]

    for i in 1:4
        for q_arr in [numu_q, numubar_q]
            push!(expected, collect(q_arr[i].my_predicted))
            push!(observed, collect(q_arr[i].observed))
            push!(bins, squeezed_edges(q_arr[i].energy))
        end
    end

    return expected, expected, observed, bins
end

# ===================================================
# Negative log-likelihood functions (muon neutrino)
# ===================================================

"""
Calculate negative log-likelihood for muon neutrino disappearance data.
Poisson likelihood: sum(pred - obs * log(pred))
Returns: (neutrino NLL, antineutrino NLL)
"""
function nllh(params::Dict{String, Float64}, osc_prob::Function)
    make_predictions(params, osc_prob, numu_q, numubar_q)

    numu_llh = 0.0
    numubar_llh = 0.0

    for i in 1:4
        # Skip bins with zero prediction and zero observation
        mask = .!((numu_q[i].my_predicted .== 0) .& (numu_q[i].observed .== 0))
        pred = numu_q[i].my_predicted[mask]
        obs = numu_q[i].observed[mask]
        numu_llh += sum(pred - obs .* log.(pred))

        mask = .!((numubar_q[i].my_predicted .== 0) .& (numubar_q[i].observed .== 0))
        pred = numubar_q[i].my_predicted[mask]
        obs = numubar_q[i].observed[mask]
        numubar_llh += sum(pred - obs .* log.(pred))
    end

    return numu_llh, numubar_llh
end

# ===================================================
# Fast predictions function
# ===================================================

"""
Efficiently combine signal and background components with normalization.
Optimized function that adds signal and background, applies normalization,
and optionally condenses all values into bin 3.
"""
function fast_predictions_new(signal::Vector{Float64}, backgrounds::Vector{Vector{Float64}},
                             norm_factor::Float64, condense_to_bin3::Bool=false)
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

    # Add normalized backgrounds
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

    # Condense all values into bin 3 if flag is True
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
# Rebinning functions
# ===================================================

"""
Rebin a spectrum from irregular energy bins to regular bins.
Properly accounts for partial bin overlaps.
"""
function rebin_energy_spectrum(input_data::Vector{Float64}, edges::Union{Edges, Vector{Float64}},
                               e_min::Float64=0.5, e_max::Float64=4.5, num_bins::Int=8)
    # Handle Edges object
    if edges isa Edges
        edge_values = squeezed_edges(edges)
    else
        edge_values = collect(edges)
    end

    # Create edge pairs from consecutive edges
    edge_pairs = [(edge_values[i], edge_values[i+1]) for i in 1:length(edge_values)-1]

    # Create new equally spaced bin edges
    new_edges = collect(range(e_min, stop=e_max, length=num_bins + 1))
    new_counts = zeros(Float64, num_bins)

    # For each input bin
    for i in 1:length(input_data)
        old_e_low, old_e_high = edge_pairs[i]
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
                new_counts[j] += input_data[i] * fraction
            end
        end
    end

    return new_counts
end

"""
Calculate bin edges from bin centers, maintaining proper spacing.
Matches dama's Edges.edges_from_points() for both log and linear spacing.

For logarithmic binning: uses geometric mean between adjacent centers.
For linear binning: uses arithmetic mean between adjacent centers.
Boundaries: uses first/last diff to extrapolate, matching dama's behavior.
"""
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
# Prediction functions for electron neutrinos
# ===================================================

"""
Calculate electron neutrino appearance predictions for both
neutrino and antineutrino modes.
"""
function make_predictions_nue_new(params::Dict{String, Float64}, osc_prob::Function,
                                 nue::GridData, nuebar::GridData)
    # Baseline and matter density for NOvA experiment
    L = [810.0]  # Baseline in km
    density = [2.84]  # Matter density in g/cm³
    zoa = 0.5  # Z/A ratio for matter effects
    density[1] *= zoa

    # Energy scale and bias parameters for electron neutrino analysis
    e_scale_nue = 0.65
    e_bias_nue = 0.02

    # Calculate oscillation probabilities for neutrinos
    p_energy = collect(range(0.5, stop=4.5, length=100))
    energy_centers = p_energy
    energy_edges_calc = calculate_energy_edges(energy_centers)

    p_probs = osc_prob(p_energy .* e_scale_nue .+ e_bias_nue, L, params, false, density)

    # Extract νμ→νe probability: Python uses [:, 0, 1, 0]
    # Julia [:, 1, 2, 1] = (all E, L=1, final=νe=1, initial=νμ=2)
    if ndims(p_probs) >= 4
        p_col = p_probs[:, 1, 2, 1]
    else
        p_col = vec(p_probs)
    end

    p_smeared = smearnorm(p_energy, collect(p_col), 1.0, 5, e_scale_nue, e_bias_nue)

    # Rebin probabilities to match detector binning
    num_edges_tot = length(squeezed_edges(numu_tot.energy)) - 1
    binned_means_tot = rebin_energy_spectrum(collect(p_smeared), energy_edges_calc,
                                               e_min=0.5, e_max=4.5, num_bins=num_edges_tot)
    numu_tot.p_nue = binned_means_tot / sum(binned_means_tot)
    numu_tot.signal = (numu_tot.NoOscillations_Signal) .* numu_tot.p_nue

    # Rebin signal to match electron neutrino analysis binning
    nue.p_numu = rebin_energy_spectrum(collect(numu_tot.signal), squeezed_edges(numu_tot.energy),
                                         e_min=0.5, e_max=4.5, num_bins=8)

    # Repeat for antineutrinos
    p_bar_probs = osc_prob(p_energy .* e_scale_nue .+ e_bias_nue, L, params, true, density)

    if ndims(p_bar_probs) >= 4
        p_bar_col = p_bar_probs[:, 1, 2, 1]
    else
        p_bar_col = vec(p_bar_probs)
    end

    p_bar_smeared = smearnorm(p_energy, collect(p_bar_col), 1.0, 5, e_scale_nue, e_bias_nue)

    num_edges_tot = length(squeezed_edges(numubar_tot.energy)) - 1
    binned_means_tot = rebin_energy_spectrum(collect(p_bar_smeared), energy_edges_calc,
                                               e_min=0.5, e_max=4.5, num_bins=num_edges_tot)
    numubar_tot.p_nue = binned_means_tot / sum(binned_means_tot)
    numubar_tot.signal = (numubar_tot.NoOscillations_Signal) .* numubar_tot.p_nue

    nuebar.p_numu = rebin_energy_spectrum(collect(numubar_tot.signal), squeezed_edges(numubar_tot.energy),
                                           e_min=0.5, e_max=4.5, num_bins=8)

    # Process each segment for neutrino data
    for segment in 1:3
        backgrounds = Vector{Float64}[]
        for bkg_type in ["Wrong_sign_bkg", "Beam_nue_bkg", "Cosmic_bkg"]
            bkg_name = Symbol(bkg_type * string(segment))
            bkg = getproperty(nue, bkg_name)
            backgrounds_vec = bkg isa AbstractArray ? collect(Float64, bkg) : [Float64(bkg)]
            push!(backgrounds, backgrounds_vec)
        end

        # Special handling for segment 3 (condense all values to bin 3)
        condense_mode = (segment == 3)

        signal_vec = getproperty(nue, :p_numu)
        signal_float64 = signal_vec isa AbstractArray ? collect(Float64, signal_vec) : [Float64(signal_vec)]

        prediction = fast_predictions_new(
            signal_float64,
            backgrounds,
            Float64(params["nova_norm"]),
            condense_to_bin3=condense_mode
        )

        setproperty!(nue, Symbol("my_predicted" * string(segment)), prediction)
    end

    # Process each segment for antineutrino data
    for segment in 1:3
        backgrounds = Vector{Float64}[]
        for bkg_type in ["Wrong_sign_bkg", "Beam_nue_bkg", "Cosmic_bkg"]
            bkg_name = Symbol(bkg_type * string(segment))
            bkg = getproperty(nuebar, bkg_name)
            backgrounds_vec = bkg isa AbstractArray ? collect(Float64, bkg) : [Float64(bkg)]
            push!(backgrounds, backgrounds_vec)
        end

        condense_mode = (segment == 3)

        signal_vec = getproperty(nuebar, :p_numu)
        signal_float64 = signal_vec isa AbstractArray ? collect(Float64, signal_vec) : [Float64(signal_vec)]

        prediction = fast_predictions_new(
            signal_float64,
            backgrounds,
            Float64(params["nova_norm"]),
            condense_to_bin3=condense_mode
        )

        setproperty!(nuebar, Symbol("my_predicted" * string(segment)), prediction)
    end
end

"""
Extract expected and observed event counts and bin information for electron neutrino analysis.
"""
function get_expected_observed_bins_nue_tot(params::Dict{String, Float64}, osc_prob::Function)
    make_predictions_nue_new(params, osc_prob, nue, nuebar)

    expected = Vector{Float64}[]
    observed = Vector{Float64}[]
    bins = Vector{Int}[]

    # Process neutrino data
    nue_exp = Vector{Float64}[]
    nue_obs = Vector{Float64}[]

    for segment in 1:3
        pred = getproperty(nue, Symbol("my_predicted" * string(segment)))
        obs = getproperty(nue, Symbol("observed" * string(segment)))

        pred_vec = pred isa AbstractArray ? collect(pred) : [pred]
        obs_vec = obs isa AbstractArray ? collect(obs) : [obs]

        push!(nue_exp, pred_vec .* sum(obs_vec))
        push!(nue_obs, obs_vec)
    end

    push!(expected, vcat(nue_exp...))
    push!(observed, vcat(nue_obs...))
    push!(bins, collect(1:length(vcat(nue_exp...)) + 1))

    # Process antineutrino data
    nuebar_exp = Vector{Float64}[]
    nuebar_obs = Vector{Float64}[]

    for segment in 1:3
        pred = getproperty(nuebar, Symbol("my_predicted" * string(segment)))
        obs = getproperty(nuebar, Symbol("observed" * string(segment)))

        pred_vec = pred isa AbstractArray ? collect(pred) : [pred]
        obs_vec = obs isa AbstractArray ? collect(obs) : [obs]

        push!(nuebar_exp, pred_vec .* sum(obs_vec))
        push!(nuebar_obs, obs_vec)
    end

    push!(expected, vcat(nuebar_exp...))
    push!(observed, vcat(nuebar_obs...))
    push!(bins, collect(1:length(vcat(nuebar_exp...)) + 1))

    return expected, expected, observed, bins
end

"""
Calculate negative log-likelihood for electron neutrino appearance data.
"""
function nllh_nue(params::Dict{String, Float64}, osc_prob::Function)
    make_predictions_nue_new(params, osc_prob, nue, nuebar)

    # Add small constant to prevent log(0)
    epsilon = 1e-10

    nue_llh = 0.0
    nuebar_llh = 0.0

    for i in 1:3
        pred = getproperty(nue, Symbol("my_predicted" * string(i)))
        obs = getproperty(nue, Symbol("observed" * string(i)))

        pred_vec = pred isa AbstractArray ? collect(pred) : [pred]
        obs_vec = obs isa AbstractArray ? collect(obs) : [obs]

        # Skip bins with zero prediction and zero observation
        mask = .!((pred_vec .== 0) .& (obs_vec .== 0))
        pred_nue = pred_vec[mask] .* sum(obs_vec) .+ epsilon
        obs_nue = obs_vec[mask]
        nue_llh += sum(pred_nue - obs_nue .* log.(pred_nue))

        pred = getproperty(nuebar, Symbol("my_predicted" * string(i)))
        obs = getproperty(nuebar, Symbol("observed" * string(i)))

        pred_vec = pred isa AbstractArray ? collect(pred) : [pred]
        obs_vec = obs isa AbstractArray ? collect(obs) : [obs]

        mask = .!((pred_vec .== 0) .& (obs_vec .== 0))
        pred_nuebar = pred_vec[mask] .* sum(obs_vec) .+ epsilon
        obs_nuebar = obs_vec[mask]
        nuebar_llh += sum(pred_nuebar - obs_nuebar .* log.(pred_nuebar))
    end

    return nue_llh + nuebar_llh
end

# ===================================================
# Combined NLL
# ===================================================

"""
Calculate combined negative log-likelihood for all data channels.
"""
function nllh_total_new(params::Dict{String, Float64}, osc_prob::Function)
    numu_llh, numubar_llh = nllh(params, osc_prob)
    nue_llh = nllh_nue(params, osc_prob)

    total_llh = numu_llh + numubar_llh + nue_llh
    return total_llh
end

# ===================================================
# Chi-square function
# ===================================================

"""
Calculate chi-square statistic and p-value for goodness-of-fit.
Uses Poisson error for uncertainty.
"""
function chisquare(expected::Vector{Vector{Float64}}, observed::Vector{Vector{Float64}},
                  bins::Vector{Vector{Float64}})
    total_chisq = 0.0
    dof = 0

    # Loop through all datasets
    for (exp, obs) in zip(expected, observed)
        # Ensure the arrays are Float64
        exp_arr = Float64.(exp)
        obs_arr = Float64.(obs)

        # Use Poisson error for uncertainty
        uncertainty = ifelse.(obs_arr .> 0, sqrt.(obs_arr), 1.0)

        # Calculate chi-square for this dataset
        chi_sq = sum(((obs_arr - exp_arr) ./ uncertainty) .^ 2)

        total_chisq += chi_sq
        dof += length(exp_arr)
    end

    # Calculate degrees of freedom
    # Subtracting parameters used in the fit (4 + 8 = 12 parameters)
    dof = dof - 4 - 8

    # Calculate p-value using chi-square distribution
    # Handle case where dof <= 0
    if dof <= 0
        dof = max(1, dof)
    end
    p_value = 1.0 - cdf(Chisq(dof), total_chisq)

    return total_chisq, p_value
end

# ===================================================
# Exports
# ===================================================

export smearnorm, smear_philipp, make_predictions, get_expected_observed_bins, nllh

export fast_predictions_new, rebin_energy_spectrum, calculate_energy_edges
export digitize_and_bin

export make_predictions_nue_new, get_expected_observed_bins_nue_tot, nllh_nue, nllh_total_new

export chisquare, GridData, Edges

export init_data, close_data

# ===================================================
# Atexit hook to close data files
# ===================================================

atexit() do
    close_data()
end
end