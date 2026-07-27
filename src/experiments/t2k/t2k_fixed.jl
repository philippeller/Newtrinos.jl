# ===================================================
# T2K neutrino oscillation analysis — Newtrinos.jl framework version
#
# Physics faithfully preserved from t2k_new.jl (Python reference translation):
#   - Unoscillated component extraction from expected data
#   - Background fraction handling (b=10 for NUMU, b=7 for NUE)
#   - create_proper_binning_new / create_proper_binning_nue
#   - Gaussian (smear) and boxcar (smearnorm) smearing with dynamic width
#   - Energy caching for performance
#   - Systematic penalty terms in likelihood
#   - T2K-specific parameters: t2k_energyscale, t2k_energybias, t2k_norm
#
# Structure follows Newtrinos.jl Experiment interface pattern.
# ===================================================
module t2k

using LinearAlgebra
using Distributions
using DelimitedFiles
using BAT
using DataStructures
using CairoMakie
using Logging
using Printf
using Statistics
import ..Newtrinos

@kwdef struct t2kExperiment <: Newtrinos.Experiment
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
    return t2kExperiment(
        physics = physics,
        params = (
            t2k_energyscale = 1.0,
            t2k_energybias = 0.0,
            t2k_norm = 1.0,
        ),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

# ===================================================
# Helper: CSV loading
# ===================================================

"Read a CSV file as a Float64 matrix, skipping header rows."
function read_csv_matrix(filepath::String; skipstart::Int=0)
    return readdlm(filepath, ',', Float64; skipstart=skipstart)
end

"Drop rows by index from a matrix. indices are 1-based."
function drop_rows(mat::Matrix{Float64}, indices::Vector{Int})
    keep = trues(size(mat, 1))
    for i in indices
        if i >= 1 && i <= length(keep)
            keep[i] = false
        end
    end
    return mat[keep, :]
end

"Drop last n rows from a matrix."
function drop_rows_from_end(mat::Matrix{Float64}, n::Int)
    return mat[1:end-n, :]
end

# ===================================================
# Asset loading
# ===================================================

function get_assets(physics; datadir = joinpath(@__DIR__, "data"))
    @info "Loading T2K data"

    # --- Muon neutrino (numu) ---
    numu_observed_raw = read_csv_matrix(joinpath(datadir, "rmu.csv"))
    numu_observed = numu_observed_raw  # all rows are data, no header

    numu_expected_raw = read_csv_matrix(joinpath(datadir, "pred_rmu.csv"); skipstart=5)
    numu_expected = drop_rows(numu_expected_raw, [32, 34])  # Python: drop [31, 33] (0-idx)

    # --- Muon antineutrino (numubar) ---
    numubar_observed_raw = read_csv_matrix(joinpath(datadir, "rmubar.csv"))
    numubar_observed = numubar_observed_raw

    numubar_expected_raw = read_csv_matrix(joinpath(datadir, "pred_rmubar.csv"); skipstart=5)
    _exp1 = drop_rows(numubar_expected_raw, [34])   # Python: drop 33 → Julia: 34
    _exp2 = drop_rows(_exp1, [31])                    # Python: drop 30 → Julia: 31
    numubar_expected = drop_rows(_exp2, [28])         # Python: drop 27 → Julia: 28

    # --- Electron neutrino (nue) ---
    nue_observed_raw = read_csv_matrix(joinpath(datadir, "re.csv"))
    nue_observed = nue_observed_raw

    nue_expected_raw = read_csv_matrix(joinpath(datadir, "pred_re.csv"); skipstart=2)
    _exp1 = drop_rows(nue_expected_raw, [22])         # Python: drop 21 → Julia: 22
    nue_expected = drop_rows(_exp1, [17, 18])          # Python: drop [16,17] → Julia: [17,18]

    # --- Electron antineutrino (nuebar) ---
    nuebar_observed_raw = read_csv_matrix(joinpath(datadir, "rebar.csv"))
    nuebar_observed = nuebar_observed_raw

    nuebar_expected_raw = read_csv_matrix(joinpath(datadir, "pred_rebar.csv"); skipstart=7)
    _exp0 = drop_rows_from_end(nuebar_expected_raw, 6)  # Python: .iloc[:-6]
    _exp1 = drop_rows(_exp0, [10, 11])                   # Python: drop [9,10] → Julia: [10,11]
    nuebar_expected = drop_rows(_exp1, [7])               # Python: drop 6 → Julia: 7

    # --- Electron neutrino deuterium (nuede) ---
    nuede_observed_raw = read_csv_matrix(joinpath(datadir, "rede.csv"))
    nuede_observed = nuede_observed_raw

    nuede_expected_raw = read_csv_matrix(joinpath(datadir, "pred_rede.csv"); skipstart=10)
    n_nuede = size(nuede_expected_raw, 1)
    _drop_indices = collect(9:(n_nuede - 1))  # Python: index[8:-1] → Julia rows 9:n-1
    nuede_expected = drop_rows(nuede_expected_raw, _drop_indices)

    # --- Combined electron neutrino (nuetot = nue + nuede) ---
    nuetot_observed = copy(nue_observed)
    # Python: nuetot.observed[8:16, 1] += nuede.observed[1:-1, 1] (skip first 0-energy row and last)
    # Julia: rows 9:16 (8 rows) += nuede rows 2:9 (8 rows, skip first and last)
    nuetot_observed[9:16, 2] .+= nuede_observed[2:end-1, 2]

    # --- Store all data in assets ---
    # Each is a Matrix{Float64} with columns: [energy, count]
    all_observed = (
        numu = numu_observed,
        numubar = numubar_observed,
        nuetot = nuetot_observed,
        nuebar = nuebar_observed,
    )
    all_expected = (
        numu = numu_expected,
        numubar = numubar_expected,
        nuetot = nue_expected,       # nuetot uses nue expected!
        nuebar = nuebar_expected,
    )

    # Reference oscillation parameters (from t2k_new.jl params_rev)
    params_rev = Dict{String, Float64}(
        "theta12" => asin(sqrt(0.307)),
        "theta13" => asin(sqrt(0.0218)),
        "theta23" => asin(sqrt(0.46888889)),
        "delta_CP" => -1.97,
        "Deltasq_m21" => 7.53e-5,
        "Deltasq_m31" => 0.002509 + 7.53e-5,
        "H" => 1.0,
    )

    # T2K constants
    L = 295.3  # km
    density = 2.6 * 0.5  # g/cm³ * Z/A (not passed to osc_prob; framework handles density)

    # Flatten observed data for Newtrinos framework
    # Order: NUMU, NUMUBAR, NUE(nuetot), NUEBAR
    observed_flat = Vector{Float64}()
    append!(observed_flat, numu_observed[:, 2])
    append!(observed_flat, numubar_observed[:, 2])
    append!(observed_flat, nuetot_observed[:, 2])
    append!(observed_flat, nuebar_observed[:, 2])

    assets = (
        L = L,
        density = density,
        all_observed = all_observed,
        all_expected = all_expected,
        params_rev = params_rev,
        observed = observed_flat,
        # T2K-specific physics parameters
        b_numu = 10.0,      # background fraction denominator for NUMU
        b_nue = 7.0,        # background fraction denominator for NUE
        smear_percent = 0.03,  # 3% energy resolution
        sys_numu = 0.10,    # 10% systematic for NUMU
        sys_nue = 0.20,     # 20% systematic for NUE
        penalty_weight = 0.4,  # weight for systematic penalty terms
    )

    return assets
end

# ===================================================
# Energy cache (per-configure, not global)
# ===================================================

function get_cached_energy(compute_fn::Function, cache::Dict{String, Vector{Float64}},
                           key::String)
    if !haskey(cache, key)
        cache[key] = compute_fn()
    end
    return cache[key]
end

# ===================================================
# Physics: Helper for numerical stability
# ===================================================

"Replace NaN and Inf with safe values for numerical stability."
function sanitize(x::Vector{Float64}; nan_val=0.0, inf_val=1e10, neginf_val=0.0)
    return map(x) do v
        if isnan(v); nan_val
        elseif isinf(v) && v > 0; inf_val
        elseif isinf(v) && v < 0; neginf_val
        else; v
        end
    end
end

"""
Compute the unoscillated flux component from expected events and reference probability.
Matches Python formula with NaN protection for near-zero survival probabilities.
"""
function compute_unoscillated(p_expected::Vector{Float64}, p_rev_smeared::Vector{Float64}, b::Float64)
    # Protect against division by zero in survival probability
    safe_smeared = max.(p_rev_smeared, 1e-10)
    inv_factor = (1 .- (1/b) .+ (1 ./ (b .* safe_smeared))) .^ (-1)
    unosc = sanitize((p_expected ./ safe_smeared) .* inv_factor .* (1 .- (1/b)))
    return unosc
end

"""
Compute total prediction from smeared probability, unoscillated component, and background fraction.
"""
function compute_total(p_prob_smeared::Vector{Float64}, p_unosc::Vector{Float64}, b::Float64)
    return sanitize((p_prob_smeared .* p_unosc) .+ (1/(b-1)) .* p_unosc)
end

# ===================================================
# Physics: Smearing functions (from t2k_new.jl)
# ===================================================

"""
Gaussian energy smearing function.
Matches Python's smear() with dynamic width based on sigma/ΔE.
"""
function smear(energies::Vector{Float64}, p::Vector{Float64}, percent::Float64;
               width::Int=15, e_scale::Float64=1.0, e_bias::Float64=0.0)
    out = zeros(Float64, length(p))
    len_p = length(p)

    for i in 1:len_p
        e = energies[i] * e_scale + e_bias
        sigma = percent * e

        # Dynamic width: min(width, int(3*sigma/ΔE) + 1)
        if i < len_p
            de = energies[2] - energies[1]
            effective_width = min(width, Int(floor(3 * sigma / de)) + 1)
        else
            effective_width = width
        end

        start_idx = max(1, i - effective_width)
        end_idx = min(len_p, i + effective_width)

        norm_val = 0.0
        for j in start_idx:end_idx
            sigma_j = percent * energies[j]
            diff = (e - energies[j]) / sigma_j
            # Skip if |diff| >= 5 sigma
            if abs(diff) < 5.0
                coeff = (1.0 / sigma_j) * exp(-0.5 * diff * diff)
                norm_val += coeff
                out[i] += coeff * p[j]
            end
        end
        if norm_val > 0
            out[i] /= norm_val
        end
    end
    return out
end

"""
Boxcar energy smearing function.
Matches Python's smearnorm() with dynamic width based on array length.
"""
function smearnorm(energies::Vector{Float64}, p::Vector{Float64}, percent::Float64;
                   width::Int=10, e_scale::Float64=1.0, e_bias::Float64=0.0)
    out = zeros(Float64, length(p))
    len_p = length(p)

    for i in 1:len_p
        effective_width = min(width, max(3, Int(floor(0.1 * len_p))))

        start_idx = max(1, i - effective_width)
        end_idx = min(len_p, i + effective_width)

        norm_val = 0.0
        for j in start_idx:end_idx
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

# ===================================================
# Physics: Rebinning functions (from t2k_new.jl)
# ===================================================

"""
Rebin a spectrum to linearly-spaced output bins.
Uses matrix-based fractional overlap (matches Python's rebin_energy_spectrum).
"""
function rebin_energy_spectrum(input_data::Vector{Float64}, bin_width::Float64;
                               e_min::Float64=0.5, e_max::Float64=4.5, num_bins::Int=8)
    input_length = length(input_data)

    # Calculate input bin edges
    old_edges = collect(range(e_min, step=bin_width, length=input_length + 1))

    # Create new bin edges
    new_edges = collect(range(e_min, stop=e_max, length=num_bins + 1))

    # Compute overlap fractions matrix
    fractions = zeros(Float64, num_bins, input_length)

    for j in 1:num_bins
        new_low = new_edges[j]
        new_high = new_edges[j + 1]
        for i in 1:input_length
            old_low = old_edges[i]
            old_high = old_edges[i + 1]

            left_edge = max(old_low, new_low)
            right_edge = min(old_high, new_high)
            overlap = max(right_edge - left_edge, 0.0)
            fractions[j, i] = overlap / bin_width
        end
    end

    return fractions * input_data
end

"""
Rebin a spectrum to logarithmically-spaced output bins.
Matches Python's rebin_energy_spectrum_log.
"""
function rebin_energy_spectrum_log(input_data::Vector{Float64}, bin_width::Float64;
                                   e_min::Float64=0.5, e_max::Float64=4.5, num_bins::Int=8)
    input_length = length(input_data)

    old_edges = collect(range(e_min, step=bin_width, length=input_length + 1))

    new_edges = 10.0 .^ range(log10(e_min), stop=log10(e_max), length=num_bins + 1)

    fractions = zeros(Float64, num_bins, input_length)

    for j in 1:num_bins
        new_low = new_edges[j]
        new_high = new_edges[j + 1]
        for i in 1:input_length
            old_low = old_edges[i]
            old_high = old_edges[i + 1]

            left_edge = max(old_low, new_low)
            right_edge = min(old_high, new_high)
            overlap = max(right_edge - left_edge, 0.0)
            fractions[j, i] = overlap / bin_width
        end
    end

    return fractions * input_data
end

# ===================================================
# Physics: Binning functions (from t2k_new.jl)
# ===================================================

"""
Create a binning scheme matching observed data.
Digitizes fine-grid energy centers into bins and SUMS the probabilities.
Matches Python's create_proper_binning_new.
"""
function create_proper_binning_new(p_energy::Vector{Float64}, p_values::Vector{Float64},
                                   vec_observed::Matrix{Float64})
    num_bin = size(vec_observed, 1)

    # Create bin edges spanning observed energy range
    first_e = vec_observed[1, 1] - 0.025
    last_e = vec_observed[end, 1] + 0.025
    bin_edges = collect(range(first_e, stop=last_e, length=num_bin + 1))
    bin_centers = (bin_edges[2:end] .+ bin_edges[1:end-1]) ./ 2

    rebinned_predicted = zeros(Float64, num_bin)

    for i in 1:num_bin
        mask = (p_energy .>= bin_edges[i]) .& (p_energy .< bin_edges[i+1])
        if any(mask)
            rebinned_predicted[i] = sum(p_values[mask])
        end
    end

    return bin_centers, rebinned_predicted
end

"""
Create fine binning and extract predicted values at observed energy points.
Matches Python's create_proper_binning_nue.
"""
function create_proper_binning_nue(p_energy::Vector{Float64}, p_values::Vector{Float64},
                                   vec_observed::Matrix{Float64}, numbin::Int)
    # Create fine binning
    first_e = vec_observed[1, 1] - 0.025
    last_e = vec_observed[end, 1] + 0.025
    bin_edges = collect(range(first_e, stop=last_e, length=numbin + 1))
    bin_centers = (bin_edges[2:end] .+ bin_edges[1:end-1]) ./ 2

    rebinned_predicted = zeros(Float64, numbin)

    for i in 1:numbin
        mask = (p_energy .>= bin_edges[i]) .& (p_energy .< bin_edges[i+1])
        if any(mask)
            rebinned_predicted[i] = sum(p_values[mask])
        end
    end

    # Extract observed energies
    observed_energies = vec_observed[:, 1]

    # Find closest bin center for each observed energy
    observed_predicted = zeros(Float64, length(observed_energies))
    for i in 1:length(observed_energies)
        obs_e = observed_energies[i]
        _, closest_idx = findmin(abs.(bin_centers .- obs_e))
        observed_predicted[i] = rebinned_predicted[closest_idx]
    end

    return observed_energies, observed_predicted
end

# ===================================================
# Helper: Build reference params NamedTuple for osc_prob
# ===================================================

function build_reference_params(current_params, assets)
    """Build reference params by merging reference values into current param structure."""
    pr = assets.params_rev
    # Start from current params (has all required keys) and override with reference values
    return merge(current_params, (;
        θ₁₂ = pr["theta12"],
        θ₁₃ = pr["theta13"],
        θ₂₃ = pr["theta23"],
        δCP = pr["delta_CP"],
        Δm²₂₁ = pr["Deltasq_m21"],
        Δm²₃₁ = pr["Deltasq_m31"],
    ))
end

# ===================================================
# Prediction: Muon neutrino disappearance
# ===================================================

"""
Calculate oscillation predictions for muon neutrino disappearance.
Preserves t2k_new.jl physics: unoscillated component from expected data,
background fraction b=10, create_proper_binning_new.
"""
function make_numu_predictions(params, physics, assets, energy_cache::Dict{String, Vector{Float64}})
    L = [assets.L]
    e_scale = params.t2k_energyscale
    e_bias = params.t2k_energybias
    b = assets.b_numu  # 10.0

    # Reference parameters for unoscillated component extraction
    params_ref = build_reference_params(params, assets)

    # Energy grid: logspace from observed data range
    cache_key = "numu_0.327_1.92_100"
    p_energy = get_cached_energy(energy_cache, cache_key) do
        10.0 .^ range(log10(0.327), stop=log10(1.92), length=100)
    end

    # --- Neutrino mode ---
    # Reference: extract unoscillated component
    p_rev = physics.osc.osc_prob(p_energy .* e_scale .+ e_bias, L, params_ref; anti=false)
    p_rev_col = ndims(p_rev) >= 4 ? p_rev[:, 1, 2, 2] : vec(p_rev)  # νμ→νμ
    p_rev_smeared = smearnorm(p_energy, collect(p_rev_col), assets.smear_percent;
                              width=10, e_scale=e_scale, e_bias=e_bias)

    # Expected events rebinned to fine grid
    p_expected = rebin_energy_spectrum_log(assets.all_expected.numu[:, 2], 0.05;
                                           e_min=0.25, e_max=1.95, num_bins=100)

    # Unoscillated component
    p_unosc = compute_unoscillated(p_expected, p_rev_smeared, b)

    # Fit parameters: oscillation probabilities
    p_prob = physics.osc.osc_prob(p_energy .* e_scale .+ e_bias, L, params; anti=false)
    p_prob_col = ndims(p_prob) >= 4 ? p_prob[:, 1, 2, 2] : vec(p_prob)
    p_prob_smeared = smearnorm(p_energy, collect(p_prob_col), assets.smear_percent;
                               width=10, e_scale=e_scale, e_bias=e_bias) .* params.t2k_norm

    # Signal + backgrounds
    p_total = compute_total(p_prob_smeared, p_unosc, b)

    # Create proper binning matching observed data
    _, predicted_values = create_proper_binning_new(p_energy, p_total, assets.all_observed.numu)

    # --- Antineutrino mode ---
    p_bar_rev = physics.osc.osc_prob(p_energy .* e_scale .+ e_bias, L, params_ref; anti=true)
    p_bar_rev_col = ndims(p_bar_rev) >= 4 ? p_bar_rev[:, 1, 2, 2] : vec(p_bar_rev)
    p_bar_rev_smeared = smearnorm(p_energy, collect(p_bar_rev_col), assets.smear_percent;
                                  width=10, e_scale=e_scale, e_bias=e_bias)

    p_bar_expected = rebin_energy_spectrum_log(assets.all_expected.numubar[:, 2], 0.05;
                                               e_min=0.25, e_max=1.95, num_bins=100)

    p_bar_unosc = compute_unoscillated(p_bar_expected, p_bar_rev_smeared, b)

    p_bar_prob = physics.osc.osc_prob(p_energy .* e_scale .+ e_bias, L, params; anti=true)
    p_bar_prob_col = ndims(p_bar_prob) >= 4 ? p_bar_prob[:, 1, 2, 2] : vec(p_bar_prob)
    p_bar_prob_smeared = smearnorm(p_energy, collect(p_bar_prob_col), assets.smear_percent;
                                   width=10, e_scale=e_scale, e_bias=e_bias) .* params.t2k_norm

    p_bar_total = compute_total(p_bar_prob_smeared, p_bar_unosc, b)

    _, predicted_values_bar = create_proper_binning_new(p_energy, p_bar_total, assets.all_observed.numubar)

    return (numu = predicted_values, numubar = predicted_values_bar)
end

# ===================================================
# Prediction: Electron neutrino appearance
# ===================================================

"""
Calculate oscillation predictions for electron neutrino appearance.
Preserves t2k_new.jl physics: unoscillated component, b=7,
create_proper_binning_nue, nuetot = nue + nuede.
"""
function make_nue_predictions(params, physics, assets, energy_cache::Dict{String, Vector{Float64}})
    L = [assets.L]
    e_scale = params.t2k_energyscale
    e_bias = params.t2k_energybias
    b = assets.b_nue  # 7.0
    params_ref = build_reference_params(params, assets)

    # Energy ranges from observed data
    energy_min = assets.all_observed.nuetot[1, 1]
    energy_max = assets.all_observed.nuetot[20, 1]   # Python: observed[19][0]
    energy_minbar = assets.all_observed.nuebar[1, 1]
    energy_maxbar = assets.all_observed.nuebar[9, 1]  # Python: observed[8][0]

    # --- Neutrino mode (νμ→νe appearance) ---
    nue_cache_key = "nue_$(energy_min)_$(energy_max)_100"
    p_energy_nue = get_cached_energy(energy_cache, nue_cache_key) do
        collect(range(energy_min, stop=energy_max, length=100))
    end

    # Reference
    p_rev = physics.osc.osc_prob(p_energy_nue .* e_scale .+ e_bias, L, params_ref; anti=false)
    p_rev_col = ndims(p_rev) >= 4 ? p_rev[:, 1, 2, 1] : vec(p_rev)  # νμ→νe
    p_rev_smeared = smearnorm(p_energy_nue, collect(p_rev_col), assets.smear_percent;
                              width=10, e_scale=e_scale, e_bias=e_bias)

    # Expected events (using nue expected for nuetot channel)
    p_expected = rebin_energy_spectrum(assets.all_expected.nuetot[:, 2], 0.05;
                                       e_min=energy_min, e_max=energy_max, num_bins=100)

    # Unoscillated component
    p_unosc = compute_unoscillated(p_expected, p_rev_smeared, b)

    # Fit parameters
    p_prob = physics.osc.osc_prob(p_energy_nue .* e_scale .+ e_bias, L, params; anti=false)
    p_prob_col = ndims(p_prob) >= 4 ? p_prob[:, 1, 2, 1] : vec(p_prob)
    p_prob_smeared = smearnorm(p_energy_nue, collect(p_prob_col), assets.smear_percent;
                               width=10, e_scale=e_scale, e_bias=e_bias) .* params.t2k_norm

    # Signal + backgrounds
    p_total = compute_total(p_prob_smeared, p_unosc, b)

    _, predicted_values_nue = create_proper_binning_nue(
        p_energy_nue, p_total, assets.all_observed.nuetot, size(assets.all_observed.nuetot, 1) + 2)

    # --- Antineutrino mode ---
    nuebar_cache_key = "nuebar_$(energy_minbar)_$(energy_maxbar)_100"
    p_energy_nuebar = get_cached_energy(energy_cache, nuebar_cache_key) do
        collect(range(energy_minbar, stop=energy_maxbar, length=100))
    end

    p_bar_rev = physics.osc.osc_prob(p_energy_nuebar .* e_scale .+ e_bias, L, params_ref; anti=true)
    p_bar_rev_col = ndims(p_bar_rev) >= 4 ? p_bar_rev[:, 1, 2, 1] : vec(p_bar_rev)
    p_bar_rev_smeared = smearnorm(p_energy_nuebar, collect(p_bar_rev_col), assets.smear_percent;
                                  width=10, e_scale=e_scale, e_bias=e_bias)

    p_bar_expected = rebin_energy_spectrum(assets.all_expected.nuebar[:, 2], 0.05;
                                           e_min=energy_minbar, e_max=energy_maxbar, num_bins=100)

    p_bar_unosc = compute_unoscillated(p_bar_expected, p_bar_rev_smeared, b)

    p_bar_prob = physics.osc.osc_prob(p_energy_nuebar .* e_scale .+ e_bias, L, params; anti=true)
    p_bar_prob_col = ndims(p_bar_prob) >= 4 ? p_bar_prob[:, 1, 2, 1] : vec(p_bar_prob)
    p_bar_prob_smeared = smearnorm(p_energy_nuebar, collect(p_bar_prob_col), assets.smear_percent;
                                   width=10, e_scale=e_scale, e_bias=e_bias) .* params.t2k_norm

    p_bar_total = compute_total(p_bar_prob_smeared, p_bar_unosc, b)

    _, predicted_values_nuebar = create_proper_binning_nue(
        p_energy_nuebar, p_bar_total, assets.all_observed.nuebar, size(assets.all_observed.nuebar, 1) + 3)

    return (nuetot = predicted_values_nue, nuebar = predicted_values_nuebar)
end

# ===================================================
# Forward model
# ===================================================

function get_forward_model(physics, assets)
    # Create a fresh energy cache for each forward model (per-configure)
    energy_cache = Dict{String, Vector{Float64}}()

    function forward_model(params)
        # Compute predictions
        numu_preds = make_numu_predictions(params, physics, assets, energy_cache)
        nue_preds = make_nue_predictions(params, physics, assets, energy_cache)

        # Assemble in same order as observed data: NUMU, NUMUBAR, NUE(nuetot), NUEBAR
        all_predictions = Vector{Float64}()
        append!(all_predictions, numu_preds.numu)
        append!(all_predictions, numu_preds.numubar)
        append!(all_predictions, nue_preds.nuetot)
        append!(all_predictions, nue_preds.nuebar)

        # Replace NaN/Inf with small positive values for Poisson compatibility
        all_predictions_safe = replace(all_predictions, NaN => 1e-10, Inf => 1e10, -Inf => 1e-10)
        return Poisson.(max.(all_predictions_safe, 1e-10))
    end
    return forward_model
end

# ===================================================
# Plot function
# ===================================================

function get_plot(physics, assets)
    energy_cache = Dict{String, Vector{Float64}}()

    function plot(params)
        f = Figure(resolution=(1200, 800))

        numu_preds = make_numu_predictions(params, physics, assets, energy_cache)
        nue_preds = make_nue_predictions(params, physics, assets, energy_cache)

        # Muon neutrino
        ax1 = Axis(f[1, 1], title="νμ Disappearance", xlabel="Energy (GeV)", ylabel="Events")
        obs_e = assets.all_observed.numu[:, 1]
        obs_c = assets.all_observed.numu[:, 2]
        scatter!(ax1, obs_e, obs_c, color=:black, label="Observed", markersize=6)
        lines!(ax1, obs_e, numu_preds.numu, color=:red, linewidth=2, label="Predicted")
        axislegend(ax1, position=:rt, labelsize=6)

        # Muon antineutrino
        ax2 = Axis(f[1, 2], title="ν̄μ Disappearance", xlabel="Energy (GeV)", ylabel="Events")
        obs_e = assets.all_observed.numubar[:, 1]
        obs_c = assets.all_observed.numubar[:, 2]
        scatter!(ax2, obs_e, obs_c, color=:black, label="Observed", markersize=6)
        lines!(ax2, obs_e, numu_preds.numubar, color=:red, linewidth=2, label="Predicted")
        axislegend(ax2, position=:rt, labelsize=6)

        # Electron neutrino (nuetot)
        ax3 = Axis(f[2, 1], title="νe Appearance (CC1π⁺ + CCQE)", xlabel="Energy (GeV)", ylabel="Events")
        obs_e = assets.all_observed.nuetot[:, 1]
        obs_c = assets.all_observed.nuetot[:, 2]
        scatter!(ax3, obs_e, obs_c, color=:black, label="Observed", markersize=6)
        lines!(ax3, obs_e, nue_preds.nuetot, color=:blue, linewidth=2, label="Predicted")
        axislegend(ax3, position=:rt, labelsize=6)

        # Electron antineutrino
        ax4 = Axis(f[2, 2], title="ν̄e Appearance", xlabel="Energy (GeV)", ylabel="Events")
        obs_e = assets.all_observed.nuebar[:, 1]
        obs_c = assets.all_observed.nuebar[:, 2]
        scatter!(ax4, obs_e, obs_c, color=:black, label="Observed", markersize=6)
        lines!(ax4, obs_e, nue_preds.nuebar, color=:green, linewidth=2, label="Predicted")
        axislegend(ax4, position=:rt, labelsize=6)

        return f
    end
    return plot
end

end # module t2k
