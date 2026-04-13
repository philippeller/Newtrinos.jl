#!/usr/bin/env julia
# Fit LogNormal mixture (K=2) energy + vMF cosZ response parameters for all bins
# using unoscillated MC. Saves precomputed parameters to JLD2 files so that
# super_k.jl can build the response matrix without repeating these fits.
#
# Run once (or whenever MC changes):
#   julia --project generate_response_params.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../../.."))

using Newtrinos
using JLD2
using Printf

const DATADIR = @__DIR__
const QP = [0.023, 0.159, 0.5, 0.841, 0.977]

# Load unoscillated MC
println("Loading unoscillated MC...")
mc = (
    nue     = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins", "unoscillated", "sk_2023_MCNueNoOsc.txt")),
    numu    = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins", "unoscillated", "sk_2023_MCNumuNoOsc.txt")),
    nutau   = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins", "unoscillated", "sk_2023_MCNutauNoOsc.txt")),
    nuebar  = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins", "unoscillated", "sk_2023_MCNueBarNoOsc.txt")),
    numubar = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins", "unoscillated", "sk_2023_MCNumuBarNoOsc.txt")),
    nunc    = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins", "unoscillated", "sk_2023_MCNCNoOsc.txt")),
)

n_bins = 930
flavors = keys(mc)

function get_energy_quantiles(bin)
    [bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent,
     bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent,
     bin.EnergyQuantile97_7Percent]
end

# ─── LogNormal mixture (K=2) energy fits ───
println("\n--- Fitting LogNormal mixture K=2 (logE) ---")
mix_result = Dict{Symbol, Dict{String, Vector{Float64}}}()

for flav in flavors
    mc_comp = mc[flav]
    mu1_vec = zeros(n_bins); mu2_vec = zeros(n_bins)
    sigma1_vec = zeros(n_bins); sigma2_vec = zeros(n_bins)
    w1_vec = zeros(n_bins); w2_vec = zeros(n_bins)

    t0 = time()
    for bin_idx in 1:n_bins
        bin = mc_comp[bin_idx, :]
        bin.Counts == 0 && continue
        eq = get_energy_quantiles(bin)
        any(eq .<= 0) && continue
        bin.EnergyAvg <= 0 && continue
        bin.EnergyRMS <= 0 && continue

        μs, σs, ws = Newtrinos.super_k.fit_energy_response(eq, bin.EnergyAvg, bin.EnergyRMS, QP)
        mu1_vec[bin_idx] = μs[1]; mu2_vec[bin_idx] = μs[2]
        sigma1_vec[bin_idx] = σs[1]; sigma2_vec[bin_idx] = σs[2]
        w1_vec[bin_idx] = ws[1]; w2_vec[bin_idx] = ws[2]
    end
    dt = time() - t0

    mix_result[flav] = Dict(
        "mu1" => mu1_vec, "mu2" => mu2_vec,
        "sigma1" => sigma1_vec, "sigma2" => sigma2_vec,
        "w1" => w1_vec, "w2" => w2_vec
    )
    println(@sprintf "  %8s: %.1f s" flav dt)
end

# ─── vMF cosZ fits (two stages) ───
# Stage 1: fit constant vMF (mu_z, kappa) per bin
println("\n--- Fitting vMF cosZ (stage 1: constant κ) ---")
vmf_const = Dict{Symbol, Dict{String, Vector{Float64}}}()

loge_grid = LinRange(-1, 3, 201)

for flav in flavors
    mc_comp = mc[flav]
    mu_z_vec = zeros(n_bins); kappa_vec = zeros(n_bins)

    t0 = time()
    for bin_idx in 1:n_bins
        bin = mc_comp[bin_idx, :]
        bin.Counts == 0 && continue

        cz_q = [bin.CosZQuantile2_3Percent, bin.CosZQuantile15_9Percent,
                bin.CosZQuantile50_0Percent, bin.CosZQuantile84_1Percent,
                bin.CosZQuantile97_7Percent]

        mu_z, kappa = Newtrinos.super_k.fit_vmf_cosz(cz_q, QP)
        mu_z_vec[bin_idx] = mu_z; kappa_vec[bin_idx] = kappa
    end
    dt = time() - t0

    vmf_const[flav] = Dict("mu_z" => mu_z_vec, "kappa" => kappa_vec)
    println(@sprintf "  %8s: %.1f s" flav dt)
end

# Stage 2: fit E-dependent κ(E) = κ₀ × (E/E_ref)^β per bin
println("\n--- Fitting vMF cosZ (stage 2: E-dependent κ) ---")
vmf_result = Dict{Symbol, Dict{String, Vector{Float64}}}()

for flav in flavors
    mc_comp = mc[flav]
    mu_z_vec = copy(vmf_const[flav]["mu_z"])
    kappa_vec = copy(vmf_const[flav]["kappa"])
    beta_vec = zeros(n_bins)
    E_ref_vec = ones(n_bins)

    t0 = time()
    for bin_idx in 1:n_bins
        bin = mc_comp[bin_idx, :]
        bin.Counts == 0 && continue
        kappa_vec[bin_idx] <= 0 && continue

        eq = get_energy_quantiles(bin)
        any(eq .<= 0) && continue
        bin.EnergyAvg <= 0 && continue
        bin.EnergyRMS <= 0 && continue

        cz_q = [bin.CosZQuantile2_3Percent, bin.CosZQuantile15_9Percent,
                bin.CosZQuantile50_0Percent, bin.CosZQuantile84_1Percent,
                bin.CosZQuantile97_7Percent]

        kappa0, beta, E_ref = Newtrinos.super_k.fit_vmf_edep(
            cz_q, QP, mu_z_vec[bin_idx], kappa_vec[bin_idx],
            collect(loge_grid), mix_result[flav], bin_idx)

        kappa_vec[bin_idx] = kappa0
        beta_vec[bin_idx] = beta
        E_ref_vec[bin_idx] = E_ref
    end
    dt = time() - t0

    vmf_result[flav] = Dict("mu_z" => mu_z_vec, "kappa" => kappa_vec,
                             "beta" => beta_vec, "E_ref" => E_ref_vec)
    println(@sprintf "  %8s: %.1f s" flav dt)
end

# ─── Regularization: smooth parameters across neighboring cosZ bins ───
# Within each (sample, momentum bin), the response should vary smoothly in cosZ.
# First canonicalize K=2 component ordering (μ₁ < μ₂), then smooth across cosZ neighbors.
using CSV, DataFrames, Statistics
bininfo = CSV.read(joinpath(DATADIR, "bins/sk_2023_BinInfo.txt"), DataFrame; delim=' ', ignorerepeated=true, comment="#", header=false)
rename!(bininfo, [:Sample, :logPMin, :logPMax, :CosZMin, :CosZMax])

# Step 1: Canonicalize K=2 component ordering so μ₁ < μ₂ in every bin
println("\n--- Canonicalizing K=2 component ordering (μ₁ < μ₂) ---")
for flav in flavors
    d = mix_result[flav]
    n_swapped = 0
    for i in 1:n_bins
        d["sigma1"][i] == 0 && continue
        if d["mu1"][i] > d["mu2"][i]
            # Swap components
            d["mu1"][i], d["mu2"][i] = d["mu2"][i], d["mu1"][i]
            d["sigma1"][i], d["sigma2"][i] = d["sigma2"][i], d["sigma1"][i]
            d["w1"][i], d["w2"][i] = d["w2"][i], d["w1"][i]
            n_swapped += 1
        end
    end
    println(@sprintf "  %8s: swapped %d bins" flav n_swapped)
end

# Step 2: Build cosZ neighbor groups — bins with same (sample, logPMin, logPMax)
# These are the cosZ slices within each momentum bin of each sample
cz_groups = Dict{Tuple{String,Float64,Float64}, Vector{Int}}()
for i in 1:n_bins
    key = (bininfo.Sample[i], bininfo.logPMin[i], bininfo.logPMax[i])
    haskey(cz_groups, key) || (cz_groups[key] = Int[])
    push!(cz_groups[key], i)
end

# Regularization strength: 0 = no smoothing, 1 = full average with neighbors
const REG_STRENGTH = 0.5

println("\n--- Regularizing across cosZ neighbors (strength=$REG_STRENGTH) ---")

function smooth_param_cz!(result, key, groups, mc_data)
    for (grp_key, idxs) in groups
        length(idxs) < 3 && continue
        original = [result[key][i] for i in idxs]
        for pos in 1:length(idxs)
            i = idxs[pos]
            mc_data[i, :].Counts == 0 && continue
            original[pos] == 0 && continue

            # Collect active cosZ neighbors (only prev/next in cosZ)
            neighbors = Float64[]
            for np in max(1, pos-1):min(length(idxs), pos+1)
                np == pos && continue
                ni = idxs[np]
                mc_data[ni, :].Counts == 0 && continue
                original[np] == 0 && continue
                push!(neighbors, original[np])
            end
            isempty(neighbors) && continue

            avg = mean(neighbors)
            result[key][i] = (1 - REG_STRENGTH) * original[pos] + REG_STRENGTH * avg
        end
    end
end

for flav in flavors
    mc_comp = mc[flav]
    # Smooth energy params (both components, now correctly ordered)
    for key in ["mu1", "mu2", "sigma1", "sigma2"]
        smooth_param_cz!(mix_result[flav], key, cz_groups, mc_comp)
    end
    # Smooth cosZ params
    for key in ["kappa", "beta"]
        smooth_param_cz!(vmf_result[flav], key, cz_groups, mc_comp)
    end
    println(@sprintf "  %8s: smoothed" flav)
end

# ─── Save ───
e_path = joinpath(DATADIR, "energy_response_params.jld2")
jldsave(e_path; mix_logE=mix_result, quantile_probs=QP, mc_source="unoscillated")
println(@sprintf "\nSaved energy params to %s" e_path)

cz_path = joinpath(DATADIR, "vmf_cosz_params.jld2")
jldsave(cz_path; vmf_params=vmf_result, quantile_probs=QP, mc_source="unoscillated")
println(@sprintf "Saved cosZ params to %s" cz_path)

# ─── Summary ───
println("\nSummary:")
for flav in flavors
    d = mix_result[flav]; v = vmf_result[flav]
    n_e = count(d["sigma1"] .> 0); n_cz = count(v["kappa"] .> 0)
    println(@sprintf "  %8s: %d energy bins, %d cosZ bins" flav n_e n_cz)
end
