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
