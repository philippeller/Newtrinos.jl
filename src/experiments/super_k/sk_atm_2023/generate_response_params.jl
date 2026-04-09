#!/usr/bin/env julia
# Fit DSCB energy + vMF cosZ response parameters for all bins and channels
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

# ─── DSCB energy fits ───
println("\n--- Fitting DSCB (logE) ---")
dscb_result = Dict{Symbol, Dict{String, Vector{Float64}}}()

for flav in flavors
    mc_comp = mc[flav]
    mu_vec = zeros(n_bins); sigma_vec = zeros(n_bins)
    alphaL_vec = zeros(n_bins); nL_vec = zeros(n_bins)
    alphaR_vec = zeros(n_bins); nR_vec = zeros(n_bins)

    t0 = time()
    for bin_idx in 1:n_bins
        bin = mc_comp[bin_idx, :]
        bin.Counts == 0 && continue
        mu, sigma, aL, nL, aR, nR = Newtrinos.super_k.fit_dscb(get_energy_quantiles(bin), QP)
        mu_vec[bin_idx] = mu; sigma_vec[bin_idx] = sigma
        alphaL_vec[bin_idx] = aL; nL_vec[bin_idx] = nL
        alphaR_vec[bin_idx] = aR; nR_vec[bin_idx] = nR
    end
    dt = time() - t0

    dscb_result[flav] = Dict(
        "mu" => mu_vec, "sigma" => sigma_vec,
        "alphaL" => alphaL_vec, "nL" => nL_vec,
        "alphaR" => alphaR_vec, "nR" => nR_vec
    )
    println(@sprintf "  %8s: %.1f s" flav dt)
end

# ─── vMF cosZ fits ───
println("\n--- Fitting vMF cosZ ---")
vmf_result = Dict{Symbol, Dict{String, Vector{Float64}}}()

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

    vmf_result[flav] = Dict("mu_z" => mu_z_vec, "kappa" => kappa_vec)
    println(@sprintf "  %8s: %.1f s" flav dt)
end

# ─── Save ───
e_path = joinpath(DATADIR, "energy_response_params.jld2")
jldsave(e_path; dscb_logE=dscb_result, quantile_probs=QP, mc_source="unoscillated")
println(@sprintf "\nSaved energy params to %s" e_path)

cz_path = joinpath(DATADIR, "vmf_cosz_params.jld2")
jldsave(cz_path; vmf_params=vmf_result, quantile_probs=QP, mc_source="unoscillated")
println(@sprintf "Saved cosZ params to %s" cz_path)

# ─── Summary ───
println("\nSummary:")
for flav in flavors
    d = dscb_result[flav]; v = vmf_result[flav]
    n_e = count(d["sigma"] .> 0); n_cz = count(v["kappa"] .> 0)
    println(@sprintf "  %8s: %d energy bins, %d cosZ bins" flav n_e n_cz)
end
