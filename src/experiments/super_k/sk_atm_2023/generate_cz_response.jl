#!/usr/bin/env julia
# Fit vMF cosZ response parameters for all bins and channels
# using unoscillated MC. Stores (mu_z, kappa) per bin per flavor
# so the R matrix can be reconstructed without repeating the expensive fit.

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

# Fit vMF parameters for each bin and flavor
result = Dict{Symbol, Dict{String, Vector{Float64}}}()

for flav in flavors
    println("Fitting $flav...")
    mc_comp = mc[flav]

    mu_z_vec = zeros(n_bins)
    kappa_vec = zeros(n_bins)

    t0 = time()
    for bin_idx in 1:n_bins
        bin = mc_comp[bin_idx, :]
        if bin.Counts == 0
            continue
        end

        cz_q = [bin.CosZQuantile2_3Percent, bin.CosZQuantile15_9Percent,
                bin.CosZQuantile50_0Percent, bin.CosZQuantile84_1Percent,
                bin.CosZQuantile97_7Percent]

        mu_z, kappa = Newtrinos.super_k.fit_vmf_cosz(cz_q, QP)
        mu_z_vec[bin_idx] = mu_z
        kappa_vec[bin_idx] = kappa
    end
    dt = time() - t0

    result[flav] = Dict("mu_z" => mu_z_vec, "kappa" => kappa_vec)
    println(@sprintf "  Done in %.1f s" dt)
end

# Save
outpath = joinpath(DATADIR, "vmf_cosz_params.jld2")
jldsave(outpath; vmf_params=result, quantile_probs=QP, mc_source="unoscillated")
println("\nSaved to $outpath")

# Print summary statistics
println("\nSummary:")
for flav in flavors
    mu = result[flav]["mu_z"]
    kap = result[flav]["kappa"]
    active = kap .> 0
    n_active = count(active)
    println(@sprintf "  %8s: %d active bins, kappa range [%.1f, %.1f], median kappa=%.1f" flav n_active minimum(kap[active]) maximum(kap[active]) sort(kap[active])[n_active÷2])
end
