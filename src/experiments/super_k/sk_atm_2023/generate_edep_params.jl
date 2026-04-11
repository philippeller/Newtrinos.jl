#!/usr/bin/env julia
# Fit E-dependent β per bin using existing energy + vMF params
# β ∈ [0, 2], κ₀ scanned per bin at best β

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../../.."))

using Newtrinos
using JLD2
using Printf

const DATADIR = @__DIR__
const QP = [0.023, 0.159, 0.5, 0.841, 0.977]

println("Loading existing params...")
e_params = load(joinpath(DATADIR, "energy_response_params.jld2"))
mix_result = e_params["mix_logE"]
vmf_data = load(joinpath(DATADIR, "vmf_cosz_params.jld2"))
vmf_existing = vmf_data["vmf_params"]

mc = (
    nue     = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins/unoscillated/sk_2023_MCNueNoOsc.txt")),
    numu    = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins/unoscillated/sk_2023_MCNumuNoOsc.txt")),
    nutau   = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins/unoscillated/sk_2023_MCNutauNoOsc.txt")),
    nuebar  = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins/unoscillated/sk_2023_MCNueBarNoOsc.txt")),
    numubar = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins/unoscillated/sk_2023_MCNumuBarNoOsc.txt")),
    nunc    = Newtrinos.super_k.read_sk_file(joinpath(DATADIR, "bins/unoscillated/sk_2023_MCNCNoOsc.txt")),
)

n_bins = 930
flavors = keys(mc)
loge_grid = collect(LinRange(-1, 3, 201))

println("\n--- Fitting per-bin E-dependent β ∈ [0, 2] ---")
vmf_result = Dict{Symbol, Dict{String, Vector{Float64}}}()

for flav in flavors
    mu_z_vec = copy(vmf_existing[flav]["mu_z"])
    kappa_vec = copy(vmf_existing[flav]["kappa"])
    beta_vec = zeros(n_bins)
    E_ref_vec = ones(n_bins)

    t0 = time()
    for bin_idx in 1:n_bins
        bin = mc[flav][bin_idx, :]
        bin.Counts == 0 && continue
        kappa_vec[bin_idx] <= 0 && continue

        eq = [bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent,
              bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent,
              bin.EnergyQuantile97_7Percent]
        any(eq .<= 0) && continue
        bin.EnergyAvg <= 0 && continue
        bin.EnergyRMS <= 0 && continue

        cz_q = [bin.CosZQuantile2_3Percent, bin.CosZQuantile15_9Percent,
                bin.CosZQuantile50_0Percent, bin.CosZQuantile84_1Percent,
                bin.CosZQuantile97_7Percent]

        kappa0, beta, E_ref = Newtrinos.super_k.fit_vmf_edep(
            cz_q, QP, mu_z_vec[bin_idx], kappa_vec[bin_idx],
            loge_grid, mix_result[flav], bin_idx)

        kappa_vec[bin_idx] = kappa0
        beta_vec[bin_idx] = beta
        E_ref_vec[bin_idx] = E_ref
    end
    dt = time() - t0

    vmf_result[flav] = Dict("mu_z" => mu_z_vec, "kappa" => kappa_vec,
                             "beta" => beta_vec, "E_ref" => E_ref_vec)
    n_edep = count(beta_vec .> 0)
    @printf "  %8s: %.1f s  (%d bins with β>0)\n" flav dt n_edep
end

cz_path = joinpath(DATADIR, "vmf_cosz_params.jld2")
jldsave(cz_path; vmf_params=vmf_result, quantile_probs=QP, mc_source="unoscillated")
@printf "\nSaved to %s\n" cz_path
