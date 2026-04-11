#!/usr/bin/env julia
# Fit E-dependent κ using per-class β values
# β is fixed per sample class (from NoOsc→NO roundtrip optimization)
# κ₀ is fitted per bin via 1D scan to match cosZ quantiles

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

using CSV, DataFrames
bininfo = CSV.read(joinpath(DATADIR, "bins/sk_2023_BinInfo.txt"), DataFrame; delim=' ', ignorerepeated=true, comment="#", header=false)
rename!(bininfo, [:Sample, :logPMin, :logPMax, :CosZMin, :CosZMax])

n_bins = 930
flavors = keys(mc)
loge_grid = collect(LinRange(-1, 3, 201))

# Per-class β values (from NoOsc→NO roundtrip optimization)
function get_class_beta(sample)
    occursin("upmu_thru_nonshowering", sample) && return 2.21
    occursin("upmu_thru_showering", sample) && return 3.19
    occursin("upmu_stop", sample) && return 1.02
    occursin("pc_thru", sample) && return 1.15
    occursin("pc_stop", sample) && return 0.35
    return 0.5  # default for all other samples
end

println("\n--- Fitting E-dependent κ with per-class β ---")
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

        beta = get_class_beta(bininfo.Sample[bin_idx])

        # Use fit_vmf_edep but with fixed beta — just scan κ₀
        # We override the beta search by passing it directly
        kappa0, _, E_ref = Newtrinos.super_k.fit_vmf_edep(
            cz_q, QP, mu_z_vec[bin_idx], kappa_vec[bin_idx],
            loge_grid, mix_result[flav], bin_idx)

        # Now rescan κ₀ at the fixed class beta
        μs = (mix_result[flav]["mu1"][bin_idx], mix_result[flav]["mu2"][bin_idx])
        σs = (mix_result[flav]["sigma1"][bin_idx], mix_result[flav]["sigma2"][bin_idx])
        ws = (mix_result[flav]["w1"][bin_idx], mix_result[flav]["w2"][bin_idx])
        E_grid = 10.0 .^ loge_grid
        c_e = [Newtrinos.super_k.lognormal_mix_cdf(e, μs, σs, ws) for e in E_grid]
        p_E = max.(diff(c_e), 0.0)
        s = sum(p_E); s > 0 && (p_E ./= s)
        cdf_e = cumsum(p_E)
        med_idx = clamp(searchsortedfirst(cdf_e, 0.5), 1, length(p_E))
        E_ref = 10.0^((loge_grid[med_idx] + loge_grid[med_idx+1]) / 2)

        active = findall(p_E .> 1e-4 * maximum(p_E))
        E_mids = [10.0^((loge_grid[i] + loge_grid[i+1]) / 2) for i in active]
        p_active = p_E[active]

        # Precompute CDF lookup
        kappa_init = kappa_vec[bin_idx]
        mu_z = mu_z_vec[bin_idx]
        n_kgrid = 200
        log_k_min = log(max(kappa_init * 0.01, 0.1))
        log_k_max = log(min(kappa_init * 100.0, 1e6))
        log_k_grid = range(log_k_min, log_k_max, length=n_kgrid)
        cdf_lookup = zeros(5, n_kgrid)
        for ki in 1:n_kgrid
            k = exp(log_k_grid[ki])
            ct, cd = Newtrinos.super_k._vmf_cdf_table(mu_z, k; n_pts=100)
            for qi in 1:5
                cdf_lookup[qi, ki] = Newtrinos.super_k._vmf_cdf_at(ct, cd, cz_q[qi])
            end
        end

        function interp_cdf(qi, log_k)
            log_k <= log_k_grid[1] && return cdf_lookup[qi, 1]
            log_k >= log_k_grid[end] && return cdf_lookup[qi, end]
            t = (log_k - log_k_grid[1]) / (log_k_grid[end] - log_k_grid[1]) * (n_kgrid - 1) + 1
            i = clamp(floor(Int, t), 1, n_kgrid - 1)
            f = t - i
            cdf_lookup[qi, i] * (1 - f) + cdf_lookup[qi, i+1] * f
        end

        # Scan κ₀ at fixed beta
        best_k = kappa_init; best_loss = Inf
        for log_k in range(log(max(kappa_init * 0.1, 0.1)), log(kappa_init * 10.0), length=30)
            kappa0 = exp(log_k)
            l = 0.0
            for qi in 1:5
                c = 0.0
                for j in eachindex(active)
                    log_k_eff = log(kappa0) + beta * log(E_mids[j] / E_ref)
                    c += p_active[j] * interp_cdf(qi, log_k_eff)
                end
                (isnan(c) || c <= 0 || c >= 1) && (l = 1e10; break)
                l += (c - QP[qi])^2
            end
            l < best_loss && (best_loss = l; best_k = kappa0)
        end

        kappa_vec[bin_idx] = best_k
        beta_vec[bin_idx] = beta
        E_ref_vec[bin_idx] = E_ref
    end
    dt = time() - t0

    vmf_result[flav] = Dict("mu_z" => mu_z_vec, "kappa" => kappa_vec,
                             "beta" => beta_vec, "E_ref" => E_ref_vec)
    @printf "  %8s: %.1f s\n" flav dt
end

cz_path = joinpath(DATADIR, "vmf_cosz_params.jld2")
jldsave(cz_path; vmf_params=vmf_result, quantile_probs=QP, mc_source="unoscillated")
@printf "\nSaved to %s\n" cz_path
