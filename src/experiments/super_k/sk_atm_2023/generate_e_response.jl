#!/usr/bin/env julia
# Fit energy response distribution parameters for all bins and channels
# using unoscillated MC. Stores fit parameters for each method so the
# R matrix can be reconstructed without repeating the expensive fitting.
#
# Methods fitted:
#   - DSCB (logE): Double-sided Crystal Ball in log10(E) space (6 params)
#   - Novosibirsk (logE): Novosibirsk function in log10(E) space (3 params)
#   - Novosibirsk (linE): Novosibirsk function in linear E space (3 params)

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

# ─── DSCB fits ───
println("\n--- Fitting DSCB (logE) ---")
dscb_result = Dict{Symbol, Dict{String, Vector{Float64}}}()

for flav in flavors
    mc_comp = mc[flav]
    mu_vec = zeros(n_bins)
    sigma_vec = zeros(n_bins)
    alphaL_vec = zeros(n_bins)
    nL_vec = zeros(n_bins)
    alphaR_vec = zeros(n_bins)
    nR_vec = zeros(n_bins)

    t0 = time()
    for bin_idx in 1:n_bins
        bin = mc_comp[bin_idx, :]
        bin.Counts == 0 && continue
        e_q = get_energy_quantiles(bin)
        mu, sigma, aL, nL, aR, nR = Newtrinos.super_k.fit_dscb(e_q, QP)
        mu_vec[bin_idx] = mu
        sigma_vec[bin_idx] = sigma
        alphaL_vec[bin_idx] = aL
        nL_vec[bin_idx] = nL
        alphaR_vec[bin_idx] = aR
        nR_vec[bin_idx] = nR
    end
    dt = time() - t0

    dscb_result[flav] = Dict(
        "mu" => mu_vec, "sigma" => sigma_vec,
        "alphaL" => alphaL_vec, "nL" => nL_vec,
        "alphaR" => alphaR_vec, "nR" => nR_vec
    )
    println(@sprintf "  %8s: %.1f s" flav dt)
end

# ─── Novosibirsk (logE) fits ───
println("\n--- Fitting Novosibirsk (logE) ---")
novo_logE_result = Dict{Symbol, Dict{String, Vector{Float64}}}()

for flav in flavors
    mc_comp = mc[flav]
    x0_vec = zeros(n_bins)
    sigma_vec = zeros(n_bins)
    tau_vec = zeros(n_bins)

    t0 = time()
    for bin_idx in 1:n_bins
        bin = mc_comp[bin_idx, :]
        bin.Counts == 0 && continue
        e_q = get_energy_quantiles(bin)
        x0, sigma, tau = Newtrinos.super_k.fit_novosibirsk(e_q, QP)
        x0_vec[bin_idx] = x0
        sigma_vec[bin_idx] = sigma
        tau_vec[bin_idx] = tau
    end
    dt = time() - t0

    novo_logE_result[flav] = Dict("x0" => x0_vec, "sigma" => sigma_vec, "tau" => tau_vec)
    println(@sprintf "  %8s: %.1f s" flav dt)
end

# ─── Novosibirsk (linE) fits ───
println("\n--- Fitting Novosibirsk (linE) ---")
novo_linE_result = Dict{Symbol, Dict{String, Vector{Float64}}}()

for flav in flavors
    mc_comp = mc[flav]
    x0_vec = zeros(n_bins)
    sigma_vec = zeros(n_bins)
    tau_vec = zeros(n_bins)

    t0 = time()
    for bin_idx in 1:n_bins
        bin = mc_comp[bin_idx, :]
        bin.Counts == 0 && continue
        e_q = get_energy_quantiles(bin)

        # Fit in linear E space
        x0_init = e_q[3]
        sigma_init = max((e_q[4] - e_q[2]) / 2.0, 1e-6)

        best = (x0_init, sigma_init, 0.0)
        best_err = Inf
        e_max = maximum(e_q) * 3

        for tau in range(-2.0, 2.0, length=41)
            for sigma_f in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
                for x0_f in [-0.1, -0.05, 0.0, 0.05, 0.1]
                    x0 = x0_init * (1 + x0_f)
                    sigma = sigma_init * sigma_f
                    xt, ct = Newtrinos.super_k.novosibirsk_cdf_table(x0, sigma, tau; x_range=(0.0, e_max), n_pts=500)
                    err = sum((Newtrinos.super_k.novosibirsk_cdf_at.(Ref(xt), Ref(ct), e_q) .- QP).^2)
                    if err < best_err
                        best_err = err
                        best = (x0, sigma, tau)
                    end
                end
            end
        end

        x0_0, sig_0, tau_0 = best
        for tau in range(tau_0 - 0.2, tau_0 + 0.2, length=15)
            for sigma in range(max(1e-6, sig_0 * 0.9), sig_0 * 1.1, length=10)
                for x0 in range(x0_0 * 0.97, x0_0 * 1.03, length=10)
                    xt, ct = Newtrinos.super_k.novosibirsk_cdf_table(x0, sigma, tau; x_range=(0.0, e_max), n_pts=500)
                    err = sum((Newtrinos.super_k.novosibirsk_cdf_at.(Ref(xt), Ref(ct), e_q) .- QP).^2)
                    if err < best_err
                        best_err = err
                        best = (x0, sigma, tau)
                    end
                end
            end
        end

        x0_vec[bin_idx], sigma_vec[bin_idx], tau_vec[bin_idx] = best
    end
    dt = time() - t0

    novo_linE_result[flav] = Dict("x0" => x0_vec, "sigma" => sigma_vec, "tau" => tau_vec)
    println(@sprintf "  %8s: %.1f s" flav dt)
end

# ─── Save ───
outpath = joinpath(DATADIR, "energy_response_params.jld2")
jldsave(outpath;
    dscb_logE = dscb_result,
    novosibirsk_logE = novo_logE_result,
    novosibirsk_linE = novo_linE_result,
    quantile_probs = QP,
    mc_source = "unoscillated"
)
println(@sprintf "\nSaved to %s" outpath)

# ─── Summary ───
println("\nSummary:")
for flav in flavors
    d = dscb_result[flav]
    active = d["sigma"] .> 0
    n = count(active)
    println(@sprintf "  %8s: %d active bins" flav n)
end
