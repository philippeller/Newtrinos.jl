module super_k

using CSV, DataFrames
using Interpolations
using CairoMakie
using Distributions
using DensityInterface
using BAT
using Accessors
using StatsBase
using Statistics: mean
using Printf
using SpecialFunctions: besseli
using JLD2
using Optim
using ..Newtrinos

@kwdef struct SuperKAtm <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics()
    # Spray propagation: sigma_E matched to logE grid for Nyquist (no aliasing),
    # sigma_h = 10 km atmospheric production height uncertainty
    dlogE = 4.0 / 200.0  # logE grid step
    sigma_E = 10.0^dlogE - 1.0  # ~ 0.047, fractional energy smearing
    propagation = Newtrinos.osc.Spray(averaging=:gaussian, σ_E=sigma_E, σ_h=10.0)
    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI(), propagation=propagation))
    atm_flux = Newtrinos.atm_flux.configure(Newtrinos.atm_flux.AtmFluxConfig(nominal_model=Newtrinos.atm_flux.HKKM("kam-ally-20-01-mtn-solmin.d")))
    earth_layers = Newtrinos.earth_layers.configure(Newtrinos.earth_layers.VariableDensity())
    xsec = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA())
    (; osc, atm_flux, earth_layers, xsec)
end

function configure(physics=default_physics())
    physics = (;physics.osc, physics.atm_flux, physics.earth_layers, physics.xsec)
    assets = get_assets(physics)
    return SuperKAtm(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function read_sk_file(filepath::String)
    df = CSV.read(filepath, DataFrame; delim=' ', ignorerepeated=true, comment="#", header=false)
    rename!(df, [
        :Counts, :EnergyAvg, :EnergyRMS, :EnergyQuantile2_3Percent, :EnergyQuantile15_9Percent,
        :EnergyQuantile50_0Percent, :EnergyQuantile84_1Percent, :EnergyQuantile97_7Percent,
        :CosZAvg, :CosZRMS, :CosZQuantile2_3Percent, :CosZQuantile15_9Percent,
        :CosZQuantile50_0Percent, :CosZQuantile84_1Percent, :CosZQuantile97_7Percent
    ])
    return df
end


# 2-component LogNormal mixture for energy response.
# Equivalent to a 2-Gaussian mixture in log₁₀(E) space.
# Fits all 7 data release statistics: 5 quantiles + mean + std (in linear E).
# Two overlapping components produce smooth, nearly-unimodal distributions with
# enough flexibility to capture asymmetry, while analytic moments keep fitting fast.
# Falls back to single LogNormal when K=2 produces spurious tails.

const _LN10 = log(10.0)

function lognormal_mix_cdf(E, μs, σs, ws)
    x = log10(E)
    ws[1] * cdf(Normal(μs[1], σs[1]), x) + ws[2] * cdf(Normal(μs[2], σs[2]), x)
end

function lognormal_mix_mean(μs, σs, ws)
    ws[1] * exp(μs[1] * _LN10 + 0.5 * σs[1]^2 * _LN10^2) +
    ws[2] * exp(μs[2] * _LN10 + 0.5 * σs[2]^2 * _LN10^2)
end

function lognormal_mix_std(μs, σs, ws)
    m1 = lognormal_mix_mean(μs, σs, ws)
    m2 = ws[1] * exp(2 * μs[1] * _LN10 + 2 * σs[1]^2 * _LN10^2) +
         ws[2] * exp(2 * μs[2] * _LN10 + 2 * σs[2]^2 * _LN10^2)
    sqrt(max(m2 - m1^2, 0.0))
end

function _mix_unpack(x, σ_min)
    μs = (x[1], x[2])
    σs = (σ_min + exp(x[3]), σ_min + exp(x[4]))
    w1 = 1.0 / (1.0 + exp(-x[5]))
    ws = (w1, 1.0 - w1)
    μs, σs, ws
end

function _mix_loss(x, eq, qp, mean_obs, std_obs, σ_min)
    μs, σs, ws = _mix_unpack(x, σ_min)
    loss = 0.0
    for i in 1:5
        c = lognormal_mix_cdf(eq[i], μs, σs, ws)
        (isnan(c) || c <= 0.0 || c >= 1.0) && return 1e10
        loss += (c - qp[i])^2
    end
    m = lognormal_mix_mean(μs, σs, ws)
    (isnan(m) || m <= 0.0 || isinf(m)) && return 1e10
    loss += 5.0 * ((m - mean_obs) / mean_obs)^2
    s = lognormal_mix_std(μs, σs, ws)
    (isnan(s) || s <= 0.0 || isinf(s)) && return 1e10
    loss += 5.0 * ((s - std_obs) / std_obs)^2
    loss
end

function _fit_lognormal_mixture(eq, mean_obs, std_obs, qp, σ_min; n_restarts=20)
    # Fit 2-component Gaussian mixture in log₁₀(E) to 5 quantiles + mean + std
    log_q = log10.(max.(eq, 1e-30))
    log_median = log_q[3]
    log_iqr = max((log_q[4] - log_q[2]) / 2.0, 0.03)
    log_range = max(log_q[5] - log_q[1], 0.05)
    log_mean = log10(max(mean_obs, 1e-30))
    to_raw_σ(σ) = log(max(σ - σ_min, 1e-4))

    best_x = nothing
    best_loss = Inf

    inits = [
        ([log_median, log_mean], [log_iqr, log_iqr*0.5], [1.0]),
        ([log_median-0.3*log_iqr, log_median+0.3*log_iqr], [log_iqr*0.8, log_iqr*0.8], [0.0]),
        ([log_q[2], log_q[4]], [log_iqr*0.5, log_iqr*0.5], [0.5]),
        ([log_median, log_q[1]+0.3*log_range], [log_iqr*1.2, log_iqr*0.3], [2.0]),
        ([log_median, log_q[5]-0.3*log_range], [log_iqr*1.0, log_iqr*0.5], [1.5]),
        ([log_median-0.05*log_iqr, log_median+0.05*log_iqr], [log_iqr, log_iqr], [0.0]),
        ([log_q[1]+0.2*log_range, log_q[5]-0.2*log_range], [log_iqr*0.6, log_iqr*0.6], [0.5]),
    ]

    for restart in 1:n_restarts
        if restart <= length(inits)
            μ_init, σ_init, w_init = inits[restart]
        else
            μ_init = [log_median + randn()*0.3*log_iqr, log_median + randn()*0.5*log_iqr]
            σ_init = [log_iqr*(0.3+0.7*rand()), log_iqr*(0.3+0.7*rand())]
            w_init = [randn()]
        end
        x0 = vcat(μ_init, to_raw_σ.(σ_init), w_init)

        try
            result = optimize(x -> _mix_loss(x, eq, qp, mean_obs, std_obs, σ_min), x0, NelderMead(),
                              Optim.Options(iterations=20000, f_reltol=1e-15))
            if Optim.minimum(result) < best_loss
                best_loss = Optim.minimum(result)
                best_x = Optim.minimizer(result)
            end
        catch
            continue
        end

        if best_x !== nothing && restart > 5 && restart % 3 == 0
            try
                result = optimize(x -> _mix_loss(x, eq, qp, mean_obs, std_obs, σ_min),
                                  best_x .+ randn(5) .* 0.02, NelderMead(),
                                  Optim.Options(iterations=20000, f_reltol=1e-15))
                if Optim.minimum(result) < best_loss
                    best_loss = Optim.minimum(result)
                    best_x = Optim.minimizer(result)
                end
            catch
            end
        end
    end

    if best_x === nothing
        return ([log_median, log_median], [log_iqr, log_iqr], [0.5, 0.5])
    end

    μs, σs, ws = _mix_unpack(best_x, σ_min)
    return (collect(μs), collect(σs), collect(ws))
end

function _fit_single_lognormal(eq, qp; n_restarts=10)
    # Fallback: single Gaussian in log₁₀(E), fits quantiles only (no moments).
    # Always produces a clean unimodal PDF with no spurious tails.
    log_q = log10.(max.(eq, 1e-30))
    log_median = log_q[3]
    log_iqr = max((log_q[4] - log_q[2]) / 2.0, 0.01)

    best_mu = log_median; best_sigma = log_iqr; best_loss = Inf
    for restart in 1:n_restarts
        mu0 = log_median + randn() * 0.05 * log_iqr
        sigma0_raw = log(log_iqr * (0.7 + 0.6 * rand()))
        try
            result = optimize(
                x -> begin
                    mu, sigma = x[1], exp(x[2])
                    sigma < 0.005 && return 1e10
                    loss = 0.0
                    for i in 1:5
                        c = cdf(Normal(mu, sigma), log_q[i])
                        (isnan(c) || c <= 0 || c >= 1) && return 1e10
                        loss += (c - qp[i])^2
                    end
                    loss
                end, [mu0, sigma0_raw], NelderMead(),
                Optim.Options(iterations=5000, f_reltol=1e-14))
            if Optim.minimum(result) < best_loss
                best_loss = Optim.minimum(result)
                best_mu = Optim.minimizer(result)[1]
                best_sigma = exp(Optim.minimizer(result)[2])
            end
        catch; end
    end
    return best_mu, best_sigma
end

function fit_energy_response(eq, mean_obs, std_obs, qp; n_restarts=20)
    # Try K=2 mixture first. If edge quantiles are badly off (spurious tails),
    # fall back to single LogNormal.
    log_q = log10.(max.(eq, 1e-30))
    log_range = max(log_q[5] - log_q[1], 0.05)
    σ_min = max(0.02, 0.05 * log_range)

    μs, σs, ws = _fit_lognormal_mixture(eq, mean_obs, std_obs, qp, σ_min; n_restarts)

    c_lo = lognormal_mix_cdf(eq[1], μs, σs, ws)
    c_hi = lognormal_mix_cdf(eq[5], μs, σs, ws)
    if abs(c_lo - qp[1]) > 0.03 || abs(c_hi - qp[5]) > 0.03
        mu, sigma = _fit_single_lognormal(eq, qp)
        return ([mu, mu], [sigma, sigma], [0.5, 0.5])
    end

    return (μs, σs, ws)
end


# Von Mises-Fisher marginal distribution in cosZ
# For vMF on S^2 with mean direction at zenith angle theta_0:
# f(cz) proportional to exp(kappa * mu_z * cz) * I_0(kappa * sqrt(1-mu_z^2) * sqrt(1-cz^2))
# where mu_z = cos(theta_0)

function _log_besseli0(x)
    x = abs(x)
    x < 500.0 ? log(besseli(0, x)) : x - 0.5 * log(2 * pi * x)
end

function _vmf_marginal_log_pdf(cz, mu_z, kappa)
    mu_perp = sqrt(max(0.0, 1.0 - mu_z^2))
    kappa * mu_z * cz + _log_besseli0(kappa * mu_perp * sqrt(max(0.0, 1.0 - cz^2)))
end

function _vmf_cdf_table(mu_z, kappa; n_pts=500)
    cz_pts = collect(range(-1.0, 1.0, length=n_pts))
    dcz = 2.0 / (n_pts - 1)
    lp = [_vmf_marginal_log_pdf(cz, mu_z, kappa) for cz in cz_pts]
    pdfs = exp.(lp .- maximum(lp))
    cumul = zeros(n_pts)
    for i in 2:n_pts
        cumul[i] = cumul[i-1] + 0.5 * (pdfs[i] + pdfs[i-1]) * dcz
    end
    cumul ./= cumul[end]
    cz_pts, cumul
end

function _vmf_cdf_at(ct, cd, x)
    x <= ct[1] && return 0.0
    x >= ct[end] && return 1.0
    i = searchsortedlast(ct, x)
    i = clamp(i, 1, length(ct) - 1)
    cd[i] + (x - ct[i]) / (ct[i+1] - ct[i]) * (cd[i+1] - cd[i])
end

function fit_vmf_cosz(cz_quantiles, qp)
    # Stage 1: mu_z from median, 1D search over kappa
    mu_z = clamp(cz_quantiles[3], -0.999, 0.999)

    sigma_cz = (cz_quantiles[4] - cz_quantiles[2]) / 2.0
    sigma_cz = max(sigma_cz, 0.001)

    best_kappa = 1.0 / sigma_cz^2
    best_mu = mu_z
    best_err = Inf

    for log_k in range(log(0.01), log(20000.0), length=50)
        k = exp(log_k)
        ct, cd = _vmf_cdf_table(mu_z, k)
        err = sum((_vmf_cdf_at.(Ref(ct), Ref(cd), cz_quantiles) .- qp).^2)
        if err < best_err
            best_err = err
            best_kappa = k
            best_mu = mu_z
        end
    end

    # Stage 2: 2D refinement around (mu_z, kappa)
    for _ in 1:2
        mu0, k0 = best_mu, best_kappa
        mu_halfwidth = min(0.1, 3.0 / sqrt(max(k0, 1.0)))
        for m in range(max(-0.999, mu0 - mu_halfwidth), min(0.999, mu0 + mu_halfwidth), length=11)
            for log_k in range(log(max(0.01, k0 * 0.7)), log(k0 * 1.4), length=11)
                k = exp(log_k)
                ct, cd = _vmf_cdf_table(m, k)
                err = sum((_vmf_cdf_at.(Ref(ct), Ref(cd), cz_quantiles) .- qp).^2)
                if err < best_err
                    best_err = err
                    best_kappa = k
                    best_mu = m
                end
            end
        end
    end

    return best_mu, best_kappa
end

function fit_vmf_edep(cz_quantiles, qp, mu_z, kappa_init, logE_grid, energy_params, bin_idx)
    # Fast fit: precompute vMF CDF on a κ grid, then binary search β + κ₀ scan
    # using interpolation instead of recomputing CDF tables.
    μs = (energy_params["mu1"][bin_idx], energy_params["mu2"][bin_idx])
    σs = (energy_params["sigma1"][bin_idx], energy_params["sigma2"][bin_idx])
    ws = (energy_params["w1"][bin_idx], energy_params["w2"][bin_idx])
    E_grid = 10 .^ logE_grid
    c_e = [lognormal_mix_cdf(e, μs, σs, ws) for e in E_grid]
    p_E = max.(diff(c_e), 0.0)
    s = sum(p_E); s > 0 && (p_E ./= s)

    cdf_e = cumsum(p_E)
    med_idx = clamp(searchsortedfirst(cdf_e, 0.5), 1, length(p_E))
    E_ref = 10.0^((logE_grid[med_idx] + logE_grid[med_idx+1]) / 2)

    active = findall(p_E .> 1e-4 * maximum(p_E))
    E_mids = [10.0^((logE_grid[i] + logE_grid[i+1]) / 2) for i in active]
    p_active = p_E[active]

    # Precompute vMF CDF at the 5 quantile points for a grid of log(κ) values.
    # κ range: kappa_init * 0.01 to kappa_init * 100 (covers β∈[0,2] with E ratios)
    n_kgrid = 200
    log_k_min = log(max(kappa_init * 0.01, 0.1))
    log_k_max = log(min(kappa_init * 100.0, 1e6))
    log_k_grid = range(log_k_min, log_k_max, length=n_kgrid)
    # cdf_lookup[qi, ki] = CDF of vMF(mu_z, κ_ki) at cz_quantiles[qi]
    cdf_lookup = zeros(5, n_kgrid)
    for ki in 1:n_kgrid
        k = exp(log_k_grid[ki])
        ct, cd = _vmf_cdf_table(mu_z, k; n_pts=100)
        for qi in 1:5
            cdf_lookup[qi, ki] = _vmf_cdf_at(ct, cd, cz_quantiles[qi])
        end
    end

    # Interpolate CDF from precomputed grid
    function interp_cdf(qi, log_k)
        log_k <= log_k_grid[1] && return cdf_lookup[qi, 1]
        log_k >= log_k_grid[end] && return cdf_lookup[qi, end]
        # Linear interpolation
        t = (log_k - log_k_grid[1]) / (log_k_grid[end] - log_k_grid[1]) * (n_kgrid - 1) + 1
        i = clamp(floor(Int, t), 1, n_kgrid - 1)
        f = t - i
        cdf_lookup[qi, i] * (1 - f) + cdf_lookup[qi, i+1] * f
    end

    function loss_for_params(kappa0, beta)
        l = 0.0
        for qi in 1:5
            c = 0.0
            for j in eachindex(active)
                log_k = log(kappa0) + beta * log(E_mids[j] / E_ref)
                c += p_active[j] * interp_cdf(qi, log_k)
            end
            (isnan(c) || c <= 0 || c >= 1) && return 1e10
            l += (c - qp[qi])^2
        end
        l
    end

    function best_kappa_for_beta(beta)
        best_l = Inf; best_k = kappa_init
        for log_k in range(log(max(kappa_init * 0.3, 1.0)), log(kappa_init * 3.0), length=15)
            l = loss_for_params(exp(log_k), beta)
            if l < best_l; best_l = l; best_k = exp(log_k); end
        end
        best_l, best_k
    end

    # Binary search for β in [0, 2]
    lo = 0.0; hi = 2.0
    for _ in 1:6
        m1 = (2lo + hi) / 3; m2 = (lo + 2hi) / 3
        l1, _ = best_kappa_for_beta(m1)
        l2, _ = best_kappa_for_beta(m2)
        l1 < l2 ? (hi = m2) : (lo = m1)
    end
    best_beta = (lo + hi) / 2
    best_loss, best_kappa = best_kappa_for_beta(best_beta)

    # Check β=0
    loss_0, kappa_0 = best_kappa_for_beta(0.0)
    if loss_0 <= best_loss
        return kappa_0, 0.0, E_ref
    end

    return best_kappa, best_beta, E_ref
end


function _build_R_from_params(MC_component, logE_grid, cosZ_grid, energy_params, vmf_params)
    # Build response matrix from precomputed energy + E-dependent vMF cosZ parameters
    n_bins = size(MC_component, 1)
    n_logE = length(logE_grid) - 1
    n_cosZ = length(cosZ_grid) - 1
    E_grid = 10 .^ logE_grid

    response_matrix = zeros(Float64, n_bins, n_logE, n_cosZ)

    for bin_idx in 1:n_bins
        counts = MC_component[bin_idx, :].Counts
        counts == 0 && continue

        # Energy: LogNormal mixture CDF (K=2)
        μs = (energy_params["mu1"][bin_idx], energy_params["mu2"][bin_idx])
        σs = (energy_params["sigma1"][bin_idx], energy_params["sigma2"][bin_idx])
        ws = (energy_params["w1"][bin_idx], energy_params["w2"][bin_idx])
        (σs[1] == 0 && σs[2] == 0) && continue
        c_e = [lognormal_mix_cdf(e, μs, σs, ws) for e in E_grid]
        p_e = diff(c_e)

        # CosZ: E-dependent vMF — κ(E) = κ₀ × (E/E_ref)^β
        kappa0 = vmf_params["kappa"][bin_idx]
        kappa0 <= 0 && continue
        mu_z = vmf_params["mu_z"][bin_idx]
        beta = vmf_params["beta"][bin_idx]
        E_ref = vmf_params["E_ref"][bin_idx]

        if beta == 0
            # Factorized: single vMF for all energies
            ct, cd = _vmf_cdf_table(mu_z, kappa0)
            c_cosz = [_vmf_cdf_at(ct, cd, x) for x in cosZ_grid]
            p_cosz = diff(c_cosz)
            response_matrix[bin_idx, :, :] .= p_e * p_cosz'
        else
            # E-dependent: different κ per energy bin
            for i in 1:n_logE
                p_e[i] < 1e-15 && continue
                E_mid = 10.0^((logE_grid[i] + logE_grid[i+1]) / 2)
                k = kappa0 * (E_mid / E_ref)^beta
                k = clamp(k, 0.1, 1e6)
                ct, cd = _vmf_cdf_table(mu_z, k; n_pts=200)
                c_cosz = [_vmf_cdf_at(ct, cd, x) for x in cosZ_grid]
                p_cosz = diff(c_cosz)
                response_matrix[bin_idx, i, :] .= p_e[i] .* p_cosz
            end
        end
    end
    return response_matrix
end


function contract_R(R_flat, weighted_flux)
    # R_flat is (n_bins, n_E*n_cz), weighted_flux is (n_E, n_cz)
    R_flat * vec(weighted_flux)
end

function make_gaussian_kernel_matrix(n, sigma_bins)
    # Precompute a banded Gaussian convolution matrix for 1D smoothing.
    # K[i,j] = normalized Gaussian weight from bin j contributing to bin i.
    # Truncate at 3σ for efficiency. Returns a dense Float64 matrix.
    K = zeros(n, n)
    hw = ceil(Int, 3 * sigma_bins)
    for i in 1:n
        s = 0.0
        for j in max(1, i-hw):min(n, i+hw)
            v = exp(-0.5 * ((i - j) / sigma_bins)^2)
            K[i, j] = v
            s += v
        end
        K[i, :] ./= s
    end
    return K
end

# Energy scale via reco bin overlap method
struct EnergyGroup
    indices::Vector{Int}
    logP_edges::Vector{Float64}
end

function build_energy_groups(bininfo, bin_mask)
    groups = EnergyGroup[]
    masked_indices = findall(bin_mask)
    sub_bininfo = bininfo[masked_indices, :]
    for key in unique(zip(sub_bininfo.Sample, sub_bininfo.CosZMin, sub_bininfo.CosZMax))
        local_mask = (sub_bininfo.Sample .== key[1]) .& (sub_bininfo.CosZMin .== key[2]) .& (sub_bininfo.CosZMax .== key[3])
        local_idxs = findall(local_mask)
        order = sortperm(sub_bininfo.logPMin[local_idxs])
        sorted_local = local_idxs[order]
        global_idxs = masked_indices[sorted_local]
        edges = vcat(sub_bininfo.logPMin[sorted_local], [sub_bininfo.logPMax[sorted_local[end]]])
        push!(groups, EnergyGroup(global_idxs, edges))
    end
    return groups
end

function apply_energy_scale(counts, energy_groups, delta)
    result = copy(counts)
    for group in energy_groups
        n = length(group.indices)
        edges = group.logP_edges
        for i in 1:n
            # Clamp edge bins to group boundaries for count conservation
            shifted_lo = (i == 1) ? edges[1] : edges[i] + delta
            shifted_hi = (i == n) ? edges[end] : edges[i+1] + delta
            acc = zero(eltype(counts))
            for j in 1:n
                overlap = max(zero(eltype(counts)), min(shifted_hi, edges[j+1]) - max(shifted_lo, edges[j]))
                nom_width = edges[j+1] - edges[j]
                acc += (overlap / nom_width) * counts[group.indices[j]]
            end
            result[group.indices[i]] = acc
        end
    end
    return result
end

function apply_all_energy_scales(counts, assets, params)
    # Absolute energy scale: shifts events between momentum bins
    delta_i_iii = log10(params.sk_i_iii_energy_scale)
    delta_iv_v = log10(params.sk_iv_v_energy_scale)
    result = apply_energy_scale(counts, assets.energy_groups_sk_i_iii, delta_i_iii)
    result = apply_energy_scale(result, assets.energy_groups_sk_iv_v, delta_iv_v)

    # Up/down energy scale: relative normalization of upgoing vs downgoing FC+PC events
    # (Wester thesis Sec 5.2.3: "varies the normalization of upward-going and
    # downward-going FC and PC events")
    delta_ud_i_iii = params.sk_i_iii_updown_energy_scale - one(params.sk_i_iii_updown_energy_scale)
    delta_ud_iv_v = params.sk_iv_v_updown_energy_scale - one(params.sk_iv_v_updown_energy_scale)
    result = result .* (one(delta_ud_i_iii) .+ delta_ud_i_iii .* assets.masks.sk_i_iii_updown .+ delta_ud_iv_v .* assets.masks.sk_iv_v_updown)
    return result
end

function flux_norm_sigma_low(logE)
    # 25% at logE=-1 (0.1 GeV), linear in logE to 7% at logE=0 (1 GeV), zero above
    logE < zero(logE) ? max(0.07, 0.25 - 0.18 * (logE + 1)) : zero(logE)
end

function flux_norm_sigma_high(logE)
    # Zero below 1 GeV; 7% flat from 1-10 GeV; linear in logE to 20% at 1 TeV
    logE < zero(logE) ? zero(logE) : (logE ≤ 1 ? oftype(logE, 0.07) : 0.07 + 0.065 * (logE - 1))
end

function calc_weights(params, assets, physics)

    E = 10. .^midpoints(assets.loge_grid)
    logE = midpoints(assets.loge_grid)

    layers = haskey(params, :electron_density_scale) ? Newtrinos.earth_layers.scale_densities(assets.nominal_layers, params.electron_density_scale) : assets.nominal_layers
    paths = physics.earth_layers.compute_paths(assets.cz_midpoints, layers)

    p = physics.osc.osc_prob(E, paths, layers, params)
    p_anti = physics.osc.osc_prob(E, paths, layers, params; anti=true)

    flux = physics.atm_flux.sys_flux(assets.flux_nominal, params)

    s = (size(p)[1], size(p)[2])

    # Energy-dependent flux normalization (bathtub shape, split at 1 GeV)
    fnl = haskey(params, :sk_flux_norm_low) ? params.sk_flux_norm_low : zero(eltype(E))
    fnh = haskey(params, :sk_flux_norm_high) ? params.sk_flux_norm_high : zero(eltype(E))
    flux_norm = 1 .+ fnl .* flux_norm_sigma_low.(logE) .+ fnh .* flux_norm_sigma_high.(logE)

    xsec_nue     = physics.xsec.dσdE(E, :nue,   :CC, false, params)
    xsec_numu    = physics.xsec.dσdE(E, :numu,  :CC, false, params)
    xsec_nutau   = physics.xsec.dσdE(E, :nutau, :CC, false, params)
    xsec_nuebar  = physics.xsec.dσdE(E, :nue,   :CC, true,  params)
    xsec_numubar = physics.xsec.dσdE(E, :numu,  :CC, true,  params)
    xsec_nutaubar= physics.xsec.dσdE(E, :nutau, :CC, true,  params)
    xsec_nc      = physics.xsec.dσdE(E, :nue,   :NC, false, params)

    # HKKM flux is differential: Φ(E) in (m² s sr GeV)⁻¹.
    # dσdE returns σ/E (cross-section per nucleon per GeV, divided by E).
    # On our logE grid: event rate ∝ Φ(E) × σ(E) × dE = Φ(E) × (σ/E × E) × (E × dlogE × ln10)
    # = Φ(E) × σ/E × E² × dlogE × ln10
    # So we need E² after reshape: one E from the Jacobian, one E from σ/E → σ.
    flux_nue    = reshape(flux.nue,    s) .* E .* E
    flux_numu   = reshape(flux.numu,   s) .* E .* E
    flux_nuebar = reshape(flux.nuebar, s) .* E .* E
    flux_numubar= reshape(flux.numubar,s) .* E .* E

    nue_flux   = (flux_nue .* p[:, :, 1, 1] .+
                  flux_numu .* p[:, :, 2, 1]) .* xsec_nue .* flux_norm
    numu_flux  = (flux_nue .* p[:, :, 1, 2] .+
                  flux_numu .* p[:, :, 2, 2]) .* xsec_numu .* flux_norm
    nutau_flux = (flux_nue .* p[:, :, 1, 3] .+
                  flux_numu .* p[:, :, 2, 3]) .* xsec_nutau .* flux_norm
    nuebar_flux  = (flux_nuebar .* p_anti[:, :, 1, 1] .+
                    flux_numubar .* p_anti[:, :, 2, 1]) .* xsec_nuebar .* flux_norm
    numubar_flux = (flux_nuebar .* p_anti[:, :, 1, 2] .+
                    flux_numubar .* p_anti[:, :, 2, 2]) .* xsec_numubar .* flux_norm
    nutaubar_flux = (flux_nuebar .* p_anti[:, :, 1, 3] .+
                     flux_numubar .* p_anti[:, :, 2, 3]) .* xsec_nutaubar .* flux_norm

    nue     = contract_R(assets.R.nue,     nue_flux)
    numu    = contract_R(assets.R.numu,    numu_flux)
    nuebar  = contract_R(assets.R.nuebar,  nuebar_flux)
    numubar = contract_R(assets.R.numubar, numubar_flux)

    # nutau: SK MC lumps nutau + nutaubar CC into one channel.
    # Use precomputed nu/nubar mixture fractions to combine with correct proportions.
    nutau_combined = nutau_flux .* assets.nutau_nu_frac .+ nutaubar_flux .* (1 .- assets.nutau_nu_frac)
    nutau   = contract_R(assets.R.nutau,   nutau_combined)

    # NC: SK MC lumps all NC (nu + nubar, all flavors) into one channel.
    # Use precomputed nu/nubar mixture fractions with proper cross-sections.
    xsec_nc_anti = physics.xsec.dσdE(E, :nue, :NC, true, params)
    flux_nu_total = flux_nue .+ flux_numu
    flux_nubar_total = flux_nuebar .+ flux_numubar
    nc_nu_flux = flux_nu_total .* xsec_nc
    nc_nubar_flux = flux_nubar_total .* xsec_nc_anti
    nc_combined = nc_nu_flux .* assets.nc_nu_frac .+ nc_nubar_flux .* (1 .- assets.nc_nu_frac)
    nunc    = contract_R(assets.R.nunc,    nc_combined .* flux_norm)

    return (; nue, numu, nutau, nuebar, numubar, nunc)
end

safe_div(a, b, ε=1e-10) = a / (b + ε)

function get_assets(physics; datadir = @__DIR__)
    @info "Loading Super-K Data"

    bininfo = CSV.read(joinpath(datadir, "bins/sk_2023_BinInfo.txt"), DataFrame; delim=' ', ignorerepeated=true, comment="#", header=false);
    rename!(bininfo, [:Sample, :logPMin, :logPMax, :CosZMin, :CosZMax]);
    bad_entries = findall(bininfo.CosZMin .> bininfo.CosZMax)

    bininfo[bad_entries[1], :].CosZMax = 0.0
    bininfo[bad_entries[2], :].CosZMax = 0.0
    bininfo[bad_entries[3], :].CosZMax = 1.0    

    masks = (
        fc = occursin.("_fc_", bininfo.Sample),
        pc = occursin.("_pc_", bininfo.Sample),
        upmu = occursin.("_upmu_", bininfo.Sample),
        pc_stop = occursin.("_pc_stop", bininfo.Sample),
        pc_thru = occursin.("_pc_thru", bininfo.Sample),
        # Directional PC stop/thru masks (top/barrel/bottom exit)
        # cosZ midpoint: bottom < -0.22, barrel in [-0.22, 0.22], top > 0.22
        pc_stop_bottom = occursin.("_pc_stop", bininfo.Sample) .& ((bininfo.CosZMin .+ bininfo.CosZMax) ./ 2 .< -0.22),
        pc_thru_bottom = occursin.("_pc_thru", bininfo.Sample) .& ((bininfo.CosZMin .+ bininfo.CosZMax) ./ 2 .< -0.22),
        pc_stop_barrel = occursin.("_pc_stop", bininfo.Sample) .& ((bininfo.CosZMin .+ bininfo.CosZMax) ./ 2 .>= -0.22) .& ((bininfo.CosZMin .+ bininfo.CosZMax) ./ 2 .<= 0.22),
        pc_thru_barrel = occursin.("_pc_thru", bininfo.Sample) .& ((bininfo.CosZMin .+ bininfo.CosZMax) ./ 2 .>= -0.22) .& ((bininfo.CosZMin .+ bininfo.CosZMax) ./ 2 .<= 0.22),
        pc_stop_top = occursin.("_pc_stop", bininfo.Sample) .& ((bininfo.CosZMin .+ bininfo.CosZMax) ./ 2 .> 0.22),
        pc_thru_top = occursin.("_pc_thru", bininfo.Sample) .& ((bininfo.CosZMin .+ bininfo.CosZMax) ./ 2 .> 0.22),
        umpmu_stop = occursin.("_upmu_stop", bininfo.Sample),
        upmu_thru = occursin.("_upmu_thru", bininfo.Sample),
        upmu_shower = occursin.(r"_upmu_.*_showering",  bininfo.Sample),
        upmu_nonshower = occursin.(r"_upmu_.*_nonshowering", bininfo.Sample),
        mu_indices = occursin.("_numu", bininfo.Sample),
        sk_i_iii_elike_0decay_e = occursin.(r"sk1-3_.*elike_0decaye", bininfo.Sample),
        sk_i_iii_elike_1decay_e = occursin.(r"sk1-3_.*elike_1decaye", bininfo.Sample),
        sk_i_iii_mulike_0decay_e = occursin.(r"sk1-3_.*mulike_0decaye", bininfo.Sample),
        sk_i_iii_mulike_1decay_e = occursin.(r"sk1-3_.*mulike_1decaye", bininfo.Sample),
        sk_i_iii_mulike_2decay_e = occursin.(r"sk1-3_.*mulike_2decaye", bininfo.Sample),
        sk_iv_v_0decay_e = occursin.(r"sk4-5_fc_.*_nuebarlike",  bininfo.Sample),
        sk_iv_v_1decay_e = occursin.(r"sk4-5_fc_.*_nuelike",  bininfo.Sample),
        sk_iv_v_subgev_0neutron = occursin.(r"sk4-5_fc_subgev.*(_0neutron|numulike)", bininfo.Sample),
        sk_iv_v_subgev_1neutron = occursin.(r"sk4-5_fc_subgev.*(_1neutron|numubarlike)", bininfo.Sample),
        sk_iv_v_multigev_0neutron = occursin.(r"sk4-5_fc_multigev.*(_0neutron|numulike)", bininfo.Sample),
        sk_iv_v_multigev_1neutron = occursin.(r"sk4-5_fc_multigev.*(_1neutron|numubarlike)", bininfo.Sample),
        sk_i_v_multigev_multiring_nue = occursin.("sk1-5_fc_multigev_multiring_nuelike", bininfo.Sample),
        sk_i_v_multigev_multiring_nuebar = occursin.("sk1-5_fc_multigev_multiring_nuebarlike", bininfo.Sample),
        sk_i_v_multigev_multiring_mu = occursin.("sk1-5_fc_multigev_multiring_mulike", bininfo.Sample),
        sk_i_v_multigev_multiring_other = occursin.("sk1-5_fc_multigev_multiring_other", bininfo.Sample),
        # PID migration masks
        sk_i_iii_subgev_elike = occursin.(r"sk1-3_fc_subgev_1ring_elike", bininfo.Sample),
        sk_i_iii_subgev_mulike = occursin.(r"sk1-3_fc_subgev_1ring_mulike", bininfo.Sample),
        sk_iv_v_subgev_elike = occursin.(r"sk4-5_fc_subgev_1ring_nue", bininfo.Sample),
        sk_iv_v_subgev_mulike = occursin.(r"sk4-5_fc_subgev_1ring_numu", bininfo.Sample),
        sk_i_iii_multigev_1ring_elike = occursin.(r"sk1-3_fc_multigev_1ring_(elike|nue)", bininfo.Sample),
        sk_i_iii_multigev_1ring_mulike = occursin.(r"sk1-3_fc_multigev_1ring_(mulike|numu)", bininfo.Sample),
        sk_iv_v_multigev_1ring_elike = occursin.(r"sk4-5_fc_multigev_1ring_nue", bininfo.Sample),
        sk_iv_v_multigev_1ring_mulike = occursin.(r"sk4-5_fc_multigev_1ring_numu", bininfo.Sample),
        # Ring counting masks
        sk_1ring = occursin.(r"_1ring_", bininfo.Sample),
        sk_multiring = occursin.(r"_(2ring|multiring)_", bininfo.Sample),
        # E-like mask (for nue contamination and NC pi0)
        sk_elike = occursin.(r"(elike|nuebarlike)", bininfo.Sample) .| occursin.(r"_nuelike", bininfo.Sample),
        # SK phase masks (for split energy scale)
        sk_i_iii_bins = occursin.(r"^sk1-3_", bininfo.Sample),
        sk_iv_v_bins = .!occursin.(r"^sk1-3_", bininfo.Sample),
        # Multi-GeV FC mask (for relative normalization)
        fc_multigev = occursin.(r"_fc_multigev_", bininfo.Sample),
        # PC + Up-mu mask (for relative normalization)
        pc_upmu = occursin.("_pc_", bininfo.Sample) .| occursin.("_upmu_", bininfo.Sample),
        # FC multi-GeV mu-like single-ring (for FC/PC separation)
        fc_multigev_mulike = occursin.(r"_fc_multigev_1ring_mu", bininfo.Sample),
        # pi0 samples
        sk_1ring_pi0 = occursin.("_1ring_ncpi0", bininfo.Sample),
        sk_2ring_pi0 = occursin.("_2ring_ncpi0", bininfo.Sample),
        # Ring separation sub-GeV vs multi-GeV
        sk_subgev_1ring = occursin.(r"_fc_subgev_1ring_", bininfo.Sample),
        sk_subgev_multiring = occursin.(r"_fc_subgev.*(2ring|multiring)", bininfo.Sample),
        sk_multigev_1ring = occursin.(r"_fc_multigev_1ring_", bininfo.Sample),
        sk_multigev_multiring = occursin.(r"_fc_multigev.*(2ring|multiring)", bininfo.Sample),
    )

    data = CSV.read(joinpath(datadir, "bins/sk_2023_Data.txt"), DataFrame; delim=' ', ignorerepeated=true, comment="#", header=false)
    observed = round.(data.Column1);

    MC = (nue=read_sk_file(joinpath(datadir, "bins/normal/sk_2023_MCNueNO.txt")),
        numu=read_sk_file(joinpath(datadir, "bins/normal/sk_2023_MCNumuNO.txt")),
        nutau=read_sk_file(joinpath(datadir, "bins/normal/sk_2023_MCNutauNO.txt")),
        nuebar=read_sk_file(joinpath(datadir, "bins/normal/sk_2023_MCNueBarNO.txt")),
        numubar=read_sk_file(joinpath(datadir, "bins/normal/sk_2023_MCNumuBarNO.txt")),
        nunc=read_sk_file(joinpath(datadir, "bins/normal/sk_2023_MCNCNO.txt")))

        
    loge_grid = LinRange(-1,3,201)
    cz_grid = LinRange(-1.0,1.0,101)

    # Bestfit from SK atm 2023 paper
    params_nominal = Newtrinos.get_params(physics)
    @reset params_nominal.Δm²₃₁ = 2.475e-3
    @reset params_nominal.θ₂₃ = asin(sqrt(0.45))
    @reset params_nominal.θ₁₃ = asin(sqrt(0.02))
    @reset params_nominal.δCP = -1.89

    nominal_layers = physics.earth_layers.compute_layers()
    cz_midpoints = midpoints(cz_grid)
    paths = physics.earth_layers.compute_paths(cz_midpoints, nominal_layers)
    flux_nominal = physics.atm_flux.nominal_flux(10. .^midpoints(loge_grid), cz_midpoints)

    flatten_R(R3d) = NamedTuple(key => reshape(R3d[key], size(R3d[key], 1), :) for key in keys(R3d))

    # Load precomputed response parameters (from generate_response_params.jl)
    vmf_data = load(joinpath(datadir, "vmf_cosz_params.jld2"))
    vmf_params = vmf_data["vmf_params"]

    e_params = load(joinpath(datadir, "energy_response_params.jld2"))
    energy_params = e_params["mix_logE"]

    # Build response matrices from precomputed energy + E-dependent vMF cosZ parameters
    R_3d = NamedTuple(key => _build_R_from_params(
        MC[key], loge_grid, cz_grid,
        energy_params[key], vmf_params[key]) for key in keys(MC))

    R = flatten_R(R_3d)

    # Compute nu/nubar mixture fractions for nutau CC and NC channels.
    # The SK MC lumps nutau+nutaubar CC, and all NC, into single channels.
    # We estimate the nu vs nubar fraction from nominal flux * actual cross-sections.
    # For nutau CC, use numu cross-sections (nutau xsec not available separately).
    E_mid = 10.0 .^ midpoints(loge_grid)

    # Get actual cross-section curves from the xsec data file
    xsec_data = load(joinpath(dirname(dirname(dirname(datadir))), "physics", "xsec_genie_data.jld2"))
    xsec_E = xsec_data["E_grid"]
    wester = xsec_data["wester_xsec"]
    cc_channels = ("CC1p1h", "CC2p2h", "CC1pi", "CCDIS", "CCother")

    # Total CC xsec (sigma/E) for numu and numubar (used for nutau since nutau xsec ~ numu xsec)
    numu_cc_total = sum(wester["numu"][ch] for ch in cc_channels)
    numubar_cc_total = sum(wester["numubar"][ch] for ch in cc_channels)
    # NC xsec for nu and nubar
    numu_nc = wester["numu"]["NC"]
    numubar_nc = wester["numubar"]["NC"]

    # Interpolate to our energy grid
    itp_numu_cc = extrapolate(interpolate((xsec_E,), numu_cc_total, Gridded(Linear())), Interpolations.Flat())
    itp_numubar_cc = extrapolate(interpolate((xsec_E,), numubar_cc_total, Gridded(Linear())), Interpolations.Flat())
    itp_nu_nc = extrapolate(interpolate((xsec_E,), numu_nc, Gridded(Linear())), Interpolations.Flat())
    itp_nubar_nc = extrapolate(interpolate((xsec_E,), numubar_nc, Gridded(Linear())), Interpolations.Flat())

    sigma_numu_cc = itp_numu_cc.(E_mid) .* E_mid      # sigma = (sigma/E) * E
    sigma_numubar_cc = itp_numubar_cc.(E_mid) .* E_mid
    sigma_nu_nc = itp_nu_nc.(E_mid) .* E_mid
    sigma_nubar_nc = itp_nubar_nc.(E_mid) .* E_mid

    # Atmospheric flux averaged over cosZ
    flux_nom = physics.atm_flux.nominal_flux(E_mid, cz_midpoints)
    s_flux = (length(E_mid), length(cz_midpoints))
    flux_nu_E = vec(mean(reshape(flux_nom.nue, s_flux) .+ reshape(flux_nom.numu, s_flux), dims=2))
    flux_nubar_E = vec(mean(reshape(flux_nom.nuebar, s_flux) .+ reshape(flux_nom.numubar, s_flux), dims=2))

    # nutau CC mixture: use numu xsec as proxy for nutau
    nutau_nu_rate = flux_nu_E .* sigma_numu_cc
    nutau_nubar_rate = flux_nubar_E .* sigma_numubar_cc
    nutau_nu_frac = nutau_nu_rate ./ (nutau_nu_rate .+ nutau_nubar_rate .+ 1e-30)

    # NC mixture
    nc_nu_rate = flux_nu_E .* sigma_nu_nc
    nc_nubar_rate = flux_nubar_E .* sigma_nubar_nc
    nc_nu_frac = nc_nu_rate ./ (nc_nu_rate .+ nc_nubar_rate .+ 1e-30)

    nominal_weights = calc_weights(params_nominal, (;R, flux_nominal, paths, nominal_layers, loge_grid, cz_grid, cz_midpoints, nutau_nu_frac, nc_nu_frac), physics)

    # Build energy groups for reco bin overlap energy scale method
    sk_i_iii_mask = masks.sk_i_iii_bins
    sk_iv_v_mask = masks.sk_iv_v_bins

    energy_groups_sk_i_iii = build_energy_groups(bininfo, sk_i_iii_mask)
    energy_groups_sk_iv_v = build_energy_groups(bininfo, sk_iv_v_mask)

    # Up/down normalization masks for FC+PC bins:
    # +1 for upgoing (CosZMax ≤ 0), -1 for downgoing (CosZMin ≥ 0), 0 otherwise
    fc_pc_mask = masks.fc .| masks.pc
    upgoing = bininfo.CosZMax .<= 0
    downgoing = bininfo.CosZMin .>= 0

    sk_i_iii_updown = Float64.(sk_i_iii_mask .& fc_pc_mask .& upgoing) .- Float64.(sk_i_iii_mask .& fc_pc_mask .& downgoing)
    sk_iv_v_updown = Float64.(sk_iv_v_mask .& fc_pc_mask .& upgoing) .- Float64.(sk_iv_v_mask .& fc_pc_mask .& downgoing)

    masks = (; masks..., sk_i_iii_updown, sk_iv_v_updown)


    return (; MC, R, flux_nominal, nominal_layers, loge_grid, cz_grid, cz_midpoints, nominal_weights, observed, bininfo, masks,
              energy_groups_sk_i_iii, energy_groups_sk_iv_v, nutau_nu_frac, nc_nu_frac)

end



    
function get_params()
    params = (
        sk_i_iii_energy_scale = 1.0,
        sk_iv_v_energy_scale = 1.0,
        sk_i_iii_updown_energy_scale = 1.0,
        sk_iv_v_updown_energy_scale = 1.0,
        sk_fc_norm = 1.0,
        sk_pc_norm = 1.0,
        sk_upmu_norm = 1.0,
        sk_fiducial_norm = 1.0,
        sk_nc_mu_norm = 1.0,
        sk_pc_stop_thru_top = 1.0,
        sk_pc_stop_thru_barrel = 1.0,
        sk_pc_stop_thru_bottom = 1.0,
        sk_upmu_stopping_vs_througoing = 1.0,
        sk_upmu_nonshower_vs_shower = 1.0,
        sk_i_iii_decay_e_tag_eff = 1.0,
        sk_iv_v_decay_e_tag_eff = 1.0,
        sk_iv_v_subgev_neutron_tag_eff = 1.0,
        sk_iv_v_multigev_neutron_tag_eff = 1.0,
        sk_i_v_bdt_1 = 1.0,
        sk_i_v_bdt_2 = 1.0,
        sk_i_v_bdt_3 = 1.0,
        sk_i_iii_subgev_pid = 1.0,
        sk_iv_v_subgev_pid = 1.0,
        sk_i_iii_multigev_pid = 1.0,
        sk_iv_v_multigev_pid = 1.0,
        sk_ring_counting = 1.0,
        sk_nue_contamination = 1.0,
        sk_ncpi0_norm = 1.0,
        # Relative normalizations (flux model differences at high energy)
        sk_fc_multigev_rel_norm = 1.0,
        sk_pc_upmu_rel_norm = 1.0,
        # FC/PC separation
        sk_fc_pc_separation = 1.0,
        # pi0 selection
        sk_pi0_norm = 1.0,
        # Split ring counting: sub-GeV and multi-GeV
        sk_subgev_ring_counting = 1.0,
        sk_multigev_ring_counting = 1.0,
        # Energy-dependent flux normalization (bathtub shape, split at 1 GeV)
        sk_flux_norm_low = 0.0,
        sk_flux_norm_high = 0.0,
        )
end

function get_priors()
    priors = (
        # Energy scales from Table 5.6 (conventional FV), exposure-weighted by livetime
        # SK I: 3.3% (~1489d), SK II: 2.0% (~799d), SK III: 2.4% (~518d) → exposure-weighted ~2.8%, use SK I-dominated 3.3%
        # SK IV: 2.1% (~3244d), SK V: 1.8% (~2970d) → exposure-weighted ~2.0%
        sk_i_iii_energy_scale = Truncated(Normal(1.0, 0.033), 0.5, 1.5),
        sk_iv_v_energy_scale = Truncated(Normal(1.0, 0.021), 0.5, 1.5),
        # Up/down energy scale from Table 5.6, split by phase group
        # SK I: 1.3%, SK II: 0.6%, SK III: 0.7% → exposure-weighted ~1.0%
        # SK IV: 0.5%, SK V: 0.7% → exposure-weighted ~0.6%
        sk_i_iii_updown_energy_scale = Truncated(Normal(1.0, 0.01), 0.5, 1.5),
        sk_iv_v_updown_energy_scale = Truncated(Normal(1.0, 0.006), 0.5, 1.5),
        sk_fc_norm = Normal(1.0, 0.015),
        sk_pc_norm = Normal(1.0, 0.03),
        sk_upmu_norm = Normal(1.0, 0.01),
        sk_fiducial_norm = Normal(1.0, 0.02),
        sk_nc_mu_norm = Normal(1.0, 0.1),
        sk_pc_stop_thru_top = Normal(1.0, 0.2),
        sk_pc_stop_thru_barrel = Normal(1.0, 0.2),
        sk_pc_stop_thru_bottom = Normal(1.0, 0.2),
        sk_upmu_stopping_vs_througoing = Normal(1.0, 0.01),
        sk_upmu_nonshower_vs_shower = Normal(1.0, 0.04),
        sk_i_iii_decay_e_tag_eff = Normal(1.0, 0.015),
        sk_iv_v_decay_e_tag_eff = Normal(1.0, 0.008),
        sk_iv_v_subgev_neutron_tag_eff = Normal(1.0, 0.12),
        sk_iv_v_multigev_neutron_tag_eff = Normal(1.0, 0.12),
        sk_i_v_bdt_1 = Normal(1, 0.05),
        sk_i_v_bdt_2 = Normal(1, 0.05),
        sk_i_v_bdt_3 = Normal(1, 0.05),
        # PID: thesis shows <1% for most phases, up to ~2-3% for some
        # Sub-GeV PID is better constrained than multi-GeV
        sk_i_iii_subgev_pid = Normal(1, 0.02),
        sk_iv_v_subgev_pid = Normal(1, 0.02),
        sk_i_iii_multigev_pid = Normal(1, 0.03),
        sk_iv_v_multigev_pid = Normal(1, 0.03),
        # Ring counting: split into sub-GeV (better constrained) and multi-GeV
        sk_ring_counting = Normal(1, 0.05),
        sk_nue_contamination = Normal(1, 0.05),
        sk_ncpi0_norm = Normal(1, 0.1),
        # Relative normalizations (Section 5.2.1): 5% for multi-GeV FC and PC+upmu
        sk_fc_multigev_rel_norm = Normal(1, 0.05),
        sk_pc_upmu_rel_norm = Normal(1, 0.05),
        # FC/PC separation: ~1% migration
        sk_fc_pc_separation = Normal(1, 0.01),
        # pi0 selection uncertainty
        sk_pi0_norm = Normal(1, 0.1),
        # Split ring counting
        sk_subgev_ring_counting = Normal(1, 0.03),
        sk_multigev_ring_counting = Normal(1, 0.05),
        # Energy-dependent flux normalization (bathtub shape)
        # Low-E: 25% at 0.1 GeV, linear in logE to 7% at 1 GeV
        # High-E: 7% flat from 1-10 GeV, linear in logE to 20% at 1 TeV
        sk_flux_norm_low = Truncated(Normal(0, 1), -3, 3),
        sk_flux_norm_high = Truncated(Normal(0, 1), -3, 3),
        )
end


function reweight(params, physics, assets)
    weights = calc_weights(params, assets, physics)
    return map((mc, w, nw) -> mc.Counts .* safe_div.(w, nw), assets.MC, weights, assets.nominal_weights)
end

function get_factor(mask, factor)
    mask * factor .+ .!mask 
end

function get_double_factor(total, mask1, mask2, factor1)
    total1 = sum(total[mask1])
    total2 = sum(total[mask2])
    new_total1 = factor1 * total1
    new_total2 = total2 + total1 - new_total1
    factor2 = new_total2 / total2

    factor = (mask1 * factor1 .+ .!mask1) .* (mask2 * factor2 .+ .!mask2)

    return factor
end

function get_all_factors(params, assets, total)
    # Returns the sum of deviations Σ_j f_ij for the linearized formalism.
    # Applied in get_expected as: expected_i × (1 + common_deviations + sample_deviations)
    return (
        (get_factor(assets.masks.fc, params.sk_fc_norm * params.sk_fiducial_norm) .- 1) .+
        (get_factor(assets.masks.pc, params.sk_pc_norm * params.sk_fiducial_norm) .- 1) .+
        (get_factor(assets.masks.upmu, params.sk_upmu_norm) .- 1) .+
        (get_double_factor(total, assets.masks.pc_stop_top, assets.masks.pc_thru_top, params.sk_pc_stop_thru_top) .- 1) .+
        (get_double_factor(total, assets.masks.pc_stop_barrel, assets.masks.pc_thru_barrel, params.sk_pc_stop_thru_barrel) .- 1) .+
        (get_double_factor(total, assets.masks.pc_stop_bottom, assets.masks.pc_thru_bottom, params.sk_pc_stop_thru_bottom) .- 1) .+
        (get_double_factor(total, assets.masks.umpmu_stop, assets.masks.upmu_thru, params.sk_upmu_stopping_vs_througoing) .- 1) .+
        (get_double_factor(total, assets.masks.upmu_nonshower, assets.masks.upmu_shower, params.sk_upmu_nonshower_vs_shower) .- 1) .+
        (get_double_factor(total, assets.masks.sk_i_iii_elike_1decay_e, assets.masks.sk_i_iii_elike_0decay_e, params.sk_i_iii_decay_e_tag_eff) .- 1) .+
        (get_double_factor(total, assets.masks.sk_i_iii_mulike_1decay_e, assets.masks.sk_i_iii_mulike_0decay_e, params.sk_i_iii_decay_e_tag_eff) .- 1) .+
        (get_double_factor(total, assets.masks.sk_i_iii_mulike_2decay_e, assets.masks.sk_i_iii_mulike_1decay_e, params.sk_i_iii_decay_e_tag_eff) .- 1) .+
        (get_double_factor(total, assets.masks.sk_iv_v_1decay_e, assets.masks.sk_iv_v_0decay_e, params.sk_iv_v_decay_e_tag_eff) .- 1) .+
        (get_double_factor(total, assets.masks.sk_iv_v_subgev_0neutron, assets.masks.sk_iv_v_subgev_1neutron, params.sk_iv_v_subgev_neutron_tag_eff) .- 1) .+
        (get_double_factor(total, assets.masks.sk_iv_v_multigev_0neutron, assets.masks.sk_iv_v_multigev_1neutron, params.sk_iv_v_multigev_neutron_tag_eff) .- 1) .+
        (get_double_factor(total, assets.masks.sk_i_v_multigev_multiring_nuebar, assets.masks.sk_i_v_multigev_multiring_nue, params.sk_i_v_bdt_1) .- 1) .+
        (get_double_factor(total, assets.masks.sk_i_v_multigev_multiring_nue, assets.masks.sk_i_v_multigev_multiring_mu, params.sk_i_v_bdt_2) .- 1) .+
        (get_double_factor(total, assets.masks.sk_i_v_multigev_multiring_mu, assets.masks.sk_i_v_multigev_multiring_other, params.sk_i_v_bdt_3) .- 1) .+
        # PID migration: e-like ↔ mu-like
        (get_double_factor(total, assets.masks.sk_i_iii_subgev_elike, assets.masks.sk_i_iii_subgev_mulike, params.sk_i_iii_subgev_pid) .- 1) .+
        (get_double_factor(total, assets.masks.sk_iv_v_subgev_elike, assets.masks.sk_iv_v_subgev_mulike, params.sk_iv_v_subgev_pid) .- 1) .+
        (get_double_factor(total, assets.masks.sk_i_iii_multigev_1ring_elike, assets.masks.sk_i_iii_multigev_1ring_mulike, params.sk_i_iii_multigev_pid) .- 1) .+
        (get_double_factor(total, assets.masks.sk_iv_v_multigev_1ring_elike, assets.masks.sk_iv_v_multigev_1ring_mulike, params.sk_iv_v_multigev_pid) .- 1) .+
        # Ring counting migration: overall + split by energy
        (get_double_factor(total, assets.masks.sk_1ring, assets.masks.sk_multiring, params.sk_ring_counting) .- 1) .+
        (get_double_factor(total, assets.masks.sk_subgev_1ring, assets.masks.sk_subgev_multiring, params.sk_subgev_ring_counting) .- 1) .+
        (get_double_factor(total, assets.masks.sk_multigev_1ring, assets.masks.sk_multigev_multiring, params.sk_multigev_ring_counting) .- 1) .+
        # Relative normalizations for high-energy samples
        (get_factor(assets.masks.fc_multigev, params.sk_fc_multigev_rel_norm) .- 1) .+
        (get_factor(assets.masks.pc_upmu, params.sk_pc_upmu_rel_norm) .- 1) .+
        # FC/PC separation: FC multi-GeV mu-like ↔ PC
        (get_double_factor(total, assets.masks.fc_multigev_mulike, assets.masks.pc, params.sk_fc_pc_separation) .- 1) .+
        # pi0 selection
        (get_double_factor(total, assets.masks.sk_1ring_pi0, assets.masks.sk_2ring_pi0, params.sk_pi0_norm) .- 1)
    )
end

function get_expected(params, physics, assets)
    expected = reweight(params, physics, assets)

    total = reduce(+, values(expected))

    # Common systematic deviations (shared across all samples)
    common = get_all_factors(params, assets, total)

    # Per-sample scale: 1 + common_deviations + sample-specific deviations
    nue = apply_all_energy_scales(expected.nue .* (1 .+ common), assets, params)
    numu = apply_all_energy_scales(expected.numu .* (1 .+ common .+ (get_factor(assets.masks.sk_elike, params.sk_nue_contamination) .- 1)), assets, params)
    nutau = apply_all_energy_scales(expected.nutau .* (1 .+ common), assets, params)
    nuebar = apply_all_energy_scales(expected.nuebar .* (1 .+ common), assets, params)
    numubar = apply_all_energy_scales(expected.numubar .* (1 .+ common .+ (get_factor(assets.masks.sk_elike, params.sk_nue_contamination) .- 1)), assets, params)
    nunc = apply_all_energy_scales(expected.nunc .* (1 .+ common .+ (get_factor(assets.masks.mu_indices, params.sk_nc_mu_norm) .- 1) .+ (get_factor(assets.masks.sk_elike, params.sk_ncpi0_norm) .- 1)), assets, params)

    return (; nue, numu, nutau, nuebar, numubar, nunc)
end

function get_forward_model(physics, assets)
    function fwd_model(params)
        expected = get_expected(params, physics, assets)
        total = reduce(+, values(expected))
        clamped = max.(1e-3, total)
        distprod(Poisson.(clamped))
    end
end


function get_plot(physics, assets)

    function format_plot_title(raw::String)
        # Replace underscores with spaces
        title = replace(raw, "_" => " ")

        # Replace known abbreviations with readable forms
        replacements = Dict(
            "fc" => "FC",
            "pc" => "PC",
            "subgev" => "Sub-GeV",
            "multigev" => "Multi-GeV",
            "1ring" => "1-Ring",
            "decaye" => "Decay-e",
            "sk1-3" => "SKI-III",
            "sk1-5" => "SKI-V",
            "sk4-5" => "SKIV-V",
            "nuelike" => "νe-like",
            "nuebarlike" => "νe-bar-like",
            "numubarlike" => "‾νμ-bar-like",
            "numulike" => "νμ-like",
        )

        for (key, val) in replacements
            title = replace(title, key => val)
        end

        # Capitalize first letter of each word
        #title = join(uppercasefirst.(split(title)), " ")

        return title
    end

    plot_order = [:nunc, :numubar, :nuebar, :nutau, :numu, :nue]
    plot_color = Dict(zip(plot_order, [:gray80, :paleturquoise, :lightpink, :purple, :steelblue3, :red3]))
    plot_labels = Dict(zip(plot_order, [L"NC", L"$\bar{\nu}_\mu$", L"$\bar{\nu}_e$", L"$\nu_\tau$", L"$\nu_\mu$", L"$\nu_e$"]))

    function plot(params, data=assets.observed)

        bininfo = assets.bininfo
        expected = get_expected(params, physics, assets)

        fig = Figure()
        for (i,sample) in enumerate(unique(bininfo.Sample))
            grid_idx = (Int(floor((i-1)/5))+1, (i-1)%5+1)
            inds = findall(bininfo.Sample .== sample)
            e = NamedTuple(key => expected[key][inds] for key in keys(expected))
            o = data[inds]
            ax = Axis(fig[grid_idx...]; title=format_plot_title(sample), width = 200, height = 150, titlesize=10)
            if all(bininfo.CosZMin[inds] .== -1.0)
                bins = vcat(bininfo.logPMin[inds], [bininfo.logPMax[inds][end]])
                bottom = first(e) * 0.0
                for key in plot_order
                    hist!(ax, midpoints(bins), bins=bins, weights=e[key], offset=bottom, label=plot_labels[key], color=plot_color[key])
                    bottom .+= e[key]
                end
                scatter!(ax, midpoints(bins), o, color=:black)
            else
                bins = vcat(unique(bininfo.CosZMin[inds]), [bininfo.CosZMax[inds][end]])
                bottom = fit(Histogram, bininfo.CosZMin[inds], weights(first(e)), bins).weights * 0.0

                for key in plot_order
                    hist!(ax, bininfo.CosZMin[inds], bins=bins, weights=e[key], offset=bottom, label=plot_labels[key], color=plot_color[key])
                    bottom .+= fit(Histogram, bininfo.CosZMin[inds], weights(e[key]), bins).weights
                end
                h = fit(Histogram, bininfo.CosZMin[inds], weights(o), bins)
                scatter!(ax, midpoints(bins), h.weights, color=:black, label="Data")
            end

            total_e = sum(e[key] for key in keys(e))
            t_e = sum(total_e)
            t_o = sum(o)
            chi2_ndf = sum((total_e .- o).^2 ./ total_e) / size(o)[1]
            text!(ax, 0, 1, text = @sprintf("χ²/n.d.f = %.2f\nTotal MC: %.1f\nTotal Data: %.1f", chi2_ndf, t_e, t_o), space=:relative, fontsize=10, align = (:left, :top), offset = (4, -2))

        end
        Legend(fig[6,5], fig.content[1]; position=:rb, nbanks=2)
        resize_to_layout!(fig)
        fig
    end
end

end
