module super_k

using CSV, DataFrames
using MonotonicSplines
using Interpolations
using CairoMakie
using DataStructures
using Distributions
using DensityInterface
using BAT
using LaTeXStrings
using Accessors
using StatsBase
using Statistics: mean
using Printf
using SpecialFunctions: besseli
using JLD2
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

function configure(physics=default_physics(); energy_cdf=:logE)
    physics = (;physics.osc, physics.atm_flux, physics.earth_layers, physics.xsec)
    assets = get_assets(physics; energy_cdf)
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

"""
    fit_metalog(qv, qp)

Fit a 5-term metalog distribution from 5 quantile points.
The metalog quantile function is:
  Q(y) = a₁ + a₂·ln(y/(1-y)) + a₃·(y-0.5)·ln(y/(1-y)) + a₄·(y-0.5) + a₅·(y-0.5)²

This is a linear system in a₁...a₅, so the fit is exact (no optimization needed).
The CDF is obtained by numerical root-finding of Q(y) = x.
"""
function fit_metalog(qv, qp)
    # Build the 5×5 basis matrix
    n = length(qv)
    M = zeros(n, n)
    for i in 1:n
        y = qp[i]
        logit = log(y / (1 - y))
        M[i, 1] = 1.0
        M[i, 2] = logit
        M[i, 3] = (y - 0.5) * logit
        M[i, 4] = y - 0.5
        M[i, 5] = (y - 0.5)^2
    end
    a = M \ qv
    return a
end

function metalog_quantile(a, y)
    logit = log(y / (1 - y))
    ymh = y - 0.5
    a[1] + a[2] * logit + a[3] * ymh * logit + a[4] * ymh + a[5] * ymh^2
end

function metalog_cdf_at(a, x; tol=1e-10, maxiter=100)
    # Find y such that Q(y) = x via bisection
    lo, hi = 1e-8, 1.0 - 1e-8
    # Check bounds
    metalog_quantile(a, lo) >= x && return 0.0
    metalog_quantile(a, hi) <= x && return 1.0

    for _ in 1:maxiter
        mid = (lo + hi) / 2
        (hi - lo) < tol && return mid
        metalog_quantile(a, mid) < x ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

function make_metalog_e_cdf(bin; resolution_scale=1.0)
    e = [bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent, bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent, bin.EnergyQuantile97_7Percent]

    median_e = e[3]
    e = median_e .+ resolution_scale .* (e .- median_e)

    # Fit in log-space for better behavior on highly skewed energy distributions
    log_e = log10.(max.(e, 1e-30))
    qp = [0.023, 0.159, 0.5, 0.841, 0.977]
    a = fit_metalog(log_e, qp)

    # CDF takes linear energy, converts to log10, then evaluates metalog CDF
    f = x -> begin
        x <= 0 && return 0.0
        metalog_cdf_at(a, log10(x))
    end
    return f
end



# Double-sided Crystal Ball distribution in log(E)
# PDF: Gaussian core with power-law tails on both sides
# Parameters: mu, sigma, alphaL, nL, alphaR, nR
# Transition: left tail at t < -alphaL, right tail at t > alphaR (t = (x-mu)/sigma)

function dscb_pdf_unnorm(t, alphaL, nL, alphaR, nR)
    if t < -alphaL
        A = (nL / alphaL)^nL * exp(-alphaL^2 / 2)
        B = nL / alphaL - alphaL
        return A * (B - t)^(-nL)
    elseif t > alphaR
        A = (nR / alphaR)^nR * exp(-alphaR^2 / 2)
        B = nR / alphaR - alphaR
        return A * (B + t)^(-nR)
    else
        return exp(-t^2 / 2)
    end
end

function dscb_cdf_unnorm(t, alphaL, nL, alphaR, nR)
    # Integral of unnormalized PDF from -inf to t
    if t < -alphaL
        A = (nL / alphaL)^nL * exp(-alphaL^2 / 2)
        B = nL / alphaL - alphaL
        # Integral of A*(B-t')^(-nL) from -inf to t = A/(nL-1) * (B-t)^(1-nL)
        return A / (nL - 1) * (B - t)^(1 - nL)
    else
        # Left tail integral from -inf to -alphaL
        A_L = (nL / alphaL)^nL * exp(-alphaL^2 / 2)
        B_L = nL / alphaL - alphaL
        left_tail = A_L / (nL - 1) * (B_L + alphaL)^(1 - nL)

        if t <= alphaR
            # Gaussian core integral from -alphaL to t
            # = sqrt(2pi) * (Phi(t) - Phi(-alphaL))
            gauss_part = sqrt(2 * pi) * (cdf(Normal(), t) - cdf(Normal(), -alphaL))
            return left_tail + gauss_part
        else
            # Full Gaussian core from -alphaL to alphaR
            gauss_full = sqrt(2 * pi) * (cdf(Normal(), alphaR) - cdf(Normal(), -alphaL))
            # Right tail integral from alphaR to t
            A_R = (nR / alphaR)^nR * exp(-alphaR^2 / 2)
            B_R = nR / alphaR - alphaR
            right_part = A_R / (nR - 1) * ((B_R + alphaR)^(1 - nR) - (B_R + t)^(1 - nR))
            return left_tail + gauss_full + right_part
        end
    end
end

function dscb_cdf(t, alphaL, nL, alphaR, nR)
    # Normalized CDF: divide by total integral (from -inf to +inf)
    total = dscb_cdf_unnorm(1e6, alphaL, nL, alphaR, nR)
    return dscb_cdf_unnorm(t, alphaL, nL, alphaR, nR) / total
end

function fit_dscb(qv, qp)
    # Fit DSCB in log-space: qv are energy quantiles, qp are probabilities
    # Work in log10(E) space
    log_q = log10.(max.(qv, 1e-30))

    # Initial estimates: mu from median, sigma from IQR
    mu = log_q[3]
    sigma = (log_q[4] - log_q[2]) / 2.0
    sigma = max(sigma, 1e-6)

    best = (mu, sigma, 1.5, 5.0, 1.5, 5.0)
    best_err = Inf

    # Scan alphaL, alphaR, nL, nR
    for alphaL in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        for alphaR in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
            for nL in [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
                for nR in [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
                    err = 0.0
                    valid = true
                    for i in 1:5
                        t = (log_q[i] - mu) / sigma
                        c = dscb_cdf(t, alphaL, nL, alphaR, nR)
                        (isnan(c) || c <= 0 || c >= 1) && (valid = false; break)
                        err += (c - qp[i])^2
                    end
                    valid || continue
                    if err < best_err
                        best = (mu, sigma, alphaL, nL, alphaR, nR)
                        best_err = err
                    end
                end
            end
        end
    end

    # Refine: also vary mu and sigma slightly
    mu0, sigma0, aL0, nL0, aR0, nR0 = best
    for mu_f in [0.98, 0.99, 1.0, 1.01, 1.02]
        for sigma_f in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
            mu_t = mu * mu_f
            sigma_t = sigma * sigma_f
            for alphaL in LinRange(max(0.3, aL0*0.7), aL0*1.4, 8)
                for alphaR in LinRange(max(0.3, aR0*0.7), aR0*1.4, 8)
                    for nL in LinRange(max(2.0, nL0*0.5), nL0*2.0, 8)
                        for nR in LinRange(max(2.0, nR0*0.5), nR0*2.0, 8)
                            err = 0.0
                            valid = true
                            for i in 1:5
                                t = (log_q[i] - mu_t) / sigma_t
                                c = dscb_cdf(t, alphaL, nL, alphaR, nR)
                                (isnan(c) || c <= 0 || c >= 1) && (valid = false; break)
                                err += (c - qp[i])^2
                            end
                            valid || continue
                            if err < best_err
                                best = (mu_t, sigma_t, alphaL, nL, alphaR, nR)
                                best_err = err
                            end
                        end
                    end
                end
            end
        end
    end

    return best
end

function make_dscb_e_cdf(bin; resolution_scale=1.0)
    e = [bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent, bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent, bin.EnergyQuantile97_7Percent]
    median_e = e[3]
    e = median_e .+ resolution_scale .* (e .- median_e)
    qp = [0.023, 0.159, 0.5, 0.841, 0.977]

    mu, sigma, alphaL, nL, alphaR, nR = fit_dscb(e, qp)

    f = x -> begin
        x <= 0 && return 0.0
        t = (log10(x) - mu) / sigma
        dscb_cdf(t, alphaL, nL, alphaR, nR)
    end
    return f
end



# Novosibirsk function for energy CDF (in log10(E) space)
# PDF: f(x) = A * exp(-0.5 * ln(1 + Lambda*tau*(x-x0))^2 / tau^2 - tau^2/2)
# where Lambda = sinh(tau*sqrt(ln4)) / (sigma*tau*sqrt(ln4))
# 3 parameters: x0 (peak), sigma (width), tau (tail asymmetry)
# tau > 0: right tail heavier, tau < 0: left tail heavier, tau = 0: Gaussian

function novosibirsk_log_pdf(x, x0, sigma, tau)
    if abs(tau) < 1e-7
        # Gaussian limit
        return -0.5 * ((x - x0) / sigma)^2
    end
    sq2ln4 = sqrt(2 * log(2))  # sqrt(2*ln2) = sqrt(ln4)
    arg = 1.0 + tau * sq2ln4 * (x - x0) / sigma
    arg <= 0 && return -1e10  # outside support
    lnarg = log(arg)
    return -0.5 * (lnarg / tau)^2 - 0.5 * tau^2 + lnarg  # +lnarg is the Jacobian correction
end

function novosibirsk_cdf_table(x0, sigma, tau; x_range=(-1.0, 3.0), n_pts=500)
    xs = collect(range(x_range[1], x_range[2], length=n_pts))
    dx = (x_range[2] - x_range[1]) / (n_pts - 1)
    lp = [novosibirsk_log_pdf(x, x0, sigma, tau) for x in xs]
    lp_max = maximum(lp)
    pdfs = exp.(lp .- lp_max)
    cumul = zeros(n_pts)
    for i in 2:n_pts
        cumul[i] = cumul[i-1] + 0.5 * (pdfs[i] + pdfs[i-1]) * dx
    end
    cumul ./= cumul[end]
    return xs, cumul
end

function novosibirsk_cdf_at(xt, ct, x)
    x <= xt[1] && return 0.0
    x >= xt[end] && return 1.0
    i = searchsortedlast(xt, x)
    i = clamp(i, 1, length(xt) - 1)
    ct[i] + (x - xt[i]) / (xt[i+1] - xt[i]) * (ct[i+1] - ct[i])
end

function fit_novosibirsk(qv, qp)
    # Fit in log10(E) space
    log_q = log10.(max.(qv, 1e-30))

    x0_init = log_q[3]  # median
    sigma_init = (log_q[4] - log_q[2]) / 2.0
    sigma_init = max(sigma_init, 0.01)

    best = (x0_init, sigma_init, 0.0)
    best_err = Inf

    # Coarse scan
    for tau in range(-2.0, 2.0, length=41)
        for sigma_f in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
            for x0_f in [-0.1, -0.05, 0.0, 0.05, 0.1]
                x0 = x0_init + x0_f
                sigma = sigma_init * sigma_f
                xt, ct = novosibirsk_cdf_table(x0, sigma, tau)
                err = sum((novosibirsk_cdf_at.(Ref(xt), Ref(ct), log_q) .- qp).^2)
                if err < best_err
                    best_err = err
                    best = (x0, sigma, tau)
                end
            end
        end
    end

    # Refine
    x0_0, sig_0, tau_0 = best
    for tau in range(tau_0 - 0.2, tau_0 + 0.2, length=15)
        for sigma in range(max(0.01, sig_0 * 0.9), sig_0 * 1.1, length=10)
            for x0 in range(x0_0 - 0.03, x0_0 + 0.03, length=10)
                xt, ct = novosibirsk_cdf_table(x0, sigma, tau)
                err = sum((novosibirsk_cdf_at.(Ref(xt), Ref(ct), log_q) .- qp).^2)
                if err < best_err
                    best_err = err
                    best = (x0, sigma, tau)
                end
            end
        end
    end

    return best
end

function make_novosibirsk_e_cdf(bin; resolution_scale=1.0)
    e = [bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent, bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent, bin.EnergyQuantile97_7Percent]
    median_e = e[3]
    e = median_e .+ resolution_scale .* (e .- median_e)
    qp = [0.023, 0.159, 0.5, 0.841, 0.977]

    x0, sigma, tau = fit_novosibirsk(e, qp)
    xt, ct = novosibirsk_cdf_table(x0, sigma, tau)

    f = x -> begin
        x <= 0 && return 0.0
        novosibirsk_cdf_at(xt, ct, log10(x))
    end
    return f
end


function make_novosibirsk_linE_e_cdf(bin; resolution_scale=1.0)
    e = [bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent, bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent, bin.EnergyQuantile97_7Percent]
    median_e = e[3]
    e = median_e .+ resolution_scale .* (e .- median_e)
    qp = [0.023, 0.159, 0.5, 0.841, 0.977]

    # Fit directly in linear E space
    x0_init = e[3]
    sigma_init = (e[4] - e[2]) / 2.0
    sigma_init = max(sigma_init, 1e-6)

    best = (x0_init, sigma_init, 0.0)
    best_err = Inf

    for tau in range(-2.0, 2.0, length=41)
        for sigma_f in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
            for x0_f in [-0.1, -0.05, 0.0, 0.05, 0.1]
                x0 = x0_init * (1 + x0_f)
                sigma = sigma_init * sigma_f
                xt, ct = novosibirsk_cdf_table(x0, sigma, tau; x_range=(0.0, maximum(e)*3), n_pts=500)
                err = sum((novosibirsk_cdf_at.(Ref(xt), Ref(ct), e) .- qp).^2)
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
                xt, ct = novosibirsk_cdf_table(x0, sigma, tau; x_range=(0.0, maximum(e)*3), n_pts=500)
                err = sum((novosibirsk_cdf_at.(Ref(xt), Ref(ct), e) .- qp).^2)
                if err < best_err
                    best_err = err
                    best = (x0, sigma, tau)
                end
            end
        end
    end

    x0, sigma, tau = best
    xt, ct = novosibirsk_cdf_table(x0, sigma, tau; x_range=(0.0, maximum(e)*3), n_pts=500)

    f = x -> begin
        x <= 0 && return 0.0
        novosibirsk_cdf_at(xt, ct, x)
    end
    return f
end

function make_log_e_cdf(bin; resolution_scale=1.0)
    log_e = log10.([bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent, bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent, bin.EnergyQuantile97_7Percent])  # extrapolate tails

    # Scale quantile distances from median to broaden/narrow the resolution
    median = log_e[3]
    log_e = median .+ resolution_scale .* (log_e .- median)

    log_energy_quantiles = [2*log_e[1] - log_e[2], log_e... , 2*log_e[end] - log_e[end-1]]  # extrapolate tails
    #log_energy_quantiles = [log_e[1] - mean(diff(log_e)), log_e... , log_e[end] + mean(diff(log_e))]  # extrapolate tails
    quantile_probs = [0.0, 0.023, 0.159, 0.5, 0.841, 0.977, 1.]  # corresponding probabilities

    dy_dx = MonotonicSplines.estimate_dYdX(log_energy_quantiles, quantile_probs)
    dy_dx[1] = 0
    dy_dx[end] = 0
    f = RQSpline(log_energy_quantiles, quantile_probs, dy_dx)


    f_save = x -> begin
        if x < log_energy_quantiles[1]
            return 0.0
        elseif x > log_energy_quantiles[end]
            return 1.0
        else
            return f(x)
        end
    end

    return f_save
end

function make_e_cdf(bin; resolution_scale=1.0)
    e = [bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent, bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent, bin.EnergyQuantile97_7Percent]

    # Scale quantile distances from median to broaden/narrow the resolution
    median = e[3]
    e = median .+ resolution_scale .* (e .- median)

    # Extrapolate tails, ensuring strict monotonicity
    lo = max(0.0, 2*e[1] - e[2])
    if lo >= e[1]
        lo = e[1] * 0.5  # fallback: half the lowest quantile
    end
    hi = 2*e[end] - e[end-1]
    energy_quantiles = [lo, e..., hi]
    quantile_probs = [0.0, 0.023, 0.159, 0.5, 0.841, 0.977, 1.]

    # Ensure strict monotonicity by nudging any non-increasing knots
    for i in 2:length(energy_quantiles)
        if energy_quantiles[i] <= energy_quantiles[i-1]
            energy_quantiles[i] = energy_quantiles[i-1] + 1e-6
        end
    end

    dy_dx = MonotonicSplines.estimate_dYdX(energy_quantiles, quantile_probs)
    dy_dx[1] = 0
    dy_dx[end] = 0
    f = RQSpline(energy_quantiles, quantile_probs, dy_dx)

    f_save = x -> begin
        if x < energy_quantiles[1]
            return 0.0
        elseif x > energy_quantiles[end]
            return 1.0
        else
            return f(x)
        end
    end

    return f_save
end

function make_cosz_cdf(bin; resolution_scale=1.0)
    cosz = [bin.CosZQuantile2_3Percent, bin.CosZQuantile15_9Percent, bin.CosZQuantile50_0Percent, bin.CosZQuantile84_1Percent, bin.CosZQuantile97_7Percent]  # extrapolate tails

    # Scale quantile distances from median to broaden/narrow the resolution
    median = cosz[3]
    cosz = median .+ resolution_scale .* (cosz .- median)

    cosz_quantiles = [-1, cosz..., 1]  # extrapolate tails
    cosz_quantiles = [min(-1, 2*cosz[1] - cosz[2]), cosz... , max(1, 2*cosz[end] - cosz[end-1])]  # extrapolate tails
    quantile_probs = [0., 0.023, 0.159, 0.5, 0.841, 0.977, 1.]  # corresponding probabilities

    dy_dx = MonotonicSplines.estimate_dYdX(cosz_quantiles, quantile_probs)
    #dy_dx[1] = 0
    #dy_dx[end] = 0
    f = RQSpline(cosz_quantiles, quantile_probs, dy_dx)
    f_save = x -> begin
        if x <= cosz_quantiles[1]
            return 0.0
        elseif x > cosz_quantiles[end]
            return 1.0
        else
            return f(x)
        end
    end

    return f_save
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

function make_vmf_cosz_cdf(bin; resolution_scale=1.0)
    cosz = [bin.CosZQuantile2_3Percent, bin.CosZQuantile15_9Percent, bin.CosZQuantile50_0Percent, bin.CosZQuantile84_1Percent, bin.CosZQuantile97_7Percent]
    med = cosz[3]
    cosz = med .+ resolution_scale .* (cosz .- med)
    qp = [0.023, 0.159, 0.5, 0.841, 0.977]

    mu_z, kappa = fit_vmf_cosz(cosz, qp)
    ct, cd = _vmf_cdf_table(mu_z, kappa)

    f = x -> _vmf_cdf_at(ct, cd, x)
    return f
end


function _build_R_from_params(MC_component, logE_grid, cosZ_grid, dscb_params, vmf_params, reco_logP_width, reco_cz_width; mc_data_ratio=5.0)
    # Build response matrix from precomputed DSCB energy + vMF cosZ parameters
    # mc_data_ratio: assumed ratio of raw MC events to reported (scaled) counts
    n_bins = size(MC_component, 1)
    n_logE = length(logE_grid) - 1
    n_cosZ = length(cosZ_grid) - 1
    E_grid = 10 .^ logE_grid
    dlogE = Float64(logE_grid[2] - logE_grid[1])

    response_matrix = zeros(Float64, n_bins, n_logE, n_cosZ)

    for bin_idx in 1:n_bins
        counts = MC_component[bin_idx, :].Counts
        counts == 0 && continue

        # Energy: DSCB CDF in logE space
        mu = dscb_params["mu"][bin_idx]
        sigma = dscb_params["sigma"][bin_idx]
        sigma == 0 && continue
        aL = dscb_params["alphaL"][bin_idx]
        nL = dscb_params["nL"][bin_idx]
        aR = dscb_params["alphaR"][bin_idx]
        nR = dscb_params["nR"][bin_idx]
        c_e = [dscb_cdf((log10(e) - mu) / sigma, aL, nL, aR, nR) for e in E_grid]
        p_e = diff(c_e)

        # Per-bin energy blur based on MC statistics
        # SK uses ~10 nearest neighbors in energy for per-event osc prob averaging.
        # With N_MC events distributed across a reco bin of width delta_logP,
        # the 10 nearest neighbors span ~ delta_logP * min(10, N_MC) / N_MC.
        # sigma ~ that span / sqrt(12).
        n_mc = mc_data_ratio * counts
        n_neighbors = min(10.0, n_mc)
        sigma_blur_logE = reco_logP_width[bin_idx] * sqrt(n_neighbors / max(n_mc, 1.0)) / sqrt(12.0)
        sigma_blur_bins = sigma_blur_logE / dlogE
        K_e = make_gaussian_kernel_matrix(n_logE, sigma_blur_bins)
        p_e = K_e * p_e

        # CosZ: vMF CDF
        kappa = vmf_params["kappa"][bin_idx]
        if kappa > 0
            ct, cd = _vmf_cdf_table(vmf_params["mu_z"][bin_idx], kappa)
            c_cosz = [_vmf_cdf_at(ct, cd, x) for x in cosZ_grid]
        else
            c_cosz = collect(range(0, 1, length=length(cosZ_grid)))
        end
        p_cosz = diff(c_cosz)

        # CosZ blur: same logic with reco cosZ bin width
        sigma_blur_cz = reco_cz_width[bin_idx] * sqrt(n_neighbors / max(n_mc, 1.0)) / sqrt(12.0)
        dCZ = Float64(cosZ_grid[2] - cosZ_grid[1])
        sigma_blur_cz_bins = sigma_blur_cz / dCZ
        K_cz = make_gaussian_kernel_matrix(n_cosZ, sigma_blur_cz_bins)
        p_cosz = K_cz * p_cosz

        response_matrix[bin_idx, :, :] .= p_e * p_cosz'
    end
    return response_matrix
end

function make_response_matrix(MC_component, logE_grid, cosZ_grid; resolution_scale=1.0, energy_cdf=:logE, vmf_mu_z=nothing, vmf_kappa=nothing)
    n_bins = size(MC_component, 1)
    n_logE = length(logE_grid)
    n_cosZ = length(cosZ_grid)

    response_matrix = zeros(Float64, n_bins, n_logE-1, n_cosZ-1)

    # For linear-E / johnsonSU CDF, convert logE grid edges to linear E
    E_grid = energy_cdf in (:linearE, :metalog, :dscb, :novosibirsk, :novosibirsk_linE) ? 10 .^ logE_grid : nothing

    for bin_idx in 1:n_bins
        bin = MC_component[bin_idx, :]

        if bin.Counts == 0
            continue
        end

        if energy_cdf == :novosibirsk_linE
            e_cdf = make_novosibirsk_linE_e_cdf(bin; resolution_scale)
            c_e = e_cdf.(E_grid)
        elseif energy_cdf == :novosibirsk
            e_cdf = make_novosibirsk_e_cdf(bin; resolution_scale)
            c_e = e_cdf.(E_grid)
        elseif energy_cdf == :dscb
            e_cdf = make_dscb_e_cdf(bin; resolution_scale)
            c_e = e_cdf.(E_grid)
        elseif energy_cdf == :metalog
            e_cdf = make_metalog_e_cdf(bin; resolution_scale)
            c_e = e_cdf.(E_grid)
        elseif energy_cdf == :linearE
            e_cdf = make_e_cdf(bin; resolution_scale)
            c_e = e_cdf.(E_grid)
        else
            log_e_cdf = make_log_e_cdf(bin; resolution_scale)
            c_e = log_e_cdf.(logE_grid)
        end
        p_e = diff(c_e)

        if vmf_mu_z !== nothing && vmf_kappa !== nothing && vmf_kappa[bin_idx] > 0
            # Use precomputed vMF parameters
            ct, cd = _vmf_cdf_table(vmf_mu_z[bin_idx], vmf_kappa[bin_idx])
            c_cosz = [_vmf_cdf_at(ct, cd, x) for x in cosZ_grid]
            p_cosz = diff(c_cosz)
        else
            # Fallback to spline
            cosz_fn = make_cosz_cdf(bin; resolution_scale)
            c_cosz = cosz_fn.(cosZ_grid)
            p_cosz = diff(c_cosz)
        end

        response_matrix[bin_idx, :, :] .= p_e * p_cosz'
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

function smooth_osc_prob(p, K_E, K_CZ)
    # Apply separable 2D Gaussian smoothing to oscillation probabilities.
    # p is (n_E, n_CZ, n_flav_in, n_flav_out)
    # K_E is (n_E, n_E) convolution matrix, K_CZ is (n_CZ, n_CZ).
    # Smoothing: p_smooth = K_E * p * K_CZ'  (for each flavor pair)
    n_E, n_CZ, n_in, n_out = size(p)
    p_smooth = similar(p)
    for a in 1:n_in, b in 1:n_out
        # K_E * p[:,:,a,b] * K_CZ' — separable 2D convolution via matrix multiply
        p_smooth[:, :, a, b] = K_E * p[:, :, a, b] * K_CZ'
    end
    return p_smooth
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

    xsec_nue     = physics.xsec.scale(E, :nue,   :CC, false, params)
    xsec_numu    = physics.xsec.scale(E, :numu,  :CC, false, params)
    xsec_nutau   = physics.xsec.scale(E, :nutau, :CC, false, params)
    xsec_nuebar  = physics.xsec.scale(E, :nue,   :CC, true,  params)
    xsec_numubar = physics.xsec.scale(E, :numu,  :CC, true,  params)
    xsec_nutaubar= physics.xsec.scale(E, :nutau, :CC, true,  params)
    xsec_nc      = physics.xsec.scale(E, :nue,   :NC, false, params)

    # HKKM flux is differential: Φ(E) in (m² s sr GeV)⁻¹.
    # On our logE grid, bin content ∝ Φ(E) × E (Jacobian dE/dlogE = E ln10).
    # Multiply by E after reshape to properly weight the energy integration.
    flux_nue    = reshape(flux.nue,    s) .* E
    flux_numu   = reshape(flux.numu,   s) .* E
    flux_nuebar = reshape(flux.nuebar, s) .* E
    flux_numubar= reshape(flux.numubar,s) .* E

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
    xsec_nc_anti = physics.xsec.scale(E, :nue, :NC, true, params)
    flux_nu_total = flux_nue .+ flux_numu
    flux_nubar_total = flux_nuebar .+ flux_numubar
    nc_nu_flux = flux_nu_total .* xsec_nc
    nc_nubar_flux = flux_nubar_total .* xsec_nc_anti
    nc_combined = nc_nu_flux .* assets.nc_nu_frac .+ nc_nubar_flux .* (1 .- assets.nc_nu_frac)
    nunc    = contract_R(assets.R.nunc,    nc_combined .* flux_norm)

    return (; nue, numu, nutau, nuebar, numubar, nunc)
end

safe_div(a, b, ε=1e-10) = a / (b + ε)

function get_assets(physics; datadir = @__DIR__, energy_cdf=:logE)
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

    # Load precomputed response parameters (from generate_cz_response.jl and generate_e_response.jl)
    vmf_data = load(joinpath(datadir, "vmf_cosz_params.jld2"))
    vmf_params = vmf_data["vmf_params"]

    e_response_file = joinpath(datadir, "energy_response_params.jld2")
    e_params = isfile(e_response_file) ? load(e_response_file) : nothing

    # Build response matrices from precomputed distribution parameters
    if e_params !== nothing && energy_cdf == :dscb && haskey(e_params, "dscb_logE")
        # Use precomputed DSCB energy params + vMF cosZ params
        dscb_params = e_params["dscb_logE"]
        reco_logP_width = bininfo.logPMax .- bininfo.logPMin
        reco_cz_width = abs.(bininfo.CosZMax .- bininfo.CosZMin)
        R_3d = NamedTuple(key => _build_R_from_params(
            MC[key], loge_grid, cz_grid,
            dscb_params[key], vmf_params[key], reco_logP_width, reco_cz_width) for key in keys(MC))
    else
        # Fallback: fit energy CDF on the fly, use precomputed vMF cosZ
        R_3d = NamedTuple(key => make_response_matrix(MC[key], loge_grid, cz_grid;
            resolution_scale=1.0, energy_cdf,
            vmf_mu_z=vmf_params[key]["mu_z"], vmf_kappa=vmf_params[key]["kappa"]) for key in keys(MC))
    end
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
    itp_numu_cc = extrapolate(interpolate((xsec_E,), numu_cc_total, Gridded(Linear())), Flat())
    itp_numubar_cc = extrapolate(interpolate((xsec_E,), numubar_cc_total, Gridded(Linear())), Flat())
    itp_nu_nc = extrapolate(interpolate((xsec_E,), numu_nc, Gridded(Linear())), Flat())
    itp_nubar_nc = extrapolate(interpolate((xsec_E,), numubar_nc, Gridded(Linear())), Flat())

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

    # Old F_ij method (commented out — replaced by reco bin overlap method)
    #loge_grid_plus = loge_grid .+ log10(1.02)
    #loge_grid_minus = loge_grid .+ log10(0.98)
    #R_plus_3d = NamedTuple(key => make_response_matrix(MC[key], loge_grid_plus, cz_grid; resolution_scale=1.01) for key in keys(MC))
    #R_minus_3d = NamedTuple(key => make_response_matrix(MC[key], loge_grid_minus, cz_grid; resolution_scale=1.01) for key in keys(MC))
    #weights_plus = calc_weights(params_nominal, (;R=flatten_R(R_plus_3d), flux_nominal, paths, nominal_layers, loge_grid=loge_grid_plus, cz_midpoints), physics)
    #weights_minus = calc_weights(params_nominal, (;R=flatten_R(R_minus_3d), flux_nominal, paths, nominal_layers, loge_grid=loge_grid_minus, cz_midpoints), physics)
    #Fij = NamedTuple(key => safe_div.((weights_plus[key] .- weights_minus[key]), (2*0.02 .* nominal_weights[key])) for key in keys(nominal_weights))
    #
    ## Up/down Fij: downgoing uses nominal R and nominal E, upgoing uses shifted R and shifted E
    #nE = length(midpoints(loge_grid))
    #ncz_half = length(cz_midpoints) ÷ 2
    #ncols_half = nE * ncz_half
    #
    #cz_down = cz_midpoints[1:ncz_half]
    #cz_up = cz_midpoints[ncz_half+1:end]
    #flux_down = flux_nominal[1:ncols_half]
    #flux_up = flux_nominal[ncols_half+1:end]
    #
    #slice_R_cols(R_nt, cols) = NamedTuple(key => R_nt[key][:, cols] for key in keys(R_nt))
    #R_down_nom = slice_R_cols(R, 1:ncols_half)
    #R_up_plus = slice_R_cols(flatten_R(R_plus_3d), ncols_half+1:2*ncols_half)
    #R_up_minus = slice_R_cols(flatten_R(R_minus_3d), ncols_half+1:2*ncols_half)
    #
    #weights_down = calc_weights(params_nominal, (;R=R_down_nom, flux_nominal=flux_down, nominal_layers, loge_grid, cz_midpoints=cz_down), physics)
    #weights_up_plus = calc_weights(params_nominal, (;R=R_up_plus, flux_nominal=flux_up, nominal_layers, loge_grid=loge_grid_plus, cz_midpoints=cz_up), physics)
    #weights_up_minus = calc_weights(params_nominal, (;R=R_up_minus, flux_nominal=flux_up, nominal_layers, loge_grid=loge_grid_minus, cz_midpoints=cz_up), physics)
    #
    #weights_updown_plus = NamedTuple(key => weights_down[key] .+ weights_up_plus[key] for key in keys(weights_down))
    #weights_updown_minus = NamedTuple(key => weights_down[key] .+ weights_up_minus[key] for key in keys(weights_down))
    #Fij_updown = NamedTuple(key => safe_div.((weights_updown_plus[key] .- weights_updown_minus[key]), (2*0.02 .* nominal_weights[key])) for key in keys(nominal_weights))

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


    # Compute per-bin, per-flavor R matrix coverage fraction (sum of R row).
    # Events outside the energy grid have coverage < 1 and should not be reweighted.
    R_coverage = NamedTuple(key => vec(sum(R_3d[key], dims=(2,3))) for key in keys(R_3d))

    return (; MC, R, R_3d, flux_nominal, nominal_layers, loge_grid, cz_grid, cz_midpoints, nominal_weights, observed, bininfo, masks,
              energy_groups_sk_i_iii, energy_groups_sk_iv_v, nutau_nu_frac, nc_nu_frac, R_coverage)

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
        sk_pc_stopping_vs_througoing = 1.0,
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
        sk_pc_stopping_vs_througoing = Normal(1.0, 0.2),
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
    if haskey(assets, :R_coverage)
        # Account for events outside the energy grid: they keep weight 1.0
        # reweighted = MC * [f * (w_new/w_nom) + (1-f)]
        return map((mc, w, nw, cov) -> mc.Counts .* (cov .* safe_div.(w, nw) .+ (1 .- cov)),
                   assets.MC, weights, assets.nominal_weights, assets.R_coverage)
    else
        return map((mc, w, nw) -> mc.Counts .* safe_div.(w, nw), assets.MC, weights, assets.nominal_weights)
    end
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
        (get_double_factor(total, assets.masks.pc_stop, assets.masks.pc_thru, params.sk_pc_stopping_vs_througoing) .- 1) .+
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

# Old F_ij functions (commented out — replaced by reco bin overlap method)
#function get_Fij_factor(Fij, param)
#    factor = 1 .+ Fij .* (1 - param)
#end
#
#function get_Fij_factor_escale(Fij, masks, params)
#    # Split energy scale by SK phase: SK I-III and SK IV-V bins get independent scales
#    1 .+ Fij .* ((1 - params.sk_i_iii_energy_scale) .* masks.sk_i_iii_bins .+
#                  (1 - params.sk_iv_v_energy_scale) .* masks.sk_iv_v_bins)
#end
#
#function get_Fij_factor_updown(Fij_updown, masks, params)
#    # Split up/down energy scale by SK phase
#    1 .+ Fij_updown .* ((1 - params.sk_i_iii_updown_energy_scale) .* masks.sk_i_iii_bins .+
#                         (1 - params.sk_iv_v_updown_energy_scale) .* masks.sk_iv_v_bins)
#end

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
