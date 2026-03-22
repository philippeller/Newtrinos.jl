#!/usr/bin/env julia
"""
Estimate the impact of oscillation probability smoothing on Δm² fit in Super-K.

SK averages osc_prob over 5 energy points × 20 production heights per event
(Wester 2023, section 5.3.1). We approximate this with a Gaussian low-pass
filter on the osc_prob grid and measure how it shifts the effective Δm².
"""

using Newtrinos
using StatsBase: midpoints
using Accessors
using CairoMakie
using Printf

# --- Gaussian smoothing kernel (separable, no external deps) ---

function gaussian_kernel_1d(σ, T=Float64)
    σ <= 0 && return T[1.0]
    hw = ceil(Int, 3σ)
    k = T[exp(-0.5 * (i / σ)^2) for i in -hw:hw]
    k ./= sum(k)
    return k
end

"""
    smooth_1d!(out, x, kernel)

1D convolution along first dimension with edge-padding, in-place into `out`.
"""
function smooth_1d!(out, x, kernel)
    hw = length(kernel) ÷ 2
    n = size(x, 1)
    m = size(x, 2)
    @inbounds for j in 1:m
        for i in 1:n
            s = zero(eltype(x))
            for k in -hw:hw
                idx = clamp(i + k, 1, n)
                s += kernel[k + hw + 1] * x[idx, j]
            end
            out[i, j] = s
        end
    end
    return out
end

"""
    smooth_osc_prob(p, σ_E, σ_cz)

Apply separable Gaussian smoothing to osc_prob array p (nE, ncz, nf, nf).
σ_E, σ_cz are in grid-bin units.
"""
function smooth_osc_prob(p, σ_E, σ_cz)
    nE, ncz, nf1, nf2 = size(p)
    ps = similar(p)
    tmp = similar(p, nE, ncz)
    kE = gaussian_kernel_1d(σ_E)
    kcz = gaussian_kernel_1d(σ_cz)
    for b in 1:nf2, a in 1:nf1
        slice = @view p[:, :, a, b]
        smooth_1d!(tmp, slice, kE)  # smooth along E
        if σ_cz > 0
            tmp2 = similar(tmp)
            # smooth along cz (dim 2) — transpose trick
            smooth_1d!(tmp2', tmp', kcz)
            ps[:, :, a, b] .= tmp2
        else
            ps[:, :, a, b] .= tmp
        end
    end
    return ps
end

# --- Helper: compute bin predictions with optional osc_prob smoothing ---

function compute_prediction(params, assets, physics; σ_E=0.0, σ_cz=0.0)
    E = 10.0 .^ midpoints(assets.loge_grid)
    logE = midpoints(assets.loge_grid)

    layers = haskey(params, :electron_density_scale) ?
        Newtrinos.earth_layers.scale_densities(assets.nominal_layers, params.electron_density_scale) :
        assets.nominal_layers
    paths = physics.earth_layers.compute_paths(assets.cz_midpoints, layers)

    p = physics.osc.osc_prob(E, paths, layers, params)
    p_anti = physics.osc.osc_prob(E, paths, layers, params, anti=true)

    # Apply smoothing
    if σ_E > 0 || σ_cz > 0
        p = smooth_osc_prob(p, σ_E, σ_cz)
        p_anti = smooth_osc_prob(p_anti, σ_E, σ_cz)
    end

    flux = physics.atm_flux.sys_flux(assets.flux_nominal, params)
    s = (size(p, 1), size(p, 2))

    fnl = haskey(params, :sk_flux_norm_low) ? params.sk_flux_norm_low : zero(eltype(E))
    fnh = haskey(params, :sk_flux_norm_high) ? params.sk_flux_norm_high : zero(eltype(E))
    flux_norm = 1 .+ fnl .* Newtrinos.super_k.flux_norm_sigma_low.(logE) .+
                      fnh .* Newtrinos.super_k.flux_norm_sigma_high.(logE)

    xsec_nue     = physics.xsec.scale(E, :nue,   :CC, false, params)
    xsec_numu    = physics.xsec.scale(E, :numu,  :CC, false, params)
    xsec_nutau   = physics.xsec.scale(E, :nutau, :CC, false, params)
    xsec_nuebar  = physics.xsec.scale(E, :nue,   :CC, true,  params)
    xsec_numubar = physics.xsec.scale(E, :numu,  :CC, true,  params)
    xsec_nutaubar= physics.xsec.scale(E, :nutau, :CC, true,  params)
    xsec_nc      = physics.xsec.scale(E, :nue,   :NC, false, params)

    nue_flux   = (reshape(flux.nue,    s) .* p[:, :, 1, 1] .+
                  reshape(flux.numu,   s) .* p[:, :, 2, 1]) .* xsec_nue .* flux_norm
    numu_flux  = (reshape(flux.nue,    s) .* p[:, :, 1, 2] .+
                  reshape(flux.numu,   s) .* p[:, :, 2, 2]) .* xsec_numu .* flux_norm
    nutau_flux = (reshape(flux.nue,    s) .* p[:, :, 1, 3] .+
                  reshape(flux.numu,   s) .* p[:, :, 2, 3]) .* xsec_nutau .* flux_norm
    nuebar_flux  = (reshape(flux.nuebar,  s) .* p_anti[:, :, 1, 1] .+
                    reshape(flux.numubar, s) .* p_anti[:, :, 2, 1]) .* xsec_nuebar .* flux_norm
    numubar_flux = (reshape(flux.nuebar,  s) .* p_anti[:, :, 1, 2] .+
                    reshape(flux.numubar, s) .* p_anti[:, :, 2, 2]) .* xsec_numubar .* flux_norm
    nutaubar_flux = (reshape(flux.nuebar,  s) .* p_anti[:, :, 1, 3] .+
                     reshape(flux.numubar, s) .* p_anti[:, :, 2, 3]) .* xsec_nutaubar .* flux_norm

    R = assets.R
    nue     = R.nue * vec(nue_flux)
    numu    = R.numu * vec(numu_flux)
    nutau   = R.nutau * vec(nutau_flux)
    nuebar  = R.nuebar * vec(nuebar_flux)
    numubar = R.numubar * vec(numubar_flux)
    nunc    = R.nunc * vec(ones(eltype(nue_flux), s) .* xsec_nc .* flux_norm)

    return nue .+ numu .+ nutau .+ nuebar .+ numubar .+ nunc
end

# --- Main ---

function main()
    println("Configuring Super-K...")
    sk = Newtrinos.super_k.configure()
    physics = sk.physics
    assets = sk.assets

    # Nominal parameters (SK bestfit)
    params = Newtrinos.get_params((super_k=sk,))
    @reset params.Δm²₃₁ = 2.475e-3
    @reset params.θ₂₃ = asin(sqrt(0.45))
    @reset params.θ₁₃ = asin(sqrt(0.02))
    @reset params.δCP = -1.89

    # --- Part 1: Compare smoothed vs unsmoothed at nominal Δm² ---
    println("\n=== Part 1: Effect of smoothing at nominal Δm² = 2.475e-3 ===")

    pred_nominal = compute_prediction(params, assets, physics)
    nbins = length(pred_nominal)
    println("Total bins: $nbins")
    println("Total predicted events (no smoothing): $(@sprintf("%.1f", sum(pred_nominal)))")

    smoothing_configs = [
        (σ_E=1.0, σ_cz=0.0, label="σ_E=1 bin (2% logE)"),
        (σ_E=2.0, σ_cz=0.0, label="σ_E=2 bins (4% logE)"),
        (σ_E=3.0, σ_cz=0.0, label="σ_E=3 bins (6% logE)"),
        (σ_E=5.0, σ_cz=0.0, label="σ_E=5 bins (10% logE)"),
        (σ_E=2.0, σ_cz=1.0, label="σ_E=2, σ_cz=1"),
        (σ_E=3.0, σ_cz=1.0, label="σ_E=3, σ_cz=1"),
        (σ_E=5.0, σ_cz=2.0, label="σ_E=5, σ_cz=2"),
    ]

    for cfg in smoothing_configs
        pred_smooth = compute_prediction(params, assets, physics; σ_E=cfg.σ_E, σ_cz=cfg.σ_cz)
        diff = pred_smooth .- pred_nominal
        frac_diff = diff ./ max.(pred_nominal, 1e-10)
        println("\n$(cfg.label):")
        println("  Total events: $(@sprintf("%.1f", sum(pred_smooth)))  (Δ = $(@sprintf("%+.1f", sum(diff))))")
        println("  Max |frac diff|: $(@sprintf("%.4f", maximum(abs.(frac_diff))))")
        println("  RMS frac diff:   $(@sprintf("%.4f", sqrt(sum(frac_diff.^2) / nbins)))")
        println("  Mean frac diff:  $(@sprintf("%+.4f", sum(frac_diff) / nbins))")
    end

    # --- Part 2: Δm² scan to find the shift ---
    println("\n\n=== Part 2: Δm² scan — finding effective shift from smoothing ===")

    # Use σ_E=3 as the baseline smoothing (moderate, ~6% in logE)
    σ_E_ref = 3.0
    σ_cz_ref = 1.0
    println("Reference smoothing: σ_E=$(σ_E_ref), σ_cz=$(σ_cz_ref)")

    pred_smooth_ref = compute_prediction(params, assets, physics; σ_E=σ_E_ref, σ_cz=σ_cz_ref)

    # Scan Δm² values
    dm2_values = LinRange(2.2e-3, 2.7e-3, 51)
    chi2_unsmoothed = Float64[]
    chi2_smoothed = Float64[]

    for dm2 in dm2_values
        p_scan = @set params.Δm²₃₁ = dm2
        pred_unsm = compute_prediction(p_scan, assets, physics)
        pred_sm = compute_prediction(p_scan, assets, physics; σ_E=σ_E_ref, σ_cz=σ_cz_ref)

        # Simple χ² against observed data
        obs = assets.observed
        c2_unsm = sum((pred_unsm .- obs).^2 ./ max.(pred_unsm, 1.0))
        c2_sm   = sum((pred_sm .- obs).^2 ./ max.(pred_sm, 1.0))
        push!(chi2_unsmoothed, c2_unsm)
        push!(chi2_smoothed, c2_sm)
    end

    # Find minima
    idx_min_unsm = argmin(chi2_unsmoothed)
    idx_min_sm = argmin(chi2_smoothed)
    dm2_best_unsm = dm2_values[idx_min_unsm]
    dm2_best_sm = dm2_values[idx_min_sm]

    println("\nBest-fit Δm² (unsmoothed): $(@sprintf("%.4e", dm2_best_unsm)) eV²")
    println("Best-fit Δm² (smoothed):   $(@sprintf("%.4e", dm2_best_sm)) eV²")
    println("Shift: $(@sprintf("%+.4e", dm2_best_sm - dm2_best_unsm)) eV²")
    println("Shift: $(@sprintf("%+.2f", (dm2_best_sm - dm2_best_unsm)*1e3)) × 10⁻³ eV²")

    # --- Part 3: Compare smoothed prediction vs Δm²-shifted unsmoothed ---
    println("\n\n=== Part 3: What Δm² shift in unsmoothed model matches the smoothed prediction? ===")

    for (σ_E, σ_cz, label) in [(2.0, 0.0, "σ_E=2"), (3.0, 1.0, "σ_E=3,σ_cz=1"), (5.0, 2.0, "σ_E=5,σ_cz=2")]
        target = compute_prediction(params, assets, physics; σ_E=σ_E, σ_cz=σ_cz)

        best_dm2 = 0.0
        best_chi2 = Inf
        for dm2 in LinRange(2.2e-3, 2.7e-3, 101)
            p_scan = @set params.Δm²₃₁ = dm2
            pred = compute_prediction(p_scan, assets, physics)
            c2 = sum((pred .- target).^2 ./ max.(target, 1.0))
            if c2 < best_chi2
                best_chi2 = c2
                best_dm2 = dm2
            end
        end
        shift = best_dm2 - 2.475e-3
        println("$label: best-match Δm² = $(@sprintf("%.4e", best_dm2))  (shift = $(@sprintf("%+.4e", shift)) = $(@sprintf("%+.2f", shift*1e3))×10⁻³)")
    end

    # --- Part 4: Plots ---
    println("\n\nGenerating plots...")

    fig = Figure(size=(1200, 800))

    # Plot 1: χ² vs Δm²
    ax1 = Axis(fig[1, 1], xlabel="Δm²₃₁ [eV²]", ylabel="χ²",
               title="χ² scan: smoothed vs unsmoothed")
    lines!(ax1, collect(dm2_values), chi2_unsmoothed .- minimum(chi2_unsmoothed),
           label="Unsmoothed", color=:blue)
    lines!(ax1, collect(dm2_values), chi2_smoothed .- minimum(chi2_smoothed),
           label="Smoothed (σ_E=$σ_E_ref, σ_cz=$σ_cz_ref)", color=:red)
    vlines!(ax1, [dm2_best_unsm], color=:blue, linestyle=:dash, alpha=0.5)
    vlines!(ax1, [dm2_best_sm], color=:red, linestyle=:dash, alpha=0.5)
    axislegend(ax1, position=:rt)

    # Plot 2: Fractional difference by bin
    ax2 = Axis(fig[1, 2], xlabel="Bin index", ylabel="Fractional difference",
               title="(Smoothed - Nominal) / Nominal")
    for (σ_E, σ_cz, col, label) in [(2.0, 0.0, :green, "σ_E=2"),
                                      (3.0, 1.0, :orange, "σ_E=3,cz=1"),
                                      (5.0, 2.0, :red, "σ_E=5,cz=2")]
        pred_sm = compute_prediction(params, assets, physics; σ_E=σ_E, σ_cz=σ_cz)
        frac = (pred_sm .- pred_nominal) ./ max.(pred_nominal, 1e-10)
        lines!(ax2, 1:nbins, frac, label=label, color=col)
    end
    hlines!(ax2, [0.0], color=:black, linestyle=:dash, alpha=0.3)
    axislegend(ax2, position=:rt)

    # Plot 3: osc_prob slice before/after smoothing at cz=-0.5
    E = 10.0 .^ midpoints(assets.loge_grid)
    logE = midpoints(assets.loge_grid)
    layers = assets.nominal_layers
    paths = physics.earth_layers.compute_paths(assets.cz_midpoints, layers)

    p_raw = physics.osc.osc_prob(E, paths, layers, params)
    p_sm3 = smooth_osc_prob(p_raw, 3.0, 1.0)
    p_sm5 = smooth_osc_prob(p_raw, 5.0, 2.0)

    # cz=-0.5 → index 25 (out of 100 cz bins, cz from -1 to 1)
    cz_idx = 25
    ax3 = Axis(fig[2, 1], xlabel="log₁₀(E/GeV)", ylabel="P(νμ→νμ)",
               title="P(νμ→νμ) at cos(θ) ≈ $(@sprintf("%.2f", assets.cz_midpoints[cz_idx]))")
    lines!(ax3, collect(logE), p_raw[:, cz_idx, 2, 2], label="Raw", color=:blue, alpha=0.7)
    lines!(ax3, collect(logE), p_sm3[:, cz_idx, 2, 2], label="σ_E=3, σ_cz=1", color=:orange, linewidth=2)
    lines!(ax3, collect(logE), p_sm5[:, cz_idx, 2, 2], label="σ_E=5, σ_cz=2", color=:red, linewidth=2)
    axislegend(ax3, position=:rb)

    # Plot 4: same at cz=-0.9 (more upgoing)
    cz_idx2 = 5
    ax4 = Axis(fig[2, 2], xlabel="log₁₀(E/GeV)", ylabel="P(νμ→νμ)",
               title="P(νμ→νμ) at cos(θ) ≈ $(@sprintf("%.2f", assets.cz_midpoints[cz_idx2]))")
    lines!(ax4, collect(logE), p_raw[:, cz_idx2, 2, 2], label="Raw", color=:blue, alpha=0.7)
    lines!(ax4, collect(logE), p_sm3[:, cz_idx2, 2, 2], label="σ_E=3, σ_cz=1", color=:orange, linewidth=2)
    lines!(ax4, collect(logE), p_sm5[:, cz_idx2, 2, 2], label="σ_E=5, σ_cz=2", color=:red, linewidth=2)
    axislegend(ax4, position=:rb)

    outpath = joinpath(@__DIR__, "osc_smoothing_study.png")
    save(outpath, fig, px_per_unit=2)
    println("Saved plot to $outpath")

    println("\nDone.")
end

main()
