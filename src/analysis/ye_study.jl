#!/usr/bin/env julia
"""
Estimate the impact of proper layer-dependent Ye (electron fraction) on Δm² fit in Super-K.

The matter effect depends on electron density Ne = Ye × ρ × NA. Previously Ye=0.5
was used for all layers. In reality:
  - Mantle (silicate rock): Ye ≈ 0.494–0.496
  - Core (iron-nickel):     Ye ≈ 0.466–0.468

The core Ye is ~7% lower than 0.5, reducing the matter potential for core-crossing
(upgoing) neutrinos. This script quantifies the effect on Super-K bin predictions
and the resulting shift in best-fit Δm².
"""

using Newtrinos
using StatsBase: midpoints
using Accessors
using CairoMakie
using Printf

# --- Helper: compute bin predictions with a given PREM config ---

function compute_prediction(params, assets, physics, layers)
    E = 10.0 .^ midpoints(assets.loge_grid)
    logE = midpoints(assets.loge_grid)

    paths = physics.earth_layers.compute_paths(assets.cz_midpoints, layers)

    p = physics.osc.osc_prob(E, paths, layers, params)
    p_anti = physics.osc.osc_prob(E, paths, layers, params, anti=true)

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

    # Configure with proper Ye (new default)
    sk = Newtrinos.super_k.configure()
    physics = sk.physics
    assets = sk.assets

    params = Newtrinos.get_params((super_k=sk,))
    @reset params.Δm²₃₁ = 2.475e-3
    @reset params.θ₂₃ = asin(sqrt(0.45))
    @reset params.θ₁₃ = asin(sqrt(0.02))
    @reset params.δCP = -1.89

    # Compute layers with old (Ye=0.5) and new (proper Ye) values
    prem_old = Newtrinos.earth_layers.PREM(p_fractions=[0.5, 0.5, 0.5, 0.5])
    prem_new = Newtrinos.earth_layers.PREM()  # uses new defaults [0.496, 0.494, 0.468, 0.466]
    layers_old = Newtrinos.earth_layers.get_compute_layers(prem_old)()
    layers_new = Newtrinos.earth_layers.get_compute_layers(prem_new)()

    println("\n=== Layer comparison ===")
    println("Layer | Radius   | Old p_den | New p_den | Old n_den | New n_den | Old Ye  | New Ye")
    for i in 1:length(layers_old.radius)
        pd_old = layers_old.p_density[i]
        nd_old = layers_old.n_density[i]
        pd_new = layers_new.p_density[i]
        nd_new = layers_new.n_density[i]
        rho_old = pd_old + nd_old
        rho_new = pd_new + nd_new
        ye_old = rho_old > 0 ? pd_old / rho_old : 0.5
        ye_new = rho_new > 0 ? pd_new / rho_new : 0.5
        @printf("  %2d  | %8.1f | %8.4f  | %8.4f  | %8.4f  | %8.4f  | %6.4f  | %6.4f\n",
                i, layers_old.radius[i], pd_old, pd_new, nd_old, nd_new, ye_old, ye_new)
    end

    # --- Part 1: Compare predictions at nominal Δm² ---
    println("\n=== Part 1: Effect of proper Ye on bin predictions at Δm² = 2.475e-3 ===")

    pred_old = compute_prediction(params, assets, physics, layers_old)
    pred_new = compute_prediction(params, assets, physics, layers_new)
    nbins = length(pred_old)

    diff = pred_new .- pred_old
    frac_diff = diff ./ max.(pred_old, 1e-10)

    println("Total bins: $nbins")
    println("Total events (Ye=0.5):    $(@sprintf("%.1f", sum(pred_old)))")
    println("Total events (proper Ye): $(@sprintf("%.1f", sum(pred_new)))")
    println("ΔN total: $(@sprintf("%+.1f", sum(diff)))")
    println("Max |frac diff|: $(@sprintf("%.5f", maximum(abs.(frac_diff))))")
    println("RMS frac diff:   $(@sprintf("%.5f", sqrt(sum(frac_diff.^2) / nbins)))")

    # --- Part 2: Δm² scan ---
    println("\n=== Part 2: Δm² scan — measuring shift from proper Ye ===")

    dm2_values = LinRange(2.2e-3, 2.7e-3, 101)
    chi2_old = Float64[]
    chi2_new = Float64[]
    obs = assets.observed

    for dm2 in dm2_values
        p_scan = @set params.Δm²₃₁ = dm2
        pred_o = compute_prediction(p_scan, assets, physics, layers_old)
        pred_n = compute_prediction(p_scan, assets, physics, layers_new)

        c2_o = sum((pred_o .- obs).^2 ./ max.(pred_o, 1.0))
        c2_n = sum((pred_n .- obs).^2 ./ max.(pred_n, 1.0))
        push!(chi2_old, c2_o)
        push!(chi2_new, c2_n)
    end

    idx_min_old = argmin(chi2_old)
    idx_min_new = argmin(chi2_new)
    dm2_best_old = dm2_values[idx_min_old]
    dm2_best_new = dm2_values[idx_min_new]

    println("Best-fit Δm² (Ye=0.5):    $(@sprintf("%.4e", dm2_best_old)) eV²")
    println("Best-fit Δm² (proper Ye): $(@sprintf("%.4e", dm2_best_new)) eV²")
    println("Shift: $(@sprintf("%+.4e", dm2_best_new - dm2_best_old)) eV²")
    println("Shift: $(@sprintf("%+.3f", (dm2_best_new - dm2_best_old)*1e3)) × 10⁻³ eV²")

    # --- Part 3: What Δm² shift in old model best matches new prediction? ---
    println("\n=== Part 3: What Δm² shift compensates for proper Ye? ===")
    target = pred_new
    best_dm2 = 0.0
    best_chi2 = Inf
    for dm2 in LinRange(2.2e-3, 2.7e-3, 501)
        p_scan = @set params.Δm²₃₁ = dm2
        pred = compute_prediction(p_scan, assets, physics, layers_old)
        c2 = sum((pred .- target).^2 ./ max.(target, 1.0))
        if c2 < best_chi2
            best_chi2 = c2
            best_dm2 = dm2
        end
    end
    shift = best_dm2 - 2.475e-3
    println("Δm² in old model that best matches new prediction: $(@sprintf("%.4e", best_dm2))")
    println("Shift: $(@sprintf("%+.4e", shift)) eV² = $(@sprintf("%+.3f", shift*1e3)) × 10⁻³ eV²")

    # --- Part 4: osc_prob comparison for core-crossing trajectories ---
    println("\n=== Part 4: Oscillation probability differences ===")

    E = 10.0 .^ midpoints(assets.loge_grid)
    paths_old = physics.earth_layers.compute_paths(assets.cz_midpoints, layers_old)
    paths_new = physics.earth_layers.compute_paths(assets.cz_midpoints, layers_new)

    p_old = physics.osc.osc_prob(E, paths_old, layers_old, params)
    p_new = physics.osc.osc_prob(E, paths_new, layers_new, params)
    p_anti_old = physics.osc.osc_prob(E, paths_old, layers_old, params, anti=true)
    p_anti_new = physics.osc.osc_prob(E, paths_new, layers_new, params, anti=true)

    # νμ→νμ survival probability
    pdiff_numu = p_new[:, :, 2, 2] .- p_old[:, :, 2, 2]
    println("P(νμ→νμ) max |diff|: $(@sprintf("%.5f", maximum(abs.(pdiff_numu))))")
    println("P(νμ→νμ) mean diff:  $(@sprintf("%+.5f", mean(pdiff_numu)))")

    # --- Plot ---
    println("\nGenerating plots...")

    fig = Figure(size=(1400, 1000))

    # Plot 1: χ² scan
    ax1 = Axis(fig[1, 1], xlabel="Δm²₃₁ [eV²]", ylabel="Δχ²",
               title="χ² scan: Ye=0.5 vs proper Ye")
    lines!(ax1, collect(dm2_values), chi2_old .- minimum(chi2_old),
           label="Ye = 0.5 (old)", color=:blue)
    lines!(ax1, collect(dm2_values), chi2_new .- minimum(chi2_new),
           label="Ye per layer (new)", color=:red)
    vlines!(ax1, [dm2_best_old], color=:blue, linestyle=:dash, alpha=0.5)
    vlines!(ax1, [dm2_best_new], color=:red, linestyle=:dash, alpha=0.5)
    axislegend(ax1, position=:rt)

    # Plot 2: Fractional bin prediction difference
    ax2 = Axis(fig[1, 2], xlabel="Bin index", ylabel="(new - old) / old",
               title="Fractional change in bin predictions")
    lines!(ax2, 1:nbins, frac_diff, color=:red)
    hlines!(ax2, [0.0], color=:black, linestyle=:dash, alpha=0.3)

    # Plot 3: P(νμ→νμ) difference map (E vs cos θ)
    logE = midpoints(assets.loge_grid)
    cz = assets.cz_midpoints
    ax3 = Axis(fig[2, 1], xlabel="cos(θ)", ylabel="log₁₀(E/GeV)",
               title="ΔP(νμ→νμ) = P(proper Ye) − P(Ye=0.5)")
    hm = heatmap!(ax3, collect(cz), collect(logE), pdiff_numu', colormap=:RdBu)
    Colorbar(fig[2, 1][1, 2], hm)

    # Plot 4: P(νμ→νμ) slice at cos θ ≈ -0.9 (core-crossing)
    cz_idx = argmin(abs.(cz .- (-0.9)))
    ax4 = Axis(fig[2, 2], xlabel="log₁₀(E/GeV)", ylabel="P(νμ→νμ)",
               title="P(νμ→νμ) at cos(θ) ≈ $(@sprintf("%.2f", cz[cz_idx]))")
    lines!(ax4, collect(logE), p_old[:, cz_idx, 2, 2], label="Ye = 0.5", color=:blue, alpha=0.7)
    lines!(ax4, collect(logE), p_new[:, cz_idx, 2, 2], label="Proper Ye", color=:red, linewidth=2)
    axislegend(ax4, position=:rb)

    outpath = joinpath(@__DIR__, "ye_study.png")
    save(outpath, fig, px_per_unit=2)
    println("Saved plot to $outpath")

    println("\nDone.")
end

main()
