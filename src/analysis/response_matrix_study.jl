#!/usr/bin/env julia
"""
Compare spline vs split-Gaussian response matrix construction for Super-K.

The current approach fits RQSpline CDFs through 5 quantile points + 2 linearly
extrapolated tail points. This study compares against a split-Gaussian CDF
constructed from the same quantiles, to check whether the spline tails
underestimate smearing.
"""

using Newtrinos
using StatsBase: midpoints
using Accessors
using CairoMakie
using Printf
using Distributions: Normal, cdf as dist_cdf

# ── Split-Gaussian CDF from quantiles ──────────────────────────────────────

"""
    make_split_gaussian_log_e_cdf(bin)

Build a split-Gaussian CDF in log10(E) from the 5 quantile points.
μ = Q50 in log10, σ_low from lower quantiles, σ_high from upper quantiles.
"""
function make_split_gaussian_log_e_cdf(bin)
    q = log10.([bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent,
                bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent,
                bin.EnergyQuantile97_7Percent])

    μ = q[3]
    # Average of 1σ and 2σ estimates for each side
    σ_low  = 0.5 * ((q[3] - q[2]) + (q[3] - q[1]) / 2)
    σ_high = 0.5 * ((q[4] - q[3]) + (q[5] - q[3]) / 2)

    # Guard against degenerate bins
    σ_low  = max(σ_low, 1e-6)
    σ_high = max(σ_high, 1e-6)

    return x -> begin
        if x <= μ
            dist_cdf(Normal(μ, σ_low), x)
        else
            dist_cdf(Normal(μ, σ_high), x)
        end
    end
end

"""
    make_split_gaussian_cosz_cdf(bin)

Build a split-Gaussian CDF in cos(zenith) from the 5 quantile points.
"""
function make_split_gaussian_cosz_cdf(bin)
    q = [bin.CosZQuantile2_3Percent, bin.CosZQuantile15_9Percent,
         bin.CosZQuantile50_0Percent, bin.CosZQuantile84_1Percent,
         bin.CosZQuantile97_7Percent]

    μ = q[3]
    σ_low  = 0.5 * ((q[3] - q[2]) + (q[3] - q[1]) / 2)
    σ_high = 0.5 * ((q[4] - q[3]) + (q[5] - q[3]) / 2)
    σ_low  = max(σ_low, 1e-6)
    σ_high = max(σ_high, 1e-6)

    return x -> begin
        if x <= μ
            dist_cdf(Normal(μ, σ_low), x)
        else
            dist_cdf(Normal(μ, σ_high), x)
        end
    end
end

# ── Response matrix with split-Gaussian CDFs ───────────────────────────────

function make_response_matrix_splitgauss(MC_component, logE_grid, cosZ_grid)
    n_bins = size(MC_component, 1)
    n_logE = length(logE_grid)
    n_cosZ = length(cosZ_grid)

    response_matrix = zeros(Float64, n_bins, n_logE-1, n_cosZ-1)

    for bin_idx in 1:n_bins
        bin = MC_component[bin_idx, :]
        bin.Counts == 0 && continue

        log_e_cdf = make_split_gaussian_log_e_cdf(bin)
        cosz_cdf = make_split_gaussian_cosz_cdf(bin)

        c_e = log_e_cdf.(logE_grid)
        p_e = diff(c_e)

        c_cosz = cosz_cdf.(cosZ_grid)
        p_cosz = diff(c_cosz)

        response_matrix[bin_idx, :, :] .= p_e * p_cosz'

        s = sum(response_matrix[bin_idx, :, :])
        s == 0 && continue
        response_matrix[bin_idx, :, :] ./= s
    end
    return response_matrix
end

# ── Helper: numerical mean & RMS from CDF on a grid ────────────────────────

function cdf_mean_rms(cdf_func, grid)
    # PDF from CDF differences
    c = cdf_func.(grid)
    p = diff(c)
    mids = midpoints(grid)
    # Normalize (in case CDF doesn't span [0,1] on grid)
    w = sum(p)
    w < 1e-12 && return (NaN, NaN)
    μ = sum(mids .* p) / w
    σ = sqrt(max(0.0, sum((mids .- μ).^2 .* p) / w))
    return (μ, σ)
end

# ── Main ───────────────────────────────────────────────────────────────────

function main()
    println("Configuring Super-K...")
    sk = Newtrinos.super_k.configure()
    physics = sk.physics
    assets = sk.assets
    MC = assets.MC

    params = Newtrinos.get_params((super_k=sk,))
    @reset params.Δm²₃₁ = 2.475e-3
    @reset params.θ₂₃ = asin(sqrt(0.45))
    @reset params.θ₁₃ = asin(sqrt(0.02))
    @reset params.δCP = -1.89

    loge_grid = assets.loge_grid
    cz_grid = assets.cz_grid

    # ── Part 1: CDF/PDF comparison for representative bins ─────────────

    println("\n=== Part 1: CDF/PDF comparison for representative bins ===")

    # Pick representative bins from nue MC component (index into the 930-row table)
    bininfo = assets.bininfo
    # Choose bins spanning different sample types and resolution quality
    representative = [
        (idx=1,   label="Sub-GeV e-like (bin 1)"),
        (idx=50,  label="Sub-GeV μ-like (bin 50)"),
        (idx=200, label="Multi-GeV 1-ring (bin 200)"),
        (idx=400, label="Multi-GeV multi-ring (bin 400)"),
        (idx=700, label="PC through-going (bin 700)"),
        (idx=900, label="Upmu (bin 900)"),
    ]
    # Filter to bins with nonzero counts in nue MC
    representative = filter(r -> MC.numu[r.idx, :].Counts > 0, representative)
    nrep = length(representative)
    println("Using $nrep representative bins")

    # Fine grids for CDF/PDF evaluation
    loge_fine = LinRange(-1.5, 3.5, 500)
    cosz_fine = LinRange(-1.0, 1.0, 500)

    fig = Figure(size=(1600, 3600))

    # Rows 1-2: Energy CDF, Rows 3-4: Energy PDF
    for (i, rep) in enumerate(representative)
        bin = MC.numu[rep.idx, :]

        # Build energy CDFs
        spline_cdf = Newtrinos.super_k.make_log_e_cdf(bin)
        gauss_cdf = make_split_gaussian_log_e_cdf(bin)

        # Quantile data points
        q_vals = log10.([bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent,
                         bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent,
                         bin.EnergyQuantile97_7Percent])
        q_probs = [0.023, 0.159, 0.5, 0.841, 0.977]

        # Energy CDF plot
        col = (i - 1) % 3 + 1
        row = (i - 1) ÷ 3 + 1
        ax_cdf = Axis(fig[row, col], xlabel="log₁₀(E/GeV)", ylabel="CDF",
                       title=rep.label, titlesize=12)
        cdf_s = spline_cdf.(loge_fine)
        cdf_g = gauss_cdf.(loge_fine)
        lines!(ax_cdf, collect(loge_fine), cdf_s, label="RQSpline", color=:blue, linewidth=2)
        lines!(ax_cdf, collect(loge_fine), cdf_g, label="Split-Gaussian", color=:red, linewidth=2)
        scatter!(ax_cdf, q_vals, q_probs, color=:black, markersize=8, label="Quantile data")
        if i == 1
            axislegend(ax_cdf, position=:rb, labelsize=10)
        end

        # Energy PDF plot (numerical derivative)
        Δx = Float64(loge_fine[2] - loge_fine[1])
        pdf_s = diff(cdf_s) ./ Δx
        pdf_g = diff(cdf_g) ./ Δx
        pdf_x = collect(midpoints(loge_fine))

        ax_pdf = Axis(fig[row + 2, col], xlabel="log₁₀(E/GeV)", ylabel="PDF",
                       title=rep.label * " (PDF)", titlesize=12)
        lines!(ax_pdf, pdf_x, pdf_s, label="RQSpline", color=:blue, linewidth=2)
        lines!(ax_pdf, pdf_x, pdf_g, label="Split-Gaussian", color=:red, linewidth=2)
        if i == 1
            axislegend(ax_pdf, position=:rt, labelsize=10)
        end
    end

    Label(fig[0, :], "Part 1a: CDF and PDF comparison (energy, νμ MC)", fontsize=16, font=:bold)

    # Rows 5-6: CosZ CDF, Rows 7-8: CosZ PDF
    for (i, rep) in enumerate(representative)
        bin = MC.numu[rep.idx, :]

        # Build cosZ CDFs
        spline_cdf_cz = Newtrinos.super_k.make_cosz_cdf(bin)
        gauss_cdf_cz = make_split_gaussian_cosz_cdf(bin)

        # CosZ quantile data points
        q_cz_vals = [bin.CosZQuantile2_3Percent, bin.CosZQuantile15_9Percent,
                     bin.CosZQuantile50_0Percent, bin.CosZQuantile84_1Percent,
                     bin.CosZQuantile97_7Percent]
        q_probs = [0.023, 0.159, 0.5, 0.841, 0.977]

        col = (i - 1) % 3 + 1
        row_offset = 4
        row = (i - 1) ÷ 3 + 1 + row_offset

        # CosZ CDF plot
        ax_cdf_cz = Axis(fig[row, col], xlabel="cos(θ)", ylabel="CDF",
                          title=rep.label, titlesize=12)
        cdf_s_cz = spline_cdf_cz.(cosz_fine)
        cdf_g_cz = gauss_cdf_cz.(cosz_fine)
        lines!(ax_cdf_cz, collect(cosz_fine), cdf_s_cz, label="RQSpline", color=:blue, linewidth=2)
        lines!(ax_cdf_cz, collect(cosz_fine), cdf_g_cz, label="Split-Gaussian", color=:red, linewidth=2)
        scatter!(ax_cdf_cz, q_cz_vals, q_probs, color=:black, markersize=8, label="Quantile data")
        if i == 1
            axislegend(ax_cdf_cz, position=:rb, labelsize=10)
        end

        # CosZ PDF plot
        Δcz = Float64(cosz_fine[2] - cosz_fine[1])
        pdf_s_cz = diff(cdf_s_cz) ./ Δcz
        pdf_g_cz = diff(cdf_g_cz) ./ Δcz
        pdf_x_cz = collect(midpoints(cosz_fine))

        ax_pdf_cz = Axis(fig[row + 2, col], xlabel="cos(θ)", ylabel="PDF",
                          title=rep.label * " (PDF)", titlesize=12)
        lines!(ax_pdf_cz, pdf_x_cz, pdf_s_cz, label="RQSpline", color=:blue, linewidth=2)
        lines!(ax_pdf_cz, pdf_x_cz, pdf_g_cz, label="Split-Gaussian", color=:red, linewidth=2)
        if i == 1
            axislegend(ax_pdf_cz, position=:rt, labelsize=10)
        end
    end

    Label(fig[5, :], "Part 1b: CDF and PDF comparison (cos θ, νμ MC)", fontsize=16, font=:bold)

    # ── Part 2: Mean/RMS validation ────────────────────────────────────

    println("\n=== Part 2: Mean/RMS validation ===")

    loge_eval = LinRange(-1.5, 3.5, 1000)
    cosz_eval = LinRange(-1.0, 1.0, 1000)
    mc_keys = keys(MC)

    # Energy
    mean_diff_spline = Float64[]
    mean_diff_gauss = Float64[]
    rms_diff_spline = Float64[]
    rms_diff_gauss = Float64[]

    # CosZ
    cz_mean_diff_spline = Float64[]
    cz_mean_diff_gauss = Float64[]
    cz_rms_diff_spline = Float64[]
    cz_rms_diff_gauss = Float64[]

    for key in mc_keys
        mc = MC[key]
        for bin_idx in 1:size(mc, 1)
            bin = mc[bin_idx, :]
            bin.Counts == 0 && continue

            # Energy validation
            true_mean = log10(bin.EnergyAvg)
            true_rms = bin.EnergyRMS / bin.EnergyAvg / log(10)  # Approximate RMS in log10 space

            s_cdf = Newtrinos.super_k.make_log_e_cdf(bin)
            s_mean, s_rms = cdf_mean_rms(s_cdf, loge_eval)

            g_cdf = make_split_gaussian_log_e_cdf(bin)
            g_mean, g_rms = cdf_mean_rms(g_cdf, loge_eval)

            if !isnan(s_mean) && !isnan(g_mean)
                push!(mean_diff_spline, s_mean - true_mean)
                push!(mean_diff_gauss, g_mean - true_mean)
                push!(rms_diff_spline, s_rms - true_rms)
                push!(rms_diff_gauss, g_rms - true_rms)
            end

            # CosZ validation
            true_cz_mean = bin.CosZAvg
            true_cz_rms = bin.CosZRMS

            s_cdf_cz = Newtrinos.super_k.make_cosz_cdf(bin)
            s_cz_mean, s_cz_rms = cdf_mean_rms(s_cdf_cz, cosz_eval)

            g_cdf_cz = make_split_gaussian_cosz_cdf(bin)
            g_cz_mean, g_cz_rms = cdf_mean_rms(g_cdf_cz, cosz_eval)

            if !isnan(s_cz_mean) && !isnan(g_cz_mean)
                push!(cz_mean_diff_spline, s_cz_mean - true_cz_mean)
                push!(cz_mean_diff_gauss, g_cz_mean - true_cz_mean)
                push!(cz_rms_diff_spline, s_cz_rms - true_cz_rms)
                push!(cz_rms_diff_gauss, g_cz_rms - true_cz_rms)
            end
        end
    end

    println("  Energy — Bins analyzed: $(length(mean_diff_spline))")
    println("  Spline mean offset: $(@sprintf("%.4f ± %.4f", mean(mean_diff_spline), std(mean_diff_spline)))")
    println("  Gauss  mean offset: $(@sprintf("%.4f ± %.4f", mean(mean_diff_gauss), std(mean_diff_gauss)))")
    println("  Spline RMS offset:  $(@sprintf("%.4f ± %.4f", mean(rms_diff_spline), std(rms_diff_spline)))")
    println("  Gauss  RMS offset:  $(@sprintf("%.4f ± %.4f", mean(rms_diff_gauss), std(rms_diff_gauss)))")

    println("  CosZ — Bins analyzed: $(length(cz_mean_diff_spline))")
    println("  Spline mean offset: $(@sprintf("%.4f ± %.4f", mean(cz_mean_diff_spline), std(cz_mean_diff_spline)))")
    println("  Gauss  mean offset: $(@sprintf("%.4f ± %.4f", mean(cz_mean_diff_gauss), std(cz_mean_diff_gauss)))")
    println("  Spline RMS offset:  $(@sprintf("%.4f ± %.4f", mean(cz_rms_diff_spline), std(cz_rms_diff_spline)))")
    println("  Gauss  RMS offset:  $(@sprintf("%.4f ± %.4f", mean(cz_rms_diff_gauss), std(cz_rms_diff_gauss)))")

    # Energy Mean/RMS validation histograms
    ax_mean = Axis(fig[9, 1], xlabel="CDF mean - log₁₀(EnergyAvg)", ylabel="Count",
                    title="Energy mean validation", titlesize=12)
    hist!(ax_mean, mean_diff_spline, bins=100, color=(:blue, 0.5), label="RQSpline")
    hist!(ax_mean, mean_diff_gauss, bins=100, color=(:red, 0.5), label="Split-Gaussian")
    axislegend(ax_mean, position=:rt, labelsize=10)

    ax_rms = Axis(fig[9, 2], xlabel="CDF RMS - approx true RMS (log₁₀)", ylabel="Count",
                   title="Energy RMS validation", titlesize=12)
    hist!(ax_rms, rms_diff_spline, bins=100, color=(:blue, 0.5), label="RQSpline")
    hist!(ax_rms, rms_diff_gauss, bins=100, color=(:red, 0.5), label="Split-Gaussian")
    axislegend(ax_rms, position=:rt, labelsize=10)

    # CosZ Mean/RMS validation histograms
    ax_cz_mean = Axis(fig[10, 1], xlabel="CDF mean - CosZAvg", ylabel="Count",
                       title="CosZ mean validation", titlesize=12)
    hist!(ax_cz_mean, cz_mean_diff_spline, bins=100, color=(:blue, 0.5), label="RQSpline")
    hist!(ax_cz_mean, cz_mean_diff_gauss, bins=100, color=(:red, 0.5), label="Split-Gaussian")
    axislegend(ax_cz_mean, position=:rt, labelsize=10)

    ax_cz_rms = Axis(fig[10, 2], xlabel="CDF RMS - CosZRMS", ylabel="Count",
                      title="CosZ RMS validation", titlesize=12)
    hist!(ax_cz_rms, cz_rms_diff_spline, bins=100, color=(:blue, 0.5), label="RQSpline")
    hist!(ax_cz_rms, cz_rms_diff_gauss, bins=100, color=(:red, 0.5), label="Split-Gaussian")
    axislegend(ax_cz_rms, position=:rt, labelsize=10)

    # Summary text
    summary_text = @sprintf("Energy — Spline: mean=%+.4f±%.4f, RMS=%+.4f±%.4f | Gauss: mean=%+.4f±%.4f, RMS=%+.4f±%.4f\nCosZ — Spline: mean=%+.4f±%.4f, RMS=%+.4f±%.4f | Gauss: mean=%+.4f±%.4f, RMS=%+.4f±%.4f",
        mean(mean_diff_spline), std(mean_diff_spline), mean(rms_diff_spline), std(rms_diff_spline),
        mean(mean_diff_gauss), std(mean_diff_gauss), mean(rms_diff_gauss), std(rms_diff_gauss),
        mean(cz_mean_diff_spline), std(cz_mean_diff_spline), mean(cz_rms_diff_spline), std(cz_rms_diff_spline),
        mean(cz_mean_diff_gauss), std(cz_mean_diff_gauss), mean(cz_rms_diff_gauss), std(cz_rms_diff_gauss))
    Label(fig[10, 3], summary_text, fontsize=9, halign=:left)

    # ── Part 3: Full response matrix comparison ────────────────────────

    println("\n=== Part 3: Response matrix and prediction comparison ===")

    # Build split-Gaussian response matrices
    flatten_R(R3d) = NamedTuple(key => reshape(R3d[key], size(R3d[key], 1), :) for key in keys(R3d))

    R_gauss_3d = NamedTuple(key => make_response_matrix_splitgauss(MC[key], loge_grid, cz_grid) for key in keys(MC))
    R_gauss = flatten_R(R_gauss_3d)

    # Compute predictions with each response matrix
    assets_gauss = merge(assets, (; R=R_gauss))
    weights_spline = Newtrinos.super_k.calc_weights(params, assets, physics)
    weights_gauss = Newtrinos.super_k.calc_weights(params, assets_gauss, physics)

    pred_spline = map((mc, w, nw) -> mc.Counts .* Newtrinos.super_k.safe_div.(w, nw),
                      MC, weights_spline, assets.nominal_weights)
    # Need nominal weights for Gaussian too
    nominal_weights_gauss = Newtrinos.super_k.calc_weights(params, assets_gauss, physics)
    pred_gauss = map((mc, w, nw) -> mc.Counts .* Newtrinos.super_k.safe_div.(w, nw),
                     MC, weights_gauss, nominal_weights_gauss)

    total_spline = reduce(+, values(pred_spline))
    total_gauss = reduce(+, values(pred_gauss))

    println("  Total events (spline):         $(@sprintf("%.1f", sum(total_spline)))")
    println("  Total events (split-Gaussian):  $(@sprintf("%.1f", sum(total_gauss)))")
    println("  Total events (observed):        $(@sprintf("%.1f", sum(assets.observed)))")

    # Plot 4: Fractional difference in predicted bin counts
    frac_diff = (total_gauss .- total_spline) ./ max.(total_spline, 1e-3)
    nbins = length(total_spline)

    ax_frac = Axis(fig[11, 1:2], xlabel="Bin index", ylabel="(Gauss − Spline) / Spline",
                    title="Fractional difference in predicted bin counts", titlesize=12)
    scatter!(ax_frac, 1:nbins, frac_diff, markersize=3, color=:black)
    hlines!(ax_frac, [0.0], color=:gray, linestyle=:dash)
    text!(ax_frac, 0.02, 0.95, text=@sprintf("Mean: %+.4f\nRMS: %.4f\nMax |Δ|: %.4f",
          mean(frac_diff), sqrt(mean(frac_diff.^2)), maximum(abs.(frac_diff))),
          space=:relative, fontsize=10, align=(:left, :top))

    # Histogram of fractional differences
    ax_fhist = Axis(fig[11, 3], xlabel="(Gauss − Spline) / Spline", ylabel="Count",
                     title="Distribution of fractional differences", titlesize=12)
    hist!(ax_fhist, frac_diff, bins=50, color=(:steelblue, 0.7))

    # ── Part 4: χ² scan over Δm² ──────────────────────────────────────

    println("\n=== Part 4: χ² scan over Δm² ===")

    dm2_values = LinRange(2.0e-3, 3.0e-3, 51)
    chi2_spline = Float64[]
    chi2_gauss = Float64[]
    obs = assets.observed

    for dm2 in dm2_values
        p_scan = @set params.Δm²₃₁ = dm2

        # Spline prediction
        w_s = Newtrinos.super_k.calc_weights(p_scan, assets, physics)
        pred_s = reduce(+, values(map((mc, w, nw) -> mc.Counts .* Newtrinos.super_k.safe_div.(w, nw),
                                       MC, w_s, assets.nominal_weights)))

        # Gaussian prediction
        w_g = Newtrinos.super_k.calc_weights(p_scan, assets_gauss, physics)
        pred_g = reduce(+, values(map((mc, w, nw) -> mc.Counts .* Newtrinos.super_k.safe_div.(w, nw),
                                       MC, w_g, nominal_weights_gauss)))

        c2_s = sum((pred_s .- obs).^2 ./ max.(pred_s, 1.0))
        c2_g = sum((pred_g .- obs).^2 ./ max.(pred_g, 1.0))
        push!(chi2_spline, c2_s)
        push!(chi2_gauss, c2_g)
    end

    idx_s = argmin(chi2_spline)
    idx_g = argmin(chi2_gauss)
    println("  Best-fit Δm² (spline):         $(@sprintf("%.4e", dm2_values[idx_s])) eV²")
    println("  Best-fit Δm² (split-Gaussian):  $(@sprintf("%.4e", dm2_values[idx_g])) eV²")
    println("  Shift: $(@sprintf("%+.4e", dm2_values[idx_g] - dm2_values[idx_s])) eV²")

    # Likelihood at nominal params
    ll_spline = sum(obs .* log.(max.(total_spline, 1e-10)) .- total_spline)
    ll_gauss  = sum(obs .* log.(max.(total_gauss, 1e-10)) .- total_gauss)
    println("  Log-likelihood at nominal (spline):  $(@sprintf("%.1f", ll_spline))")
    println("  Log-likelihood at nominal (Gaussian): $(@sprintf("%.1f", ll_gauss))")
    println("  ΔLL: $(@sprintf("%+.1f", ll_gauss - ll_spline))")

    # Plot 5: χ² scan
    ax_chi2 = Axis(fig[12, 1:2], xlabel="Δm²₃₁ [eV²]", ylabel="Δχ²",
                    title="χ² scan over Δm²₃₁", titlesize=12)
    lines!(ax_chi2, collect(dm2_values), chi2_spline .- minimum(chi2_spline),
           label="RQSpline", color=:blue, linewidth=2)
    lines!(ax_chi2, collect(dm2_values), chi2_gauss .- minimum(chi2_gauss),
           label="Split-Gaussian", color=:red, linewidth=2)
    vlines!(ax_chi2, [dm2_values[idx_s]], color=:blue, linestyle=:dash, alpha=0.5)
    vlines!(ax_chi2, [dm2_values[idx_g]], color=:red, linestyle=:dash, alpha=0.5)
    hlines!(ax_chi2, [1.0], color=:gray, linestyle=:dot, alpha=0.5)
    axislegend(ax_chi2, position=:rt, labelsize=10)

    # Plot: absolute χ² comparison
    ax_chi2_abs = Axis(fig[12, 3], xlabel="Δm²₃₁ [eV²]", ylabel="χ²",
                        title="Absolute χ²", titlesize=12)
    lines!(ax_chi2_abs, collect(dm2_values), chi2_spline, label="RQSpline", color=:blue, linewidth=2)
    lines!(ax_chi2_abs, collect(dm2_values), chi2_gauss, label="Split-Gaussian", color=:red, linewidth=2)
    axislegend(ax_chi2_abs, position=:rt, labelsize=10)

    Label(fig[13, :], @sprintf("Spline: %.1f events, LL=%.1f | Gaussian: %.1f events, LL=%.1f | ΔLL=%+.1f",
          sum(total_spline), ll_spline, sum(total_gauss), ll_gauss, ll_gauss - ll_spline),
          fontsize=12)

    outpath = joinpath(@__DIR__, "response_matrix_study.png")
    save(outpath, fig, px_per_unit=2)
    println("\nSaved plot to $outpath")
    println("Done.")
end

# Use Statistics for mean/std
using Statistics: mean, std

main()
