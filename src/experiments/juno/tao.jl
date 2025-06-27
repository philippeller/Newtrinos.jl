module tao

using DataFrames
using CSV
using Distributions
using LinearAlgebra
using Statistics
using BAT
using Interpolations
using Base
using CairoMakie
import ..Newtrinos

@kwdef struct TAO <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function configure(physics)
    physics = (;physics.osc)
    assets = get_assets(physics)
    return TAO(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_params()
    params = (
        # shared systematics with juno, nmaes match
        flux_scale = 1.0,   
        energy_scale = 1.0,
        shape_eps = 0.0,
        
        # TAO only systematics
        tao_detection_epsilon = 1.0,
        tao_res_a = 0.015,
        tao_res_b = 0.0,
        tao_res_c = 0.0,
        
        # TAO specific backgrounds
        tao_accidental_norm = 1.0,
        tao_fast_neutron_norm = 1.0,
        tao_lihe_norm = 1.0,
    )
end

fixed_params = (
    δCP = 0.0,
    θ₂₃ = 0.8556288707523761,
)

function get_priors()
    priors = (
        flux_scale = Normal(1.0, 0.02), 
        energy_scale = Normal(1.0, 0.005), 
        shape_eps = Normal(0,1),

        tao_detection_epsilon = Normal(1.0, 0.005),
        tao_res_a = Normal(0.015, 0.015 * 0.05),
        tao_res_b = Normal(0.0, 0.001),
        tao_res_c = Normal(0.0, 0.001),
        
        tao_accidental_norm = Normal(1.0, 0.20),
        tao_fast_neutron_norm = Normal(1.0, 0.30),
        tao_lihe_norm = Normal(1.0, 0.30),
    )
end

function get_assets(physics, datadir = @__DIR__)
    @info "Loading TAO data"

    DATA_LIVETIME_YEARS = 6.5 
    DAYS_IN_DATA = DATA_LIVETIME_YEARS * 365
    
    LIVETIME_YEARS = 6.0
    LIVETIME_DAYS = LIVETIME_YEARS * 365
    BASELINE_M = 30.0

    reactors = DataFrame(
        Reactor = ["Taishan"], Core = [1], Power_GW_th = [4.6], Baseline_m = [BASELINE_M],
    )
    reactors.Flux_Factor = [1.0]

    flav = physics.osc.cfg.flavour
    if     hasproperty(flav, :ordering)
        ord_sym = flav.ordering
    elseif hasproperty(flav, :three_flavour) && hasproperty(flav.three_flavour, :ordering)
        ord_sym = flav.three_flavour.ordering
    else
        error("Couldnt find `ordering` in flavour config: $flav")
    end
    ord_str = uppercase(string(ord_sym))

    observed_fname = joinpath(datadir, "tao_$(ord_str)_observed.csv")
    df_observed = CSV.read(observed_fname, DataFrame;
                           header=true,
                           types=Dict("E_vis_MeV" => Float64,
                                      "Counts"   => Float64))
    observed = convert(Vector{Float64}, df_observed.Counts)  

    df_no_osc_juno = CSV.read(joinpath(datadir,"spectrum_noosc.csv"), DataFrame, header=["E [MeV]", "Events"])
    sort!(df_no_osc_juno, "E [MeV]")
    df_no_osc_processed = combine(groupby(df_no_osc_juno, "E [MeV]"), "Events" => mean => "Events")
    
    E_nu_grid_no_osc = df_no_osc_processed."E [MeV]"
    no_osc_shape_raw = df_no_osc_processed.Events
    
    integral_of_shape = sum( (no_osc_shape_raw[1:end-1] .+ no_osc_shape_raw[2:end]) ./ 2 .* diff(E_nu_grid_no_osc) )
    TARGET_SIGNAL_RATE_PER_DAY = 1000.0
    tao_normalization = TARGET_SIGNAL_RATE_PER_DAY / integral_of_shape
    no_osc_interp = LinearInterpolation(E_nu_grid_no_osc, no_osc_shape_raw .* tao_normalization; extrapolation_bc=0)
    
    df_acc = CSV.read(joinpath(datadir, "tao_bg_accidentals.csv"), DataFrame, header=["E_vis_MeV", "total_events_6_5_years"])
    df_acc.daily_rate_per_bin = df_acc.total_events_6_5_years ./ DAYS_IN_DATA
    accidental_bkg_interp = LinearInterpolation(df_acc.E_vis_MeV, df_acc.daily_rate_per_bin; extrapolation_bc=0.0)
    
    df_fn = CSV.read(joinpath(datadir, "tao_bg_fn.csv"), DataFrame, header=["E_vis_MeV", "total_events_6_5_years"])
    df_fn.daily_rate_per_bin = df_fn.total_events_6_5_years ./ DAYS_IN_DATA
    fast_neutron_bkg_interp = LinearInterpolation(df_fn.E_vis_MeV, df_fn.daily_rate_per_bin; extrapolation_bc=0.0)
    
    df_lihe = CSV.read(joinpath(datadir, "tao_bg_lihe.csv"), DataFrame, header=["E_vis_MeV", "total_events_6_5_years"])
    df_lihe.daily_rate_per_bin = df_lihe.total_events_6_5_years ./ DAYS_IN_DATA
    lihe_bkg_interp = LinearInterpolation(df_lihe.E_vis_MeV, df_lihe.daily_rate_per_bin; extrapolation_bc=0.0)

    E_bins_visible = 1.0:0.02:8.0

    df_response = CSV.read(joinpath(datadir,"detector_nonlinear_response.csv"), DataFrame, header=["E_deposited_MeV", "response"])
    df_response_sorted = combine(DataFrames.groupby(df_response, "E_deposited_MeV"), :response => mean => :response)
    sort!(df_response_sorted, "E_deposited_MeV") 
    E_deposited_MeV_raw = df_response_sorted.E_deposited_MeV
    response_arr_raw = df_response_sorted.response
    
    IBD_KINEMATIC_OFFSET = 0.782
    E_arr_deposited_for_response = E_nu_grid_no_osc .- IBD_KINEMATIC_OFFSET
    
    nonlinear_response_interp = LinearInterpolation(E_deposited_MeV_raw, response_arr_raw, extrapolation_bc=Interpolations.Flat())
    E_arr_visible_from_raw = nonlinear_response_interp.(E_arr_deposited_for_response) .* E_arr_deposited_for_response
    
    sort_indices = sortperm(E_arr_visible_from_raw)
    E_arr_visible_from_raw_sorted = E_arr_visible_from_raw[sort_indices]
    E_arr_neutrino_for_inverse_sorted = E_nu_grid_no_osc[sort_indices]
    
    unique_indices = unique(i -> E_arr_visible_from_raw_sorted[i], 1:length(E_arr_visible_from_raw_sorted))
    E_arr_visible_final_unique = E_arr_visible_from_raw_sorted[unique_indices]
    E_arr_neutrino_for_inverse_final_unique = E_arr_neutrino_for_inverse_sorted[unique_indices]
    
    visible_to_neutrino_interp = LinearInterpolation(E_arr_visible_final_unique, E_arr_neutrino_for_inverse_final_unique, extrapolation_bc=Interpolations.Flat()) 

    df_shape = CSV.read(joinpath(datadir,"shape_uncertainty_TAO.csv"), DataFrame)
    rename!(df_shape, [:energy_MeV, :rel_unc])
    df_shape.rel_unc .= df_shape.rel_unc ./ 100.0
    df_shape2 = combine(DataFrames.groupby(df_shape, :energy_MeV), :rel_unc => mean => :Δshape)
    sort!(df_shape2, :energy_MeV)
    shape_unc_interp = LinearInterpolation(df_shape2.energy_MeV, df_shape2.Δshape; extrapolation_bc = 0.0)

    assets = (;
        reactors,
        no_osc_interp,
        E_bins_visible,
        accidental_bkg_interp,
        fast_neutron_bkg_interp,
        lihe_bkg_interp,
        visible_to_neutrino_interp,
        shape_unc_interp,
        LIVETIME_DAYS,
        observed
    )

end

function smear(E_arr_smear_local, smear_arr_in, sigma_arr; width=10, E_scale=1.0, E_bias=0.0)
    l = length(smear_arr_in)
    out = zeros(eltype(smear_arr_in), l)

    for i in 1:l
        e_center = E_arr_smear_local[i] * E_scale + E_bias 
        norm_val = 0.0
        sum_val = 0.0
        j_min_loop = max(1, i - width)
        j_max_loop = min(l, i + width)

        for j in j_min_loop:j_max_loop
            if sigma_arr[j] > 1e-10
                coeff = (1 / sigma_arr[j]) * exp(-0.5 * ((e_center - E_arr_smear_local[j]) / sigma_arr[j])^2)
                norm_val += coeff
                sum_val += coeff * smear_arr_in[j]
            end
        end
        if norm_val > 1e-10 
            out[i] = sum_val / norm_val
        else
            out[i] = 0.0 
        end
    end
    return out
end


function get_expected(params, physics, assets)
    
    E_vis_corr = assets.E_bins_visible .* params.energy_scale
    bin_width_mev = step(assets.E_bins_visible)
    
    E_nu = assets.visible_to_neutrino_interp.(E_vis_corr)
    L_km = assets.reactors.Baseline_m ./ 1e3
    P_ee = physics.osc.osc_prob(E_nu./1e3, L_km, params; anti=true)[:, :, 1, 1]
    prob_weighted_flat = vec(sum(P_ee .* assets.reactors.Flux_Factor', dims=2))
    
    unosc_counts_density = assets.no_osc_interp.(E_nu)
    spectrum_before_smearing = (unosc_counts_density .* prob_weighted_flat) .* assets.LIVETIME_DAYS .* bin_width_mev

    sigma_res_val = @. sqrt(params.tao_res_a^2 * abs(E_vis_corr) + params.tao_res_b^2 * E_vis_corr^2 + params.tao_res_c^2)
    smeared_signal = smear(collect(assets.E_bins_visible), spectrum_before_smearing, sigma_res_val, width=200)
    final_signal_counts = smeared_signal .* params.flux_scale .* params.tao_detection_epsilon

    final_accidental_counts = (assets.accidental_bkg_interp.(E_vis_corr) .* params.tao_accidental_norm) .* assets.LIVETIME_DAYS
    final_fast_neutron_counts = (assets.fast_neutron_bkg_interp.(E_vis_corr) .* params.tao_fast_neutron_norm) .* assets.LIVETIME_DAYS
    final_lihe_counts = (assets.lihe_bkg_interp.(E_vis_corr) .* params.tao_lihe_norm) .* assets.LIVETIME_DAYS
    
    total_backgrounds = final_accidental_counts .+ final_fast_neutron_counts .+ final_lihe_counts
    total_events = final_signal_counts .+ total_backgrounds

    return max.(total_events, 0.0)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        full_params = merge(params, fixed_params)
        exp_events = get_expected(full_params, physics, assets)
        return distprod(Poisson.(exp_events))
    end
    return forward_model
end

function get_plot(physics, assets)
    function plot_spectra(params; title_suffix::String="")

        E_vis = assets.E_bins_visible
        bin_width_mev = step(E_vis)
     
        E_vis_corr = E_vis .* params.energy_scale
        E_nu = assets.visible_to_neutrino_interp.(E_vis_corr)
        L_km = assets.reactors.Baseline_m ./ 1e3
        P_ee = physics.osc.osc_prob(E_nu./1e3, L_km, params; anti=true)[:, :, 1, 1]
        prob_weighted_flat = vec(sum(P_ee .* assets.reactors.Flux_Factor', dims=2))
        unosc_counts_density = assets.no_osc_interp.(E_nu)
        spectrum_before_smearing = (unosc_counts_density .* prob_weighted_flat) .* assets.LIVETIME_DAYS .* bin_width_mev
        sigma_res_val = @. sqrt(params.tao_res_a^2 * abs(E_vis_corr) + params.tao_res_b^2 * E_vis_corr^2 + params.tao_res_c^2)
        smeared_signal = smear(collect(E_vis), spectrum_before_smearing, sigma_res_val, width=200)
        final_signal_counts = smeared_signal .* params.flux_scale .* params.tao_detection_epsilon

        final_accidental_counts = (assets.accidental_bkg_interp.(E_vis .* params.energy_scale) .* params.tao_accidental_norm) .* assets.LIVETIME_DAYS
        final_fast_neutron_counts = (assets.fast_neutron_bkg_interp.(E_vis .* params.energy_scale) .* params.tao_fast_neutron_norm) .* assets.LIVETIME_DAYS
        final_lihe_counts = (assets.lihe_bkg_interp.(E_vis .* params.energy_scale) .* params.tao_lihe_norm) .* assets.LIVETIME_DAYS
        
        total_backgrounds = final_accidental_counts .+ final_fast_neutron_counts .+ final_lihe_counts
        total_model = final_signal_counts .+ total_backgrounds

        fig = Figure(size = (1200, 900), fontsize = 14)
        ax_spec = Axis(fig[1, 1], title="TAO Stacked Spectrum", ylabel="Events / Bin")
        band!(ax_spec, E_vis, 0, final_lihe_counts, color=(:purple, 0.5), label="⁹Li/⁸He")
        band!(ax_spec, E_vis, final_lihe_counts, final_lihe_counts .+ final_fast_neutron_counts, color=(:green, 0.5), label="Fast Neutron")
        band!(ax_spec, E_vis, final_lihe_counts .+ final_fast_neutron_counts, total_backgrounds, color=(:orange, 0.5), label="Accidental")
        band!(ax_spec, E_vis, total_backgrounds, total_model, color=(:red, 0.5), label="ν Signal")
        lines!(ax_spec, E_vis, total_model, color=:black, label="Total Model")
        scatter!(ax_spec, E_vis, assets.observed, color=:black, markersize=3, label="Asimov Data")
        axislegend(ax_spec, position=:rt, merge=true, unique=true)
        ylims!(ax_spec, low=0)
        
        ax_paper = Axis(fig[1, 2], title="TAO Background Components")
        band!(ax_paper, E_vis, 0, final_accidental_counts, color=(:brown, 0.6), label="Accidentals")
        band!(ax_paper, E_vis, 0, final_fast_neutron_counts, color=(:blue, 0.6), label="Fast Neutron")
        band!(ax_paper, E_vis, 0, final_lihe_counts, color=(:red, 0.6), label="⁹Li/⁸He")
        lines!(ax_paper, E_vis, final_signal_counts, color=:green, linewidth=2.5, label="ν Signal")
        axislegend(ax_paper, position=:rt, merge=true, unique=true)
        ylims!(ax_paper, low=0)
        
        ax_res_ratio = Axis(fig[2, 1], title="Smearing Effect", xlabel=L"E_{vis} \text{ (MeV)}", ylabel="Smeared / Unsmeared")
        ratio_smearing = smeared_signal ./ (spectrum_before_smearing .+ 1e-9)
        lines!(ax_res_ratio, E_vis, ratio_smearing, label="Resolution Effect")
        hlines!(ax_res_ratio, [1.0], color=:black, linestyle=:dash)
        ylims!(ax_res_ratio, 0.95, 1.05)
        
        ax_osc_dis = Axis(fig[2, 2], title="Disappearance Probability", xlabel=L"E_{\nu} \text{ (MeV)}", ylabel=L"1 - P(\bar{\nu}_e \rightarrow \bar{\nu}_e)", yscale=log10)
        disappearance_prob = 1.0 .- prob_weighted_flat
        lines!(ax_osc_dis, E_nu, disappearance_prob)
        ylims!(ax_osc_dis, 1e-6, 1e-2)
        linkxaxes!(ax_spec, ax_paper, ax_res_ratio)
        hidexdecorations!(ax_spec, grid = false)
        hidexdecorations!(ax_paper, grid = false)

        return fig
    end
    return plot_spectra
end

end 