module juno

using DataFrames
using CSV
using Distributions
using Statistics
using BAT
using CairoMakie
using Logging
using Interpolations
import ValueShapes
import ..Newtrinos

@kwdef struct JUNO <: Newtrinos.Experiment
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
    return JUNO(
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
        flux_scale = 1.0,   
        energy_scale = 1.0,      
        detection_epsilon = 1.0,
        
        res_a = 0.0261,
        res_b = 0.0082 ,
        res_c = 0.0123,
    
        shape_eps   = 0.0,
        geo_shape_eps = 0.0,
        
        geo_rate_norm = 1.0,
        accidental_norm = 1.0,
        world_reactor_norm = 1.0,
        lihe_norm = 1.0,
        co_norm = 1.0,
        atmnc_norm = 1.0,
        fast_neutron_norm = 1.0,
        )
end

fixed_params =  (δCP = 0.0,
        θ₂₃ = 0.8556288707523761,
    
        #flux_scale = 1.0,   
        #energy_scale = 1.0,      
        #detection_epsilon = 1.0,

        #res_a = 0.0261,
        #res_b = 0.0082,
        #res_c = 0.0123,
    
        #shape_eps=0,
        #geo_shape_eps=0,
        #geo_rate_norm = 1.0,
        #accidental_norm = 1.0,
        #world_reactor_norm = 1.0,
        #lihe_norm = 1.0,
        #co_norm = 1.0,
        #atmnc_norm = 1.0,
        #fast_neutron_norm = 1.0,
    )

function get_priors()
    priors = (
        flux_scale = Normal(1.0, 0.02), 
        energy_scale = Normal(1.0, 0.005),
        detection_epsilon = Normal(1.0, 0.01),

        res_a = Normal(0.0261, 0.0002),
        res_b = Normal(0.0082, 0.0001),
        res_c = Normal(0.0123, 0.0004),
        
        shape_eps = Normal(0,1),
        geo_shape_eps = Normal(0,1),
        
        geo_rate_norm = Normal(1.0, 0.30),
        accidental_norm = Normal(1.0, 0.01),     
        world_reactor_norm = Normal(1.0, 0.02),  
        lihe_norm = Normal(1.0, 0.20),      
        co_norm = Normal(1.0, 0.50),         
        atmnc_norm = Normal(1.0, 0.50),   
        fast_neutron_norm = Normal(1.0, 1.0), 
          )
end



function get_assets(physics, datadir = @__DIR__)
    @info "Loading juno data"

    L_JUNO = 52.5   
    L_JUNO_m = L_JUNO * 1e3 
    
    resolution_a = 0.0261
    resolution_b = 0.0082 
    resolution_c = 0.0123 

    nominal_livetime = 6.0
    analysis_livetime = 6.0
    LIVETIME_DAYS = analysis_livetime * 365

    Delta_E = 20e-3   # 20 keV bins
 
    GEO_SHAPE_UNC_FRACTION = 0.05 

    flav = physics.osc.cfg.flavour
    if     hasproperty(flav, :ordering)
        ord_sym = flav.ordering
    elseif hasproperty(flav, :three_flavour) && hasproperty(flav.three_flavour, :ordering)
        ord_sym = flav.three_flavour.ordering
    else
        error("Could not find `ordering` in flavour config: $flav")
    end
    ord_str = uppercase(string(ord_sym))

    observed_fname = joinpath(datadir, "juno_$(ord_str)_observed.csv")
    df_observed = CSV.read(observed_fname, DataFrame;
                           header=true,
                           types=Dict("E_vis_MeV" => Float64,
                                      "Counts"   => Float64))
    observed = convert(Vector{Float64}, df_observed.Counts)  
    
    df_no_osc = CSV.read(joinpath(datadir,"spectrum_noosc.csv"), DataFrame, header=["E [MeV]", "Events"])
    df_shape = CSV.read(joinpath(datadir,"shape_uncertainty_TAO.csv"), DataFrame)
    rename!(df_shape, [:energy_MeV, :rel_unc])
    df_response = CSV.read(joinpath(datadir,"detector_nonlinear_response.csv"), DataFrame, header=["E_deposited_MeV", "response"])
    df_backgrounds = CSV.read(joinpath(datadir,"geoneutrino_background.csv"), DataFrame, header=["E_visible_MeV", "events_per_day_per_20keV"])

    df_acc = CSV.read(joinpath(datadir, "bg_accidentals.csv"), DataFrame, header=["E_vis_MeV", "rate_density"])
    df_acc.total_events_nominal = df_acc.rate_density .* LIVETIME_DAYS
    sort!(df_acc, :E_vis_MeV)   
    accidental_bkg_interp = LinearInterpolation(df_acc.E_vis_MeV, df_acc.total_events_nominal; extrapolation_bc = 0.0)
    
    df_wr = CSV.read(joinpath(datadir, "bg_world_reactors.csv"), DataFrame, header=["E_vis_MeV", "rate_density"])
    df_wr.total_events_nominal = df_wr.rate_density .* LIVETIME_DAYS
    sort!(df_wr, :E_vis_MeV)   
    world_reactor_bkg_interp = LinearInterpolation(df_wr.E_vis_MeV, df_wr.total_events_nominal; extrapolation_bc = 0.0)

    df_lihe = CSV.read(joinpath(datadir, "bg_LiHe.csv"), DataFrame, header=["E_vis_MeV", "rate_density"])
    df_lihe.total_events_nominal = df_lihe.rate_density .* LIVETIME_DAYS
    sort!(df_lihe, :E_vis_MeV)
    lihe_bkg_interp = LinearInterpolation(df_lihe.E_vis_MeV, df_lihe.total_events_nominal; extrapolation_bc=0.0)

    df_co = CSV.read(joinpath(datadir, "bg_CO.csv"), DataFrame, header=["E_vis_MeV", "rate_density"])
    df_co.total_events_nominal = df_co.rate_density .* LIVETIME_DAYS
    sort!(df_co, :E_vis_MeV)
    co_bkg_interp = LinearInterpolation(df_co.E_vis_MeV, df_co.total_events_nominal; extrapolation_bc=0.0)

    df_atmnc = CSV.read(joinpath(datadir, "bg_atm_NC.csv"), DataFrame, header=["E_vis_MeV", "rate_density"])
    df_atmnc.total_events_nominal = df_atmnc.rate_density .* LIVETIME_DAYS
    sort!(df_atmnc, :E_vis_MeV)
    atmnc_bkg_interp = LinearInterpolation(df_atmnc.E_vis_MeV, df_atmnc.total_events_nominal; extrapolation_bc=0.0)

    df_fn = CSV.read(joinpath(datadir, "bg_fast_neutrons.csv"), DataFrame, header=["E_vis_MeV", "rate_density"])
    df_fn.total_events_nominal = df_fn.rate_density .* LIVETIME_DAYS
    sort!(df_fn, :E_vis_MeV)
    fast_neutron_bkg_interp = LinearInterpolation(df_fn.E_vis_MeV, df_fn.total_events_nominal; extrapolation_bc=0.0)

    
    df_shape.rel_unc .= df_shape.rel_unc ./ 100.0
    
    df_shape2 = combine(DataFrames.groupby(df_shape, :energy_MeV),
                        :rel_unc => mean => :Δshape)
    sort!(df_shape2, :energy_MeV)
    
    E_shape = df_shape2.energy_MeV
    Δshape = df_shape2.Δshape
    
    shape_unc_interp = LinearInterpolation(E_shape, Δshape; extrapolation_bc = 0.0)

    df_no_osc.Events = df_no_osc.Events .* (analysis_livetime / nominal_livetime) .* (20.0 / 1000.0)  # adjst no_osc events for livetime and bin width (20 keV)
        
    df_no_osc_sorted = combine(DataFrames.groupby(df_no_osc, "E [MeV]"), :Events => sum => :Events)
    sort!(df_no_osc_sorted, "E [MeV]") 
    
    E_arr_neutrino_raw = df_no_osc_sorted."E [MeV]"
    no_osc_arr_raw = df_no_osc_sorted.Events
    
    no_osc_interp = LinearInterpolation(E_arr_neutrino_raw, no_osc_arr_raw, extrapolation_bc=0)
        
    df_response_sorted = combine(DataFrames.groupby(df_response, "E_deposited_MeV"), :response => mean => :response)
    sort!(df_response_sorted, "E_deposited_MeV") 
    E_deposited_MeV_raw = df_response_sorted.E_deposited_MeV
    response_arr_raw = df_response_sorted.response

    E_arr_deposited_for_response = E_arr_neutrino_raw .- 0.782   # E_vis = E_nu - (m_n - m_p - m_e) ≈ 0.782 MeV
    
    nonlinear_response_interp = LinearInterpolation(E_deposited_MeV_raw, response_arr_raw, extrapolation_bc=Interpolations.Flat())
    E_arr_visible_from_raw = nonlinear_response_interp.(E_arr_deposited_for_response) .* E_arr_deposited_for_response
    
    sort_indices = sortperm(E_arr_visible_from_raw)
    E_arr_visible_from_raw_sorted = E_arr_visible_from_raw[sort_indices]
    E_arr_neutrino_for_inverse_sorted = E_arr_neutrino_raw[sort_indices]
    
    unique_indices = unique(i -> E_arr_visible_from_raw_sorted[i], 1:length(E_arr_visible_from_raw_sorted))
    E_arr_visible_final_unique = E_arr_visible_from_raw_sorted[unique_indices]
    E_arr_neutrino_for_inverse_final_unique = E_arr_neutrino_for_inverse_sorted[unique_indices]
    
    
    visible_to_neutrino_interp = LinearInterpolation(E_arr_visible_final_unique, E_arr_neutrino_for_inverse_final_unique, extrapolation_bc=Interpolations.Flat()) 
    
    df_backgrounds_sorted = combine(DataFrames.groupby(df_backgrounds, "E_visible_MeV"), :events_per_day_per_20keV => sum => :events_per_day_per_20keV)
    sort!(df_backgrounds_sorted, "E_visible_MeV")
    E_visible_MeV_bkg = df_backgrounds_sorted.E_visible_MeV
    background_arr_bkg = df_backgrounds_sorted.events_per_day_per_20keV .* analysis_livetime .* 365
    
    background_interp = LinearInterpolation(E_visible_MeV_bkg, background_arr_bkg, extrapolation_bc=0)
    
    reactors = DataFrame(
        Reactor = ["Taishan", "Taishan", "Yangjiang", "Yangjiang", "Yangjiang", "Yangjiang", "Yangjiang", "Yangjiang", "Daya Bay"],
        Core = [1, 2, 1, 2, 3, 4, 5, 6, 0],
        Power_GW_th = [4.6, 4.6, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 17.4],
        Baseline_m = 1e3 .* [52.77, 52.64, 52.74, 52.82, 52.41, 52.49, 52.11, 52.19, 215.0],
        IBD_Rate_per_day = [7.5, 7.6, 4.8, 4.7, 4.8, 4.8, 4.9, 4.9, 3.0], 
        Relative_Flux_percent = [16.0, 16.1, 10.1, 10.1, 10.3, 10.2, 10.4, 10.4, 6.4] 
    )
    
    fluxes_val = reactors.Power_GW_th ./ (reactors.Baseline_m .^ 2)
    reactors.Flux_Factor = fluxes_val ./ sum(fluxes_val)
    
    N_points = round(Int, (E_arr_visible_final_unique[end] - E_arr_visible_final_unique[1]) / Delta_E)
    
    if N_points <= 0
        @warn "N_points for E_bins_visible is $N_points. Setting to 1."
        N_points = 1 
    end
    
    E_bins_visible = if N_points > 1
        collect(range(E_arr_visible_final_unique[1], E_arr_visible_final_unique[end], length=N_points))
    else
        [(E_arr_visible_final_unique[1] + E_arr_visible_final_unique[end]) / 2.0] 
    end
    
    E_bins_neutrino = visible_to_neutrino_interp.(E_bins_visible)
        
    N_smear = 400
    E_smear_start = isempty(E_arr_visible_final_unique) ? 1.0 : E_arr_visible_final_unique[1] - Delta_E/2
    E_smear_end = isempty(E_arr_visible_final_unique) ? 8.0 : E_arr_visible_final_unique[end] + Delta_E/2
    
    if E_smear_start >= E_smear_end 
        @warn "E_arr_smear start is not less than end. Adjsting."
        E_smear_end = E_smear_start + N_smear * 1e-3 
    end
    
    E_arr_smear = collect(range(E_smear_start, E_smear_end, length=N_smear))
    
    assets = (;
        visible_to_neutrino_interp,
        no_osc_interp,
        shape_unc_interp,
        background_interp,
        reactors,
        GEO_SHAPE_UNC_FRACTION,
        E_bins_visible,
        accidental_bkg_interp,
        world_reactor_bkg_interp,
        lihe_bkg_interp,
        co_bkg_interp,
        atmnc_bkg_interp,
        fast_neutron_bkg_interp,
        observed,
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

            coeff = (1 / sigma_arr[j]) * exp(-0.5 * ((e_center - E_arr_smear_local[j]) / sigma_arr[j])^2)
            norm_val += coeff
            sum_val += coeff * smear_arr_in[j]
            
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
    E_nu = assets.visible_to_neutrino_interp.(E_vis_corr)

    L_km = assets.reactors.Baseline_m ./ 1e3
    
    P_ee = physics.osc.osc_prob(E_nu./1e3, L_km, params; anti=true)[:, :, 1, 1] 
    
    prob_weighted_flat = vec(sum(P_ee .* assets.reactors.Flux_Factor', dims=2))

    unosc_counts_at_E_nu = assets.no_osc_interp.(E_nu)

    spectrum_before_reactor_shape_and_smearing = unosc_counts_at_E_nu .* prob_weighted_flat

    Δshape_values_signal = assets.shape_unc_interp.(E_vis_corr)
    spectrum_with_reactor_shape_sys = spectrum_before_reactor_shape_and_smearing .* (1.0 .+ params.shape_eps .* Δshape_values_signal)

    current_res_a = params.res_a
    current_res_b = params.res_b
    current_res_c = params.res_c
    
    sigma_res_val = @. sqrt(current_res_a^2 * abs(E_vis_corr) + current_res_b^2 * E_vis_corr^2 + current_res_c^2)
    sigma_res_val = max.(sigma_res_val, 1e-9)

    smeared_signal_spectrum = smear(E_vis_corr, spectrum_with_reactor_shape_sys, sigma_res_val, width=200) # width = 15 befofe wd?
    
    signal_counts = smeared_signal_spectrum
    signal_counts .*= params.flux_scale * params.detection_epsilon

    final_signal_counts = max.(signal_counts, 0.0)
    
    nominal_geo_counts = assets.background_interp.(E_vis_corr)
    geo_counts_scaled_rate = nominal_geo_counts .* params.geo_rate_norm

    final_geo_counts = geo_counts_scaled_rate .* (1.0 .+ params.geo_shape_eps .* assets.GEO_SHAPE_UNC_FRACTION) 
    
    final_world_reactor_counts = assets.world_reactor_bkg_interp.(E_vis_corr) .* params.world_reactor_norm
    final_accidental_counts = assets.accidental_bkg_interp.(E_vis_corr) .* params.accidental_norm
    final_lihe_counts = assets.lihe_bkg_interp.(E_vis_corr) .* params.lihe_norm
    final_co_counts = assets.co_bkg_interp.(E_vis_corr) .* params.co_norm
    final_atmnc_counts = assets.atmnc_bkg_interp.(E_vis_corr) .* params.atmnc_norm
    final_fast_neutron_counts = assets.fast_neutron_bkg_interp.(E_vis_corr) .* params.fast_neutron_norm    
    
    other_background_counts = final_world_reactor_counts .+ final_accidental_counts .+ final_lihe_counts .+ final_co_counts .+ final_atmnc_counts .+ final_fast_neutron_counts
    
    total_events = final_signal_counts .+ final_geo_counts .+ other_background_counts 

    return max.(total_events, 0.0)

end


function get_forward_model(physics, assets)
    function forward_model(params)
        full_params = merge(params, fixed_params)
        exp_events = get_expected(full_params, physics, assets)
        distprod(Poisson.(exp_events))
    end
end


function get_plot(physics, assets)
    function plot(params; title_suffix::String = "")

        E_vis = assets.E_bins_visible
        E_nu  = assets.visible_to_neutrino_interp.(E_vis)
        
        L_km  = assets.reactors.Baseline_m ./ 1e3
        P = physics.osc.osc_prob(E_nu ./ 1e3, L_km, params; anti=true)
        Pavg = vec(sum(P[:, :, 1, 1] .* assets.reactors.Flux_Factor', dims=2))

        unosc_counts = assets.no_osc_interp.(E_nu) .* params.flux_scale .* params.detection_epsilon
        osc_unsmear = unosc_counts .* Pavg
        total_model = get_expected(params, physics, assets)

        bkg = Dict(
            "Geo"      => assets.background_interp.(E_vis) .* params.geo_rate_norm,
            "Accident" => assets.accidental_bkg_interp.(E_vis) .* params.accidental_norm,
            "WorldR"   => assets.world_reactor_bkg_interp.(E_vis) .* params.world_reactor_norm,
            "LiHe"     => assets.lihe_bkg_interp.(E_vis) .* params.lihe_norm,
            "CO"       => assets.co_bkg_interp.(E_vis) .* params.co_norm,
            "ATMNC"    => assets.atmnc_bkg_interp.(E_vis) .* params.atmnc_norm,
            "FastN"    => assets.fast_neutron_bkg_interp.(E_vis) .* params.fast_neutron_norm
        )
        total_bkg = reduce(+, values(bkg))
        smeared_signal = total_model .- total_bkg

        fig = Figure(resolution=(1200, 1350), fontsize=14)

        ax1 = Axis(fig[1, 1], title="No-osc vs Oscillated $title_suffix",
                   xlabel="E₍vis₎ [MeV]", ylabel="Counts/bin")
        lines!(ax1, E_vis, unosc_counts,  label="No-osc", linestyle=:dash, color=:black)
        lines!(ax1, E_vis, osc_unsmear,  label="Osc unsmeared",  color=:blue)
        axislegend(ax1)

        ax2 = Axis(fig[1, 2], title="Oscillated + Backgrounds + Data",
                   xlabel="E₍vis₎ [MeV]", ylabel="Counts/bin")
        lines!(ax2, E_vis, smeared_signal, label="Signal (smeared)", color=:red, linewidth=2)
        for (name, arr) in bkg
            lines!(ax2, E_vis, arr, label=name, linestyle=:dot)
        end
        lines!(ax2, E_vis, total_model, label="Signal + all bkgs", color=:magenta, linewidth=3)
      
        scatter!(ax2, E_vis, assets.observed, color=:black, label="Observed Data", markersize=3)

        ax3 = Axis(fig[2, 1], title="Unsmeared vs Smeared",
                   xlabel="E₍vis₎ [MeV]", ylabel="Counts/bin")
        lines!(ax3, E_vis, osc_unsmear, label="Unsmeared (osc)", color=:blue)
        lines!(ax3, E_vis, smeared_signal, label="Smeared", color=:green)
        axislegend(ax3)

        ax4 = Axis(fig[2, 2], title="⟨Pₑₑ⟩ vs Eₙᵤ",
                   xlabel="Eₙᵤ [MeV]", ylabel="⟨Pₑₑ⟩")
        lines!(ax4, E_nu, Pavg, label="⟨Pₑₑ⟩", color=:orange, linewidth=2)
        axislegend(ax4)

        ax5 = Axis(fig[3, 1:2], title="Residuals",
                   xlabel="E₍vis₎ [MeV]", ylabel="(Data-Model)/√Model")
        residuals = (assets.observed .- total_model) ./ sqrt.(total_model .+ 1e-9)
        hlines!(ax5, 0.0, color=:red, linestyle=:dash)
        ylims!(ax5, -5, 5) 

        errorbars!(ax5, E_vis, residuals, ones(length(residuals)), color=:black)
        scatter!(ax5, E_vis, residuals, color=:black)

        return fig
    end

    return plot
end


end 