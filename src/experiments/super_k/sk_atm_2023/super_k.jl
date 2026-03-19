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
using Printf
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
    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI()))
    atm_flux = Newtrinos.atm_flux.configure(Newtrinos.atm_flux.AtmFluxConfig(nominal_model=Newtrinos.atm_flux.HKKM("kam-ally-20-01-mtn-solmin.d")))
    earth_layers = Newtrinos.earth_layers.configure()
    xsec = Newtrinos.xsec.configure(Newtrinos.xsec.Differential_H2O())
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

function make_log_e_cdf(bin)
    log_e = log10.([bin.EnergyQuantile2_3Percent, bin.EnergyQuantile15_9Percent, bin.EnergyQuantile50_0Percent, bin.EnergyQuantile84_1Percent, bin.EnergyQuantile97_7Percent])  # extrapolate tails

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


function make_cosz_cdf(bin)
    cosz = [bin.CosZQuantile2_3Percent, bin.CosZQuantile15_9Percent, bin.CosZQuantile50_0Percent, bin.CosZQuantile84_1Percent, bin.CosZQuantile97_7Percent]  # extrapolate tails

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

function make_response_matrix(MC_component, logE_grid, cosZ_grid)
    n_bins = size(MC_component, 1)
    n_logE = length(logE_grid)
    n_cosZ = length(cosZ_grid)

    response_matrix = zeros(Float64, n_bins, n_logE-1, n_cosZ-1)

    for bin_idx in 1:n_bins
        bin = MC_component[bin_idx, :]

        if bin.Counts == 0
            continue
        end

        log_e_cdf = make_log_e_cdf(bin)
        cosz_cdf = make_cosz_cdf(bin)

        c_e = log_e_cdf.(logE_grid)
        p_e = diff(c_e)

        c_cosz = cosz_cdf.(cosZ_grid)
        p_cosz = diff(c_cosz)

        response_matrix[bin_idx, :, :] .= p_e * p_cosz'

        sum_response = sum(response_matrix[bin_idx, :, :])
        if sum_response == 0
            continue
        end
        response_matrix[bin_idx, :, :] .= response_matrix[bin_idx, :, :] ./ sum_response #* bin.Counts
    end
    return response_matrix
end

function contract_R(R_flat, weighted_flux)
    # R_flat is (n_bins, n_E*n_cz), weighted_flux is (n_E, n_cz)
    R_flat * vec(weighted_flux)
end

function calc_weights(params, assets, physics)

    E = 10. .^midpoints(assets.loge_grid)

    p = physics.osc.osc_prob(E, assets.paths, assets.layers, params);
    p_anti = physics.osc.osc_prob(E, assets.paths, assets.layers, params, anti=true);

    flux = physics.atm_flux.sys_flux(assets.flux_nominal, params)

    s = (size(p)[1], size(p)[2])

    xsec_nue     = physics.xsec.scale(E, :nue,   :CC, false, params)
    xsec_numu    = physics.xsec.scale(E, :numu,  :CC, false, params)
    xsec_nutau   = physics.xsec.scale(E, :nutau, :CC, false, params)
    xsec_nuebar  = physics.xsec.scale(E, :nue,   :CC, true,  params)
    xsec_numubar = physics.xsec.scale(E, :numu,  :CC, true,  params)
    xsec_nutaubar= physics.xsec.scale(E, :nutau, :CC, true,  params)
    xsec_nc      = physics.xsec.scale(E, :nue,   :NC, false, params)

    nue_flux   = (reshape(flux.nue,    s) .* p[:, :, 1, 1] .+
                  reshape(flux.numu,   s) .* p[:, :, 2, 1]) .* xsec_nue
    numu_flux  = (reshape(flux.nue,    s) .* p[:, :, 1, 2] .+
                  reshape(flux.numu,   s) .* p[:, :, 2, 2]) .* xsec_numu
    nutau_flux = (reshape(flux.nue,    s) .* p[:, :, 1, 3] .+
                  reshape(flux.numu,   s) .* p[:, :, 2, 3]) .* xsec_nutau
    nuebar_flux  = (reshape(flux.nuebar,  s) .* p_anti[:, :, 1, 1] .+
                    reshape(flux.numubar, s) .* p_anti[:, :, 2, 1]) .* xsec_nuebar
    numubar_flux = (reshape(flux.nuebar,  s) .* p_anti[:, :, 1, 2] .+
                    reshape(flux.numubar, s) .* p_anti[:, :, 2, 2]) .* xsec_numubar
    nutaubar_flux = (reshape(flux.nuebar,  s) .* p_anti[:, :, 1, 3] .+
                     reshape(flux.numubar, s) .* p_anti[:, :, 2, 3]) .* xsec_nutaubar

    nue     = contract_R(assets.R.nue,     nue_flux)
    numu    = contract_R(assets.R.numu,    numu_flux)
    nutau   = contract_R(assets.R.nutau,   nutau_flux)
    nuebar  = contract_R(assets.R.nuebar,  nuebar_flux)
    numubar = contract_R(assets.R.numubar, numubar_flux)
    nunc    = contract_R(assets.R.nunc,    ones(eltype(nue_flux), s) .* xsec_nc)

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

    layers = physics.earth_layers.compute_layers()
    paths = physics.earth_layers.compute_paths(midpoints(cz_grid), layers)
    flux_nominal = physics.atm_flux.nominal_flux(10. .^midpoints(loge_grid), midpoints(cz_grid))

    flatten_R(R3d) = NamedTuple(key => reshape(R3d[key], size(R3d[key], 1), :) for key in keys(R3d))

    R_3d = NamedTuple(key => make_response_matrix(MC[key], loge_grid, cz_grid) for key in keys(MC))
    R = flatten_R(R_3d)
    nominal_weights = calc_weights(params_nominal, (;R, flux_nominal, paths, layers, loge_grid), physics)

    R_plus_3d = NamedTuple(key => make_response_matrix(MC[key], loge_grid .+ log(1.02), cz_grid) for key in keys(MC))
    R_minus_3d = NamedTuple(key => make_response_matrix(MC[key], loge_grid .+ log(0.98), cz_grid) for key in keys(MC))
    weights_plus = calc_weights(params_nominal, (;R=flatten_R(R_plus_3d), flux_nominal, paths, layers, loge_grid), physics)
    weights_minus = calc_weights(params_nominal, (;R=flatten_R(R_minus_3d), flux_nominal, paths, layers, loge_grid), physics)
    Fij = NamedTuple(key => safe_div.((weights_plus[key] .- weights_minus[key]), (2*0.02 .* nominal_weights[key])) for key in keys(nominal_weights))

    for key in keys(R_3d)
        R_plus_3d[key][:,:,1:50] .= R_3d[key][:,:,1:50]
        R_minus_3d[key][:,:,1:50] .= R_3d[key][:,:,1:50]
    end
    weights_plus = calc_weights(params_nominal, (;R=flatten_R(R_plus_3d), flux_nominal, paths, layers, loge_grid), physics)
    weights_minus = calc_weights(params_nominal, (;R=flatten_R(R_minus_3d), flux_nominal, paths, layers, loge_grid), physics)
    Fij_updown = NamedTuple(key => safe_div.((weights_plus[key] .- weights_minus[key]), (2*0.02 .* nominal_weights[key])) for key in keys(nominal_weights))

    return (; MC, R, Fij, Fij_updown, flux_nominal, paths, layers, loge_grid, cz_grid, nominal_weights, observed, bininfo, masks)

end



    
function get_params()
    params = (
        sk_i_iii_energy_scale = 1.0,
        sk_iv_v_energy_scale = 1.0,
        sk_updown_energy_scale = 1.0,
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
        sk_i_v_btd_1 = 1.0,
        sk_i_v_btd_2 = 1.0,
        sk_i_v_btd_3 = 1.0,
        sk_i_iii_subgev_pid = 1.0,
        sk_iv_v_subgev_pid = 1.0,
        sk_i_iii_multigev_pid = 1.0,
        sk_iv_v_multigev_pid = 1.0,
        sk_ring_counting = 1.0,
        sk_nue_contamination = 1.0,
        sk_ncpi0_norm = 1.0,
        )
end

function get_priors()
    priors = (
        sk_i_iii_energy_scale = Normal(1.0, 0.03),
        sk_iv_v_energy_scale = Normal(1.0, 0.018),
        sk_updown_energy_scale = Normal(1.0, 0.01),
        sk_fc_norm = Normal(1.0, 0.05),
        sk_pc_norm = Normal(1.0, 0.05),
        sk_upmu_norm = Normal(1.0, 0.05),
        sk_fiducial_norm = Normal(1.0, 0.02),
        sk_nc_mu_norm = Normal(1.0, 0.1),
        sk_pc_stopping_vs_througoing = Normal(1.0, 0.2),
        sk_upmu_stopping_vs_througoing = Normal(1.0, 0.01),
        sk_upmu_nonshower_vs_shower = Normal(1.0, 0.04),
        sk_i_iii_decay_e_tag_eff = Normal(1.0, 0.015),
        sk_iv_v_decay_e_tag_eff = Normal(1.0, 0.008),
        sk_iv_v_subgev_neutron_tag_eff = Normal(1.0, 0.12),
        sk_iv_v_multigev_neutron_tag_eff = Normal(1.0, 0.12),
        sk_i_v_btd_1 = Normal(1, 0.05),
        sk_i_v_btd_2 = Normal(1, 0.05),
        sk_i_v_btd_3 = Normal(1, 0.05),
        sk_i_iii_subgev_pid = Normal(1, 0.03),
        sk_iv_v_subgev_pid = Normal(1, 0.03),
        sk_i_iii_multigev_pid = Normal(1, 0.03),
        sk_iv_v_multigev_pid = Normal(1, 0.03),
        sk_ring_counting = Normal(1, 0.05),
        sk_nue_contamination = Normal(1, 0.05),
        sk_ncpi0_norm = Normal(1, 0.1),
        )
end


function reweight(params, physics, assets)
    weights = calc_weights(params, assets, physics)
    return NamedTuple(key => assets.MC[key].Counts .* safe_div.(weights[key], assets.nominal_weights[key]) for key in keys(assets.MC))
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
    return (
        get_factor(assets.masks.fc, params.sk_fc_norm * params.sk_fiducial_norm) .*
        get_factor(assets.masks.pc, params.sk_pc_norm * params.sk_fiducial_norm) .*
        get_factor(assets.masks.upmu, params.sk_upmu_norm) .*
        get_double_factor(total, assets.masks.pc_stop, assets.masks.pc_thru, params.sk_pc_stopping_vs_througoing) .*
        get_double_factor(total, assets.masks.umpmu_stop, assets.masks.upmu_thru, params.sk_upmu_stopping_vs_througoing) .*
        get_double_factor(total, assets.masks.upmu_nonshower, assets.masks.upmu_shower, params.sk_upmu_nonshower_vs_shower) .* 
        get_double_factor(total, assets.masks.sk_i_iii_elike_1decay_e, assets.masks.sk_i_iii_elike_0decay_e, params.sk_i_iii_decay_e_tag_eff) .*
        get_double_factor(total, assets.masks.sk_i_iii_mulike_1decay_e, assets.masks.sk_i_iii_mulike_0decay_e, params.sk_i_iii_decay_e_tag_eff) .*
        get_double_factor(total, assets.masks.sk_i_iii_mulike_2decay_e, assets.masks.sk_i_iii_mulike_1decay_e, params.sk_i_iii_decay_e_tag_eff) .*
        get_double_factor(total, assets.masks.sk_iv_v_1decay_e, assets.masks.sk_iv_v_0decay_e, params.sk_iv_v_decay_e_tag_eff) .*
        get_double_factor(total, assets.masks.sk_iv_v_subgev_0neutron, assets.masks.sk_iv_v_subgev_1neutron, params.sk_iv_v_subgev_neutron_tag_eff) .*
        get_double_factor(total, assets.masks.sk_iv_v_multigev_0neutron, assets.masks.sk_iv_v_multigev_1neutron, params.sk_iv_v_multigev_neutron_tag_eff) .*
        get_double_factor(total, assets.masks.sk_i_v_multigev_multiring_nuebar, assets.masks.sk_i_v_multigev_multiring_nue, params.sk_i_v_btd_1) .*
        get_double_factor(total, assets.masks.sk_i_v_multigev_multiring_nue, assets.masks.sk_i_v_multigev_multiring_mu, params.sk_i_v_btd_2) .*
        get_double_factor(total, assets.masks.sk_i_v_multigev_multiring_mu, assets.masks.sk_i_v_multigev_multiring_other, params.sk_i_v_btd_3) .*
        # PID migration: e-like ↔ mu-like
        get_double_factor(total, assets.masks.sk_i_iii_subgev_elike, assets.masks.sk_i_iii_subgev_mulike, params.sk_i_iii_subgev_pid) .*
        get_double_factor(total, assets.masks.sk_iv_v_subgev_elike, assets.masks.sk_iv_v_subgev_mulike, params.sk_iv_v_subgev_pid) .*
        get_double_factor(total, assets.masks.sk_i_iii_multigev_1ring_elike, assets.masks.sk_i_iii_multigev_1ring_mulike, params.sk_i_iii_multigev_pid) .*
        get_double_factor(total, assets.masks.sk_iv_v_multigev_1ring_elike, assets.masks.sk_iv_v_multigev_1ring_mulike, params.sk_iv_v_multigev_pid) .*
        # Ring counting migration: single-ring ↔ multi-ring
        get_double_factor(total, assets.masks.sk_1ring, assets.masks.sk_multiring, params.sk_ring_counting)
    )
end

function get_Fij_factor(Fij, param)
    factor = 1 .+ Fij .* (1 - param)
end

function get_Fij_factor_escale(Fij, masks, params)
    # Split energy scale by SK phase: SK I-III and SK IV-V bins get independent scales
    1 .+ Fij .* ((1 - params.sk_i_iii_energy_scale) .* masks.sk_i_iii_bins .+
                  (1 - params.sk_iv_v_energy_scale) .* masks.sk_iv_v_bins)
end

function get_expected(params, physics, assets)
    expected = reweight(params, physics, assets)

    total = sum([expected[key] for key in keys(expected)])

    factors = get_all_factors(params, assets, total)

    nunc = expected.nunc .* factors .* get_factor(assets.masks.mu_indices, params.sk_nc_mu_norm) .* get_factor(assets.masks.sk_elike, params.sk_ncpi0_norm) .* get_Fij_factor_escale(assets.Fij.nunc, assets.masks, params) .* get_Fij_factor(assets.Fij_updown.nunc, params.sk_updown_energy_scale)

    nue = expected.nue .* factors .* get_Fij_factor_escale(assets.Fij.nue, assets.masks, params) .* get_Fij_factor(assets.Fij_updown.nue, params.sk_updown_energy_scale)
    numu = expected.numu .* factors .* get_factor(assets.masks.sk_elike, params.sk_nue_contamination) .* get_Fij_factor_escale(assets.Fij.numu, assets.masks, params) .* get_Fij_factor(assets.Fij_updown.numu, params.sk_updown_energy_scale)
    nutau = expected.nutau .* factors .* get_Fij_factor_escale(assets.Fij.nutau, assets.masks, params) .* get_Fij_factor(assets.Fij_updown.nutau, params.sk_updown_energy_scale)
    nuebar = expected.nuebar .* factors .* get_Fij_factor_escale(assets.Fij.nuebar, assets.masks, params) .* get_Fij_factor(assets.Fij_updown.nuebar, params.sk_updown_energy_scale)
    numubar = expected.numubar .* factors .* get_factor(assets.masks.sk_elike, params.sk_nue_contamination) .* get_Fij_factor_escale(assets.Fij.numubar, assets.masks, params) .* get_Fij_factor(assets.Fij_updown.numubar, params.sk_updown_energy_scale)

    return (; nue, numu, nutau, nuebar, numubar, nunc)
end

function get_forward_model(physics, assets)
    function fwd_model(params)
        expected = get_expected(params, physics, assets)
        total = sum(expected[key] for key in keys(expected))
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
