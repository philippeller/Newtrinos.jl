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
    earth_layers = Newtrinos.earth_layers.configure(Newtrinos.earth_layers.VariableDensity())
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

    layers = haskey(params, :matter_density_scale) ? Newtrinos.earth_layers.scale_densities(assets.nominal_layers, params.matter_density_scale) : assets.nominal_layers
    paths = physics.earth_layers.compute_paths(assets.cz_midpoints, layers)

    p = physics.osc.osc_prob(E, paths, layers, params);
    p_anti = physics.osc.osc_prob(E, paths, layers, params, anti=true);

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

    nue     = contract_R(assets.R.nue,     nue_flux)
    numu    = contract_R(assets.R.numu,    numu_flux)
    nutau   = contract_R(assets.R.nutau,   nutau_flux)
    nuebar  = contract_R(assets.R.nuebar,  nuebar_flux)
    numubar = contract_R(assets.R.numubar, numubar_flux)
    nunc    = contract_R(assets.R.nunc,    ones(eltype(nue_flux), s) .* xsec_nc .* flux_norm)

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

    R_3d = NamedTuple(key => make_response_matrix(MC[key], loge_grid, cz_grid) for key in keys(MC))
    R = flatten_R(R_3d)
    nominal_weights = calc_weights(params_nominal, (;R, flux_nominal, paths, nominal_layers, loge_grid, cz_midpoints), physics)

    R_plus_3d = NamedTuple(key => make_response_matrix(MC[key], loge_grid .+ log(1.02), cz_grid) for key in keys(MC))
    R_minus_3d = NamedTuple(key => make_response_matrix(MC[key], loge_grid .+ log(0.98), cz_grid) for key in keys(MC))
    weights_plus = calc_weights(params_nominal, (;R=flatten_R(R_plus_3d), flux_nominal, paths, nominal_layers, loge_grid, cz_midpoints), physics)
    weights_minus = calc_weights(params_nominal, (;R=flatten_R(R_minus_3d), flux_nominal, paths, nominal_layers, loge_grid, cz_midpoints), physics)
    Fij = NamedTuple(key => safe_div.((weights_plus[key] .- weights_minus[key]), (2*0.02 .* nominal_weights[key])) for key in keys(nominal_weights))

    for key in keys(R_3d)
        R_plus_3d[key][:,:,1:50] .= R_3d[key][:,:,1:50]
        R_minus_3d[key][:,:,1:50] .= R_3d[key][:,:,1:50]
    end
    weights_plus = calc_weights(params_nominal, (;R=flatten_R(R_plus_3d), flux_nominal, paths, nominal_layers, loge_grid, cz_midpoints), physics)
    weights_minus = calc_weights(params_nominal, (;R=flatten_R(R_minus_3d), flux_nominal, paths, nominal_layers, loge_grid, cz_midpoints), physics)
    Fij_updown = NamedTuple(key => safe_div.((weights_plus[key] .- weights_minus[key]), (2*0.02 .* nominal_weights[key])) for key in keys(nominal_weights))

    return (; MC, R, Fij, Fij_updown, flux_nominal, nominal_layers, loge_grid, cz_grid, cz_midpoints, nominal_weights, observed, bininfo, masks)

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
        sk_i_iii_energy_scale = Normal(1.0, 0.033),
        sk_iv_v_energy_scale = Normal(1.0, 0.021),
        # Up/down energy scale from Table 5.6, split by phase group
        # SK I: 1.3%, SK II: 0.6%, SK III: 0.7% → exposure-weighted ~1.0%
        # SK IV: 0.5%, SK V: 0.7% → exposure-weighted ~0.6%
        sk_i_iii_updown_energy_scale = Normal(1.0, 0.01),
        sk_iv_v_updown_energy_scale = Normal(1.0, 0.006),
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
        get_double_factor(total, assets.masks.sk_i_v_multigev_multiring_nuebar, assets.masks.sk_i_v_multigev_multiring_nue, params.sk_i_v_bdt_1) .*
        get_double_factor(total, assets.masks.sk_i_v_multigev_multiring_nue, assets.masks.sk_i_v_multigev_multiring_mu, params.sk_i_v_bdt_2) .*
        get_double_factor(total, assets.masks.sk_i_v_multigev_multiring_mu, assets.masks.sk_i_v_multigev_multiring_other, params.sk_i_v_bdt_3) .*
        # PID migration: e-like ↔ mu-like
        get_double_factor(total, assets.masks.sk_i_iii_subgev_elike, assets.masks.sk_i_iii_subgev_mulike, params.sk_i_iii_subgev_pid) .*
        get_double_factor(total, assets.masks.sk_iv_v_subgev_elike, assets.masks.sk_iv_v_subgev_mulike, params.sk_iv_v_subgev_pid) .*
        get_double_factor(total, assets.masks.sk_i_iii_multigev_1ring_elike, assets.masks.sk_i_iii_multigev_1ring_mulike, params.sk_i_iii_multigev_pid) .*
        get_double_factor(total, assets.masks.sk_iv_v_multigev_1ring_elike, assets.masks.sk_iv_v_multigev_1ring_mulike, params.sk_iv_v_multigev_pid) .*
        # Ring counting migration: overall + split by energy
        get_double_factor(total, assets.masks.sk_1ring, assets.masks.sk_multiring, params.sk_ring_counting) .*
        get_double_factor(total, assets.masks.sk_subgev_1ring, assets.masks.sk_subgev_multiring, params.sk_subgev_ring_counting) .*
        get_double_factor(total, assets.masks.sk_multigev_1ring, assets.masks.sk_multigev_multiring, params.sk_multigev_ring_counting) .*
        # Relative normalizations for high-energy samples
        get_factor(assets.masks.fc_multigev, params.sk_fc_multigev_rel_norm) .*
        get_factor(assets.masks.pc_upmu, params.sk_pc_upmu_rel_norm) .*
        # FC/PC separation: FC multi-GeV mu-like ↔ PC
        get_double_factor(total, assets.masks.fc_multigev_mulike, assets.masks.pc, params.sk_fc_pc_separation) .*
        # pi0 selection
        get_double_factor(total, assets.masks.sk_1ring_pi0, assets.masks.sk_2ring_pi0, params.sk_pi0_norm)
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

function get_Fij_factor_updown(Fij_updown, masks, params)
    # Split up/down energy scale by SK phase
    1 .+ Fij_updown .* ((1 - params.sk_i_iii_updown_energy_scale) .* masks.sk_i_iii_bins .+
                         (1 - params.sk_iv_v_updown_energy_scale) .* masks.sk_iv_v_bins)
end

function get_expected(params, physics, assets)
    expected = reweight(params, physics, assets)

    total = reduce(+, values(expected))

    factors = get_all_factors(params, assets, total)

    nunc = expected.nunc .* factors .* get_factor(assets.masks.mu_indices, params.sk_nc_mu_norm) .* get_factor(assets.masks.sk_elike, params.sk_ncpi0_norm) .* get_Fij_factor_escale(assets.Fij.nunc, assets.masks, params) .* get_Fij_factor_updown(assets.Fij_updown.nunc, assets.masks, params)

    nue = expected.nue .* factors .* get_Fij_factor_escale(assets.Fij.nue, assets.masks, params) .* get_Fij_factor_updown(assets.Fij_updown.nue, assets.masks, params)
    numu = expected.numu .* factors .* get_factor(assets.masks.sk_elike, params.sk_nue_contamination) .* get_Fij_factor_escale(assets.Fij.numu, assets.masks, params) .* get_Fij_factor_updown(assets.Fij_updown.numu, assets.masks, params)
    nutau = expected.nutau .* factors .* get_Fij_factor_escale(assets.Fij.nutau, assets.masks, params) .* get_Fij_factor_updown(assets.Fij_updown.nutau, assets.masks, params)
    nuebar = expected.nuebar .* factors .* get_Fij_factor_escale(assets.Fij.nuebar, assets.masks, params) .* get_Fij_factor_updown(assets.Fij_updown.nuebar, assets.masks, params)
    numubar = expected.numubar .* factors .* get_factor(assets.masks.sk_elike, params.sk_nue_contamination) .* get_Fij_factor_escale(assets.Fij.numubar, assets.masks, params) .* get_Fij_factor_updown(assets.Fij_updown.numubar, assets.masks, params)

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
