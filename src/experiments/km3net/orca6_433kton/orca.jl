module orca

using LinearAlgebra
using Distributions
using DataStructures
using TypedTables
using HDF5
using StatsBase
using Statistics: mean
using CairoMakie
using BAT
using Printf
using ..Newtrinos

@kwdef struct ORCA <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics(; datadir = @__DIR__)
    # Read energy grid to set Nyquist-matched spray σ_E (one log-E bin width)
    h5file = h5open(joinpath(datadir, "ORCA6_433kton_v_0_5.h5"), "r")
    e_fine = read(h5file["E_true_axis"]["centers"])
    close(h5file)
    N_sub = 5  # oversampling factor used in get_assets
    dlogE   = mean(diff(log10.(e_fine))) / N_sub
    sigma_E = 10.0^dlogE - 1.0   # fractional energy smearing matched to oversampled sub-bin width
    @info "ORCA spray propagation: dlogE=$(round(dlogE,digits=4)), σ_E=$(round(sigma_E,digits=4)), σ_h=10 km"
    propagation = Newtrinos.osc.Spray(averaging=:gaussian, σ_E=sigma_E, σ_h=10.0)
    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI(), propagation=propagation))
    atm_flux = Newtrinos.atm_flux.configure(Newtrinos.atm_flux.AtmFluxConfig(nominal_model=Newtrinos.atm_flux.HKKM("frj-ally-20-01-mtn-solmin.d")))
    earth_layers = Newtrinos.earth_layers.configure()
    xsec = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA(mc_nominal=:G00_00a))
    (; osc, atm_flux, earth_layers, xsec)
end

function configure(physics=default_physics())
    physics = (;physics.osc, physics.atm_flux, physics.earth_layers, physics.xsec)
    assets = get_assets(physics)
    return ORCA(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)
    h5file = h5open(joinpath(datadir, "ORCA6_433kton_v_0_5.h5"), "r")
    f = read(h5file)
    mc_nu = FlexTable(Dict(Symbol(key) => f["binned_nu_response"][key] for key in keys(f["binned_nu_response"])))
    muons = Table(Dict(Symbol(key) => f["binned_muon"][key] for key in keys(f["binned_muon"])))
    data = Table(Dict(Symbol(key) => f["binned_data"][key] for key in keys(f["binned_data"])))

    binning = (
        e_fine = f["E_true_axis"]["centers"],
        cz_fine = f["Ct_true_axis"]["centers"],
        e_reco = f["E_reco_axis"]["centers"],
        cz_reco = f["Ct_reco_axis"]["centers"],
        e_fine_edges = f["E_true_axis"]["edges"],
        cz_fine_edges = f["Ct_true_axis"]["edges"],
        e_reco_edges = f["E_reco_axis"]["edges"],
        cz_reco_edges = f["Ct_reco_axis"]["edges"]
    )

    flux_nominal = physics.atm_flux.nominal_flux(binning.e_fine, binning.cz_fine)
    nominal_layers = physics.earth_layers.compute_layers()

    N_sub = 5
    log_e_edges = log10.(binning.e_fine_edges)
    n_e_fine = length(binning.e_fine)
    e_oversample = Float64[10^(log_e_edges[i] + (j - 0.5) / N_sub * (log_e_edges[i+1] - log_e_edges[i]))
                           for i in 1:n_e_fine for j in 1:N_sub]
    flux_nominal_os = physics.atm_flux.nominal_flux(e_oversample, binning.cz_fine)

    true_shape = (length(binning.e_fine), length(binning.cz_fine))
    reco_shape = (length(binning.e_reco), length(binning.cz_reco), 3, 2)

    mc_nu.he_mask = ((mc_nu.IsCC .== 0) .& (mc_nu.E_true_bin_center .> 100)) .| ((mc_nu.IsCC .== 1) .& (mc_nu.E_true_bin_center .> 500))

    mc = (
        nue = Table(mc_nu[mc_nu.Pdg .== 12, :]),
        nuebar = Table(mc_nu[mc_nu.Pdg .== -12, :]),
        numu = Table(mc_nu[mc_nu.Pdg .== 14, :]),
        numubar = Table(mc_nu[mc_nu.Pdg .== -14, :]),
        nutau = Table(mc_nu[mc_nu.Pdg .== 16, :]),
        nutaubar = Table(mc_nu[mc_nu.Pdg .== -16, :])
        )

    rs = [2, 1, 3]
    data_hist = permutedims(reshape(data.W, reco_shape[rs]), rs)
    muon_hist = permutedims(reshape(muons.W, reco_shape[rs]), rs);

    xsec_eval = if !isnothing(physics.xsec.grid_weights)
        gw = physics.xsec.grid_weights
        @info "Precomputing per-energy xsec weight arrays for ORCA MC (oversampled grid)"
        (
            nue_cc      = gw(e_oversample, :nue,   :CC, false),
            nuebar_cc   = gw(e_oversample, :nue,   :CC, true),
            numu_cc     = gw(e_oversample, :numu,  :CC, false),
            numubar_cc  = gw(e_oversample, :numu,  :CC, true),
            nutau_cc    = gw(e_oversample, :nutau, :CC, false),
            nutaubar_cc = gw(e_oversample, :nutau, :CC, true),
            nue_nc      = gw(e_oversample, :nue,   :NC, false),
            nuebar_nc   = gw(e_oversample, :nue,   :NC, true),
            numu_nc     = gw(e_oversample, :numu,  :NC, false),
            numubar_nc  = gw(e_oversample, :numu,  :NC, true),
            nutau_nc    = gw(e_oversample, :nutau, :NC, false),
            nutaubar_nc = gw(e_oversample, :nutau, :NC, true),
        )
    else
        nothing
    end

    log_e_reco_edges = log10.(binning.e_reco_edges)

    assets = (;mc, muon_hist, observed=cut(data_hist), binning, true_shape, reco_shape, nominal_layers,
               flux_nominal, flux_nominal_os, e_oversample, N_sub, xsec_eval, log_e_reco_edges)
end

function get_params()
    params = (
        orca_energy_scale = 1.,
        orca_norm_all = 1.,
        orca_norm_hpt = 1.,
        orca_norm_showers = 1.,
        orca_norm_muons = 1.,
        orca_norm_he = 1.,
        )
end

function get_priors()
    priors = (
        orca_energy_scale = Truncated(Normal(1., 0.09), 0.7, 1.3),
        orca_norm_all = Uniform(0.5, 1.5),
        orca_norm_hpt = Uniform(0.5, 1.5),
        orca_norm_showers = Uniform(0.5, 1.5),
        orca_norm_muons = Uniform(0., 2.),
        orca_norm_he = Truncated(Normal(1, 0.5), 0., 3.),
        )
end

# Function that should NOT allocate
function gather_flux(p_flux, ef, cf, j)
    result = Vector{eltype(p_flux)}(undef, length(ef))
    @inbounds for i in eachindex(ef)
        result[i] = p_flux[ef[i], cf[i], j]
    end
    result
end


function make_hist(e_idx, c_idx, p_idx, t_idx, w, size=(8,8,2,2))
    hist = similar(w, size)
    fill!(hist, zero(eltype(hist)))
    for i in 1:length(w)
        hist[e_idx[i], c_idx[i], p_idx[i], t_idx[i]] += w[i]
    end
    hist
end

function apply_energy_scale(counts, log_e_edges, delta)
    n_e = length(log_e_edges) - 1
    T = [begin
            shifted_lo = (i == 1) ? log_e_edges[1] : log_e_edges[i] + delta
            shifted_hi = (i == n_e) ? log_e_edges[end] : log_e_edges[i+1] + delta
            overlap = max(zero(delta), min(shifted_hi, log_e_edges[j+1]) - max(shifted_lo, log_e_edges[j]))
            nom_width = log_e_edges[j+1] - log_e_edges[j]
            overlap / nom_width
        end for i in 1:n_e, j in 1:n_e]
    flat = reshape(counts, n_e, :)
    return reshape(T * flat, size(counts))
end

function make_hist_per_channel(mc, osc_flux_cc, osc_flux_nc, lifetime_seconds, params, assets)
    osc_flux = ifelse.(mc.IsCC .== 1, osc_flux_cc, osc_flux_nc)
    w = lifetime_seconds * mc.W .* osc_flux .* (mc.he_mask * (params.orca_norm_he - 1.) .+ 1.0)
    make_hist(mc.E_reco_bin, mc.Ct_reco_bin, mc.AnaClass, mc.IsCC .+ 1, w, assets.reco_shape)
end


function reweight(params, physics, assets)

    sys_flux_os = physics.atm_flux.sys_flux(assets.flux_nominal_os, params)

    layers = haskey(params, :electron_density_scale) ? Newtrinos.earth_layers.scale_densities(assets.nominal_layers, params.electron_density_scale) : assets.nominal_layers
    paths = physics.earth_layers.compute_paths(assets.binning.cz_fine, layers)

    N_sub = assets.N_sub
    n_e = length(assets.binning.e_fine)
    n_cz = length(assets.binning.cz_fine)
    s_os = (length(assets.e_oversample), n_cz)

    xsec_w_os = isnothing(assets.xsec_eval) ? nothing : map(f -> f(params), assets.xsec_eval)

    function downsample(arr_os)
        dropdims(mean(reshape(arr_os, N_sub, n_e, n_cz, 3), dims=1), dims=1)
    end

    function make_pair(p_flux_os, xsec_cc_key, xsec_nc_key, mc_table, flavor_idx)
        if isnothing(xsec_w_os)
            p = downsample(p_flux_os)
            v = gather_flux(p, mc_table.E_true_bin, mc_table.Ct_true_bin, flavor_idx)
            return (cc=v, nc=v)
        else
            xcc = getfield(xsec_w_os, xsec_cc_key)
            xnc = getfield(xsec_w_os, xsec_nc_key)
            cc = gather_flux(downsample(p_flux_os .* reshape(xcc, :, 1, 1)), mc_table.E_true_bin, mc_table.Ct_true_bin, flavor_idx)
            nc = gather_flux(downsample(p_flux_os .* reshape(xnc, :, 1, 1)), mc_table.E_true_bin, mc_table.Ct_true_bin, flavor_idx)
            return (cc=cc, nc=nc)
        end
    end

    p = physics.osc.osc_prob(assets.e_oversample, paths, layers, params)
    p_flux_os = reshape(sys_flux_os.nue, s_os) .* p[:, :, 1, :] .+ reshape(sys_flux_os.numu, s_os) .* p[:, :, 2, :]

    nus = (
        nue    = make_pair(p_flux_os, :nue_cc,      :nue_nc,      assets.mc.nue,    1),
        numu   = make_pair(p_flux_os, :numu_cc,     :numu_nc,     assets.mc.numu,   2),
        nutau  = make_pair(p_flux_os, :nutau_cc,    :nutau_nc,    assets.mc.nutau,  3),
    )

    p = physics.osc.osc_prob(assets.e_oversample, paths, layers, params, anti=true)
    p_flux_os = reshape(sys_flux_os.nuebar, s_os) .* p[:, :, 1, :] .+ reshape(sys_flux_os.numubar, s_os) .* p[:, :, 2, :]

    nubars = (
        nuebar   = make_pair(p_flux_os, :nuebar_cc,   :nuebar_nc,   assets.mc.nuebar,   1),
        numubar  = make_pair(p_flux_os, :numubar_cc,  :numubar_nc,  assets.mc.numubar,  2),
        nutaubar = make_pair(p_flux_os, :nutaubar_cc, :nutaubar_nc, assets.mc.nutaubar, 3),
    )

    merge(nus, nubars)
end

function get_expected(params, physics, assets)

    osc_flux = reweight(params, physics, assets)

    lifetime_seconds = 1.

    hists = (
        nue      = make_hist_per_channel(assets.mc.nue,     osc_flux.nue.cc,     osc_flux.nue.nc,     lifetime_seconds, params, assets),
        nuebar   = make_hist_per_channel(assets.mc.nuebar,  osc_flux.nuebar.cc,  osc_flux.nuebar.nc,  lifetime_seconds, params, assets),
        numu     = make_hist_per_channel(assets.mc.numu,    osc_flux.numu.cc,    osc_flux.numu.nc,    lifetime_seconds, params, assets),
        numubar  = make_hist_per_channel(assets.mc.numubar, osc_flux.numubar.cc, osc_flux.numubar.nc, lifetime_seconds, params, assets),
        nutau    = make_hist_per_channel(assets.mc.nutau,   osc_flux.nutau.cc,   osc_flux.nutau.nc,   lifetime_seconds, params, assets),
        nutaubar = make_hist_per_channel(assets.mc.nutaubar,osc_flux.nutaubar.cc,osc_flux.nutaubar.nc,lifetime_seconds, params, assets),
    )

    hists_nc = reduce(+, map(h -> h[:, :, :, 1], values(hists)))
    hists_cc = hists.nue[:, :, :, 2] .+ hists.nuebar[:, :, :, 2] .+ hists.numu[:, :, :, 2] .+ hists.numubar[:, :, :, 2] .+ hists.nutau[:, :, :, 2] .+ hists.nutaubar[:, :, :, 2]
    expected = (assets.muon_hist * params.orca_norm_muons .+ hists_nc .+ hists_cc) * params.orca_norm_all

    delta = log10(params.orca_energy_scale)
    expected = apply_energy_scale(expected, assets.log_e_reco_edges, delta)

    # Poisson > 0; also replace NaN (from extreme param combos) since max(1e-2, NaN) = NaN
    floor_val = one(eltype(expected)) * 1e-2
    expected = ifelse.(isnan.(expected), floor_val, max.(floor_val, expected))

    c = cut(expected)
    
    return (
        hpt = c.hpt * params.orca_norm_hpt,
        showers = c.showers * params.orca_norm_showers,
        lpt = c.lpt
    )
        
end

function cut(hist)
    (
    hpt = hist[1:end-1, 1:10, 1],
    showers = hist[1:end, 1:10, 2],
    lpt = hist[1:end-1, 1:10, 3]
        )
end


function get_forward_model(physics, assets)
    function forward_model(params)
        exp_events = get_expected(params, physics, assets)
        #distprod(Poisson.(exp_events))
        distprod(map(e -> distprod(Poisson.(e)), exp_events))
    end
end

function get_plot(physics, assets)

    function plot(params, data=assets.observed)
        expected = get_expected(params, physics, assets)
    
        fig = Figure(size=(1000, 800))

        channels = [:hpt, :lpt, :showers]

        cz_bin_edges = assets.binning.cz_reco_edges[1:11]
        
        for j in 1:3
            key = channels[j]
            for i in 1:15
                ax = Axis(fig[i,j], yticklabelsize=10)
                if i > size(expected[key])[1] continue end
                stephist!(ax, midpoints(cz_bin_edges), bins=cz_bin_edges, weights=expected[key][i, :])
                scatter!(ax, midpoints(cz_bin_edges), data[key][i, :], color=:black)
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
                ax.xlabel = ""
                up = maximum((maximum(data[key][i, :]), maximum(expected[key][i, :]))) * 1.2
                ylims!(ax, 0, up)
                e_low = assets.binning.e_reco_edges[i]
                e_high = assets.binning.e_reco_edges[i+1]
                text!(ax, 0.5, 0, text=@sprintf("E in [%.1f, %.1f] GeV", e_low, e_high), align = (:center, :bottom), space = :relative)
            end
        end
        for i in [15, 30, 45]
            ax = fig.content[i]
            ax.xticklabelsvisible = true
            ax.xticksvisible = true
            ax.xlabel="cos(zenith)"
        end
        fig.content[1].title = "High-purity Tracks"
        fig.content[16].title = "Low-purity Tracks"
        fig.content[31].title = "Showers"
        rowgap!(fig.layout, 0)
        linkxaxes!(fig.content...)
        fig
    end
end

end

    
    
