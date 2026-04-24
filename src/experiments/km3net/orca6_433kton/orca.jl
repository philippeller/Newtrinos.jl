module orca

using LinearAlgebra
using Distributions
using DataStructures
using TypedTables
using HDF5
using StatsBase
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

function default_physics()
    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI()))
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
        @info "Precomputing per-energy xsec weight arrays for ORCA MC"
        (
            nue_cc      = gw(binning.e_fine, :nue,   :CC, false),
            nuebar_cc   = gw(binning.e_fine, :nue,   :CC, true),
            numu_cc     = gw(binning.e_fine, :numu,  :CC, false),
            numubar_cc  = gw(binning.e_fine, :numu,  :CC, true),
            nutau_cc    = gw(binning.e_fine, :nutau, :CC, false),
            nutaubar_cc = gw(binning.e_fine, :nutau, :CC, true),
            nue_nc      = gw(binning.e_fine, :nue,   :NC, false),
            nuebar_nc   = gw(binning.e_fine, :nue,   :NC, true),
            numu_nc     = gw(binning.e_fine, :numu,  :NC, false),
            numubar_nc  = gw(binning.e_fine, :numu,  :NC, true),
            nutau_nc    = gw(binning.e_fine, :nutau, :NC, false),
            nutaubar_nc = gw(binning.e_fine, :nutau, :NC, true),
        )
    else
        nothing
    end

    assets = (;mc, muon_hist, observed=cut(data_hist), binning, true_shape, reco_shape, nominal_layers, flux_nominal, xsec_eval)
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

function make_hist_per_channel(mc, osc_flux, lifetime_seconds, params, assets, xsec_cc=nothing, xsec_nc=nothing)
    w = lifetime_seconds * mc.W .* osc_flux .* (mc.he_mask * (params.orca_norm_he - 1.) .+ 1.0)
    if !isnothing(xsec_cc) && !isnothing(xsec_nc)
        w = w .* ifelse.(mc.IsCC .== 1, xsec_cc[mc.E_true_bin], xsec_nc[mc.E_true_bin])
    end
    make_hist(mc.E_reco_bin, mc.Ct_reco_bin, mc.AnaClass, mc.IsCC .+ 1, w, assets.reco_shape)
end


function reweight(params, physics, assets)

    flux = physics.atm_flux.nominal_flux(assets.binning.e_fine * params.orca_energy_scale, assets.binning.cz_fine)
    sys_flux = physics.atm_flux.sys_flux(flux, params)

    s = assets.true_shape

    layers = haskey(params, :electron_density_scale) ? Newtrinos.earth_layers.scale_densities(assets.nominal_layers, params.electron_density_scale) : assets.nominal_layers
    paths = physics.earth_layers.compute_paths(assets.binning.cz_fine, layers)

    p = physics.osc.osc_prob(assets.binning.e_fine * params.orca_energy_scale, paths, layers, params)
    p_flux = reshape(sys_flux.nue, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numu, s) .* p[:, :, 2, :]

    nus = (
        nue = gather_flux(p_flux, assets.mc.nue.E_true_bin, assets.mc.nue.Ct_true_bin, 1),
        numu = gather_flux(p_flux, assets.mc.numu.E_true_bin, assets.mc.numu.Ct_true_bin, 2),
        nutau = gather_flux(p_flux, assets.mc.nutau.E_true_bin, assets.mc.nutau.Ct_true_bin, 3),
    )

    p = physics.osc.osc_prob(assets.binning.e_fine * params.orca_energy_scale, paths, layers, params, anti=true)
    p_flux = reshape(sys_flux.nuebar, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numubar, s) .* p[:, :, 2, :]

    nubars = (
        nuebar = gather_flux(p_flux, assets.mc.nuebar.E_true_bin, assets.mc.nuebar.Ct_true_bin, 1),
        numubar = gather_flux(p_flux, assets.mc.numubar.E_true_bin, assets.mc.numubar.Ct_true_bin, 2),
        nutaubar = gather_flux(p_flux, assets.mc.nutaubar.E_true_bin, assets.mc.nutaubar.Ct_true_bin, 3),
    )

    merge(nus, nubars)
end

function get_expected(params, physics, assets)

    osc_flux = reweight(params, physics, assets)

    lifetime_seconds = 1.

    xsec_w = isnothing(assets.xsec_eval) ? nothing : map(f -> f(params), assets.xsec_eval)

    function _xsec(key_cc, key_nc)
        isnothing(xsec_w) ? (nothing, nothing) : (getfield(xsec_w, key_cc), getfield(xsec_w, key_nc))
    end

    hists = (
        nue      = make_hist_per_channel(assets.mc.nue,     osc_flux.nue,     lifetime_seconds, params, assets, _xsec(:nue_cc,      :nue_nc)...),
        nuebar   = make_hist_per_channel(assets.mc.nuebar,  osc_flux.nuebar,  lifetime_seconds, params, assets, _xsec(:nuebar_cc,   :nuebar_nc)...),
        numu     = make_hist_per_channel(assets.mc.numu,    osc_flux.numu,    lifetime_seconds, params, assets, _xsec(:numu_cc,     :numu_nc)...),
        numubar  = make_hist_per_channel(assets.mc.numubar, osc_flux.numubar, lifetime_seconds, params, assets, _xsec(:numubar_cc,  :numubar_nc)...),
        nutau    = make_hist_per_channel(assets.mc.nutau,   osc_flux.nutau,   lifetime_seconds, params, assets, _xsec(:nutau_cc,    :nutau_nc)...),
        nutaubar = make_hist_per_channel(assets.mc.nutaubar,osc_flux.nutaubar,lifetime_seconds, params, assets, _xsec(:nutaubar_cc, :nutaubar_nc)...),
    )

    hists_nc = reduce(+, map(h -> h[:, :, :, 1], values(hists)))
    hists_cc = hists.nue[:, :, :, 2] .+ hists.nuebar[:, :, :, 2] .+ hists.numu[:, :, :, 2] .+ hists.numubar[:, :, :, 2] .+ hists.nutau[:, :, :, 2] .+ hists.nutaubar[:, :, :, 2]
    expected = (assets.muon_hist * params.orca_norm_muons .+ hists_nc .+ hists_cc) * params.orca_norm_all

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

    
    
