module deepcore

using LinearAlgebra
using Distributions
using DataStructures
using TypedTables
using CSV
using StatsBase
using CairoMakie
using Logging
using BAT
using Memoize
using Printf
using ..Newtrinos

@kwdef struct DeepCore <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function configure(physics)
    physics = (;physics.osc, physics.atm_flux, physics.earth_layers, physics.xsec)
    assets = get_assets(physics)
    return DeepCore(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)
    @info "Loading deepcore data"

    binning = OrderedDict()
    binning[:reco_energy_bin_edges] = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413]
    binning[:reco_coszen_bin_edges] = [-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.]
    binning[:pid_bin_edges] = -0.5:1:1.5
    binning[:type_bin_edges] = [-0.5, 0.5, 3.5]
    binning[:cz_fine_bins] = LinRange(-1,1, 201)
    binning[:log10e_fine_bins] = LinRange(0,3,201)
    binning[:e_fine_bins] = 10 .^binning[:log10e_fine_bins]
    binning[:cz_fine] = midpoints(binning[:cz_fine_bins])
    binning[:log10e_fine] = midpoints(binning[:log10e_fine_bins])
    binning[:e_fine] = 10 .^binning[:log10e_fine]
    binning[:e_ticks] = (binning[:reco_energy_bin_edges], [@sprintf("%.1f",b) for b in binning[:reco_energy_bin_edges]])
    binning = NamedTuple(binning)
    
    layers = physics.earth_layers.compute_layers()
    paths = physics.earth_layers.compute_paths(binning.cz_fine, layers)

  
    mc_nu = CSV.read(joinpath(datadir, "neutrino_mc.csv"), FlexTable; header=true);
    mc_nu.log10_true_energy = log10.(mc_nu.true_energy)
    
    function compute_indices(mc)
        mc.e_idx = searchsortedfirst.(Ref(binning.reco_energy_bin_edges), mc.reco_energy) .- 1
        mc.c_idx = searchsortedfirst.(Ref(binning.reco_coszen_bin_edges), mc.reco_coszen) .- 1
        mc.p_idx = searchsortedfirst.(Ref(binning.pid_bin_edges), mc.pid) .- 1
        mc.t_idx = searchsortedfirst.(Ref(binning.type_bin_edges), mc.type) .- 1
        mc.ef_idx = searchsortedfirst.(Ref(binning.log10e_fine_bins), mc.log10_true_energy) .- 1
        mc.cf_idx = searchsortedfirst.(Ref(binning.cz_fine_bins), mc.true_coszen) .- 1
    end
    
    compute_indices(mc_nu);
    
    mc = (
        nue = Table(mc_nu[mc_nu.pdg .== 12, :]),
        nuebar = Table(mc_nu[mc_nu.pdg .== -12, :]),
        numu = Table(mc_nu[mc_nu.pdg .== 14, :]),
        numubar = Table(mc_nu[mc_nu.pdg .== -14, :]),
        nutau = Table(mc_nu[mc_nu.pdg .== 16, :]),
        nutaubar = Table(mc_nu[mc_nu.pdg .== -16, :])
        )
    
    function read_csv_into_hist(filename)
        csv = CSV.read(joinpath(datadir, filename), Table; header=true)
        vars_to_extract = setdiff(columnnames(csv), (:reco_coszen, :reco_energy, :pid))
        d = OrderedDict()
        for var in vars_to_extract
            d[var] = fit(Histogram, (csv.reco_energy, csv.reco_coszen, csv.pid), weights(columns(csv)[var]), (binning.reco_energy_bin_edges, binning.reco_coszen_bin_edges, binning.pid_bin_edges)).weights
        end
        Table(;d...)
    end
    
    muons = read_csv_into_hist("muons.csv")
    data = read_csv_into_hist("data.csv")
    hyperplanes = (
        nuall_nc = read_csv_into_hist("hyperplanes_all_nc.csv"),
        nue_cc = read_csv_into_hist("hyperplanes_nue_cc.csv"),
        numu_cc = read_csv_into_hist("hyperplanes_numu_cc.csv"),
        nutau_cc = read_csv_into_hist("hyperplanes_nutau_cc.csv")
        )    

    flux = physics.atm_flux.nominal_flux(binning.e_fine, binning.cz_fine)

    assets = (;mc, hyperplanes, flux, muons, observed=data.count, layers, paths, binning)

end


    
# ---------- DATA IS PREPARED --------------

function get_params()
    params = (
        deepcore_lifetime = 2.5,
        deepcore_atm_muon_scale = 1.,
        deepcore_ice_absorption = 1.,
        deepcore_ice_scattering = 1.,
        deepcore_opt_eff_overall = 1.,
        deepcore_opt_eff_lateral = 0.,
        deepcore_opt_eff_headon = 0.,
        )
end

function get_priors()
    priors = (
        deepcore_lifetime = Uniform(2, 4),
        deepcore_atm_muon_scale = Uniform(0, 2),
        deepcore_ice_absorption = Truncated(Normal(1, 0.1), 0.85, 1.15),
        deepcore_ice_scattering = Truncated(Normal(1, 0.1), 0.85, 1.15),
        deepcore_opt_eff_overall = Truncated(Normal(1, 0.1), 0.8, 1.2),
        deepcore_opt_eff_lateral = Truncated(Normal(0, 1.), -2, 2),
        deepcore_opt_eff_headon = Uniform(-5, 2.),
        )
end

# ------------- Define Model --------


function make_hist(e_idx, c_idx, p_idx, t_idx, w, size=(8,8,2,2))
    hist = similar(w, size)
    for i in 1:prod(size)
        hist[i] = 0.
    end
    for i in 1:length(w)
        hist[e_idx[i], c_idx[i], p_idx[i], t_idx[i]] += w[i]
    end
    hist
end

function make_hist_per_channel(mc, osc_flux, lifetime_seconds)
    w = lifetime_seconds * mc.weight .* osc_flux
    make_hist(mc.e_idx, mc.c_idx, mc.p_idx, mc.t_idx, w)
end


# Function that should NOT allocate
function gather_flux(p_flux, ef, cf, j)
    result = Vector{eltype(p_flux)}(undef, length(ef))
    @inbounds for i in eachindex(ef)
        result[i] = p_flux[ef[i], cf[i], j]
    end
    result
end


function reweight(params, physics, assets)
    sys_flux = physics.atm_flux.sys_flux(assets.flux, params)

    s = (size(assets.binning.e_fine)[1], size(assets.binning.cz_fine)[1])

    p = physics.osc.osc_prob(assets.binning.e_fine, assets.paths, assets.layers, params)
    p_flux = reshape(sys_flux.nue, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numu, s) .* p[:, :, 2, :]
    
    nus = NamedTuple(ch=>gather_flux(p_flux, assets.mc[ch].ef_idx, assets.mc[ch].cf_idx, i) for (i, ch) in enumerate([:nue, :numu, :nutau]))
    
    p = physics.osc.osc_prob(assets.binning.e_fine, assets.paths, assets.layers, params, anti=true)
    p_flux = reshape(sys_flux.nuebar, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numubar, s) .* p[:, :, 2, :]

    nubars = NamedTuple(ch=>gather_flux(p_flux, assets.mc[ch].ef_idx, assets.mc[ch].cf_idx, i) for (i, ch) in enumerate([:nuebar, :numubar, :nutaubar]))

    merge(nus, nubars)
end

function get_hyperplane_factor(hyperplane, params)
    f = (
        hyperplane.offset .+
        (hyperplane.ice_absorption * 100*(params.deepcore_ice_absorption -1)) .+ 
        (hyperplane.ice_scattering * 100*(params.deepcore_ice_scattering -1)) .+
        (hyperplane.opt_eff_overall .* params.deepcore_opt_eff_overall) .+
        (hyperplane.opt_eff_lateral * ((params.deepcore_opt_eff_lateral*10) +25.)) .+
        (hyperplane.opt_eff_headon * params.deepcore_opt_eff_headon)
        )
    f
end

function apply_hyperplanes(hists, params, physics, hyperplanes)
    f_nc = get_hyperplane_factor(hyperplanes.nuall_nc, params)
    f_nue_cc = get_hyperplane_factor(hyperplanes.nue_cc, params)
    f_numu_cc = get_hyperplane_factor(hyperplanes.numu_cc, params)
    f_nutau_cc = get_hyperplane_factor(hyperplanes.nutau_cc, params) 

    nues = hists[:nue] .+ hists[:nuebar]
    numus = hists[:numu] .+ hists[:numubar]
    nutaus = hists[:nutau] .+ hists[:nutaubar]
    (
    (nues[:, :, :, 1] .+ numus[:, :, :, 1] .+ nutaus[:, :, :, 1]).* f_nc .* physics.xsec.scale(:Any, :NC, params) .+ 
    nues[:, :, :, 2] .* f_nue_cc .+ 
    numus[:, :, :, 2] .* f_numu_cc .+ 
    nutaus[:, :, :, 2] .* f_nutau_cc .* physics.xsec.scale(:nutau, :CC, params)
    )    
end 

function get_expected(params, physics, assets)

    osc_flux = reweight(params, physics, assets)

    lifetime_seconds = params.deepcore_lifetime * 365. * 24. * 3600.

    hists = NamedTuple(ch=>make_hist_per_channel(assets.mc[ch], osc_flux[ch], lifetime_seconds) for ch in keys(assets.mc))
    
    expected_nu = apply_hyperplanes(hists, params, physics, assets.hyperplanes)

    # set minimum number of events per bin to 1 for Poisson not to crash
    expected = max.(1., (expected_nu .+ params.deepcore_atm_muon_scale .* assets.muons.count))
end


function get_forward_model(physics, assets)
    function forward_model(params)
        exp_events = get_expected(params, physics, assets)
        distprod(Poisson.(exp_events))
    end
end

    
# function plotmap(h; colormap=Reverse(:Spectral), symm=false)
#     asset = get_assets()

#     if symm
#         colorrange = (-maximum(abs.(h)), maximum(abs.(h)))
#     else
#         colorrange = (0, maximum(h))
#     end
    
#     fig = Figure(size=(800, 400))
#     ax = Axis(fig[1,1], xscale=log10, xticks=assets.binning.e_ticks, xlabel="E (GeV)", ylabel="cos(zenith)", title="Cascades")
#     hm = heatmap!(ax, assets.binning.reco_energy_bin_edges, assets.binning.reco_coszen_bin_edges, h[:, :, 1], colormap=colormap, colorrange=colorrange)
#     ax = Axis(fig[1,2], xscale=log10, xticks=e_ticks, xlabel="E (GeV)", yticksvisible=true, yticklabelsvisible=false, title="Tracks")
#     hm = heatmap!(ax, assets.binning.reco_energy_bin_edges, assets.binning.reco_coszen_bin_edges, h[:, :, 2], colormap=colormap, colorrange=colorrange)
#     Colorbar(fig[1,3], hm)
#     fig
# end

function get_plot(physics, assets)

    function plot(params, data=assets.observed)
        expected = get_expected(params, physics, assets)
    
        fig = Figure(size=(800, 600))
        for j in 1:2
            for i in 1:size(expected)[1]
                ax = Axis(fig[i,j], yticklabelsize=10)
                stephist!(ax, midpoints(assets.binning.reco_coszen_bin_edges), bins=assets.binning.reco_coszen_bin_edges, weights=expected[i, :, j])
                scatter!(ax, midpoints(assets.binning.reco_coszen_bin_edges), data[i, :, j], color=:black)
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
                ax.xlabel = ""
                up = maximum((maximum(data[i, :, j]), maximum(expected[i, :, j]))) * 1.2
                ylims!(ax, 0, up)
                e_low = assets.binning.reco_energy_bin_edges[i]
                e_high = assets.binning.reco_energy_bin_edges[i+1]
                text!(ax, 0.5, 0, text=@sprintf("E in [%.1f, %.1f] GeV", e_low, e_high), align = (:center, :bottom), space = :relative)
            end
        end
        for i in 1:2
            ax = fig.content[size(expected)[1]*i]
            ax.xticklabelsvisible = true
            ax.xticksvisible = true
            ax.xlabel="cos(zenith)"
        end
        fig.content[1].title = "Cascades"
        fig.content[9].title = "Tracks"
        rowgap!(fig.layout, 0)
        fig
    end

end

end
