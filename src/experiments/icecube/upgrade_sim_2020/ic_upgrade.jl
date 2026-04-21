module ic_upgrade

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

@kwdef struct ICUpgrade <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics()
    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI()))
    atm_flux = Newtrinos.atm_flux.configure(Newtrinos.atm_flux.AtmFluxConfig(nominal_model=Newtrinos.atm_flux.HKKM("spl-nu-20-01-000.d")))
    earth_layers = Newtrinos.earth_layers.configure()
    xsec = Newtrinos.xsec.configure()
    (; osc, atm_flux, earth_layers, xsec)
end

function configure(physics=default_physics())
    physics = (;physics.osc, physics.atm_flux, physics.earth_layers, physics.xsec)
    assets = get_assets(physics)
    return ICUpgrade(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)
    @info "Loading IceCube Upgrade MC"

    binning = OrderedDict()
    binning[:reco_energy_bin_edges] = 10. .^LinRange(log10(1), log10(300), 16) 
    binning[:reco_coszen_bin_edges] = LinRange(-1., 0.3, 16)
    binning[:pid_bin_edges] = -0.5:1:1.5
    binning[:type_bin_edges] = [-0.5, 0.5, 1.5]
    binning[:cz_fine_bins] = LinRange(-1,1, 201)
    binning[:log10e_fine_bins] = LinRange(-1,3,301)
    binning[:e_fine_bins] = 10 .^binning[:log10e_fine_bins]
    binning[:cz_fine] = midpoints(binning[:cz_fine_bins])
    binning[:log10e_fine] = midpoints(binning[:log10e_fine_bins])
    binning[:e_fine] = 10 .^binning[:log10e_fine]
    binning[:e_ticks] = (binning[:reco_energy_bin_edges], [@sprintf("%.1f",b) for b in binning[:reco_energy_bin_edges]])
    binning = NamedTuple(binning)
    
    nominal_layers = physics.earth_layers.compute_layers()

  
    mc_nu = CSV.read(joinpath(datadir, "neutrino_mc.csv"), FlexTable; header=true);
    # cut away things that fall outside the binning
    mc_nu.log10_true_energy = log10.(mc_nu.true_energy)
    mc_nu.true_coszen = cos.(mc_nu.true_zenith)
    mc_nu.reco_coszen = cos.(mc_nu.reco_zenith)
    mc_nu = mc_nu[mc_nu.reco_coszen .< 0.3] #(mc_nu.reco_energy .> 1.) .& (mc_nu.reco_energy .< 300.) .& 
    
    function compute_indices(mc)
        mc.e_idx = searchsortedfirst.(Ref(binning.reco_energy_bin_edges), mc.reco_energy) .- 1
        mc.c_idx = searchsortedfirst.(Ref(binning.reco_coszen_bin_edges), mc.reco_coszen) .- 1
        mc.p_idx = searchsortedfirst.(Ref(binning.pid_bin_edges), mc.pid) .- 1
        mc.t_idx = searchsortedfirst.(Ref(binning.type_bin_edges), mc.current_type) .- 1
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
      

    flux_nominal = physics.atm_flux.nominal_flux(binning.e_fine, binning.cz_fine)

    assets = (;mc, nominal_layers, binning, flux_nominal)

end


    
# ---------- DATA IS PREPARED --------------

function get_params()
    params = (
        ic_upgrade_lifetime = 3.,
        ic_upgrade_energy_scale = 1.,
        )
end

function get_priors()
    priors = (
        ic_upgrade_lifetime = Uniform(2, 4),
        ic_upgrade_energy_scale = Truncated(Normal(1, 0.02), 0.5, 1.5),
        )
end

# ------------- Define Model --------


function make_hist(e_idx, c_idx, p_idx, t_idx, w, size=(15,15,2,2))
    hist = similar(w, size)
    fill!(hist, zero(eltype(hist)))
    for i in 1:length(w)
        if (e_idx[i] < 1) | (e_idx[i] > 15)
            continue
        end
        hist[e_idx[i], c_idx[i], p_idx[i], t_idx[i]] += w[i]
    end
    hist
end

function make_hist_per_channel(mc, osc_flux, lifetime_seconds, params, assets)
    w = lifetime_seconds * mc.weight .* osc_flux
    #sys_e_idx = searchsortedfirst.(Ref(assets.binning.reco_energy_bin_edges), mc.reco_energy .* params.ic_upgrade_energy_scale) .- 1
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

    sys_flux = physics.atm_flux.sys_flux(assets.flux_nominal, params)

    s = (size(assets.binning.e_fine)[1], size(assets.binning.cz_fine)[1])

    layers = haskey(params, :electron_density_scale) ? Newtrinos.earth_layers.scale_densities(assets.nominal_layers, params.electron_density_scale) : assets.nominal_layers
    paths = physics.earth_layers.compute_paths(assets.binning.cz_fine, layers)

    p = physics.osc.osc_prob(assets.binning.e_fine * params.ic_upgrade_energy_scale, paths, layers, params)
    p_flux = reshape(sys_flux.nue, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numu, s) .* p[:, :, 2, :]

    nus = (
        nue = gather_flux(p_flux, assets.mc.nue.ef_idx, assets.mc.nue.cf_idx, 1),
        numu = gather_flux(p_flux, assets.mc.numu.ef_idx, assets.mc.numu.cf_idx, 2),
        nutau = gather_flux(p_flux, assets.mc.nutau.ef_idx, assets.mc.nutau.cf_idx, 3),
    )

    p = physics.osc.osc_prob(assets.binning.e_fine * params.ic_upgrade_energy_scale, paths, layers, params, anti=true)
    p_flux = reshape(sys_flux.nuebar, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numubar, s) .* p[:, :, 2, :]

    nubars = (
        nuebar = gather_flux(p_flux, assets.mc.nuebar.ef_idx, assets.mc.nuebar.cf_idx, 1),
        numubar = gather_flux(p_flux, assets.mc.numubar.ef_idx, assets.mc.numubar.cf_idx, 2),
        nutaubar = gather_flux(p_flux, assets.mc.nutaubar.ef_idx, assets.mc.nutaubar.cf_idx, 3),
    )

    merge(nus, nubars)
end

function get_expected(params, physics, assets)

    osc_flux = reweight(params, physics, assets)

    lifetime_seconds = params.ic_upgrade_lifetime * 365. * 24. * 3600.

    hists = (
        nue = make_hist_per_channel(assets.mc.nue, osc_flux.nue, lifetime_seconds, params, assets),
        nuebar = make_hist_per_channel(assets.mc.nuebar, osc_flux.nuebar, lifetime_seconds, params, assets),
        numu = make_hist_per_channel(assets.mc.numu, osc_flux.numu, lifetime_seconds, params, assets),
        numubar = make_hist_per_channel(assets.mc.numubar, osc_flux.numubar, lifetime_seconds, params, assets),
        nutau = make_hist_per_channel(assets.mc.nutau, osc_flux.nutau, lifetime_seconds, params, assets),
        nutaubar = make_hist_per_channel(assets.mc.nutaubar, osc_flux.nutaubar, lifetime_seconds, params, assets),
    )

    nues = hists[:nue] .+ hists[:nuebar]
    numus = hists[:numu] .+ hists[:numubar]
    nutaus = hists[:nutau] .+ hists[:nutaubar]
    expected_nu = (
    (nues[:, :, :, 1] .+ numus[:, :, :, 1] .+ nutaus[:, :, :, 1]).* physics.xsec.scale(:Any, :NC, params) .+ 
    nues[:, :, :, 2]  .+ 
    numus[:, :, :, 2]  .+ 
    nutaus[:, :, :, 2] .* physics.xsec.scale(:nutau, :CC, params)
    )    

    # set minimum number of events per bin to 1 for Poisson not to crash
    expected = max.(1., expected_nu)
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
        fig.content[16].title = "Tracks"
        rowgap!(fig.layout, 0)
        fig
    end
end

end
