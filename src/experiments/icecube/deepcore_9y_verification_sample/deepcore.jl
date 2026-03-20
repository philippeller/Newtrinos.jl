module deepcore


"""
ToDos:
- hypersurface uncertainties
- MC uncertainties
- muon uncertainties
- DIS xsec
- GENIE xsecs
- DEAMON flux?
- enough energy bins??
- NC norm is fucked.....probably needs 1/3 applied or so
"""

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
    binning[:reco_energy_bin_edges] = [6.31, 8.45862141, 11.33887101, 15.19987592, 20.37559363, 27.3136977, 36.61429921, 49.08185342, 65.79474104, 88.19854278, 158.49]
    binning[:reco_coszen_bin_edges] = [-1., -0.89, -0.78, -0.67, -0.56, -0.45, -0.34, -0.23, -0.12, -0.01, 0.1]
    binning[:pid_bin_edges] = [0.55, 0.75, 1.00]
    binning[:type_bin_edges] = [-0.5, 0.5, 1.5] ## NC vs CC
    binning[:hs_dm31] = LinRange(0.0015, 0.0035, 20)
    binning[:hs_dm31_bin_edges] = LinRange(0.0015 - 0.002/38, 0.0035 + 0.002/38, 21) # approximate bins, does not matter as long as the values above are contained
    binning[:cz_fine_bins] = LinRange(-1,1, 201)
    binning[:log10e_fine_bins] = LinRange(0,4,201)
    binning[:e_fine_bins] = 10 .^binning[:log10e_fine_bins]
    binning[:cz_fine] = midpoints(binning[:cz_fine_bins])
    binning[:log10e_fine] = midpoints(binning[:log10e_fine_bins])
    binning[:e_fine] = 10 .^binning[:log10e_fine]
    binning[:e_ticks] = (binning[:reco_energy_bin_edges], [@sprintf("%.1f",b) for b in binning[:reco_energy_bin_edges]])
    binning = NamedTuple(binning)
    
    layers = physics.earth_layers.compute_layers()
    paths = physics.earth_layers.compute_paths(binning.cz_fine, layers)


    mc_flextable = (nu_nc = CSV.read(joinpath(datadir, "mc_nu_nc.csv"), FlexTable; header=true),
                    nue_cc = CSV.read(joinpath(datadir, "mc_nue_cc.csv"), FlexTable; header=true),
                    numu_cc = CSV.read(joinpath(datadir, "mc_numu_cc.csv"), FlexTable; header=true),
                    nutau_cc = CSV.read(joinpath(datadir, "mc_nutau_cc.csv"), FlexTable; header=true)
        );



    function compute_indices(mc)
        mc.e_idx = searchsortedfirst.(Ref(binning.reco_energy_bin_edges), mc.reco_energy) .- 1
        mc.c_idx = searchsortedfirst.(Ref(binning.reco_coszen_bin_edges), mc.reco_coszen) .- 1
        mc.p_idx = searchsortedfirst.(Ref(binning.pid_bin_edges), mc.pid) .- 1
        mc.ef_idx = searchsortedfirst.(Ref(binning.log10e_fine_bins), mc.log10_true_energy) .- 1
        mc.cf_idx = searchsortedfirst.(Ref(binning.cz_fine_bins), mc.true_coszen) .- 1
    end
    for key in keys(mc_flextable)
        mc_flextable[key].log10_true_energy = log10.(mc_flextable[key].true_energy)
        compute_indices(mc_flextable[key])
    end
    
    mc = NamedTuple(key => Table(mc_flextable[key]) for key in keys(mc_flextable))

    mc = (
        nu_nc = Table(mc_flextable.nu_nc[mc_flextable.nu_nc.pdg .> 0, :]),
        nubar_nc = Table(mc_flextable.nu_nc[mc_flextable.nu_nc.pdg .< 0, :]),
        nue_cc = Table(mc_flextable.nue_cc[mc_flextable.nue_cc.pdg .> 0, :]),
        nuebar_cc = Table(mc_flextable.nue_cc[mc_flextable.nue_cc.pdg .< 0, :]),
        numu_cc = Table(mc_flextable.numu_cc[mc_flextable.numu_cc.pdg .> 0, :]),
        numubar_cc = Table(mc_flextable.numu_cc[mc_flextable.numu_cc.pdg .< 0, :]),
        nutau_cc = Table(mc_flextable.nutau_cc[mc_flextable.nutau_cc.pdg .> 0, :]),
        nutaubar_cc = Table(mc_flextable.nutau_cc[mc_flextable.nutau_cc.pdg .< 0, :]),
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

    function read_hs_csv_into_hist(filename)
        csv = CSV.read(joinpath(datadir, filename), Table; header=true)
        vars_to_extract = setdiff(columnnames(csv), (:reco_coszen, :reco_energy, :pid, :deltam31))
        d = OrderedDict()
        for var in vars_to_extract
            d[var] = fit(Histogram, (csv.reco_energy, csv.reco_coszen, csv.pid, csv.deltam31), weights(columns(csv)[var]), (binning.reco_energy_bin_edges, 
                    binning.reco_coszen_bin_edges, binning.pid_bin_edges, binning.hs_dm31_bin_edges)).weights
        end
        Table(;d...)
    end
    
    muons = read_csv_into_hist("mc_mu.csv")
    data = read_csv_into_hist("data.csv")
    hypersurfaces = (
        nu_nc_nue_cc = read_hs_csv_into_hist("hs_nu_nc_nue_cc.csv"),
        numu_cc = read_hs_csv_into_hist("hs_numu_cc.csv"),
        nutau_cc = read_hs_csv_into_hist("hs_nutau_cc.csv")
        )

    
    flux = physics.atm_flux.nominal_flux(binning.e_fine, binning.cz_fine)

    assets = (;mc, hypersurfaces, flux, muons, observed=data.count, layers, paths, binning)

end

    
# ---------- DATA IS PREPARED --------------

function get_params()
    params = (
        deepcore_aeff_scale = 1.,
        deepcore_atm_muon_scale = 1.,
        deepcore_ice_absorption = 1.,
        deepcore_ice_scattering = 1.,
        deepcore_opt_eff_overall = 1.,
        deepcore_rel_eff_p0 = 0.1,
        deepcore_rel_eff_p1 = -0.05,
        )
end

function get_priors()
    priors = (
        deepcore_aeff_scale = Uniform(0.5, 2),
        deepcore_atm_muon_scale = Uniform(0, 2),
        deepcore_ice_absorption = Uniform(0.8, 1.2),
        deepcore_ice_scattering = Uniform(0.8, 1.2),
        deepcore_opt_eff_overall = Truncated(Normal(1, 0.1), 0.8, 1.2),
        deepcore_rel_eff_p0 = Uniform(-1, 0.5),
        deepcore_rel_eff_p1 = Uniform(-0.15, 0.05),
        )
end

# ------------- Define Model --------


function make_hist(e_idx, c_idx, p_idx, w, size=(10,10,2))
    hist = similar(w, size)
    fill!(hist, zero(eltype(hist)))
    for i in 1:length(w)
        hist[e_idx[i], c_idx[i], p_idx[i]] += w[i]
    end
    hist
end

function make_hist_per_channel(mc, osc_flux, lifetime_seconds)
    w = lifetime_seconds * mc.weight .* osc_flux
    make_hist(mc.e_idx, mc.c_idx, mc.p_idx, w)
end


# Function that should NOT allocate
function gather_flux(p_flux, ef, cf, j)
    result = Vector{eltype(p_flux)}(undef, length(ef))
    @inbounds for i in eachindex(ef)
        result[i] = p_flux[ef[i], cf[i], j]
    end
    result
end

# Function that should NOT allocate
function gather_flux_nc(s_flux, ef, cf)
    result = Vector{eltype(s_flux)}(undef, length(ef))
    @inbounds for i in eachindex(ef)
        result[i] = s_flux[ef[i], cf[i]] / 3
    end
    result
end


function reweight(params, physics, assets)
    sys_flux = physics.atm_flux.sys_flux(assets.flux, params)

    s = (size(assets.binning.e_fine)[1], size(assets.binning.cz_fine)[1])

    p = physics.osc.osc_prob(assets.binning.e_fine, assets.paths, assets.layers, params)
    p_flux = reshape(sys_flux.nue, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numu, s) .* p[:, :, 2, :]
    
    nus = (
        nue_cc = gather_flux(p_flux, assets.mc.nue_cc.ef_idx, assets.mc.nue_cc.cf_idx, 1),
        numu_cc = gather_flux(p_flux, assets.mc.numu_cc.ef_idx, assets.mc.numu_cc.cf_idx, 2),
        nutau_cc = gather_flux(p_flux, assets.mc.nutau_cc.ef_idx, assets.mc.nutau_cc.cf_idx, 3),
    )

    nu_ncs = (nu_nc = gather_flux_nc(reshape(sys_flux.nue .+ sys_flux.numu, s),  assets.mc[:nu_nc].ef_idx, assets.mc[:nu_nc].cf_idx),)
    
    p = physics.osc.osc_prob(assets.binning.e_fine, assets.paths, assets.layers, params, anti=true)
    p_flux = reshape(sys_flux.nuebar, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numubar, s) .* p[:, :, 2, :]

    nubars = (
        nuebar_cc = gather_flux(p_flux, assets.mc.nuebar_cc.ef_idx, assets.mc.nuebar_cc.cf_idx, 1),
        numubar_cc = gather_flux(p_flux, assets.mc.numubar_cc.ef_idx, assets.mc.numubar_cc.cf_idx, 2),
        nutaubar_cc = gather_flux(p_flux, assets.mc.nutaubar_cc.ef_idx, assets.mc.nutaubar_cc.cf_idx, 3),
    )

    nubar_ncs = (nubar_nc = gather_flux_nc(reshape(sys_flux.nue .+ sys_flux.numu, s),  assets.mc[:nubar_nc].ef_idx, assets.mc[:nubar_nc].cf_idx),)
    
    merge(nu_ncs, nus, nubar_ncs, nubars)
end

function interpolate_hypersurface(h, idx, fraction)
    (1 - fraction) * h[:, :, :, idx] .+ fraction * h[:, :, :, idx+1]
end

function get_hypersurface_factor(hypersurface, idx, fraction, params)
    f = (
        interpolate_hypersurface(hypersurface.intercept, idx, fraction) .+
        (interpolate_hypersurface(hypersurface.bulk_ice_abs, idx, fraction) * (params.deepcore_ice_absorption - 1)) .+ 
        (interpolate_hypersurface(hypersurface.bulk_ice_scatter, idx, fraction) * (params.deepcore_ice_scattering - 1)) .+
        (interpolate_hypersurface(hypersurface.dom_eff, idx, fraction) .* (params.deepcore_opt_eff_overall - 1)) .+
        (interpolate_hypersurface(hypersurface.hole_ice_p0, idx, fraction) * (params.deepcore_rel_eff_p0 - 0.1)) .+
        (interpolate_hypersurface(hypersurface.hole_ice_p0, idx, fraction) * (params.deepcore_rel_eff_p1 + 0.05))
        )
    f
end

function apply_hypersurfaces(hists, params, physics, assets)

    x = params.Δm²₃₁
    X = assets.binning.hs_dm31
    Δx = X[2] - X[1]
    idx = floor(Int, (x - X[1]) / Δx)
    fraction = (x - X[idx]) / Δx

    f_nu_nc_nue_cc = get_hypersurface_factor(assets.hypersurfaces.nu_nc_nue_cc, idx, fraction, params)
    f_numu_cc = get_hypersurface_factor(assets.hypersurfaces.numu_cc, idx, fraction, params)
    f_nutau_cc = get_hypersurface_factor(assets.hypersurfaces.nutau_cc, idx, fraction, params) 

    nu_nc = hists.nu_nc .+ hists.nubar_nc
    nue_cc = hists.nue_cc .+ hists.nuebar_cc
    numu_cc = hists.numu_cc .+ hists.numubar_cc
    nutau_cc = hists.nutau_cc .+ hists.nutaubar_cc
            
    return (
    nu_nc .* f_nu_nc_nue_cc .* physics.xsec.scale(:Any, :NC, params) .+ 
    nue_cc .* f_nu_nc_nue_cc .+ 
    numu_cc .* f_numu_cc .+ 
    nutau_cc .* f_nutau_cc .* physics.xsec.scale(:nutau, :CC, params)
    )    
end 

function get_expected(params, physics, assets)

    osc_flux = reweight(params, physics, assets)

    lifetime_seconds = params.deepcore_aeff_scale * 365. * 24. * 3600. * 7.5 * 1e-4 #(cm2 -> m2)

    hists = (
        nu_nc = make_hist_per_channel(assets.mc.nu_nc, osc_flux.nu_nc, lifetime_seconds),
        nubar_nc = make_hist_per_channel(assets.mc.nubar_nc, osc_flux.nubar_nc, lifetime_seconds),
        nue_cc = make_hist_per_channel(assets.mc.nue_cc, osc_flux.nue_cc, lifetime_seconds),
        nuebar_cc = make_hist_per_channel(assets.mc.nuebar_cc, osc_flux.nuebar_cc, lifetime_seconds),
        numu_cc = make_hist_per_channel(assets.mc.numu_cc, osc_flux.numu_cc, lifetime_seconds),
        numubar_cc = make_hist_per_channel(assets.mc.numubar_cc, osc_flux.numubar_cc, lifetime_seconds),
        nutau_cc = make_hist_per_channel(assets.mc.nutau_cc, osc_flux.nutau_cc, lifetime_seconds),
        nutaubar_cc = make_hist_per_channel(assets.mc.nutaubar_cc, osc_flux.nutaubar_cc, lifetime_seconds),
    )
    
    expected_nu = apply_hypersurfaces(hists, params, physics, assets)

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
        fig.content[1].title = "Mixed"
        fig.content[11].title = "Tracks"
        rowgap!(fig.layout, 0)
        fig
    end
end

end
