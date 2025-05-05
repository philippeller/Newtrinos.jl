module orca

using LinearAlgebra
using Distributions
using DataStructures
using TypedTables
using HDF5
using StatsBase
using CairoMakie
using BAT
using ..Newtrinos

@kwdef struct ORCA <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    #plot::Function
end

function configure(physics)
    physics = (;physics.osc, physics.atm_flux, physics.earth_layers)
    assets = get_assets(physics)
    return ORCA(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        #plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)
    h5file = h5open(joinpath(datadir, "ORCA6_433kton_v_0_5.h5"), "r")
    f = read(h5file)
    neutrinos = Table(Dict(Symbol(key) => f["binned_nu_response"][key] for key in keys(f["binned_nu_response"])))
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

    flux = physics.atm_flux.nominal_flux(binning.e_fine, binning.cz_fine)
    layers = physics.earth_layers.compute_layers()
    paths = physics.earth_layers.compute_paths(binning.cz_fine, layers)

    true_shape = (length(binning.e_fine), length(binning.cz_fine))
    reco_shape = (length(binning.e_reco), length(binning.cz_reco), 3, 2)

    mc = (
        nue = neutrinos[neutrinos.Pdg .== 12],
        nuebar = neutrinos[neutrinos.Pdg .== -12],
        numu = neutrinos[neutrinos.Pdg .== 14],
        numubar = neutrinos[neutrinos.Pdg .== -14],
        nutau = neutrinos[neutrinos.Pdg .== 16],
        nutaubar = neutrinos[neutrinos.Pdg .== -16]
        )

    rs = [2, 1, 3]
    data_hist = permutedims(reshape(data.W, reco_shape[rs]), rs)
    muon_hist = permutedims(reshape(muons.W, reco_shape[rs]), rs);

    assets = (;mc, muon_hist, observed=cut_zeros(data_hist), binning, true_shape, reco_shape, flux, layers, paths)
end

function get_params()
    params = (;
        )
end

function get_priors()
    priors = (;
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
    for i in 1:prod(size)
        hist[i] = 0.
    end
    for i in 1:length(w)
        hist[e_idx[i], c_idx[i], p_idx[i], t_idx[i]] += w[i]
    end
    hist
end

function make_hist_per_channel(mc, osc_flux, lifetime_seconds, assets)
    w = lifetime_seconds * mc.W .* osc_flux
    h = make_hist(mc.E_reco_bin, mc.Ct_reco_bin, mc.AnaClass, mc.IsCC .+ 1, w, assets.reco_shape)
end


function reweight(params, physics, assets)
    sys_flux = physics.atm_flux.sys_flux(assets.flux, params)

    s = assets.true_shape

    p = physics.osc.osc_prob(assets.binning.e_fine, assets.paths, assets.layers, params)
    p_flux = reshape(sys_flux.nue, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numu, s) .* p[:, :, 2, :]
    
    nus = NamedTuple(ch=>gather_flux(p_flux, assets.mc[ch].E_true_bin, assets.mc[ch].Ct_true_bin, i) for (i, ch) in enumerate([:nue, :numu, :nutau]))

    p = physics.osc.osc_prob(assets.binning.e_fine, assets.paths, assets.layers, params, anti=true)
    p_flux = reshape(sys_flux.nuebar, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numubar, s) .* p[:, :, 2, :]

    nubars = NamedTuple(ch=>gather_flux(p_flux, assets.mc[ch].E_true_bin, assets.mc[ch].Ct_true_bin, i) for (i, ch) in enumerate([:nuebar, :numubar, :nutaubar]))

    merge(nus, nubars)
end

function get_expected(params, physics, assets)

    osc_flux = reweight(params, physics, assets)

    lifetime_seconds = 1.

    hists = NamedTuple(ch=>make_hist_per_channel(assets.mc[ch], osc_flux[ch], lifetime_seconds, assets) for ch in keys(assets.mc))
    
    expected = assets.muon_hist .+ sum(sum(hists), dims=4)

    cut_zeros(expected)
end

function cut_zeros(hist)
    (tracks = hist[1:end-1, 1:10, 1, :],
        shower = hist[1:end, 1:10, 1, :],
        mixed = hist[1:end-1, 1:10, 1, :]
        )
end


function get_forward_model(physics, assets)
    function forward_model(params)
        exp_events = get_expected(params, physics, assets)
        distprod(NamedTuple(ch => distprod(Poisson.(exp_events[ch])) for ch in keys(exp_events)))
    end
end



end

    
    