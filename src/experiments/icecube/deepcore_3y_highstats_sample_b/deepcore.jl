module deepcore

using LinearAlgebra
using Distributions
using DataStructures
using DelimitedFiles
using TypedTables
using Interpolations
using CSV
using StatsBase
using CairoMakie
import ..earth_layers
using BAT

const datadir = @__DIR__ 
using Printf


const reco_energy_bin_edges = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413]
const reco_coszen_bin_edges = [-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.]
const pid_bin_edges = -0.5:1:1.5
const type_bin_edges = [-0.5, 0.5, 3.5]

const cz_fine_bins = LinRange(-1,1, 201)
const log10e_fine_bins = LinRange(0,3,201)
const e_fine_bins = 10 .^log10e_fine_bins

const cz_fine = midpoints(cz_fine_bins);
const log10e_fine = midpoints(log10e_fine_bins);
const e_fine = 10 .^log10e_fine;

const e_ticks = (reco_energy_bin_edges, [@sprintf("%.1f",b) for b in reco_energy_bin_edges])

const layers = earth_layers.compute_layers()
const paths = earth_layers.compute_paths(cz_fine, layers);



function prepare_data(datadir = @__DIR__)

    mc_nu = CSV.read(joinpath(datadir, "neutrino_mc.csv"), FlexTable; header=true);
    mc_nu.log10_true_energy = log10.(mc_nu.true_energy)
    
    function compute_indices(mc)
        mc.e_idx = searchsortedfirst.(Ref(reco_energy_bin_edges), mc.reco_energy) .- 1
        mc.c_idx = searchsortedfirst.(Ref(reco_coszen_bin_edges), mc.reco_coszen) .- 1
        mc.p_idx = searchsortedfirst.(Ref(pid_bin_edges), mc.pid) .- 1
        mc.t_idx = searchsortedfirst.(Ref(type_bin_edges), mc.type) .- 1
        mc.ef_idx = searchsortedfirst.(Ref(log10e_fine_bins), mc.log10_true_energy) .- 1
        mc.cf_idx = searchsortedfirst.(Ref(cz_fine_bins), mc.true_coszen) .- 1
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
            d[var] = fit(Histogram, (csv.reco_energy, csv.reco_coszen, csv.pid), weights(columns(csv)[var]), (reco_energy_bin_edges, reco_coszen_bin_edges, pid_bin_edges)).weights
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
    
    # ---------- DATA READ --------
    
    # ---------- Pre-Compute FLuxes --------
    
    function get_hkkm_flux(filename)    
    
        flux_chunks = []
        for i in 19:-1:0
            idx = i*103 + 3: (i+1)*103
            push!(flux_chunks, Float32.(readdlm(filename)[idx, 2:5]))
        end
        
        log10_energy_flux_values = LinRange(-1, 4, 101)
        
        cz_flux_bins = LinRange(-1, 1, 21);
        energy_flux_values = 10 .^ log10_energy_flux_values;
        
        cz_flux_values = LinRange(-0.95, 0.95, 20);
        
        hkkm_flux = permutedims(stack(flux_chunks), [1, 3, 2]);
        
        flux = OrderedDict{Symbol, Interpolations.Extrapolation}()
        
        flux[:numu] = cubic_spline_interpolation((log10_energy_flux_values, cz_flux_values), hkkm_flux[:, :, 1], extrapolation_bc = Line());
        flux[:numubar] = cubic_spline_interpolation((log10_energy_flux_values, cz_flux_values), hkkm_flux[:, :, 2], extrapolation_bc = Line());
        flux[:nue] = cubic_spline_interpolation((log10_energy_flux_values, cz_flux_values), hkkm_flux[:, :, 3], extrapolation_bc = Line());
        flux[:nuebar] = cubic_spline_interpolation((log10_energy_flux_values, cz_flux_values), hkkm_flux[:, :, 4], extrapolation_bc = Line());
    
        return flux
    end
    
    flux_splines = get_hkkm_flux(joinpath(datadir, "spl-nu-20-01-000.d"))
    
    function LogLogParam(true_energy::Real, y1::Real, y2::Real, x1::Real, x2::Real, use_cutoff::Bool, cutoff_value::Real)
        """ From https://github.com/icecube/pisa/blob/master/pisa/utils/barr_parameterization.py """
        nu_nubar = sign(y2)
        y1 = sign(y1) * log10(abs(y1) + 0.0001)
        y2 = log10(abs(y2 + 0.0001))
        modification = nu_nubar * 10. ^(((y2 - y1) / (x2 - x1)) * (log10(true_energy) - x1) + y1 - 2.)
        if use_cutoff
            modification *= exp(-1. * true_energy / cutoff_value)
        end
        return modification
    end
    
    
    function norm_fcn(x::Real, sigma::Real)
        """ From https://github.com/icecube/pisa/blob/master/pisa/utils/barr_parameterization.py """
        return 1. / sqrt(2 * pi * sigma^2) * exp(-x^2 / (2 * sigma^2))
    end
    
    
    flux = OrderedDict{Symbol, Table}()

    # make fine grid
    e = ones(size(cz_fine))' .* e_fine;
    log10e = ones(size(cz_fine))' .* log10e_fine;
    cz = cz_fine' .* ones(size(e_fine));
    
    for key in [:nue, :numu]
        for anti in ["", "bar"]
            fkey = Symbol(key, anti)
            f = FlexTable(true_energy=[(e...)...], log10_true_energy=[(log10e...)...], true_coszen=[(cz...)...])
            f.flux = flux_splines[fkey].(f.log10_true_energy, f.true_coszen);
            if key == :nue
                f.Barr_Ave = LogLogParam.(f.true_energy, 5.5, 53., 0.5, 3., false, 0.)
                f.Barr_LogLog = LogLogParam.(f.true_energy, 0.9, 10., 0.5, 2., true, 650.)
                f.Barr_norm_fcn = norm_fcn.(f.true_coszen, 0.36)
            else
                f.Barr_Ave = LogLogParam.(f.true_energy, 3., 43., 0.5, 3., false, 0.)
                f.Barr_LogLog = LogLogParam.(f.true_energy, 0.6, 5., 0.5, 2., true, 1000.)
                f.Barr_norm_fcn = norm_fcn.(f.true_coszen, 0.36)
            end
            flux[fkey] = Table(f)
        end
    end
    
    
    return (
            mc = mc,
            hyperplanes = hyperplanes,
            flux = NamedTuple(flux),
            muons = muons
            ),(
            data.count
            )

end

const assets, observed = prepare_data()

    
# ---------- DATA IS PREPARED --------------

params = OrderedDict()
params[:deepcore_lifetime] = 2.5
params[:deepcore_atm_muon_scale] = 1.
params[:deepcore_ice_absorption] = 1.
params[:deepcore_ice_scattering] = 1.
params[:deepcore_opt_eff_overall] = 1.
params[:deepcore_opt_eff_lateral] = 0.
params[:deepcore_opt_eff_headon] = 0.
params[:nc_norm] = 1.
params[:nutau_cc_norm] = 1.
params[:atm_flux_nunubar_ratio] = 1.
params[:atm_flux_nuenumu_ratio] = 1.
params[:atm_flux_spectral_index] = 0.
params[:Barr_uphor_ratio] = 0.0
params[:Barr_nu_nubar_ratio ] = 0.

priors = OrderedDict()
priors[:deepcore_lifetime] = Uniform(2, 4)
priors[:deepcore_atm_muon_scale ] = Uniform(0, 2)
priors[:deepcore_ice_absorption] = Truncated(Normal(1, 0.1), 0.85, 1.15)
priors[:deepcore_ice_scattering] = Truncated(Normal(1, 0.1), 0.85, 1.15)
priors[:deepcore_opt_eff_overall] = Truncated(Normal(1, 0.1), 0.8, 1.2)
priors[:deepcore_opt_eff_lateral] = Truncated(Normal(0, 1.), -2, 2)
priors[:deepcore_opt_eff_headon] = Uniform(-5, 2.)
priors[:nc_norm] = Truncated(Normal(1, 0.2), 0.4, 1.6)
priors[:nutau_cc_norm] = Uniform(0., 2.)
priors[:atm_flux_nunubar_ratio] = 1.
priors[:atm_flux_nuenumu_ratio] = Truncated(Normal(1., 0.05), 0.85, 1.15)
priors[:atm_flux_spectral_index] = Truncated(Normal(0., 0.1), -0.3, 0.3)
priors[:Barr_uphor_ratio] = Truncated(Normal(0., 1.), -3, 3)
priors[:Barr_nu_nubar_ratio] = Truncated(Normal(0., 1.), -3, 3)


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

function scale_flux(A::AbstractVector{<:Real}, B::AbstractVector{<:Real}, scale::Real)
    r = A ./ B
    total = A .+ B
    mod_B = total ./ (1 .+ r .* scale)
    mod_A = r .* scale .* mod_B
    return mod_A, mod_B  # Returns two separate vectors instead of tuples
end

function Barr_factor_nue(Barr_Ave, Barr_LogLog, Barr_norm_fcn, nubar_sys, uphor)
    """ From https://github.com/icecube/pisa/blob/master/pisa/utils/barr_parameterization.py """
    # These parameters are obtained from fits to the paper of Barr
    # E dependent ratios, max differences per flavor (Fig.7)
    
    r_uphor = 1. .- 0.3 .* uphor .* Barr_LogLog .* Barr_norm_fcn
    r = Barr_Ave .- (1.5 .* Barr_norm_fcn .- 0.7) .* Barr_LogLog
    modfactor = nubar_sys .* r

    # nue, nuebar:
    return max.(0., 1. .+ 0.5 .* modfactor) .* r_uphor, max.(0., 1. ./ (1 .+ 0.5 .* modfactor)) .* r_uphor
        
    
end

function Barr_factor_numu(Barr_Ave, Barr_LogLog, Barr_norm_fcn, nubar_sys, uphor)
    """ From https://github.com/icecube/pisa/blob/master/pisa/utils/barr_parameterization.py """
    # These parameters are obtained from fits to the paper of Barr
    # E dependent ratios, max differences per flavor (Fig.7)

    r = Barr_Ave .- (Barr_norm_fcn .- 0.6) .* 2.5 .* Barr_LogLog
    modfactor = nubar_sys .* r

    # numu, numubar:
    max.(0., 1. .+ 0.5 .* modfactor), max.(0., 1. ./ (1 .+ 0.5 .* modfactor))
    
end

function calc_sys_flux(flux, params)

    # nu-nubar ratio:
    flux_nue1, flux_nuebar1 = scale_flux(flux[:nue].flux, flux[:nuebar].flux, params.atm_flux_nunubar_ratio)
    flux_numu1, flux_numubar1 = scale_flux(flux[:numu].flux, flux[:numubar].flux, params.atm_flux_nunubar_ratio)
    
    # nue-numu ratio:
    flux_nue2, flux_numu2 = scale_flux(flux_nue1, flux_numu1, params.atm_flux_nuenumu_ratio)
    flux_nuebar2, flux_numubar2 = scale_flux(flux_nuebar1, flux_numubar1, params.atm_flux_nuenumu_ratio)

    # spectral
    f_spectral_shift = (flux[:nue].true_energy ./ 24.0900951261) .^ params.atm_flux_spectral_index

    # Barr modifiers
    f_Barr_nue, f_Barr_nuebar = Barr_factor_nue(flux[:nue].Barr_Ave, flux[:nue].Barr_LogLog, flux[:nue].Barr_norm_fcn, params.Barr_nu_nubar_ratio, params.Barr_uphor_ratio)
    f_Barr_numu, f_Barr_numubarr = Barr_factor_numu(flux[:numu].Barr_Ave, flux[:numu].Barr_LogLog, flux[:numu].Barr_norm_fcn, params.Barr_nu_nubar_ratio, params.Barr_uphor_ratio)

    # apply:
    f_nue3 = flux_nue2 .* f_spectral_shift .* f_Barr_nue
    f_nuebar3 = flux_nuebar2 .* f_spectral_shift .* f_Barr_nuebar
    f_numu3 = flux_numu2 .* f_spectral_shift .* f_Barr_numu
    f_numubar3 = flux_numubar2 .* f_spectral_shift .* f_Barr_numubarr

    s = (size(e_fine)[1], size(cz_fine)[1])

    return (reshape(f_nue3, s), reshape(f_numu3, s)), (reshape(f_nuebar3, s), reshape(f_numubar3, s)) 
end

# Function that should NOT allocate
function gather_flux(p_flux, ef, cf, j)
    result = Vector{eltype(p_flux)}(undef, length(ef))
    @inbounds for i in eachindex(ef)
        result[i] = p_flux[ef[i], cf[i], j]
    end
    result
end


function reweight(mc, flux, params, osc_prob)
    sys_flux, sys_flux_anti = calc_sys_flux(flux, params)

    p = osc_prob(e_fine, paths, layers, params)
    p_flux = sys_flux[1] .* p[:, :, 1, :] .+ sys_flux[2] .* p[:, :, 2, :]
    
    #nus = NamedTuple(ch=>[p_flux[ef_idx, cf_idx, i] for (ef_idx, cf_idx) in zip(mc[ch].ef_idx, mc[ch].cf_idx)] for (i, ch) in enumerate([:nue, :numu, :nutau]))

    nus = NamedTuple(ch=>gather_flux(p_flux, mc[ch].ef_idx, mc[ch].cf_idx, i) for (i, ch) in enumerate([:nue, :numu, :nutau]))
    
    p = osc_prob(e_fine, paths, layers, params, anti=true)
    p_flux = sys_flux_anti[1] .* p[:, :, 1, :] .+ sys_flux_anti[2] .* p[:, :, 2, :]

    #nubars = NamedTuple(ch=>[p_flux[ef_idx, cf_idx, i] for (ef_idx, cf_idx) in zip(mc[ch].ef_idx, mc[ch].cf_idx)] for (i, ch) in enumerate([:nuebar, :numubar, :nutaubar])        )
    nubars = NamedTuple(ch=>gather_flux(p_flux, mc[ch].ef_idx, mc[ch].cf_idx, i) for (i, ch) in enumerate([:nuebar, :numubar, :nutaubar]))

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

function apply_hyperplanes(hists, params, hyperplanes)
    f_nc = get_hyperplane_factor(hyperplanes.nuall_nc, params)
    f_nue_cc = get_hyperplane_factor(hyperplanes.nue_cc, params)
    f_numu_cc = get_hyperplane_factor(hyperplanes.numu_cc, params)
    f_nutau_cc = get_hyperplane_factor(hyperplanes.nutau_cc, params) 

    nues = hists[:nue] .+ hists[:nuebar]
    numus = hists[:numu] .+ hists[:numubar]
    nutaus = hists[:nutau] .+ hists[:nutaubar]
    (
    (nues[:, :, :, 1] .+ numus[:, :, :, 1] .+ nutaus[:, :, :, 1]).* f_nc .* params.nc_norm .+ 
    nues[:, :, :, 2] .* f_nue_cc .+ 
    numus[:, :, :, 2] .* f_numu_cc .+ 
    nutaus[:, :, :, 2] .* f_nutau_cc .* params.nutau_cc_norm
    )    
end 

function get_expected(params, osc_prob, assets)

    osc_flux = reweight(assets.mc, assets.flux, params, osc_prob)

    lifetime_seconds = params.deepcore_lifetime * 365. * 24. * 3600.

    hists = NamedTuple(ch=>make_hist_per_channel(assets.mc[ch], osc_flux[ch], lifetime_seconds) for ch in keys(assets.mc))
    
    expected_nu = apply_hyperplanes(hists, params, assets.hyperplanes)
    
    expected = (expected_nu .+ params.deepcore_atm_muon_scale .* assets.muons.count)
end


function forward_model(osc_prob)
    model = let this_assets = assets
        params -> begin
            exp_events = get_expected(params, osc_prob, this_assets)
            distprod(Poisson.(exp_events))
        end
    end
end

    
function plotmap(h; colormap=Reverse(:Spectral), symm=false)

    if symm
        colorrange = (-maximum(abs.(h)), maximum(abs.(h)))
    else
        colorrange = (0, maximum(h))
    end
    
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1,1], xscale=log10, xticks=e_ticks, xlabel="E (GeV)", ylabel="cos(zen)", title="cascades")
    hm = heatmap!(ax, reco_energy_bin_edges, reco_coszen_bin_edges, h[:, :, 1], colormap=colormap, colorrange=colorrange)
    ax = Axis(fig[1,2], xscale=log10, xticks=e_ticks, xlabel="E (GeV)", yticksvisible=true, yticklabelsvisible=false, title="tracks")
    hm = heatmap!(ax, reco_energy_bin_edges, reco_coszen_bin_edges, h[:, :, 2], colormap=colormap, colorrange=colorrange)
    Colorbar(fig[1,3], hm)
    fig
end

function plot(params, osc_prob)

    expected = get_expected(params, osc_prob, assets)

    plotmap(expected)
    
end

end
