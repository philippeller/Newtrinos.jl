module kamland

using DataFrames
using CSV
using Distributions
using LinearAlgebra
using Statistics
using DataStructures
using DataStructures
using BAT
using CairoMakie
using Logging


assets = undef
config = undef

function configure(;osc, kwargs...)
    global config = (;osc,)
    return true
end

function setup(datadir = @__DIR__)
    @info "Loading kamland data"

    # Important global constants
    ENERGY_RESOL_1MEV = 0.065 # Reduces as 1/sqrt(visible energy in MeV)
    RATE_UNC = 0.041 # Fully correlated event rate uncertainty
    ENERGY_SCALE_UNC = 0.019 # Uncertainty in the deposited prompt energy conversion
    ELECTRON_DENSITY_ROCK = 0.5 * 2.7 # g/cm3: approximate electron density
    NEUTRINO_POSITRON_ENERGY_SHIFT = 0.782 # MeV: energy shift between true neutrino energy and deposited positron energy
    
    no_osc = CSV.read(joinpath(datadir, "KamLAND_no_osc.csv"), DataFrame; header=false)[:, 2]
    bestfit_osc = CSV.read(joinpath(datadir, "KamLAND_best_fit_osc.csv"), DataFrame; header=false)[:, 2]
    bestfit_BG = CSV.read(joinpath(datadir, "KamLAND_best_fit_withBG.csv"), DataFrame; header=false)[:, 2]
    allBG = bestfit_BG .- bestfit_osc
    geonu = CSV.read(joinpath(datadir, "KamLAND_best_fit_geonu.csv"), DataFrame; header=false)[:, 2]
    df_reactors = CSV.read(joinpath(datadir, "Reactors.csv"), DataFrame)
    
    L = df_reactors.L
    fluxes = df_reactors.N_reactors ./ (L .^ 2)
    flux_factor = fluxes ./ sum(fluxes)
    Ep_bins = LinRange(0.9,8.5,19)[1:end-1]
    Ep = 0.5 * (Ep_bins[1:end-1] .+ Ep_bins[2:end])
    upsampling = 10
    Ep_bins_fine = LinRange(0.9,8.5,(length(Ep)+1) * upsampling + 1)[1:end-upsampling]
    Ep_fine = 0.5 * (Ep_bins_fine[1:end-1] .+ Ep_bins_fine[2:end])
    # these weights are according to very approximately a reactor nergy spectrum
    weights = pdf(LogNormal(log(3.25),0.38), Ep_fine);

    observed = round.(CSV.read(joinpath(datadir, "KamLAND_data.csv"), DataFrame; header=false)[:, 2])
    global assets = (;
        observed,
        Ep,
        Ep_bins,
        Ep_fine,
        Ep_bins_fine,
        ENERGY_RESOL_1MEV,
        RATE_UNC,
        ENERGY_SCALE_UNC,
        ELECTRON_DENSITY_ROCK,
        NEUTRINO_POSITRON_ENERGY_SHIFT,
        L,
        upsampling,
        no_osc,
        weights,
        allBG,
        geonu,
        flux_factor,        
        )

    return true
end

function smear(E, p, sigma, weights; width=10, E_scale=1.0, E_bias=0.0)
    l = length(p)
    out = similar(p, l)
    for i in 1:l
        out[i] = 0.
        e = E[i] * E_scale + E_bias
        norm = 0.0
        for j in max(1, i - width):min(l, i + width)
            coeff = 1 / sigma[j] * exp(-0.5 * ((e - E[j]) / sigma[j])^2)
            norm += coeff * weights[j]
            out[i] += coeff * weights[j] * p[j]
        end
        out[i] /= norm
    end
    return copy(out)
end

function get_expected(params, config, assets)
    E_fine = (assets.Ep_fine .+ assets.NEUTRINO_POSITRON_ENERGY_SHIFT) .* (1 .+ assets.ENERGY_SCALE_UNC .* params.kamland_energy_scale)

    BG = assets.allBG .+ assets.geonu .* (params.kamland_geonu_scale)

    prob_outer_fine = config.osc.osc_prob(E_fine .* 1e-3, assets.L, params, anti=true)[:, :, 1, 1]'

    prob_fine = dropdims(sum(assets.flux_factor .* prob_outer_fine, dims=1), dims=1)
    Ep_resolutions = assets.ENERGY_RESOL_1MEV .* sqrt.(assets.Ep_fine)

    prob_smeared = smear(assets.Ep_fine, prob_fine, Ep_resolutions, assets.weights, width=10, E_scale=1.0, E_bias=0.0)
    
    # use uniform averages:
    #prob = dropdims(mean(reshape(prob_smeared, upsampling, :), dims=1), dims=1)

    # use some approximate flux weights to do a non uniform bin average
    #prob_rs = reshape(prob_smeared, upsampling, :)
    #weights_rs = reshape(pdf(LogNormal(log(3.25),0.38), Ep_fine), upsampling, :)
    prob = dropdims(mean(reshape(prob_smeared, assets.upsampling, :), dims=1), dims=1)

    exp_events_noBG = assets.no_osc .* prob .* (1 .+ params.kamland_flux_scale .* assets.RATE_UNC)
    return exp_events_noBG .+ BG
end

function get_forward_model()
    model = let this_assets = assets, this_config = config
        params -> begin
            exp_events = get_expected(params, this_config, this_assets)
            distprod(Poisson.(exp_events))
        end
    end
end

function plot(params, data=assets.observed)

    m = mean(get_forward_model()(params))
    v = var(get_forward_model()(params))
    
    energy = assets.Ep
    energy_bins = assets.Ep_bins

    f = Figure()
    ax = Axis(f[1,1])
    
    plot!(ax, energy, data, color=:black, label="Observed")
    stephist!(ax, energy, weights=m, bins=energy_bins, label="Expected")
    barplot!(ax, energy, m .+ sqrt.(v), width=diff(energy_bins), gap=0, fillto= m .- sqrt.(v), alpha=0.5, label="Standard Deviation")
    
    ax.ylabel="Counts"
    ax.title="KamLAND"
    axislegend(ax, framevisible = false)
    
    
    ax2 = Axis(f[2,1])
    plot!(ax2, energy, data ./ m, color=:black, label="Observed")
    hlines!(ax2, 1, label="Expected")
    barplot!(ax2, energy, 1 .+ sqrt.(v) ./ m, width=diff(energy_bins), gap=0, fillto= 1 .- sqrt.(v)./m, alpha=0.5, label="Standard Deviation")
    ylims!(ax2, 0.3, 1.7)
    
    ax.xticksvisible = false
    ax.xticklabelsvisible = false
    
    rowsize!(f.layout, 1, Relative(3/4))
    rowgap!(f.layout, 1, 0)
    
    ax2.xlabel="Eₚ (MeV)"
    ax2.ylabel="Counts/Expected"
    
    xlims!(ax, minimum(energy_bins), maximum(energy_bins))
    xlims!(ax2, minimum(energy_bins), maximum(energy_bins))
    
    ylims!(ax, 0, 300)
    
    f
end


params = (
    kamland_energy_scale = 0.,
    kamland_geonu_scale = 0.,
    kamland_flux_scale = 0.,
    )

priors = (
    kamland_energy_scale = Truncated(Normal(0, 1.), -3, 3),
    kamland_geonu_scale = Uniform(-0.5, 0.5),
    kamland_flux_scale = Truncated(Normal(0, 1.), -3, 3),
    )

end