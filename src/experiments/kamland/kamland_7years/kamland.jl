module kamland

using DataFrames
using CSV
using Distributions
using LinearAlgebra
using Statistics
using DataStructures
using Zygote
using BAT

# Important global constants
const ENERGY_RESOL_1MEV = 0.065 # Reduces as 1/sqrt(visible energy in MeV)
const RATE_UNC = 0.041 # Fully correlated event rate uncertainty
const ENERGY_SCALE_UNC = 0.019 # Uncertainty in the deposited prompt energy conversion
const ELECTRON_DENSITY_ROCK = 0.5 * 2.7 # g/cm3: approximate electron density
const NEUTRINO_POSITRON_ENERGY_SHIFT = 0.782 # MeV: energy shift between true neutrino energy and deposited positron energy

function smear(E, p, sigma; width=10, E_scale=1.0, E_bias=0.0)
    l = length(p)
    out = Zygote.Buffer(p, l)
    for i in 1:l
        out[i] = 0.
        e = E[i] * E_scale + E_bias
        norm = 0.0
        for j in max(1, i - width):min(l, i + width)
            coeff = 1 / sigma[j] * exp(-0.5 * ((e - E[j]) / sigma[j])^2)
            norm += coeff
            out[i] += coeff * p[j]
        end
        out[i] /= norm
    end
    return copy(out)
end

# Import data
const datadir = @__DIR__ 

const observed = round.(CSV.read(joinpath(datadir, "KamLAND_data.csv"), DataFrame; header=false)[:, 2])
const no_osc = CSV.read(joinpath(datadir, "KamLAND_no_osc.csv"), DataFrame; header=false)[:, 2]
const bestfit_osc = CSV.read(joinpath(datadir, "KamLAND_best_fit_osc.csv"), DataFrame; header=false)[:, 2]
const bestfit_BG = CSV.read(joinpath(datadir, "KamLAND_best_fit_withBG.csv"), DataFrame; header=false)[:, 2]
const allBG = bestfit_BG .- bestfit_osc
const geonu = CSV.read(joinpath(datadir, "KamLAND_best_fit_geonu.csv"), DataFrame; header=false)[:, 2]

df_reactors = CSV.read(joinpath(datadir, "Reactors.csv"), DataFrame)
const L = df_reactors.L
const fluxes = df_reactors.N_reactors ./ (L .^ 2)
const flux_factor = fluxes ./ sum(fluxes)

const Ep_bins = LinRange(0.9,8.5,19)[1:end-1]
const Ep = 0.5 * (Ep_bins[1:end-1] .+ Ep_bins[2:end])

const upsampling = 10

const Ep_bins_fine = LinRange(0.9,8.5,(length(Ep)+1) * upsampling + 1)[1:end-upsampling]
const Ep_fine = 0.5 * (Ep_bins_fine[1:end-1] .+ Ep_bins_fine[2:end])


function get_expected(params, osc_prob)
    E_fine = (Ep_fine .+ NEUTRINO_POSITRON_ENERGY_SHIFT) .* (1 .+ ENERGY_SCALE_UNC .* params.kamland_energy_scale)

    BG = allBG .+ geonu .* (params.kamland_geonu_scale)

    prob_outer_fine = osc_prob(E_fine .* 1e-3, L, params, anti=true)[:, :, 1, 1]'

    prob_fine = dropdims(sum(flux_factor .* prob_outer_fine, dims=1), dims=1)
    Ep_resolutions = ENERGY_RESOL_1MEV .* sqrt.(Ep_fine)

    prob_smeared = smear(Ep_fine, prob_fine, Ep_resolutions; width=10, E_scale=1.0, E_bias=0.0)
    
    # use uniform averages:
    #prob = dropdims(mean(reshape(prob_smeared, upsampling, :), dims=1), dims=1)

    # use some approximate flux weights to do a non uniform bin average
    prob_rs = reshape(prob_smeared, upsampling, :)
    weights_rs = reshape(pdf(LogNormal(log(3.25),0.38), Ep_fine), upsampling, :)
    prob = dropdims(sum(prob_rs .* weights_rs, dims=1) ./ sum(weights_rs, dims=1), dims=1)

    exp_events_noBG = no_osc .* prob .* (1 .+ params.kamland_flux_scale .* RATE_UNC)
    return exp_events_noBG .+ BG
end

function forward_model(params, osc_prob)
    exp_events = get_expected(params, osc_prob)
    distprod(Poisson.(exp_events))
end

params = OrderedDict()
params[:kamland_energy_scale] = 0.
params[:kamland_geonu_scale] = 0.
params[:kamland_flux_scale] = 0.

priors = OrderedDict()
priors[:kamland_energy_scale] = Truncated(Normal(0, 1.), -3, 3)
priors[:kamland_geonu_scale] = Uniform(-0.5, 0.5)
priors[:kamland_flux_scale] = Truncated(Normal(0, 1.), -3, 3)

end