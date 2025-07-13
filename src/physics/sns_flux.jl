module sns_flux

using LinearAlgebra
using Distributions
using ..Newtrinos


@kwdef struct SNSFlux <: Newtrinos.Physics
    params::NamedTuple
    priors::NamedTuple
    flux::Function
end


function configure()
    SNSFlux(
        params = get_params(),
        priors = get_priors(),
        flux = get_flux(),
    )
end

function get_params()
    (
        sns_nu_per_POT = 0.0848,
    )
end

function get_priors()
    (
        sns_nu_per_POT = Normal(0.0848, 0.00848),
    )
end


const gf = 1.1663787e-11
const me = 0.510998
const mmu = 105.6
const mtau = 1776.86
const mpi = 139.57
const sw2 = 0.231
const gu = 1/2 - 2*2/3*sw2
const gd = -(1/2) + 2*1/3*sw2
const alph = 1/137
const ep = (mpi^2 - mmu^2) / (2*mpi)

# Returns (E, flux) arrays, matching Python's np.column_stack((E, flux))
function flux_nu_mu(E, E0, eta, bin_width)
    dE = diff(E)
    @assert all(isapprox.(dE, dE[1]; atol=1e-8))
    avg_dE = mean(dE)
    dE = fill(avg_dE, length(E))
    mask = isapprox.(E, E0; atol=avg_dE/2)
    if !any(mask)
        return (E, zeros(length(E)))
    end
    sigma = avg_dE
    weights = exp.(-0.5 .* ((E[mask] .- E0) ./ sigma).^2)
    weights ./= sum(weights .* dE[mask])
    flux = zeros(length(E))
    flux[mask] .= eta .* weights .* dE[mask]
    return (E, flux)
end

function flux_nu_e(E, eta, Emax, bin_width)
    flux = zeros(length(E))
    mask = (E .>= 0) .& (E .<= Emax)
    flux[mask] .= eta .* 192 .* (E[mask].^2 ./ mmu^3) .* (0.5 .- E[mask] ./ mmu)
    return (E, flux .* bin_width)
end

function flux_nu_mu_bar(E, eta, Emax, bin_width)
    flux = zeros(length(E))
    mask = (E .>= 0) .& (E .<= Emax)
    flux[mask] .= eta .* 64 .* (E[mask].^2 ./ mmu^3) .* (0.75 .- E[mask] ./ mmu)
    return (E, flux .* bin_width)
end

function get_flux()
    function flux(; exposure=1.0, distance=1.0, bin_width=0.1, ecut=0.5*mmu, tcut=5000.0, proton_energy=1.0, nu_per_POT=0.0848, params=nothing)
        # Use params if provided, else use nu_per_POT
        nu_per_POT_val = params !== nothing ? params.sns_nu_per_POT : nu_per_POT
        POT_per_GWhr = nu_per_POT_val * (3.6e12 / (proton_energy * 1.602e-10))
        Emax = mmu * 0.5
        E = collect(range(0.1, stop=Emax, length=Int(round((Emax-0.1)/bin_width))))
        E0 = ep
        eta = POT_per_GWhr * exposure / (4 * π * (distance * 5.07e12)^2)
        E_mu, flux_mu = flux_nu_mu(E, E0, eta, bin_width)
        E_e, flux_e = flux_nu_e(E, eta, Emax, bin_width)
        E_mu_bar, flux_mu_bar = flux_nu_mu_bar(E, eta, Emax, bin_width)
        total_flux = flux_mu .+ flux_e .+ flux_mu_bar
        return (;
            E = E,
            total_flux = total_flux,
            flux_e = flux_e,
            flux_mu = flux_mu,
            flux_mu_bar = flux_mu_bar
        )
    end
end

end