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
        flux=get_flux(),
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


const gf=1.1663787e-11
const me = 0.510998
const mmu = 105.6
const mtau=1776.86
const mpi = 139.57
const sw2 = 0.231
const gu = 1/2 - 2*2/3*sw2
const gd = -(1/2) + 2*1/3*sw2
const alph = 1/137
const ep= (mpi^2-mmu^2)/(2*mpi)


function flux_nu_mu(E, E0, eta, Emax, bin_width)
    """
    Monoenergetic NU_MU flux using Gaussian smearing on isclose-matched bins.
    Returns:
        2D numpy array:
            Two columns:
                - Column 0: Neutrino energy bin centers (enu_centers).
                - Column 1: Neutrino energy bin contents (enu_contents).
    """
    dE = diff(E)
    # assume equally spaced E
    @assert all(isapprox.(dE, dE[1]; atol=1e-8))
    avg_dE = mean(dE)
    dE = ones(eltype(E), size(E)) * avg_dE

    # Find close bins
    mask = isapprox.(E, E0; atol=avg_dE / 2)
    if !any(mask)
        return zeros(eltype(E), size(E))
    end
    # Discrete Gaussian weights centered at E0
    sigma = avg_dE
    weights = @. exp(-0.5 * ((E[mask] - E0) / sigma)^2)
    weights ./= sum(weights .* dE[mask])  # normalize to area=1

    flux = zeros(eltype(E), size(E))
    flux[mask] .= eta * weights .* dE[mask] # final flux

    return flux
end

function flux_nu_e(E, E0, eta, Emax, bin_width)
    """
    NU_E flux from muon decay: 192 * (E^2 / mmu^3) * (1/2 - E/mmu)
    Returns:
        2D numpy array:
            Two columns:
            - Column 0: Neutrino energy bin centers (enu_centers).
            - Column 1: Neutrino energy bin contents (enu_contents).
    """
    flux = zeros(eltype(E), size(E))
    mask = (E .>= 0) .& (E .<= Emax)

    @. flux[mask] = eta * 192 * (E[mask]^2 / mmu^3) * (0.5 - E[mask]/mmu)
    return flux*bin_width
end

function flux_nu_mu_bar(E, E0, eta, Emax, bin_width)
    """
    NU_MU_BAR flux from muon decay: 64 * (E^2 / mmu^3) * (3/4 - E/mmu)
    Returns:
        2D numpy array:
            Two columns:
                - Column 0: Neutrino energy bin centers (enu_centers).
                - Column 1: Neutrino energy bin contents (enu_contents).
    """
    flux = zeros(eltype(E), size(E))
    mask = @. (E >= 0) & (E <= Emax)  # Physical cutoff

    @. flux[mask] = eta * 64 * (E[mask]^2 / mmu^3) * (0.75 - E[mask]/mmu)
    return flux*bin_width
end


function get_flux()
    function flux(exposure, distance, params)

    
        bin_width=0.1
        ecut=0.5*mmu
        tcut=5000.0
        proton_energy=1.0
    
        POT_per_GWhr = params.sns_nu_per_POT * (3.6e12 / (proton_energy * 1.602e-10)) # protons on target per GWh
        
        Emax = mmu * 0.5  # endpoint
    
        E = LinRange(0.1,Emax,Int(round(((Emax-0.1)/bin_width)))) # 0.1 MeV to 60 MeV in 0.1 MeV steps
        E0 = ep  # ~29.79 MeV
        eta = POT_per_GWhr*(exposure)/(4*pi*((distance*5.07e+12)^2))
    
        flux_mu = flux_nu_mu(E, E0, eta, Emax, bin_width)
        flux_e = flux_nu_e(E, E0, eta, Emax, bin_width)
        flux_mu_bar = flux_nu_mu_bar(E, E0, eta, Emax, bin_width)
        total_flux = flux_mu .+ flux_e .+ flux_mu_bar
    
        return (;E, total_flux, flux_e, flux_mu, flux_mu_bar)
    end
end

end