module atm_flux

using DelimitedFiles
using Interpolations
using DataStructures
using Distributions
using LinearAlgebra
using TypedTables
using ..Newtrinos

export AtmFluxConfig, HKKM, Barr

const datadir = @__DIR__ 

abstract type NominalFluxModel end
@kwdef struct HKKM <: NominalFluxModel
    fname::String = "spl-nu-20-01-000.d"
end

abstract type FluxSystematicsModel end
struct Barr <: FluxSystematicsModel end

@kwdef struct AtmFluxConfig{F<:NominalFluxModel, S<:FluxSystematicsModel}
    nominal_model::F = HKKM()
    systematics_model::S = Barr()
end

@kwdef struct AtmFlux <: Newtrinos.Physics
    cfg::AtmFluxConfig
    params::NamedTuple
    priors::NamedTuple
    nominal_flux::Function
    sys_flux::Function
end

function configure(cfg::AtmFluxConfig=AtmFluxConfig())
    AtmFlux(
        cfg=cfg,
        params = get_params(cfg.systematics_model),
        priors = get_priors(cfg.systematics_model),
        nominal_flux = get_nominal_flux(cfg.nominal_model),
        sys_flux = get_sys_flux(cfg.systematics_model)
        )
end

function get_params(cfg::Barr)
    params = (
        atm_flux_nuenuebar_sigma_lo = 0.,
        atm_flux_nuenuebar_sigma_mid = 0.,
        atm_flux_nuenuebar_sigma_hi = 0.,
        atm_flux_numunumubar_sigma_lo = 0.,
        atm_flux_numunumubar_sigma_mid = 0.,
        atm_flux_numunumubar_sigma_hi = 0.,
        atm_flux_nuenumu_sigma_lo = 0.,
        atm_flux_nuenumu_sigma_mid = 0.,
        atm_flux_nuenumu_sigma_hi = 0.,
        atm_flux_delta_spectral_index = 0.,
        atm_flux_uphorizontal_sigma = 0.,
        atm_flux_updown_sigma = 0.,
        atm_flux_norm_ratio_lo = 1.,
        atm_flux_norm_ratio_hi = 1.,
        atm_flux_K_pi_ratio = 1.,
        )
end

function get_priors(cfg::Barr)
    priors = (
        atm_flux_nuenuebar_sigma_lo = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_nuenuebar_sigma_mid = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_nuenuebar_sigma_hi = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_numunumubar_sigma_lo = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_numunumubar_sigma_mid = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_numunumubar_sigma_hi = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_nuenumu_sigma_lo = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_nuenumu_sigma_mid = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_nuenumu_sigma_hi = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_delta_spectral_index = Truncated(Normal(0., 0.1), -0.3, 0.3),
        atm_flux_uphorizontal_sigma = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_updown_sigma = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_norm_ratio_lo = Normal(1, 0.15),
        atm_flux_norm_ratio_hi = Normal(1, 0.15),
        atm_flux_K_pi_ratio = Normal(1, 0.1),
        )
end

function get_hkkm_flux(filename)    

    flux_chunks = Matrix{Float32}[]
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

function get_nominal_flux(cfg::HKKM)
    function nominal_flux(energy, coszen)
        # make fine grid
        e_fine_meshgrid = vec(energy .* ones(length(coszen))')
        log10e_fine_meshgrid = log10.(e_fine_meshgrid)
        cz_fine_meshgrid = vec(ones(length(energy)) .* coszen')
    
        flux_splines = get_hkkm_flux(joinpath(datadir, cfg.fname))
        
        flux = FlexTable(true_energy=e_fine_meshgrid, log10_true_energy=log10e_fine_meshgrid, true_coszen=cz_fine_meshgrid)
        for key in keys(flux_splines)
            setproperty!(flux, key, flux_splines[key].(flux.log10_true_energy, flux.true_coszen))
        end
    
        flux = Table(flux)
        end
end

function scale_flux(A, B, scale)
    # scale a ratio between A and B
    r = A ./ B
    total = A .+ B
    mod_B = total ./ (1 .+ r .* scale)
    mod_A = r .* scale .* mod_B
    return mod_A, mod_B  # Returns two separate vectors instead of tuples
end

function uphorizontal(coszen, rel_error)
    # ratio of an ellipse to a circle
    b = rel_error
    a = 1/rel_error
    1 / sqrt((b^2 - a^2) * coszen^2 + a^2)
end

function updown(coszen, up_down_ratio)
    # Smooth transition function: ranges from -1 (down) to +1 (up)
    transition = tanh.(3 * coszen)
    # Interpolate between 1/up_down_ratio and up_down_ratio
    scale = (1 ./ up_down_ratio).^(0.5 * (1 .- transition)) .* (up_down_ratio).^(0.5 * (1 .+ transition))
    return scale
end

function get_sys_flux(cfg::Barr)
    function sys_flux(flux, params)
    
        e = flux.true_energy
        log10e = flux.log10_true_energy
        cz = flux.true_coszen

        # Energy range masks: sub-GeV (E < 1), 1-10 GeV, >10 GeV
        mask_lo = e .< 1
        mask_hi = e .>= 10
        mask_mid = .!mask_lo .& .!mask_hi

        # spectral
        f_spectral_shift = (e ./ 24.0900951261) .^ params.atm_flux_delta_spectral_index

        # Energy-dependent normalization tilt, split at 1 GeV
        f_norm_tilt = e .^ (0.15 .* ifelse.(mask_lo,
            params.atm_flux_norm_ratio_lo .- 1,
            params.atm_flux_norm_ratio_hi .- 1))

        # K/π ratio: kaons dominate above ~5 GeV, stronger effect on νe than νμ
        f_k_pi_e = 1 .+ (params.atm_flux_K_pi_ratio - 1) ./ (1 .+ exp.(-(log10e .- log10(5)) .* 3))
        f_k_pi_mu = 1 .+ 0.3 .* (params.atm_flux_K_pi_ratio - 1) ./ (1 .+ exp.(-(log10e .- log10(5)) .* 3))

        # all coefficients below come from fits to the Figs. 7 & 9 in Uncertainties in Atmospheric Neutrino Fluxes by Barr & Robbins

        # nue - nuebar (3 energy ranges)
        uncert = ((0.73 * e) .^(0.59) .+ 4.8) / 100.
        eff_sigma = ifelse.(mask_lo, params.atm_flux_nuenuebar_sigma_lo,
                    ifelse.(mask_mid, params.atm_flux_nuenuebar_sigma_mid,
                                      params.atm_flux_nuenuebar_sigma_hi))
        flux_nue1, flux_nuebar1 = scale_flux(flux.nue, flux.nuebar, 1. .+ (eff_sigma .* uncert))

        # numu - numubar (3 energy ranges)
        uncert = ((9.6 * e) .^(0.41) .-0.8) / 100.
        eff_sigma = ifelse.(mask_lo, params.atm_flux_numunumubar_sigma_lo,
                    ifelse.(mask_mid, params.atm_flux_numunumubar_sigma_mid,
                                      params.atm_flux_numunumubar_sigma_hi))
        flux_numu1, flux_numubar1 = scale_flux(flux.numu, flux.numubar, 1. .+ (eff_sigma .* uncert))

        # nue - numu (3 energy ranges)
        uncert = ((0.051 * e) .^(0.63) .+ 0.73) / 100.
        eff_sigma = ifelse.(mask_lo, params.atm_flux_nuenumu_sigma_lo,
                    ifelse.(mask_mid, params.atm_flux_nuenumu_sigma_mid,
                                      params.atm_flux_nuenumu_sigma_hi))
        flux_nue2, flux_numu2 = scale_flux(flux_nue1, flux_numu1, 1. .- (eff_sigma .* uncert))
        flux_nuebar2, flux_numubar2 = scale_flux(flux_nuebar1, flux_numubar1, 1. .- (eff_sigma .* uncert))

        #up/down
        uncert = max.(0., 7 ./ (1 .+ (e./0.5) .^2)) / 100.
        f_updown = updown(cz, 1 .+ uncert * params.atm_flux_updown_sigma)   

        # up/horizontal
        # nue
        uncert = (-0.43*log10e.^5 .+ 1.17*log10e.^4 .+ 0.89*log10e.^3 .- 0.36*log10e.^2 .- 1.59*log10e .+ 1.96) / 100.
        f_uphorizontal = uphorizontal.(cz, 1 .+ uncert * params.atm_flux_uphorizontal_sigma) 
        flux_nue3 = flux_nue2 .* f_spectral_shift .* f_uphorizontal .* f_updown .* f_norm_tilt .* f_k_pi_e
        flux_nuebar3 = flux_nuebar2 .* f_spectral_shift .* f_uphorizontal .* f_updown .* f_norm_tilt .* f_k_pi_e

        #numu
        uncert = (-0.16*log10e.^5 .+ 0.45*log10e.^4 .+ 0.48*log10e.^3 .+ 0.17*log10e.^2 .- 1.88*log10e .+ 1.88) / 100.
        f_uphorizontal = uphorizontal.(cz, 1 .+ uncert * params.atm_flux_uphorizontal_sigma)
        flux_numu3 = flux_numu2 .* f_spectral_shift .* f_uphorizontal .* f_updown .* f_norm_tilt .* f_k_pi_mu
        flux_numubar3 = flux_numubar2 .* f_spectral_shift .* f_uphorizontal .* f_updown .* f_norm_tilt .* f_k_pi_mu

        return (nue=flux_nue3, numu=flux_numu3, nuebar=flux_nuebar3, numubar=flux_numubar3)
    
    end
end

end