module atm_flux

using DelimitedFiles
using Interpolations
using DataStructures
using Distributions
using LinearAlgebra
using TypedTables


const datadir = @__DIR__ 

params = (
    atm_flux_nunubar_sigma = 0.,
    atm_flux_nuenumu_sigma = 0.,
    atm_flux_delta_spectral_index = 0.,
    atm_flux_uphorizonzal_sigma = 0.,
    )

priors = (
    atm_flux_nunubar_sigma = Truncated(Normal(0., 1.), -3, 3),
    atm_flux_nuenumu_sigma = Truncated(Normal(0., 1.), -3, 3),
    atm_flux_delta_spectral_index = Truncated(Normal(0., 0.1), -0.3, 0.3),
    atm_flux_uphorizonzal_sigma = Truncated(Normal(0., 1.), -3, 3),
    )

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

function get_nominal_flux(energy, coszen)

    # make fine grid
    e_fine_meshgrid = [((ones(size(coszen))' .* energy)...)...]
    log10e_fine_meshgrid = log10.(e_fine_meshgrid)
    cz_fine_meshgrid = [((coszen' .* ones(size(energy)))...)...]

    flux_splines = get_hkkm_flux(joinpath(datadir, "spl-nu-20-01-000.d"))
    
    flux = FlexTable(true_energy=e_fine_meshgrid, log10_true_energy=log10e_fine_meshgrid, true_coszen=cz_fine_meshgrid)
    for key in keys(flux_splines)
        setproperty!(flux, key, flux_splines[key].(flux.log10_true_energy, flux.true_coszen))
    end

    flux = Table(flux)
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

function calc_sys_flux(flux, params)

    e = flux.true_energy
    log10e =flux.log10_true_energy
    cz = flux.true_coszen

    # spectral
    f_spectral_shift = (e ./ 24.0900951261) .^ params.atm_flux_delta_spectral_index

    # all coefficients below come from fits to the Figs. 7 & 9 in Uncertainties in Atmospheric Neutrino Fluxes by Barr & Robbins
    
    # nue - nuebar
    uncert = ((0.73 * e) .^(0.59) .+ 4.8) / 100.
    flux_nue1, flux_nuebar1 = scale_flux(flux.nue, flux.nuebar, 1. .+ (params.atm_flux_nunubar_sigma .* uncert))
    
    # numu - numubar
    uncert = ((9.6 * e) .^(0.41) .-0.8) / 100.
    flux_numu1, flux_numubar1 = scale_flux(flux.numu, flux.numubar, 1. .+ (params.atm_flux_nunubar_sigma .* uncert))        

    # nue - numu
    uncert = ((0.051 * e) .^(0.63) .+ 0.73) / 100.
    flux_nue2, flux_numu2 = scale_flux(flux_nue1, flux_numu1, 1. .- (params.atm_flux_nuenumu_sigma .* uncert))
    flux_nuebar2, flux_numubar2 = scale_flux(flux_nuebar1, flux_numubar1, 1. .- (params.atm_flux_nuenumu_sigma .* uncert))

    # up/horizontal
    # nue
    uncert = (-0.43*log10e.^5 .+ 1.17*log10e.^4 .+ 0.89*log10e.^3 .- 0.36*log10e.^2 .- 1.59*log10e .+ 1.96) / 100.
    f_uphorizontal = uphorizontal.(cz, 1 .+ uncert * params.atm_flux_uphorizonzal_sigma) 
    flux_nue3 = flux_nue2 .* f_spectral_shift .* f_uphorizontal
    flux_nuebar3 = flux_nuebar2 .* f_spectral_shift .* f_uphorizontal
    
    #numu
    uncert = (-0.16*log10e.^5 .+ 0.45*log10e.^4 .+ 0.48*log10e.^3 .+ 0.17*log10e.^2 .- 1.88*log10e .+ 1.88) / 100.
    f_uphorizontal = uphorizontal.(cz, 1 .+ uncert * params.atm_flux_uphorizonzal_sigma) 
    flux_numu3 = flux_numu2 .* f_spectral_shift .* f_uphorizontal
    flux_numubar3 = flux_numubar2 .* f_spectral_shift .* f_uphorizontal

    return (nue=flux_nue3, numu=flux_numu3, nuebar=flux_nuebar3, numubar=flux_numubar3)

end


end