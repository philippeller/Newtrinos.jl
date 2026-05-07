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

"""
    NominalFluxModel

Abstract type for atmospheric neutrino flux predictions.

Each subtype provides a recipe for computing the unoscillated neutrino flux as a function
of energy and zenith angle. Currently the only implementation is [`HKKM`](@ref).
"""
abstract type NominalFluxModel end

"""
    HKKM <: NominalFluxModel

Honda–Kajita–Kasahara–Midorikawa (HKKM) atmospheric neutrino flux model.

Reads a site-specific flux table (`.d` file) and constructs cubic-spline interpolations
over ``\\log_{10}(E)`` and ``\\cos\\theta_z`` for each neutrino flavour
(``\\nu_\\mu``, ``\\bar\\nu_\\mu``, ``\\nu_e``, ``\\bar\\nu_e``).

# Fields
- `fname::String = "spl-nu-20-01-000.d"`: filename of the HKKM flux table (relative to
  the `physics/` data directory).
"""
@kwdef struct HKKM <: NominalFluxModel
    fname::String = "spl-nu-20-01-000.d"
end

"""
    FluxSystematicsModel

Abstract type for atmospheric flux systematic uncertainty models.

Each subtype defines a set of nuisance parameters and a function that modifies the
nominal flux to account for systematic uncertainties. Currently the only implementation
is [`Barr`](@ref).
"""
abstract type FluxSystematicsModel end

"""
    Barr <: FluxSystematicsModel

Barr systematic uncertainty model for atmospheric neutrino fluxes.

Parametrizes flux uncertainties as energy- and zenith-dependent modifications following
Barr et al., "Uncertainties in Atmospheric Neutrino Fluxes." The systematics cover:

| Parameter                          | Description                               |
|:---------------------------------- |:----------------------------------------- |
| `atm_flux_nuenuebar_sigma`         | ``\\nu_e / \\bar\\nu_e`` ratio uncertainty  |
| `atm_flux_numunumubar_sigma`       | ``\\nu_\\mu / \\bar\\nu_\\mu`` ratio uncertainty |
| `atm_flux_nuenumu_sigma`           | ``\\nu_e / \\nu_\\mu`` ratio uncertainty    |
| `atm_flux_delta_spectral_index`    | Spectral index tilt                       |
| `atm_flux_uphorizonzal_sigma`      | Up/horizontal anisotropy                  |
| `atm_flux_updown_sigma`            | Up/down asymmetry                         |

All sigma parameters are centred at 0 with unit Gaussian priors (truncated at ±3).
"""
struct Barr <: FluxSystematicsModel end

"""
    AtmFluxConfig{F, S}

Configuration for the atmospheric neutrino flux module.

# Fields
- `nominal_model::F = HKKM()`: nominal flux prediction model ([`NominalFluxModel`](@ref)).
- `systematics_model::S = Barr()`: systematic uncertainty model ([`FluxSystematicsModel`](@ref)).
"""
@kwdef struct AtmFluxConfig{F<:NominalFluxModel, S<:FluxSystematicsModel}
    nominal_model::F = HKKM()
    systematics_model::S = Barr()
end

"""
    AtmFlux <: Newtrinos.Physics

Configured atmospheric flux physics module, returned by [`configure`](@ref).

# Fields
- `cfg::AtmFluxConfig`: the configuration used to build this module.
- `params::NamedTuple`: default systematic parameter values.
- `priors::NamedTuple`: prior distributions for each systematic parameter.
- `nominal_flux::Function`: closure
  `nominal_flux(energy, coszen) -> Table` returning the unoscillated flux on a
  fine ``(E, \\cos\\theta_z)`` grid.
- `sys_flux::Function`: closure
  `sys_flux(flux, params) -> NamedTuple` applying systematic modifications to the
  nominal flux and returning per-flavour flux arrays.
"""
@kwdef struct AtmFlux <: Newtrinos.Physics
    cfg::AtmFluxConfig
    params::NamedTuple
    priors::NamedTuple
    nominal_flux::Function
    sys_flux::Function
end

"""
    configure(cfg::AtmFluxConfig=AtmFluxConfig()) -> AtmFlux

Create a fully configured atmospheric flux physics module.

# Arguments
- `cfg::AtmFluxConfig`: flux configuration (defaults to [`HKKM`](@ref) nominal model with
  [`Barr`](@ref) systematics).

# Returns
An [`AtmFlux`](@ref) instance.

# Examples
```julia
using Newtrinos

# Default South Pole flux
flux_physics = Newtrinos.atm_flux.configure()

# Custom site flux file
flux_physics = Newtrinos.atm_flux.configure(
    AtmFluxConfig(nominal_model=HKKM(fname="spl-nu-20-12-000.d"))
)
```
"""
function configure(cfg::AtmFluxConfig=AtmFluxConfig())
    AtmFlux(
        cfg=cfg,
        params = get_params(cfg.systematics_model),
        priors = get_priors(cfg.systematics_model),
        nominal_flux = get_nominal_flux(cfg.nominal_model),
        sys_flux = get_sys_flux(cfg.systematics_model)
        )
end

"""
    get_params(cfg::FluxSystematicsModel) -> NamedTuple

Return the default systematic parameter values for the given flux systematics model.

# Arguments
- `cfg::FluxSystematicsModel`: a [`Barr`](@ref) instance.

# Returns
A `NamedTuple` mapping parameter names to their nominal values (all default to `0.0`).
"""
function get_params(cfg::Barr)
    params = (
        atm_flux_nuenuebar_sigma = 0.,
        atm_flux_numunumubar_sigma = 0.,
        atm_flux_nuenumu_sigma = 0.,
        atm_flux_delta_spectral_index = 0.,
        atm_flux_uphorizonzal_sigma = 0.,
        atm_flux_updown_sigma = 0.,
        )
end

"""
    get_priors(cfg::FluxSystematicsModel) -> NamedTuple

Return prior distributions for each flux systematic parameter.

For [`Barr`](@ref), all sigma parameters use `Truncated(Normal(0, 1), -3, 3)` and
the spectral index uses `Truncated(Normal(0, 0.1), -0.3, 0.3)`.

# Arguments
- `cfg::FluxSystematicsModel`: a [`Barr`](@ref) instance.

# Returns
A `NamedTuple` of `Symbol => Distribution` priors.
"""
function get_priors(cfg::Barr)
    priors = (
        atm_flux_nuenuebar_sigma = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_numunumubar_sigma = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_nuenumu_sigma = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_delta_spectral_index = Truncated(Normal(0., 0.1), -0.3, 0.3),
        atm_flux_uphorizonzal_sigma = Truncated(Normal(0., 1.), -3, 3),
        atm_flux_updown_sigma = Truncated(Normal(0., 1.), -3, 3),
        )
end

"""
    get_hkkm_flux(filename) -> OrderedDict{Symbol, Extrapolation}

Read an HKKM flux table and return cubic-spline interpolations for each flavour.

Parses the fixed-format `.d` file into 20 cosine-zenith chunks of 101 energy bins each,
then builds 2D cubic-spline interpolations over ``(\\log_{10}(E/\\text{GeV}),\\, \\cos\\theta_z)``.

# Arguments
- `filename::String`: absolute path to the HKKM flux data file.

# Returns
An `OrderedDict` with keys `:numu`, `:numubar`, `:nue`, `:nuebar`, each mapping to a
`Interpolations.Extrapolation` with linear extrapolation boundary conditions.
"""
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

"""
    get_nominal_flux(cfg::NominalFluxModel) -> Function

Construct the nominal flux closure for the given flux model.

For [`HKKM`](@ref), returns a function `nominal_flux(energy, coszen) -> Table` that
evaluates the HKKM spline interpolations on a meshgrid of `energy` [GeV] and `coszen`
values. The returned `Table` contains columns `true_energy`, `log10_true_energy`,
`true_coszen`, and per-flavour flux values (`:numu`, `:numubar`, `:nue`, `:nuebar`).

# Arguments
- `cfg::NominalFluxModel`: a [`HKKM`](@ref) instance.

# Returns
A closure `nominal_flux(energy, coszen) -> Table`.
"""
function get_nominal_flux(cfg::HKKM)
    function nominal_flux(energy, coszen)
        # make fine grid
        e_fine_meshgrid = [((ones(size(coszen))' .* energy)...)...]
        log10e_fine_meshgrid = log10.(e_fine_meshgrid)
        cz_fine_meshgrid = [((coszen' .* ones(size(energy)))...)...]
    
        flux_splines = get_hkkm_flux(joinpath(datadir, cfg.fname))
        
        flux = FlexTable(true_energy=e_fine_meshgrid, log10_true_energy=log10e_fine_meshgrid, true_coszen=cz_fine_meshgrid)
        for key in keys(flux_splines)
            setproperty!(flux, key, flux_splines[key].(flux.log10_true_energy, flux.true_coszen))
        end
    
        flux = Table(flux)
        end
end

"""
    scale_flux(A, B, scale) -> (mod_A, mod_B)

Scale the ratio between two flux components while conserving their sum.

Given fluxes `A` and `B`, modifies their ratio by the factor `scale` while keeping
``A + B`` constant: ``A' / B' = (A / B) \\cdot \\text{scale}``.

# Arguments
- `A`: flux array for the first component.
- `B`: flux array for the second component.
- `scale`: multiplicative factor applied to the ``A/B`` ratio.

# Returns
A tuple `(mod_A, mod_B)` of rescaled flux arrays.
"""
function scale_flux(A, B, scale)
    # scale a ratio between A and B
    r = A ./ B
    total = A .+ B
    mod_B = total ./ (1 .+ r .* scale)
    mod_A = r .* scale .* mod_B
    return mod_A, mod_B  # Returns two separate vectors instead of tuples
end

"""
    uphorizontal(coszen, rel_error) -> Real

Compute the up/horizontal flux anisotropy correction factor.

Models the zenith-dependent flux distortion as the ratio of an ellipse to a circle,
where the ellipse semi-axes are `a = 1/rel_error` and `b = rel_error`.

# Arguments
- `coszen`: cosine of the zenith angle.
- `rel_error`: relative error parameter controlling the ellipticity.

# Returns
A multiplicative correction factor for the flux.
"""
function uphorizontal(coszen, rel_error)
    # ratio of an ellipse to a circle
    b = rel_error
    a = 1/rel_error
    1 / sqrt((b^2 - a^2) * coszen^2 + a^2)
end

"""
    updown(coszen, up_down_ratio) -> AbstractArray

Compute the up/down flux asymmetry correction factor.

Uses a smooth ``\\tanh(3\\,\\cos\\theta_z)`` transition to interpolate between
`1/up_down_ratio` (downgoing) and `up_down_ratio` (upgoing).

# Arguments
- `coszen`: cosine of the zenith angle (scalar or array).
- `up_down_ratio`: ratio of upgoing to downgoing flux modification.

# Returns
A multiplicative correction factor array.
"""
function updown(coszen, up_down_ratio)
    # Smooth transition function: ranges from -1 (down) to +1 (up)
    transition = tanh.(3 * coszen)
    # Interpolate between 1/up_down_ratio and up_down_ratio
    scale = (1 ./ up_down_ratio).^(0.5 * (1 .- transition)) .* (up_down_ratio).^(0.5 * (1 .+ transition))
    return scale
end

"""
    get_sys_flux(cfg::FluxSystematicsModel) -> Function

Construct the systematic flux modification closure for the given systematics model.

For [`Barr`](@ref), returns a function `sys_flux(flux, params) -> NamedTuple` that applies
the following energy- and zenith-dependent corrections to the nominal flux:

1. **Spectral tilt**: ``(E / E_\\text{pivot})^{\\Delta\\gamma}`` with
   ``E_\\text{pivot} \\approx 24.1`` GeV.
2. **``\\nu_e / \\bar\\nu_e`` ratio** scaling via [`scale_flux`](@ref).
3. **``\\nu_\\mu / \\bar\\nu_\\mu`` ratio** scaling via [`scale_flux`](@ref).
4. **``\\nu_e / \\nu_\\mu`` ratio** scaling via [`scale_flux`](@ref).
5. **Up/down asymmetry** via [`updown`](@ref).
6. **Up/horizontal anisotropy** via [`uphorizontal`](@ref) (separate polynomial
   uncertainty fits for ``\\nu_e`` and ``\\nu_\\mu``).

# Arguments
- `cfg::FluxSystematicsModel`: a [`Barr`](@ref) instance.

# Returns
A closure `sys_flux(flux, params) -> NamedTuple{(:nue, :numu, :nuebar, :numubar)}` of
modified flux arrays.
"""
function get_sys_flux(cfg::Barr)
    function sys_flux(flux, params)
    
        e = flux.true_energy
        log10e = flux.log10_true_energy
        cz = flux.true_coszen

        # spectral
        f_spectral_shift = (e ./ 24.0900951261) .^ params.atm_flux_delta_spectral_index

        # all coefficients below come from fits to the Figs. 7 & 9 in Uncertainties in Atmospheric Neutrino Fluxes by Barr & Robbins
        
        # nue - nuebar
        uncert = ((0.73 * e) .^(0.59) .+ 4.8) / 100.
        flux_nue1, flux_nuebar1 = scale_flux(flux.nue, flux.nuebar, 1. .+ (params.atm_flux_nuenuebar_sigma .* uncert))
        
        # numu - numubar
        uncert = ((9.6 * e) .^(0.41) .-0.8) / 100.
        flux_numu1, flux_numubar1 = scale_flux(flux.numu, flux.numubar, 1. .+ (params.atm_flux_numunumubar_sigma .* uncert))        

        # nue - numu
        uncert = ((0.051 * e) .^(0.63) .+ 0.73) / 100.
        flux_nue2, flux_numu2 = scale_flux(flux_nue1, flux_numu1, 1. .- (params.atm_flux_nuenumu_sigma .* uncert))
        flux_nuebar2, flux_numubar2 = scale_flux(flux_nuebar1, flux_numubar1, 1. .- (params.atm_flux_nuenumu_sigma .* uncert))

        #up/down
        uncert = max.(0., 7 ./ (1 .+ (e./0.5) .^2)) / 100.
        f_updown = updown(cz, 1 .+ uncert * params.atm_flux_updown_sigma)   

        # up/horizontal
        # nue
        uncert = (-0.43*log10e.^5 .+ 1.17*log10e.^4 .+ 0.89*log10e.^3 .- 0.36*log10e.^2 .- 1.59*log10e .+ 1.96) / 100.
        f_uphorizontal = uphorizontal.(cz, 1 .+ uncert * params.atm_flux_uphorizonzal_sigma) 
        flux_nue3 = flux_nue2 .* f_spectral_shift .* f_uphorizontal .* f_updown
        flux_nuebar3 = flux_nuebar2 .* f_spectral_shift .* f_uphorizontal .* f_updown
        
        #numu
        uncert = (-0.16*log10e.^5 .+ 0.45*log10e.^4 .+ 0.48*log10e.^3 .+ 0.17*log10e.^2 .- 1.88*log10e .+ 1.88) / 100.
        f_uphorizontal = uphorizontal.(cz, 1 .+ uncert * params.atm_flux_uphorizonzal_sigma) 
        flux_numu3 = flux_numu2 .* f_spectral_shift .* f_uphorizontal .* f_updown
        flux_numubar3 = flux_numubar2 .* f_spectral_shift .* f_uphorizontal .* f_updown

        return (nue=flux_nue3, numu=flux_numu3, nuebar=flux_nuebar3, numubar=flux_numubar3)
    
    end
end

end