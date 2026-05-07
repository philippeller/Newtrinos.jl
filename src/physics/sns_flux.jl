module sns_flux

using LinearAlgebra
using Distributions
using CSV
using DataFrames
using ..Newtrinos

"""
    SNSFlux <: Newtrinos.Physics

Configured Spallation Neutron Source (SNS) neutrino flux module.

Supports two modes: **data-driven** (`use_data=true`), which loads 2D energy-time flux
tables from CSV files, and **analytic** (`use_data=false`), which computes fluxes from
pion-at-rest decay kinematics.

# Fields
- `params::NamedTuple`: flux normalization parameters (`flux_norm` or `sns_nu_per_POT` (sns neutrinos per Proton On Target)).
- `priors::NamedTuple`: prior distributions for each parameter.
- `assets::NamedTuple`: precomputed energy grids, time bins, and raw flux arrays.
- `flux::Function`: closure `flux(params) -> NamedTuple` returning the scaled flux
  components and total flux.
"""
@kwdef struct SNSFlux <: Newtrinos.Physics
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    flux::Function
end
const gf = 1.1663787e-11
const me = 0.510998
const mmu = 105.6
const mtau = 1776.86
const mpi = 139.57
const alph = 1/137
const ep = (mpi^2 - mmu^2) / (2*mpi)

"""
    configure(; exposure, distance, ecut, E_bin_width, tcut, use_data, flux_folder, beam_power, proton_energy) -> SNSFlux

Create a fully configured SNS flux physics module.

# Arguments
- `exposure`: integrated exposure [GWh].
- `distance`: source-to-detector distance [m].
- `ecut::Real = mmu/2`: maximum neutrino energy [MeV] (default: ``m_\\mu / 2``).
- `E_bin_width::Real = 0.5`: energy bin width [MeV] (analytic mode only).
- `tcut::Real = 6000.0`: maximum time [ns] (data mode only).
- `use_data::Bool = true`: if `true`, load flux from CSV tables (default); if `false`, use analytic
  decay-at-rest spectra.
- `flux_folder::String`: path to the directory containing the SNS flux CSV files.
- `beam_power::Real = 1.4`: beam power [MW] (for time normalization).
- `proton_energy::Real = 1.0`: proton beam energy [GeV] (analytic mode only).

# Returns
An [`SNSFlux`](@ref) instance.
"""
function configure(;
        exposure, distance,
        ecut= mmu/2., E_bin_width=0.5, tcut=6000.0,
        use_data=true, flux_folder=joinpath(@__DIR__, "..", "experiments", "coherent", "coherent_2020", "csi", "snsFlux2D_CSV"),
        beam_power=1.4, proton_energy=1.0)
    # Call get_assets and assign the result to `assets`
    assets = get_assets(; use_data, exposure, distance, beam_power, proton_energy, flux_folder, ecut, E_bin_width, tcut)

    # Return the configured SNSFlux object
    return SNSFlux(
        params = get_params(use_data),
        priors = get_priors(use_data),
        assets = assets,
        flux = get_flux(use_data, assets),
    )
end

"""
    get_params(use_data::Bool) -> NamedTuple

Return default flux normalization parameters.

- Data mode (`use_data=true`): returns `(flux_norm=1.0,)`.
- Analytic mode (`use_data=false`): returns `(sns_nu_per_POT=0.09,)`.

# Arguments
- `use_data::Bool`: flux mode (data/analytic) selector.

# Returns
A `NamedTuple` with a single normalization parameter.
"""
function get_params(use_data)
    if use_data
        (
            flux_norm = 1.0,
        )
    else
        (
            sns_nu_per_POT = 0.09,
        )
    end
end

"""
    get_priors(use_data::Bool) -> NamedTuple

Return prior distributions for the flux normalization parameter.

- Data mode: `Truncated(Normal(1.0, 0.1), 0.7, 1.3)` for `flux_norm`.
- Analytic mode: `Truncated(Normal(0.09, 0.009), 0.05, 0.15)` for `sns_nu_per_POT`.

# Arguments
- `use_data::Bool`: flux mode (data/analytic) selector.

# Returns
A `NamedTuple` of `Symbol => Distribution` priors.
"""
function get_priors(use_data)
    if use_data
        (
            flux_norm = truncated(Normal(1.0, 0.1), 0.7, 1.3),
        )
    else
        (
            sns_nu_per_POT = truncated(Normal(0.09, 0.009), 0.05, 0.15),
        )
    end
end

"""
    get_assets(; use_data, exposure, distance, beam_power, proton_energy, flux_folder, ecut, E_bin_width, tcut) -> NamedTuple

Precompute energy grids, time bins, and raw (unnormalized) flux arrays.

In data mode, reads 2D ``(E, t)`` flux CSV files, applies energy and time cuts, rebins
time into logarithmically spaced bins, and scales by the geometric factor
``\\eta = \\text{exposure} \\cdot s_{\\text{per GWh}} \\,/\\, (4\\pi d^2)``.

In analytic mode, evaluates decay-at-rest spectra via [`flux_nu_mu`](@ref),
[`flux_nu_e`](@ref), and [`flux_nu_mu_bar`](@ref).

# Arguments
- `use_data::Bool`: flux mode selector.
- `exposure`: integrated exposure [GWh].
- `distance`: source-to-detector distance [m].
- `beam_power`: beam power [MW].
- `proton_energy`: proton beam energy [GeV].
- `flux_folder`: path to CSV flux directory.
- `ecut`: energy cutoff [MeV].
- `E_bin_width`: energy bin width [MeV] (analytic mode).
- `tcut`: time cutoff [ns] (data mode).

# Returns
A `NamedTuple` with fields `E`, `T` (data mode only), and per-flavour flux arrays.
"""
function get_assets(; use_data, exposure, distance,
                        beam_power, proton_energy,
                        flux_folder, ecut, E_bin_width, tcut)
    if use_data
        #@info "Loading SNS Flux data"
        # Function to read and rebin flux data from CSV files
        function read_flux_data(file_path, ecut, tcut)
            # Read CSV; first column = energy, remaining columns = flux values
            data = CSV.read(file_path, DataFrame; header=true, normalizenames=false)

            # Energy centers (first column)
            energy_centers = data[:, 1]

            # Time centers: column headers (strings) converted to numeric type matching energy_centers
            time_centers = parse.(eltype(energy_centers), names(data)[2:end])

            # Apply energy and time cuts
            energy_mask = energy_centers .<= ecut
            time_mask   = time_centers   .<= tcut

            # Filter centers
            energy_centers = energy_centers[energy_mask]
            time_centers   = time_centers[time_mask]

            # Extract weight submatrix
            weights = Matrix(data[:, 2:end])
            weights = weights[energy_mask, time_mask]

            # Compute bin widths
            dE = diff(energy_centers)
            dE = vcat(dE, dE[end])
            dt = diff(time_centers)
            dt = vcat(dt, dt[end])

            # Scale weights by bin widths (area-normalized)
            weights .*= dE .* dt'
            
            # Number of desired log-spaced time bins
            n_log_bins = 8
            t_start = 500.0
            # First bin: [0, t_start]
            # Remaining bins: log-spaced from t_start to tcut
            log_edges = exp10.(range(log10(t_start), log10(tcut), length=n_log_bins+1))
            # Combine: [0, t_start, ...log_edges[2:end]]
            time_edges = vcat(0.0, t_start, log_edges[2:end])
            n_time_bins = length(time_edges) - 1
            # Rebin weights into log-spaced time bins
            rebinned_weights = zeros(size(weights, 1), n_time_bins)
            for i in 1:n_time_bins
                # Find indices of original time bins whose centers fall within new bin edges
                idx = findall(c -> c >= time_edges[i] && c < time_edges[i+1], time_centers)
                if !isempty(idx)
                    rebinned_weights[:, i] = sum(weights[:, idx], dims=2)[:, 1]
                end
            end
            return (energy_centers, time_edges, rebinned_weights)
        end
        s_per_gwh = 3.6e6 / beam_power  # seconds per GWh
        distance = distance * 5.07e10 # in MeV^-1
        eta = exposure * s_per_gwh / (4 * pi * distance^2)
        # Define file paths for the flux components
        file_path_mu = joinpath(flux_folder, "convolved_energy_time_of_nu_mu.csv")
        file_path_e = joinpath(flux_folder, "convolved_energy_time_of_nu_e.csv")
        file_path_mu_bar = joinpath(flux_folder, "convolved_energy_time_of_anti_nu_mu.csv")
        file_path_e_bar = joinpath(flux_folder, "convolved_energy_time_of_anti_nu_e.csv")

        # Load flux data for each component
        E_mu, T_mu, flux_mu = read_flux_data(file_path_mu, ecut, tcut)
        E_e, T_e, flux_e = read_flux_data(file_path_e, ecut, tcut)
        E_mu_bar, T_mu_bar, flux_mu_bar = read_flux_data(file_path_mu_bar, ecut, tcut)
        E_e_bar, T_e_bar, flux_e_bar = read_flux_data(file_path_e_bar, ecut, tcut)

        # Store the loaded data as assets
        assets = (;
            E = E_mu,
            T = T_mu,
            flux_mu = eta * flux_mu,
            flux_e = eta * flux_e,
            flux_mu_bar = eta * flux_mu_bar,
            flux_e_bar = eta * flux_e_bar,
        )
    else
        #@info "Priming SNS Flux functions"
        # Uniform grid in E
        npts = Int(round((ecut - 0.5)/E_bin_width))
        E = collect(range(0.5, stop=ecut, length=max(npts, 2)))
        distance = distance * 5.07e10 # in MeV^-1
        POT_per_GWhr = (3.6e12 / (proton_energy * 1.602e-10))
        eta = POT_per_GWhr * exposure / (4 * π * (distance)^2)
        E0=ep
        E_mu, flux_mu = flux_nu_mu(E, E0, eta, E_bin_width)
        E_e, flux_e = flux_nu_e(E, eta, ecut, E_bin_width)
        E_mu_bar, flux_mu_bar = flux_nu_mu_bar(E, eta, ecut, E_bin_width)
        assets = (;
            E = E_mu,
            flux_mu,
            flux_e,
            flux_mu_bar,
        )
    end
end


"""
    flux_nu_mu(E, E0, eta, bin_width) -> (E, flux)

Compute the ``\\nu_\\mu`` flux from ``\\pi^+ \\to \\mu^+ \\nu_\\mu`` decay at rest.

The ``\\nu_\\mu`` is monoenergetic at ``E_0 = (m_\\pi^2 - m_\\mu^2)/(2 m_\\pi)``,
represented as a Gaussian-smeared delta function binned onto the energy grid.

# Arguments
- `E`: energy grid [MeV].
- `E0`: pion decay-at-rest ``\\nu_\\mu`` energy [MeV].
- `eta`: geometric normalization factor.
- `bin_width`: energy bin width [MeV].

# Returns
A tuple `(E, flux)` where `flux` is the bin-integrated ``\\nu_\\mu`` flux array.
"""
function flux_nu_mu(E, E0, eta, bin_width)
    dE = diff(E)
    @assert all(isapprox.(dE, dE[1]; atol=1e-8))
    avg_dE = mean(dE)
    dE = fill(avg_dE, length(E))

    # Element type that can hold eta (possibly Dual)
    T = promote_type(eltype(E), typeof(eta))
    flux = zeros(T, length(E))

    mask = isapprox.(E, E0; atol=avg_dE/2)
    if !any(mask)
        return (E, flux)
    end

    sigma = avg_dE
    weights = exp.(-0.5 .* ((E[mask] .- E0) ./ sigma).^2)
    norm = sum(weights .* dE[mask])
    if norm != 0
        weights ./= norm
    end
    flux[mask] .= eta .* weights .* dE[mask]
    return (E, flux)
end

"""
    flux_nu_e(E, eta, Emax, bin_width) -> (E, flux)

Compute the ``\\nu_e`` flux from ``\\mu^+ \\to e^+ \\nu_e \\bar\\nu_\\mu`` decay at rest.

The spectrum follows the Michel distribution:
``\\frac{d\\Phi}{dE} \\propto 192\\, \\frac{E^2}{m_\\mu^3} \\left(\\frac{1}{2} - \\frac{E}{m_\\mu}\\right)``
for ``0 \\le E \\le E_\\text{max}``.

# Arguments
- `E`: energy grid [MeV].
- `eta`: geometric normalization factor.
- `Emax`: maximum neutrino energy [MeV] (``\\approx m_\\mu / 2``).
- `bin_width`: energy bin width [MeV].

# Returns
A tuple `(E, flux)` where `flux` is the bin-integrated ``\\nu_e`` flux array.
"""
function flux_nu_e(E, eta, Emax, bin_width)
    # Element type that can hold eta (possibly Dual)
    T = promote_type(eltype(E), typeof(eta))
    flux = zeros(T, length(E))
    mask = (E .>= 0) .& (E .<= Emax)
    flux[mask] .= eta .* 192 .* (E[mask].^2 ./ mmu^3) .* (0.5 .- E[mask] ./ mmu)
    return (E, flux .* T(bin_width))
end

"""
    flux_nu_mu_bar(E, eta, Emax, bin_width) -> (E, flux)

Compute the ``\\bar\\nu_\\mu`` flux from ``\\mu^+ \\to e^+ \\nu_e \\bar\\nu_\\mu`` decay at rest.

The spectrum follows:
``\\frac{d\\Phi}{dE} \\propto 64\\, \\frac{E^2}{m_\\mu^3} \\left(\\frac{3}{4} - \\frac{E}{m_\\mu}\\right)``
for ``0 \\le E \\le E_\\text{max}``.

# Arguments
- `E`: energy grid [MeV].
- `eta`: geometric normalization factor.
- `Emax`: maximum neutrino energy [MeV] (``\\approx m_\\mu / 2``).
- `bin_width`: energy bin width [MeV].

# Returns
A tuple `(E, flux)` where `flux` is the bin-integrated ``\\bar\\nu_\\mu`` flux array.
"""
function flux_nu_mu_bar(E, eta, Emax, bin_width)
    # Element type that can hold eta (possibly Dual)
    T = promote_type(eltype(E), typeof(eta))
    flux = zeros(T, length(E))
    mask = (E .>= 0) .& (E .<= Emax)
    flux[mask] .= eta .* 64 .* (E[mask].^2 ./ mmu^3) .* (0.75 .- E[mask] ./ mmu)
    return (E, flux .* T(bin_width))
end

"""
    get_flux(use_data, assets) -> Function

Construct the flux evaluation closure from precomputed assets.

The returned closure `flux(params) -> NamedTuple` applies the normalization parameter
to the raw flux arrays and returns per-flavour fluxes and the total flux.

In data mode, the returned `NamedTuple` has fields: `E`, `T`, `total_flux`, `flux_e`,
`flux_mu`, `flux_mu_bar`, `flux_e_bar`.

In analytic mode: `E`, `total_flux`, `flux_e`, `flux_mu`, `flux_mu_bar`.

# Arguments
- `use_data::Bool`: flux mode (data/analytic) selector.
- `assets::NamedTuple`: precomputed energy grids and flux arrays from [`get_assets`](@ref).

# Returns
A closure `flux(params) -> NamedTuple`.
"""
function get_flux(use_data, assets)
    if use_data
        # Extract fluxes and energy/time centers from assets
        E = assets.E
        T = assets.T
        flux_mu = assets.flux_mu
        flux_e = assets.flux_e
        flux_mu_bar = assets.flux_mu_bar
        flux_e_bar = assets.flux_e_bar

        # Return the closure directly
        return function (params)
            # Extract normalization parameters
            norm = params.flux_norm

            # Compute total flux
            total = flux_mu .+ flux_e .+ flux_mu_bar .+ flux_e_bar

            return (;
                E,  # Energy grid
                T,  # Time grid
                total_flux = total .* norm,
                flux_e,
                flux_mu,
                flux_mu_bar,
                flux_e_bar,
            )
        end
    else
        # Extract fluxes and energy centers from assets
        E = assets.E
        flux_mu = assets.flux_mu
        flux_e = assets.flux_e
        flux_mu_bar = assets.flux_mu_bar

        # Return the closure directly
        return function (params)
            # Use params for nu_per_POT
            nu_per_POT_val = params.sns_nu_per_POT

            # Scale fluxes by nu_per_POT
            flux_mu_scaled = flux_mu .* nu_per_POT_val
            flux_e_scaled = flux_e .* nu_per_POT_val
            flux_mu_bar_scaled = flux_mu_bar .* nu_per_POT_val

            # Compute total flux
            total_flux = flux_mu_scaled .+ flux_e_scaled .+ flux_mu_bar_scaled

            return (;
                E,  # Energy grid
                total_flux,
                flux_e = flux_e_scaled,
                flux_mu = flux_mu_scaled,
                flux_mu_bar = flux_mu_bar_scaled,
            )
        end
    end
end

end