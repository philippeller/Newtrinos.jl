module coherent_csi

using DataFrames
using CSV
using Distributions
using SpecialFunctions
using LinearAlgebra
using Statistics
using DataStructures
using BAT
using ForwardDiff
using CairoMakie
using Logging
using StatsBase
using ..Helpers
import ..Newtrinos

@kwdef struct COHERENT_CSI <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function configure(; datadir = @__DIR__, use_flux_data::Bool = true, ff_model::Symbol = :helm, ff_kwargs::NamedTuple = (;))
    assets = get_assets(datadir)

    # Configure the SNS flux module
    sns_flux = Newtrinos.sns_flux.configure(
        exposure = assets.exposure,
        distance = assets.distance,
        use_data = use_flux_data,
    )

    # Reconfigure assets with data loaded from sns_flux
    assets = get_assets(datadir, sns_flux)

    # Configure the CEvNS cross-section module
    cevns_xsec = Newtrinos.cevns_xsec.configure(
        assets.isotopes,
        assets.er_centers .* 1e-3, # Convert keV to MeV
        sns_flux.assets.E;         # Pass the energy grid from the SNS flux assets
        ff_model = ff_model,
        ff_kwargs = ff_kwargs,
    )

    @info "Configured COHERENT CsI module."
    physics = (;sns_flux = sns_flux, cevns_xsec = cevns_xsec)

    return COHERENT_CSI(
        physics = physics,
        params = get_params(assets.ss_bkg_nom, assets.brn_nom, assets.nin_nom),
        priors = get_priors(assets.ss_bkg_nom, assets.brn_nom, assets.nin_nom),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_params(ss_bkg_nom, brn_nom, nin_nom)
    params = (
        coherent_csi_eff_a = 1.32045,
        coherent_csi_eff_b = 0.285979,
        coherent_csi_eff_c = 10.8646,
        coherent_csi_eff_d = -0.333322,
        coherent_csi_qfa_a = 0.0554628,  # QF polynomial coefficients
        coherent_csi_qfa_b = 4.30681,
        coherent_csi_qfa_c = -111.707,
        coherent_csi_qfa_d = 840.384,
        brn_norm= brn_nom,  # Normalization factor for BRN
        nin_norm= nin_nom,  # Normalization factor for NIN
        ss_bkg_norm= ss_bkg_nom,  # Normalization factor for SS background
        )
end

# TODO!
function get_priors(ss_bkg_nom, brn_nom, nin_nom)
    priors = (
        coherent_csi_eff_a = Normal(1.32045, 0.02),
        coherent_csi_eff_b = Normal(0.285979, 0.0006),
        coherent_csi_eff_c = Normal(10.8646, 1.),
        coherent_csi_eff_d = Normal(-0.333322, 0.03),
        coherent_csi_qfa_a = Normal(0.0554628, 0.0059),
        coherent_csi_qfa_b = Normal(4.30681, 0.79),
        coherent_csi_qfa_c = Normal(-111.707, 26.15),
        coherent_csi_qfa_d = Normal(840.384, 244.82),
        brn_norm= truncated(Normal(brn_nom, 0.25 * brn_nom), 0.0, brn_nom + 3 * 0.25 * brn_nom),  # Normalization factor for BRN
        nin_norm= truncated(Normal(nin_nom, 0.36 * nin_nom), 0.0, nin_nom + 3 * 0.36 * nin_nom),  # Normalization factor for NIN
        ss_bkg_norm= truncated(Normal(ss_bkg_nom, 0.021 * ss_bkg_nom), 0.0, ss_bkg_nom + 3 * 0.021 * ss_bkg_nom),  # Normalization factor for SS background
        )
end

function get_assets(datadir = @__DIR__, sns_flux = nothing)
    #@info "Loading coherent csi data"

    # Basic assets that are always loaded
    er_edges = LinRange(3, 200, Int((200 - 3) / 0.5))  # keV
    isotopes = [
        (fraction=0.49, mass=123.8e3, Z=55, N=78, Rn_key=:Rn_Cs, Rn_nom=5.7242),  # Cs-133
        (fraction=0.51, mass=118.21e3, Z=53, N=74, Rn_key=:Rn_I, Rn_nom=5.7242)   # I-127
    ]  # List of isotopes with [fraction, Nuclear mass (MeV), Z, N=A-Z, Rn_key, Rn_nom (fm)]
    Nt = 2 * (14.6 / 0.25981) * 6.023e+23
    light_yield = 13.35  # PE/keVee
    resolution = [0.0749, 9.56]  # a/Eee and b*Eee

    # Reconstruct bin edges from centers
    er_centers = midpoints(er_edges)

    pe_width = 5.0
    # Custom PE bin edges
    out_edges = [0, 12, 16, 20, 24, 28, 32, 36, 40, 46, 54, 64, 76, 100, 148, 196]
    out_centers = midpoints(out_edges)  # Bin centers: [5, 10, 15, ..., 200]

    # Initialize placeholders for binned data
    ssBkg = nothing
    observed = nothing
    brn = nothing
    nin = nothing
    ss_bkg_nom = nothing
    brn_nom = nothing
    nin_nom = nothing
    time_bins = nothing
    time_edges = nothing
    # Check if sns_flux is provided and has time bin centers
    if sns_flux !== nothing && haskey(sns_flux.assets, :T)
        @info "Loading and binning CsI data"
        @info "Configuring Flux"
        time_edges = sns_flux.assets.T  # Extract time bin-edges from sns_flux (nanoseconds)
        time_bins = midpoints(time_edges)  # Bin centers
        # Import Data
        ssBkg_df = CSV.read(joinpath(datadir, "csi/dataBeamOnAC.txt"), DataFrame, comment="#", header=false, delim=' ')  # columns: PE, timestamp
        observed_df = CSV.read(joinpath(datadir, "csi/dataBeamOnC.txt"), DataFrame, comment="#", header=false, delim=' ')  # columns: PE, timestamp
        brnPE_df = CSV.read(joinpath(datadir, "csi/brnPE.txt"), DataFrame, comment="#", header=false, delim=' ')  # columns: PE, counts
        brnTrec_df = CSV.read(joinpath(datadir, "csi/brnTrec.txt"), DataFrame, comment="#", header=false, delim=' ')  # columns: time (µs), counts
        ninPE_df = CSV.read(joinpath(datadir, "csi/ninPE.txt"), DataFrame, comment="#", header=false, delim=' ')  # columns: PE, counts
        ninTrec_df = CSV.read(joinpath(datadir, "csi/ninTrec.txt"), DataFrame, comment="#", header=false, delim=' ')  # columns: time (µs), counts

        # Convert timestamps in microseconds to nanoseconds for consistency with sns_flux.assets.T
        ssBkg_df[:, 2] .*= 1e3
        observed_df[:, 2] .*= 1e3
        brnTrec_df[:, 1] .*= 1e3
        ninTrec_df[:, 1] .*= 1e3

        # Perform 2D binning for unbinned event lists (PE, timestamp)
        #@info "Binning unbinned CsI data"

        # Filter out events outside the range of out_edges and time_bins
        valid_ssBkg = filter(row -> row[1] >= first(out_edges) && row[1] <= last(out_edges) &&
                              row[2] >= first(time_edges) && row[2] <= last(time_edges), eachrow(ssBkg_df))
        valid_observed = filter(row -> row[1] >= first(out_edges) && row[1] <= last(out_edges) &&
                                  row[2] >= first(time_edges) && row[2] <= last(time_edges), eachrow(observed_df))

        # Convert filtered data back to DataFrame
        ssBkg_df = DataFrame(valid_ssBkg)
        observed_df = DataFrame(valid_observed)

        # Perform binning
        ssBkg_hist = fit(Histogram, (ssBkg_df[:, 1], ssBkg_df[:, 2]), (out_edges, time_edges))
        observed_hist = fit(Histogram, (observed_df[:, 1], observed_df[:, 2]), (out_edges, time_edges))
        ssBkg = ssBkg_hist.weights
        observed = observed_hist.weights

        # Rebin BRN and NIN data into desired bins
        #@info "Rebinning binned CsI data"

        # Convert zipped data into an array of tuples
        brn_data = collect(zip(brnPE_df[:, 1], brnTrec_df[:, 1]))
        nin_data = collect(zip(ninPE_df[:, 1], ninTrec_df[:, 1]))

        # Filter out events outside the range of out_edges and time_bins
        valid_brn = filter(row -> row[1] >= first(out_edges) && row[1] <= last(out_edges) &&
                            row[2] >= first(time_edges) && row[2] <= last(time_edges), brn_data)
        valid_nin = filter(row -> row[1] >= first(out_edges) && row[1] <= last(out_edges) &&
                            row[2] >= first(time_edges) && row[2] <= last(time_edges), nin_data)

        # Convert filtered data back to DataFrame
        brnPE_df = DataFrame(PE=[row[1] for row in valid_brn], Time=[row[2] for row in valid_brn])
        ninPE_df = DataFrame(PE=[row[1] for row in valid_nin], Time=[row[2] for row in valid_nin])

        # Perform rebinning
        brn_hist = fit(Histogram, (brnPE_df.PE, brnPE_df.Time), (out_edges, time_edges))
        nin_hist = fit(Histogram, (ninPE_df.PE, ninPE_df.Time), (out_edges, time_edges))
        brn = brn_hist.weights
        nin = nin_hist.weights

        # Get initial nominal value for Bkg normalizations
        ss_bkg_nom = sum(ssBkg)
        #@info "Initial SS background normalization: $ss_bkg_nom"
        brn_nom = sum(brn)
        #@info "Initial BRN background normalization: $brn_nom"
        nin_nom = sum(nin)
        #@info "Initial NIN background normalization: $nin_nom"
    else
        @info "Flux is not fully configured yet."
    end

    distance = 1930  # cm
    exposure = 13.99  # GWh

    # Return assets as a NamedTuple
    return (;
        observed,
        er_edges,
        er_centers,
        time_edges,
        time_bins,
        out_edges,
        out_centers,
        isotopes,
        Nt,
        light_yield,
        resolution,
        brn,
        brn_nom,
        nin,
        nin_nom,
        ssBkg,
        ss_bkg_nom,
        distance,
        exposure,
    )
end

# QF: accepts scalar or array, returns same shape, type-generic
function qf(er, params)
    a = params.coherent_csi_qfa_a
    b = params.coherent_csi_qfa_b
    c = params.coherent_csi_qfa_c
    d = params.coherent_csi_qfa_d
    x = er .* 1e-3                         # MeV
    vals = (a .* x .+ b .* x.^2 .+ c .* x.^3 .+ d .* x.^4) .* 1e3  # keVee
    z = vals isa AbstractArray ? zero(eltype(vals)) : zero(vals)
    return max.(vals, z)
end

# Efficiency: accepts scalar or array of PE, type-generic
function eff(pe, params)
    a = params.coherent_csi_eff_a
    b = params.coherent_csi_eff_b
    c = params.coherent_csi_eff_c
    d = params.coherent_csi_eff_d
    vals = @. a / (1 + exp(-b * (pe - c))) + d
    z = vals isa AbstractArray ? zero(eltype(vals)) : zero(vals)
    return max.(vals, z)
end

# Gamma PDF (k-θ parameterization), AD-friendly
@generated function _typed_one(::Type{T}) where {T}
    :(one(T))
end
@inline function _gamma_pdf(x, k, θ)
    # Clamp arguments to safe values for AD
    x = x < eps(real(x)) ? eps(real(x)) : x
    k = k < eps(real(k)) ? eps(real(k)) : k
    θ = θ < eps(real(θ)) ? eps(real(θ)) : θ
    return exp((k - 1) * log(x) - x / θ - k * log(θ) - loggamma(k))
end

function gamma_pdf_integrated_over_bins(Eee, pe_centers, pe_edges, resolution, light_yield)
    n_bins = length(pe_centers)
    T = promote_type(eltype(pe_centers), typeof(Eee))
    probs = zeros(T, n_bins)

    if iszero(Eee) || isnan(Eee) || !isfinite(Eee)
        return probs
    end

    a = resolution[1] / Eee
    b = resolution[2] * Eee
    k = one(T) + b
    θ = inv(a * (one(T) + b))

    for i in eachindex(pe_centers)
        lo = T(pe_edges[i])
        hi = T(pe_edges[i + 1])
        mid = (lo + hi) / 2
        probs[i] = (hi - lo) / 6 * (_gamma_pdf(lo, k, θ) + 4 * _gamma_pdf(mid, k, θ) + _gamma_pdf(hi, k, θ))
        # Optional debug:
        # if isnan(probs[i]) || isinf(probs[i])
        #     @warn "NaN or Inf in gamma_pdf_integrated_over_bins: i=$i, lo=$lo, hi=$hi, k=$k, θ=$θ"
        # end
    end

    s = sum(probs)
    epsT = eps(T)
    if s > epsT && !isnan(s) && isfinite(s)
        probs ./= s
    else
        probs .= zero(T)
    end
    return probs
end

# Single ER-bin response column, AD-safe
function response_matrix_per_er_bin(keVnr, params, assets)
    keVee = qf(keVnr, params)                    # scalar (possibly Dual)
    weights = gamma_pdf_integrated_over_bins(keVee, assets.out_centers, assets.out_edges,
                                             assets.resolution, assets.light_yield)
    s = sum(weights)
    if iszero(s)
        return weights  # already zeros of correct type
    end
    eff_vals = eff(assets.out_centers, params)
    return weights .* eff_vals
end

# Full response matrix, AD-safe
function construct_response_matrix(params, assets)
    n_out = length(assets.out_centers)
    n_er = length(assets.er_centers)

    first_col = response_matrix_per_er_bin(first(assets.er_centers), params, assets)
    Tcol = eltype(first_col)
    A = Array{Tcol}(undef, n_out, n_er)
    A[:, 1] = max.(first_col, zero(eltype(first_col)))

    for j in 2:n_er
        col = response_matrix_per_er_bin(assets.er_centers[j], params, assets)
        A[:, j] = length(col) == n_out ? max.(col, zero(eltype(col))) : fill(zero(Tcol), n_out)
    end
    return A
end

function build_rate_matrix(er_centers, enu_centers, nupar, physics, params, Rn_key)
    physics.cevns_xsec.diff_xsec_csi(er_centers, enu_centers, params, nupar, Rn_key)
end

function get_rate_matrix(params, physics)
    # Simply return the dictionary of diff_xsec matrices for the given params
    return physics.cevns_xsec.diff_xsec(params)
end

function get_expected(params, physics, assets)
    # --- Step 1: Construct detector response (n_out × n_Er)
    response_matrix = construct_response_matrix(params, assets)

    # --- Step 2: Get flux and differential cross-sections
    flux = physics.sns_flux.flux(params)                  # (n_Enu, n_time)
    diff_xsec_dict = physics.cevns_xsec.diff_xsec(params) # Dict of (n_Er, n_Enu)

    # --- Step 3: Convert recoil energies from keV → MeV
    er_edges_MeV   = assets.er_edges .* 1e-3
    er_centers_MeV = assets.er_centers .* 1e-3
    dEr_MeV        = diff(er_edges_MeV)
    #dEr_MeV        = vcat(dEr_MeV, last(dEr_MeV))  # pad to match n_Er

    n_Er   = length(er_centers_MeV)
    n_time = size(flux.total_flux, 2)
    first_rate_matrix = first(values(diff_xsec_dict))
    T = eltype(first_rate_matrix)
    flux_folded_rate = zeros(T, n_Er, n_time)  # (E_r × time)
    # --- Step 4: Flux folding (sum over E_ν for each isotope)
    for iso in assets.isotopes
        rate_matrix = diff_xsec_dict[iso.Rn_key]     # (n_Er, n_Eν)
        folded_rate = rate_matrix * flux.total_flux   # (n_Er, n_time)
        flux_folded_rate .+= iso.fraction .* folded_rate
    end

    # --- Step 5: Integrate over recoil energy (multiply by ΔE_r)
    integrated_rate = flux_folded_rate .* dEr_MeV    # (n_Er × n_time)

    # --- Step 6: Multiply by number of target nuclei
    integrated_rate .*= assets.Nt                    # counts/s per E_r, per time bin

    # --- Step 7: Apply detector response
    predicted_counts = response_matrix * integrated_rate  # (n_out × n_time)

    return predicted_counts
end

function get_backgrounds(params, assets)
    scale_template(template, norm) = sum(template) > 0 ?
        norm .* (template ./ sum(template)) :
        fill(zero(norm), length(template))
    brn = scale_template(assets.brn, params.brn_norm)
    nin = scale_template(assets.nin, params.nin_norm)
    ssBkg = scale_template(assets.ssBkg, params.ss_bkg_norm)
    return (brn, nin, ssBkg)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        signal = get_expected(params, physics, assets)
        brn, nin, ssBkg = get_backgrounds(params, assets)
        
        total_bkg = brn .+ nin .+ ssBkg
        exp_events = signal .+ total_bkg
    
        distprod(Poisson.(exp_events))
    end
end

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        nothing
    end
end


end
