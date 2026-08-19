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

function rebin_1d_projection(centers, weights, edges)
    rebinned = zeros(eltype(weights), length(edges) - 1)
    for i in eachindex(rebinned)
        idx = findall(c -> c >= edges[i] && c < edges[i + 1], centers)
        if !isempty(idx)
            rebinned[i] = sum(weights[idx])
        end
    end
    return rebinned
end

function edges_from_centers(centers)
    n = length(centers)
    @assert n >= 2
    edges = similar(centers, n + 1)
    edges[2:n] .= (centers[1:end-1] .+ centers[2:end]) ./ 2
    edges[1] = centers[1] - (edges[2] - centers[1])
    edges[end] = centers[end] + (centers[end] - edges[end-1])
    return edges
end

function rebin_histogram_1d(source_weights, source_edges, target_edges)
    T = promote_type(eltype(source_weights), eltype(source_edges), eltype(target_edges))
    rebinned = zeros(T, length(target_edges) - 1)
    dt = diff(source_edges)
    density = source_weights ./ dt

    for out_bin in eachindex(rebinned)
        out_lo = target_edges[out_bin]
        out_hi = target_edges[out_bin + 1]
        for in_bin in eachindex(source_weights)
            in_lo = source_edges[in_bin]
            in_hi = source_edges[in_bin + 1]
            overlap = min(out_hi, in_hi) - max(out_lo, in_lo)
            if overlap > zero(overlap)
                rebinned[out_bin] += density[in_bin] * overlap
            end
        end
    end

    return rebinned
end

function get_timing_efficiency(time_edges)
    a_ns = 520.0
    b_per_ns = 0.0494 / 1e3
    eff = zeros(Float64, length(time_edges) - 1)

    for i in eachindex(eff)
        lo = time_edges[i]
        hi = time_edges[i + 1]
        width = hi - lo

        integral = if hi <= a_ns
            width
        elseif lo >= a_ns
            (exp(-b_per_ns * (lo - a_ns)) - exp(-b_per_ns * (hi - a_ns))) / b_per_ns
        else
            (a_ns - lo) + (1 - exp(-b_per_ns * (hi - a_ns))) / b_per_ns
        end

        eff[i] = integral / width
    end

    return eff
end

function configure(; datadir = @__DIR__, use_flux_data::Bool = true, ff_model::Symbol = :helm, ff_kwargs::NamedTuple = (;), sns_flux_kwargs::NamedTuple = (;))
    assets = get_assets(datadir)

    # Configure the SNS flux module
    sns_flux = Newtrinos.sns_flux.configure(
        exposure = assets.exposure,
        distance = assets.distance,
        use_data = use_flux_data,
        sns_flux_kwargs...
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
    coherent_csi_eff_sigma = 0.0,
    coherent_csi_qf_pc0 = 0.0,
    coherent_csi_qf_pc1 = 0.0,
        coherent_csi_brn_norm = brn_nom,  # Normalization factor for BRN
        coherent_csi_nin_norm = nin_nom,  # Normalization factor for NIN
        coherent_csi_ss_bkg_norm = ss_bkg_nom,  # Normalization factor for SS background
        )
end

# TODO!
function get_priors(ss_bkg_nom, brn_nom, nin_nom)
    priors = (
    coherent_csi_eff_sigma = truncated(Normal(0.0, 1.0), -3.0, 3.0),
    coherent_csi_qf_pc0 = truncated(Normal(0.0, 1.0), -3.0, 3.0),
    coherent_csi_qf_pc1 = truncated(Normal(0.0, 1.0), -3.0, 3.0),
        coherent_csi_brn_norm = truncated(Normal(brn_nom, 0.25 * brn_nom), 0.0, brn_nom + 3 * 0.25 * brn_nom),  # Normalization factor for BRN
        coherent_csi_nin_norm = truncated(Normal(nin_nom, 0.36 * nin_nom), 0.0, nin_nom + 3 * 0.36 * nin_nom),  # Normalization factor for NIN
        coherent_csi_ss_bkg_norm = truncated(Normal(ss_bkg_nom, 0.021 * ss_bkg_nom), 0.0, ss_bkg_nom + 3 * 0.021 * ss_bkg_nom),  # Normalization factor for SS background
        )
end

function get_assets(datadir = @__DIR__, sns_flux = nothing)
    #@info "Loading coherent csi data"

    # Basic assets that are always loaded
    er_edges = LinRange(3, 200, Int((200 - 3) / 0.5))  # keV
    isotopes = [
        (fraction=0.49, mass=123.8e3, Z=55, N=78, Rn_key=:Rn_Cs, Rn_nom=5.0),  # Cs-133
        (fraction=0.51, mass=118.21e3, Z=53, N=74, Rn_key=:Rn_I, Rn_nom=5.0)   # I-127
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
    time_efficiency = nothing
    brn_time_source = nothing
    nin_time_source = nothing
    brn_time_source_edges = nothing
    nin_time_source_edges = nothing
    brn_pe_pdf = nothing
    nin_pe_pdf = nothing
    # Check if sns_flux is provided and has time bin centers
    if sns_flux !== nothing && haskey(sns_flux.assets, :T)
        @info "Loading and binning CsI data"
        @info "Configuring Flux"
        time_edges = sns_flux.assets.T  # Extract time bin-edges from sns_flux (nanoseconds)
        time_bins = midpoints(time_edges)  # Bin centers
        time_efficiency = get_timing_efficiency(time_edges)
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

        # BRN and NIN are provided as uncorrelated PE and time projections.
        # PE distributions already include epsilon_PE, while time distributions do not include epsilon_T.
        brn_pe_pdf = rebin_1d_projection(brnPE_df[:, 1], brnPE_df[:, 2], out_edges)
        nin_pe_pdf = rebin_1d_projection(ninPE_df[:, 1], ninPE_df[:, 2], out_edges)
        brn_time_source = brnTrec_df[:, 2]
        nin_time_source = ninTrec_df[:, 2]
        brn_time_source_edges = edges_from_centers(brnTrec_df[:, 1])
        nin_time_source_edges = edges_from_centers(ninTrec_df[:, 1])

        if sum(brn_pe_pdf) > 0
            brn_pe_pdf ./= sum(brn_pe_pdf)
        end
        if sum(nin_pe_pdf) > 0
            nin_pe_pdf ./= sum(nin_pe_pdf)
        end

        brn_time_counts = rebin_histogram_1d(brn_time_source, brn_time_source_edges, time_edges)
        nin_time_counts = rebin_histogram_1d(nin_time_source, nin_time_source_edges, time_edges)
        brn = reshape(brn_pe_pdf, :, 1) .* reshape(brn_time_counts, 1, :)
        nin = reshape(nin_pe_pdf, :, 1) .* reshape(nin_time_counts, 1, :)

        # Get initial nominal value for Bkg normalizations
        ss_bkg_nom = sum(ssBkg)
        #@info "Initial SS background normalization: $ss_bkg_nom"
        brn_nom = sum(brn .* reshape(time_efficiency, 1, :))
        #@info "Initial BRN background normalization: $brn_nom"
        nin_nom = sum(nin .* reshape(time_efficiency, 1, :))
        #@info "Initial NIN background normalization: $nin_nom"
    else
        @info "Flux is not fully configured yet."
    end

    distance = 1930  # cm
    exposure = 13.99  # GWh
    eff_nominal = (1.32045, 0.285979, 10.8646, -0.333322)
    eff_delta = (0.02345, -0.000613, -1.01862, -0.023042)
    qf_nominal = (0.0554628, 4.30681, -111.707, 840.384)
    qf_pc0_delta = (0.0059004, -0.79134, 26.1515, -244.819)
    qf_pc1_delta = (-4.98e-5, -0.37084, 18.60225, -210.294)

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
        brn_pe_pdf,
        brn_nom,
        nin,
        nin_pe_pdf,
        nin_nom,
        brn_time_source,
        nin_time_source,
        brn_time_source_edges,
        nin_time_source_edges,
        ssBkg,
        ss_bkg_nom,
        distance,
        exposure,
        time_efficiency,
        eff_nominal,
        eff_delta,
        qf_nominal,
        qf_pc0_delta,
        qf_pc1_delta,
    )
end

function construct_time_response_matrix(assets)
    Diagonal(assets.time_efficiency)
end

function csi_qf_coefficients(params, assets)
    z0 = params.coherent_csi_qf_pc0
    z1 = params.coherent_csi_qf_pc1
    return ntuple(i -> assets.qf_nominal[i] + z0 * assets.qf_pc0_delta[i] + z1 * assets.qf_pc1_delta[i], 4)
end

function csi_eff_coefficients(params, assets)
    z = params.coherent_csi_eff_sigma
    return ntuple(i -> assets.eff_nominal[i] + z * assets.eff_delta[i], 4)
end

# QF: accepts scalar or array, returns same shape, type-generic
function qf(er, params, assets)
    a, b, c, d = csi_qf_coefficients(params, assets)
    x = er .* 1e-3                         # MeV
    vals = (a .* x .+ b .* x.^2 .+ c .* x.^3 .+ d .* x.^4) .* 1e3  # keVee
    z = vals isa AbstractArray ? zero(eltype(vals)) : zero(vals)
    return max.(vals, z)
end

# Efficiency: accepts scalar or array of PE, type-generic
function eff(pe, params, assets)
    a, b, c, d = csi_eff_coefficients(params, assets)
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
    keVee = qf(keVnr, params, assets)            # scalar (possibly Dual)
    weights = gamma_pdf_integrated_over_bins(keVee, assets.out_centers, assets.out_edges,
                                             assets.resolution, assets.light_yield)
    s = sum(weights)
    if iszero(s)
        return weights  # already zeros of correct type
    end
    eff_vals = eff(assets.out_centers, params, assets)
    return weights .* eff_vals
end

# Full PE response matrix, AD-safe
function construct_pe_response_matrix(params, assets)
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

function construct_detector_response(params, assets)
    return (
        pe = construct_pe_response_matrix(params, assets),
        time = construct_time_response_matrix(assets),
    )
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
    detector_response = construct_detector_response(params, assets)

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

    # --- Step 7: Apply PE response on the left and time response on the right
    predicted_counts = detector_response.pe * integrated_rate * detector_response.time  # (n_out × n_time)

    return predicted_counts
end

function shift_time_template_1d(template, time_edges, dt_shift)
    if time_edges === nothing || iszero(dt_shift)
        return template
    end

    n_time = length(template)
    T = promote_type(eltype(template), typeof(dt_shift), eltype(time_edges))
    shifted = zeros(T, n_time)
    dt = diff(time_edges)
    density = template ./ dt

    for out_bin in eachindex(template)
        out_lo = time_edges[out_bin] - dt_shift
        out_hi = time_edges[out_bin + 1] - dt_shift

        for in_bin in eachindex(template)
            in_lo = time_edges[in_bin]
            in_hi = time_edges[in_bin + 1]
            overlap = min(out_hi, in_hi) - max(out_lo, in_lo)
            if overlap > zero(overlap)
                shifted[out_bin] += density[in_bin] * overlap
            end
        end
    end

    return shifted
end

function get_backgrounds(params, assets)
    scale_template(template, norm) = sum(template) > 0 ?
        norm .* (template ./ sum(template)) :
        fill(zero(norm), length(template))
    flux_onset = if haskey(params, :flux_onset)
        params.flux_onset
    else
        @warn "Missing flux_onset in parameter set; using flux_onset = 0.0 for CsI beam-related backgrounds. Pass merged experiment+physics params to apply the shared timing shift."
        0.0
    end
    detector_response = construct_detector_response(params, assets)

    brn_time_shifted = shift_time_template_1d(assets.brn_time_source, assets.brn_time_source_edges, flux_onset)
    nin_time_shifted = shift_time_template_1d(assets.nin_time_source, assets.nin_time_source_edges, flux_onset)
    brn_time_counts = rebin_histogram_1d(brn_time_shifted, assets.brn_time_source_edges, assets.time_edges)
    nin_time_counts = rebin_histogram_1d(nin_time_shifted, assets.nin_time_source_edges, assets.time_edges)

    brn_template = reshape(assets.brn_pe_pdf, :, 1) .* reshape(brn_time_counts, 1, :)
    nin_template = reshape(assets.nin_pe_pdf, :, 1) .* reshape(nin_time_counts, 1, :)
    brn_template = brn_template * detector_response.time
    nin_template = nin_template * detector_response.time
    brn = scale_template(brn_template, params.coherent_csi_brn_norm)
    nin = scale_template(nin_template, params.coherent_csi_nin_norm)
    ssBkg = scale_template(assets.ssBkg, params.coherent_csi_ss_bkg_norm)
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
