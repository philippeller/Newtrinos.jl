module coherent_lAr

using DataFrames
using CSV
using Distributions
using Interpolations
using LinearAlgebra
using Statistics
using DataStructures
using BAT
using CairoMakie
using Logging
using StatsBase
using ..Helpers
import ..Newtrinos

@kwdef struct COHERENT_LAR <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function configure(; datadir = @__DIR__, use_flux_data::Bool = false, ff_model::Symbol = :helm, ff_kwargs::NamedTuple = (;))
    # Load assets for the experiment
    assets = get_assets(datadir)

    # Configure the SNS flux module
    sns_flux = Newtrinos.sns_flux.configure(
        exposure = assets.exposure,
        distance = assets.distance,
        use_data = use_flux_data,
    )

    # Configure the CEvNS cross-section module (pass FF choice through configure)
    cevns_xsec = Newtrinos.cevns_xsec.configure(
        assets.isotopes,
        assets.er_centers .* 1e-3,  # Convert keVnr to MeVnr
        sns_flux.assets.E;          # Pass the energy grid from the SNS flux assets
        ff_model = ff_model,
        ff_kwargs = ff_kwargs,
    )

    # Combine SNS flux and CEvNS cross-section into the physics NamedTuple
    physics = (;sns_flux = sns_flux, cevns_xsec = cevns_xsec)

    # Build experiment-specific parameters and priors
    params = get_params(assets.ss_bkg_nom, assets.pbrn_nom, assets.delbrn_nom)
    priors = get_priors(assets.ss_bkg_nom, assets.pbrn_nom, assets.delbrn_nom)

    # Return the configured COHERENT_LAR instance
    return COHERENT_LAR(
        physics = physics,
        params = params,
        priors = priors,
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

function get_params(ss_bkg_nom, pbrn_nom, delbrn_nom)
    params = (
        coherent_lar_qfa_a = 0.246,  # QF polynomial coefficients
        coherent_lar_qfa_b = 0.00078,
        coherent_lar_mass = 24.4,  # kg
        pbrn_norm= pbrn_nom,  # Normalization factor for BRN
        delbrn_norm= delbrn_nom,  # Normalization factor for delBRN
        ss_bkg_norm= ss_bkg_nom,  # Normalization factor for SS background
        )
end

# TODO!
function get_priors(ss_bkg_nom, pbrn_nom, delbrn_nom)
    priors = (
        coherent_lar_qfa_a = Normal(0.246, 0.006),
        coherent_lar_qfa_b = Normal(0.00078, 0.00009),
        coherent_lar_mass = truncated(Normal(24.4, 0.61), 0.0, 24.4 + 3 * 0.61),
        pbrn_norm= truncated(Normal(pbrn_nom, 0.32 * pbrn_nom), 0.0, pbrn_nom + 3 * 0.32 * pbrn_nom),  # Normalization factor for BRN
        delbrn_norm= truncated(Normal(delbrn_nom, 1.0 * delbrn_nom), 0.0, delbrn_nom + 3 * 1.0 * delbrn_nom),  # Normalization factor for delBRN
        ss_bkg_norm= truncated(Normal(ss_bkg_nom, 0.008 * ss_bkg_nom), 0.0, ss_bkg_nom + 3 * 0.008 * ss_bkg_nom),  # Normalization factor for SS background
        )
end

function get_assets(datadir = @__DIR__)
    @info "Loading coherent lAr data"


    er_edges = collect(3:0.5:300) # keVnr
    isotopes = [
        (fraction=1.0, mass=37.3e3, Z=18, N=22, Rn_key=:Rn_Ar, Rn_nom=4.1039) # Ar-37
    ] # List of isotopes with [fraction, Nuclear mass (GeV), Z, N=A-Z, Rn_key]
    Nt = (1.0/0.039948) * 6.023e+23
    
    resolution = 0.58  # a/Eee and b*Eee
    # Import Data
    eff_data = CSV.read(joinpath(datadir, "lAr/CENNS10AnlAEfficiency.txt"), DataFrame, comment="#", header=false, delim=' ') # columns: keVee, keVnr, efficiency
    pbrn = CSV.read(joinpath(datadir, "lAr/brnpdf.txt"), DataFrame, comment="#", header=false, delim=' ')[:, 4] # columns: keVee, F90, t, counts/bin/6.12GWh
    delbrn = CSV.read(joinpath(datadir, "lAr/delbrnpdf.txt"), DataFrame, comment="#", header=false, delim=' ')[:, 4] # columns: keVee, F90, t, counts/bin/6.12GWh
    ss_bkg = CSV.read(joinpath(datadir, "lAr/bkgpdf.txt"), DataFrame, comment="#", header=false, delim=' ')[:, 4] # columns: keVee, F90, t, counts/bin/6.12GWh
    observed = CSV.read(joinpath(datadir, "lAr/datanobkgsub.txt"), DataFrame, comment="#", header=false, delim=' ')[:, 4] # columns: keVee, F90, t, counts/bin/6.12GWh
    f90_data = CSV.read(joinpath(datadir, "lAr/f90data1d.txt"), DataFrame, skipto=2, header=false, delim=' ') # columns: Bin Center (F_{90}), SS-Sub Data, etc.
    timing_data = CSV.read(joinpath(datadir, "lAr/timingdata1d.txt"), DataFrame, skipto=2, header=false, delim=' ') # columns: Bin Center (F_{90}), SS-Sub Data, etc.

    # Extract and normalize the 4th column (CEvNS PDF) from f90data1d.txt
    f90_pdf = f90_data[:, 4]  # Extract the 4th column
    f90_pdf .= f90_pdf ./ sum(f90_pdf)  # Normalize the PDF in place

    # Extract and normalize the 4th column (CEvNS PDF) from timingdata1d.txt
    timing_pdf = timing_data[:, 4]  # Extract the 4th column
    timing_pdf .= timing_pdf ./ sum(timing_pdf)  # Normalize the PDF in place

    # Define bin centers as the average of consecutive edges
    er_centers = collect((er_edges[1:end-1] + er_edges[2:end]) ./ 2) # Reconstruct bin centers from edges

    # Output bin centers [keVee] (Match Data)
    out_width = 10.0
    out_edges = collect(0.0:out_width:120.0)         # Edges: [0, 10, 20, ..., 120]
    out_centers = midpoints(out_edges)              # Centers: [5, 15, 25, ..., 115]

    # Extract F90 and timing bin centers from the first column of f90data1d.txt and timingdata1d.txt
    f90_centers = f90_data[:, 1]  # Bin Center (F_{90})
    timing_centers = timing_data[:, 1]  # Bin Center (Timing)

    # Get initial nominal value for Bkg normalizations
    ss_bkg_nom = sum(ss_bkg)  # Sum over the last column (counts) of ss_bkg
    #@info "Initial SS background normalization: $ss_bkg_nom"

    pbrn_nom = sum(pbrn)  # Sum over the last column (counts) of pbrn
    #@info "Initial BRN background normalization: $pbrn_nom"

    delbrn_nom = sum(delbrn)  # Sum over the last column (counts) of delbrn
    #@info "Initial delBRN background normalization: $delbrn_nom"
    
    distance = 2750 # cm
    exposure = 6.12 # GWh


    assets = (;
        observed,
        er_edges,
        er_centers,
        out_centers,
        f90_centers,
        timing_centers,
        isotopes,
        Nt,
        resolution,
        eff_data,
        f90_pdf,
        timing_pdf,
        pbrn,
        pbrn_nom,
        delbrn,
        delbrn_nom,
        ss_bkg,
        ss_bkg_nom,
        distance,
        exposure,
    )
end

function qf(er_centers, params)
    a = params.coherent_lar_qfa_a
    b = params.coherent_lar_qfa_b
    vals = @. (a + b * er_centers) * er_centers
    z = vals isa AbstractArray ? zero(eltype(vals)) : zero(vals)
    return max.(vals, z)  # clamp to >= 0 with typed zero
end

# AD-safe piecewise-linear interpolation for efficiency
function eff(keVee, assets)
    x = assets.eff_data[:, 1]  # Float64 knots
    y = assets.eff_data[:, 3]  # Float64 values
    T = eltype(keVee)
    vals = similar(keVee, promote_type(T, Float64))
    xmin, xmax = first(x), last(x)
    for i in eachindex(keVee)
        u = keVee[i]
        if u <= xmin
            vals[i] = zero(u)                        # typed zero
        elseif u >= xmax
            vals[i] = one(u) * y[end]               # promote to T
        else
            # Find rightmost x[j] ≤ u
            j = findlast(x .<= u)
            if j === nothing
                vals[i] = zero(u)
            elseif j == length(x)
                vals[i] = one(u) * y[end]
            else
                x0 = x[j]; x1 = x[j + 1]
                y0 = y[j]; y1 = y[j + 1]
                t = (u - x0) / (x1 - x0)
                vals[i] = one(u) * y0 + t * (y1 - y0)  # linear interp, returns T
            end
        end
    end
    return vals
end

function sigma_keVee(keVee, assets)
    a = assets.resolution
    return @. a * sqrt(keVee)  # keVee ≥ 0 due to qf clamp; works with Duals
end

function construct_response_matrix(params, assets)
    er_centers = assets.er_centers
    out_centers = assets.out_centers
    n_out = length(out_centers)
    n_er = length(er_centers)

    # Precompute
    keVee = qf(er_centers, params)
    sigma = sigma_keVee(keVee, assets)
    eff_vals = eff(keVee, assets)

    # First column to set element type
    col1 = begin
        d = Distributions.Normal(keVee[1], sigma[1])
        v = Distributions.pdf.(d, out_centers)
        s = sum(v)
        s == 0 ? zero.(v) : (v ./ s)
    end
    A = Array{eltype(col1)}(undef, n_out, n_er)
    A[:, 1] = eff_vals[1] .* col1

    for j in 2:n_er
        d = Distributions.Normal(keVee[j], sigma[j])
        v = Distributions.pdf.(d, out_centers)
        s = sum(v)
        v = s == 0 ? zero.(v) : (v ./ s)
        A[:, j] = eff_vals[j] .* v
    end

    # Ensure non-negative with typed zero
    T = eltype(A)
    return max.(A, zero(T))
end

function get_rate_matrix(params, physics)
    # Simply return the dictionary of diff_xsec matrices for the given params
    return physics.cevns_xsec.diff_xsec(params)
end

function get_expected(params, physics, assets)
    # Step 1: Compute the 1D predicted counts
    response_matrix = construct_response_matrix(params, assets)
    flux = physics.sns_flux.flux(params)

    # Get the differential cross-section for all isotopes
    diff_xsec_dict = physics.cevns_xsec.diff_xsec(params)
    # Convert recoil energies from keV → MeV
    er_edges_MeV   = assets.er_edges .* 1e-3
    er_centers_MeV = assets.er_centers .* 1e-3
    dEr_MeV        = diff(er_edges_MeV)
    n_Er   = length(er_centers_MeV)

    flux_folded_rate = zeros(eltype(first(values(diff_xsec_dict))), n_Er)

    # --- Step 4: Flux folding (sum over E_ν for each isotope)
    for iso in assets.isotopes
        rate_matrix = diff_xsec_dict[iso.Rn_key]     # (n_Er, n_Eν)
        folded_rate = rate_matrix * flux.total_flux   # (n_Er,)
        flux_folded_rate .+= iso.fraction .* folded_rate
    end

    int_rate = params.coherent_lar_mass .* assets.Nt .* flux_folded_rate .* dEr_MeV  # Integrate over E_ν and scale
    predicted_counts = response_matrix * int_rate

    # Step 2: Break the 1D predicted counts into f90-bin and time-bin counts
    f90_pdf = assets.f90_pdf  # Normalized f90 PDF
    timing_pdf = assets.timing_pdf  # Normalized timing PDF

    n_out_bins = length(predicted_counts)  # Number of out_center bins
    n_f90_bins = length(f90_pdf)  # Number of f90 bins
    n_time_bins = length(timing_pdf)  # Number of time bins

    # Initialize the final array with the same type as `predicted_counts`
    expanded_counts = similar(predicted_counts, n_out_bins * n_f90_bins * n_time_bins)

    # Loop over each out_center bin and distribute counts
    idx = 1
    for i in 1:n_out_bins
        for j in 1:n_f90_bins
            for k in 1:n_time_bins
                # Compute the weight for each bin
                expanded_counts[idx] = predicted_counts[i] * f90_pdf[j] * timing_pdf[k]
                idx += 1
            end
        end
    end

    return expanded_counts
end

function get_backgrounds(params, assets)
    scale_template(template, norm) =
        sum(template) > 0 ?
            norm .* (template ./ sum(template)) :
            fill(zero(norm), length(template))

    # Extract the last column (counts) from the 3D binned data
    pbrn_counts = assets.pbrn
    delbrn_counts = assets.delbrn
    ss_bkg_counts = assets.ss_bkg

    # Scale the templates
    pbrn = scale_template(pbrn_counts, params.pbrn_norm)
    delbrn = scale_template(delbrn_counts, params.delbrn_norm)
    ss_bkg = scale_template(ss_bkg_counts, params.ss_bkg_norm)

    return (pbrn, delbrn, ss_bkg)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        signal = get_expected(params, physics, assets)
        bkg_pbrn, bkg_delbrn, bkg_ss_bkg = get_backgrounds(params, assets)
        total_bkg = bkg_pbrn .+ bkg_delbrn .+ bkg_ss_bkg
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
