module coherent_csi

using DataFrames
using CSV
using Distributions
using LinearAlgebra
using Statistics
using DataStructures
using BAT
using CairoMakie
using Logging
using StatsBase
import ..Newtrinos

@kwdef struct COHERENT_CSI <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function configure(physics)
    assets = get_assets(physics)
    # Dynamically build params/priors for cevns_xsec using isotope list (with Rn_nom)
    cevns_params, cevns_priors = Newtrinos.cevns_xsec.build_params_and_priors(assets.isotopes)
    # Reconfigure cevns_xsec with correct params/priors
    cevns_xsec = Newtrinos.cevns_xsec.configure(cevns_params, cevns_priors)
    # Update physics NamedTuple with new cevns_xsec
    physics = (;physics.sns_flux, cevns_xsec)
    return COHERENT_CSI(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_params()
    params = (
        coherent_csi_eff_a = 1.32045,
        coherent_csi_eff_b = 0.285979,
        coherent_csi_eff_c = 10.8646,
        coherent_csi_eff_d = -0.333322,
        coherent_csi_qfa_a = 0.0554628,  # QF polynomial coefficients
        coherent_csi_qfa_b = 4.30681,
        coherent_csi_qfa_c = -111.707,
        coherent_csi_qfa_d = 840.384,      
        )
end

# TODO!
function get_priors()
    priors = (
        coherent_csi_eff_a = Truncated(Normal(1.32045, 0.02), 0, 1),
        coherent_csi_eff_b = Truncated(Normal(0.285979, 0.0006), 0, 1),
        coherent_csi_eff_c = Truncated(Normal(10.8646, 1.), 0, 1),
        coherent_csi_eff_d = Truncated(Normal(-0.333322, 0.03), 0, 1),
        coherent_csi_qfa_a = Normal(0.0554628, 0.0059),
        coherent_csi_qfa_b = Normal(4.30681, 0.79),
        coherent_csi_qfa_c = Normal(-111.707, 26.15),
        coherent_csi_qfa_d = Normal(840.384, 244.82),
        )
end

function get_assets(physics, datadir = @__DIR__)
    @info "Loading coherent csi data"


    er_edges = LinRange(3, 100, Int((100-3)/0.5)) # keV
    isotopes = [
        (fraction=0.49, mass=123.8e3, Z=55, N=78, Rn_key=:Rn_Cs, Rn_nom=5.7242),  # Cs-133
        (fraction=0.51, mass=118.21e3, Z=53, N=74, Rn_key=:Rn_I, Rn_nom=5.7242) # I-127
    ] # List of isotopes with [fraction, Nuclear mass (GeV), Z, N=A-Z, Rn_key]
    Nt = 2 * (14.6/0.25981) * 6.023e+23
    light_yield = 13.35  # PE/keVee
    resolution = [0.0749, 9.56]  # a/Eee and b*Eee
    # Import Data
    brnPE = CSV.read(joinpath(datadir, "csi/brnPE.txt"), DataFrame, comment="#", header=false, delim=' ') # columns: PE, BRN PDF
    ninPE = CSV.read(joinpath(datadir, "csi/ninPE.txt"), DataFrame, comment="#", header=false, delim=' ') # columns: PE, NIN PDF
    ssBkg = CSV.read(joinpath(datadir, "csi/dataBeamOnAC.txt"), DataFrame, comment="#", header=false, delim=' ') # columns: PE, SS Background
    observed = CSV.read(joinpath(datadir, "csi/dataBeamOnC.txt"), DataFrame, comment="#", header=false, delim=' ') # columns: PE, Observed Counts

    # Reconstruct bin edges from centers
    er_centers = midpoints(er_edges)

    pe_width = 5.0
    out_edges = collect(2.5:pe_width:62.5)  # Bin edges: 5–60 PE, 5 PE width
    out_centers = midpoints(out_edges) # Bin centers

    distance = 19.3 # m
    exposure = 13.99 # GWh


    assets = (;
        observed,
        er_edges,
        er_centers,
        out_edges,
        out_centers,
        isotopes,
        Nt,
        light_yield,
        resolution,
        brnPE,
        ninPE,
        ssBkg,
        distance,
        exposure,
    )
end

function qf(er_centers, params)
    a = params.coherent_csi_qfa_a
    b = params.coherent_csi_qfa_b
    c = params.coherent_csi_qfa_c
    d = params.coherent_csi_qfa_d
    er_centers = er_centers * 1e-3 #Convert to MeV (no mutation)
    return @. (a * er_centers + b * er_centers ^ 2 + c * er_centers ^ 3 + d * er_centers ^ 4) * 1e3  # Convert to keVee
end

function eff(pe, params)
    a = params.coherent_csi_eff_a
    b = params.coherent_csi_eff_b
    c = params.coherent_csi_eff_c
    d = params.coherent_csi_eff_d
    return @. a / (1 + exp(-b * (pe - c))) + d
end

function gamma_pdf_integrated_over_bins(Eee, pe_centers, pe_edges, resolution, light_yield)
    if Eee <= 0
        return zeros(eltype(pe_centers), size(pe_centers))
    end

    # Resolution parameters
    a = resolution[1] / Eee
    b = resolution[2] * Eee

    shape = 1 + b
    scale = 1 / (a * (1 + b))  # mean = shape * scale = 1 / a

    mu_pe = Eee * light_yield

    # Integrate Gamma PDF over each PE bin
    CDF = cdf(Distributions.Gamma(shape, scale), pe_edges)
    pdf_vals = diff(CDF)  # P(bin i) = CDF(edge[i+1]) - CDF(edge[i])

    return pdf_vals
end


function response_matrix_per_er_bin(keVnr, params, assets)
    keVee = qf(keVnr, params)
    if keVee <= 0
        return zeros(size(assets.out_centers))
    end

    # Compute the gamma PDF values over the PE bins
    gamma_weights = gamma_pdf_integrated_over_bins(keVee, assets.out_centers, assets.out_edges, assets.resolution, assets.light_yield)
    if sum(gamma_weights) == 0
        return zeros(size(assets.out_centers)) # skip if no contribution
    end

    # Normalize first (per keVnr)
    gamma_weights ./= sum(gamma_weights)

    # Apply PE-dependent detection efficiency
    eff_vals = eff(assets.out_centers, params)
    gamma_weights .* eff_vals
end  
        
function construct_response_matrix(params, assets)
    n_out = length(assets.out_centers)
    n_er = length(assets.er_centers)
    A = zeros(n_out, n_er)
    for (j, keVnr) in enumerate(assets.er_centers)
        col = response_matrix_per_er_bin(keVnr, params, assets)
        if length(col) == n_out
            A[:, j] = max.(col, 0)  # zero out negatives
        else
            A[:, j] .= 0  # fallback if something goes wrong
        end
    end
    return A
end

function build_rate_matrix(er_centers, enu_centers, nupar, physics, params, Rn_key)
    physics.cevns_xsec.diff_xsec_csi(er_centers, enu_centers, params, nupar, Rn_key)
end


function get_expected(params, physics, assets)
    response_matrix = construct_response_matrix(params, assets)
    flux = physics.sns_flux.flux(exposure=assets.exposure, distance=assets.distance, params=params)
    dNdEr_all = zeros(eltype(assets.er_centers), size(assets.er_centers))
    for iso in assets.isotopes
        nupar = [iso.fraction, iso.mass, iso.Z, iso.N]
        rate_matrix = build_rate_matrix(
            assets.er_centers * 1e-3,  # convert to MeV
            flux.E,         # Neutrino energy centers (MeV)
            nupar,
            physics,
            params,
            iso.Rn_key,
        )
        dNdEr = sum(rate_matrix .* flux.total_flux', dims=2)
        dNdEr = dropdims(dNdEr, dims=2)
        dNdEr_all .+= iso.fraction * dNdEr
    end
    int_rate = assets.Nt * dNdEr_all .* diff(assets.er_edges * 1e-3)
    predicted_counts = response_matrix * int_rate
    return predicted_counts
end

function get_forward_model(physics, assets)
    function forward_model(params)
        exp_events = get_expected(params, physics, assets)
        distprod(Poisson.(exp_events))
    end
end

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        nothing
    end
end


end