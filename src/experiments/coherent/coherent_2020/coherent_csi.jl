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
    physics = (;physics.sns_flux, physics.cevns_xsec)
    assets = get_assets(physics)
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
        )
end

# TODO!
function get_priors()
    priors = (
        coherent_csi_eff_a = Truncated(Normal(1.32045, 0.02), 0, 1),
        coherent_csi_eff_b = Truncated(Normal(0.285979, 0.0006), 0, 1),
        coherent_csi_eff_c = Truncated(Normal(10.8646, 1.), 0, 1),
        coherent_csi_eff_d = Truncated(Normal(-0.333322, 0.03), 0, 1),
        )
end

function get_assets(physics, datadir = @__DIR__)
    @info "Loading coherent csi data"


    er_edges = LinRange(3, 100, Int((100-3)/0.5)) # keV

    isotopes = [[0.49,123.8e3,55,78,5.7242],[0.51,118.21e3,53,74,5.7242]] # List of isotopes with [fraction, Nuclear mass (GeV), Z, N=A-Z, R_n (fm)]
    Nt = 2 * (14.6/0.25981) * 6.023e+23
    qfa = [0.0554628, 4.30681, -111.707, 840.384]  # QF polynomial coefficients
    light_yield = 13.35  # PE/keVee
    resolution = [0.0749, 9.56]  # a/Eee and b*Eee
    #efficiency = [1.32045, 0.285979, 10.8646, -0.333322]  # Sigmoid + offset
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
        qfa,
        light_yield,
        resolution,
        brnPE,
        ninPE,
        ssBkg,
        distance,
        exposure,
        )
end

function qf(er_centers, qfa)
    a, b, c, d = qfa
    er_centers *= 1e-3 #Convert to MeV
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
    keVee = qf(keVnr, assets.qfa)
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
    A = stack([response_matrix_per_er_bin(keVnr, params, assets) for keVnr in assets.er_centers])
end

function build_rate_matrix(er_centers, enu_centers, nupar, physics, params)
    """
    Vectorized version: computes a stack of rate matrices for each freepar set.
    - freepar_array: shape (n_samples, 4)
    - Returns: R: shape (n_samples, n_er, n_enu)
    """
    physics.cevns_xsec.diff_xsec_csi(er_centers, enu_centers, params, nupar)
end


function get_expected(params, physics, assets)

    response_matrix = construct_response_matrix(params, assets)
        
    flux = physics.sns_flux.flux(assets.exposure, assets.distance, params)
    
    """
    Step-by-step vectorized signal model for multiple free parameter sets.
        - freepar_array: shape (n_samples, 4)
        - Returns: predicted signals: shape (n_samples, n_final_bins)
    """
    # This will hold the sum over isotopes for each parameter set
    dNdEr_all = zeros(eltype(assets.er_centers), size(assets.er_centers))

    for nupar in assets.isotopes
        # 1. Build the rate matrix for all parameter sets: (n_samples, n_er, n_enu)
        rate_matrix = build_rate_matrix(
            assets.er_centers * 1e-3,  # Convert to MeV
            flux.E,         # Neutrino energy centers
            nupar,
            physics,
            params,
        )
        #@show size(rate_matrix)
        # 2. Fold with flux for all parameter sets at once (vectorized)
        # rate_matrix: (n_samples, n_er, n_enu), self.flux[:, 1]: (n_enu,)
        #dNdEr = rate_matrix * flux.total_flux'  # (n_samples, n_er)
        dNdEr = sum(rate_matrix .* flux.total_flux', dims=2)
        dNdEr = dropdims(dNdEr, dims=2)  # result has shape (4, 193)
        
        dNdEr_all .+= nupar[1] * dNdEr  # sum over isotopes
    end
    
    # 3. Integrate over recoil energy bins (multiply by bin width)
    
    int_rate = assets.Nt * dNdEr_all .* diff(assets.er_edges * 1e-3) # (n_samples, n_er)

    # 4. Apply detector response matrix to get final predicted counts

    
    predicted_counts = response_matrix * int_rate # (n_samples, n_output_bins)

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