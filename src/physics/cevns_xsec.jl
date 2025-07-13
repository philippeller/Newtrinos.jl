module cevns_xsec

using LinearAlgebra
using Distributions
using SpecialFunctions
using ..Newtrinos

@kwdef struct CevnsXsec <: Newtrinos.Physics
    params::NamedTuple
    priors::NamedTuple
    diff_xsec_lar::Function
    diff_xsec_csi::Function
end

# Only keep the dynamic configure (isotope_keys) and the (params, priors) version
function configure(isotope_keys::Vector{Symbol})
    params, priors = build_params_and_priors(isotope_keys)
    CevnsXsec(
        params = params,
        priors = priors,
        diff_xsec_lar = get_diff_xsec_lar(),
        diff_xsec_csi = get_diff_xsec_csi(),
    )
end

function configure(params::NamedTuple, priors::NamedTuple)
    CevnsXsec(
        params = params,
        priors = priors,
        diff_xsec_lar = get_diff_xsec_lar(),
        diff_xsec_csi = get_diff_xsec_csi(),
    )
end

# Dynamic parameter/prior builder for isotope-specific Rn keys, using isotope list
function build_params_and_priors(isotopes)
    param_dict = Dict(
        :cevns_xsec_a => 0.0,
        :cevns_xsec_b => 0.0,
        :cevns_xsec_c => 0.0,
        :cevns_xsec_d => 0.0,
    )
    prior_dict = Dict{Symbol, Distributions.Distribution}(
        :cevns_xsec_a => Uniform(-1, 1),
        :cevns_xsec_b => Uniform(-1, 1),
        :cevns_xsec_c => Uniform(-1, 1),
        :cevns_xsec_d => Uniform(-1, 1),
    )
    for iso in isotopes
        param_dict[iso.Rn_key] = iso.Rn_nom
        prior_dict[iso.Rn_key] = Normal(iso.Rn_nom, 1)
    end
    return (NamedTuple(param_dict), NamedTuple(prior_dict))
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

function ffsq(er, mn, rn)
    r0 = rn / 197.326963
    arg = 2 * mn * er
    q = sqrt(max(arg, 0))
    j1 = sphericalbesselj(1, q * r0)
    ratio = ifelse.(q .* r0 .== 0, 1.0, (3 .* j1) ./ (q .* r0))
    exp_factor = exp(-((q * (0.9/197.326963))^2) / 2)
    return (ratio * exp_factor)^2
end

function ds(er, enu, params, nupar, Rn_key)
    mN = nupar[2]
    Z = nupar[3]
    N = nupar[4]
    rn = params[Rn_key]
    # Vectorized implementation
    er_grid = reshape(er, :, 1)      # (n_er, 1)
    enu_grid = reshape(enu, 1, :)    # (1, n_enu)
    C1d = (gf^2 / (4 * pi)) .* ffsq.(er, mN, rn)  # (n_er,)
    qwsq = (N - (1 - 4 * sw2) * Z)^2
    freepar_array = [params.cevns_xsec_a, params.cevns_xsec_b, params.cevns_xsec_c, params.cevns_xsec_d]
    sm_pars = [1.0, -1.0, -1.0, 1.0]
    coeffs = freepar_array .+ qwsq .* sm_pars
    # Compute basis arrays for all (er, enu) pairs
    base1 = ones(length(er), length(enu))
    base2 = er_grid ./ enu_grid
    base3 = mN .* er_grid ./ (2 .* enu_grid.^2)
    base4 = (er_grid.^2) ./ (enu_grid.^2)
    xf = coeffs[1] .* base1 .+ coeffs[2] .* base2 .+ coeffs[3] .* base3 .+ coeffs[4] .* base4
    heav = max.(xf, 0)
    # C1d is (n_er,), broadcast over columns
    res = C1d .* mN .* heav
    return res
end

function get_diff_xsec_lar()
    function diff_xsec_lar(er_centers, enu_centers, params, nupar, Rn_key)
        ds(er_centers, enu_centers, params, nupar, Rn_key)
    end
end

function get_diff_xsec_csi()
    function diff_xsec_csi(er_centers, enu_centers, params, nupar, Rn_key)
        ds(er_centers, enu_centers, params, nupar, Rn_key)
    end
end

end