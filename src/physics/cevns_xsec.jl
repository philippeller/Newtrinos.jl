module cevns_xsec

using LinearAlgebra
using Distributions
#using Bessels
using ..Newtrinos


struct CevnsXsecModel end

@kwdef struct CevnsXsec <: Newtrinos.Physics
    params::NamedTuple
    priors::NamedTuple
    diff_xsec_lar::Function
    diff_xsec_csi::Function
end


function configure()
    CevnsXsec(
        params = get_params(),
        priors = get_priors(),
        diff_xsec_lar = get_diff_xsec_lar(),
        diff_xsec_csi = get_diff_xsec_csi(),
        )
end

function get_params()
    (
        cevns_xsec_helm_csi = 5.7, # fm
        cevns_xsec_helm_lar = 4.1, # fm
        cevns_xsec_a = 0.,
        cevns_xsec_b = 0.,
        cevns_xsec_c = 0.,
        cevns_xsec_d = 0.,
    )
    
end

function get_priors()
    (
        cevns_xsec_helm_csi = Normal(5.7, 1),
        cevns_xsec_helm_lar = Normal(4.1, 1),
        cevns_xsec_a = Uniform(-1, 1),
        cevns_xsec_b = Uniform(-1, 1),
        cevns_xsec_c = Uniform(-1, 1),
        cevns_xsec_d = Uniform(-1, 1),
    )
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
    """
    form factor squared.
    
    Parameters:
      er : Recoil energy [MeV] as scalar
      mn : Nuclear mass [MeV] (scalar).
      rn : Nuclear radius [fm] (scalar).
      
    Returns:
      Form factor squared as a NumPy array. Dimensionless
    """
    
    r0 = rn / 197.326963
    # Compute q using a vectorized heaviside to ensure non-negative values.
    arg = 2 * mn * er
    q = sqrt(arg * max(arg, 0))
    
    # Compute the spherical Bessel function j1(q*r0)
    j1 = besselj1(q * r0)
    # Compute ratio; avoid division by zero by using np.where
    # For small q*r0, limit: (3*j1)/(q*r0) -> 1
    ratio = ifelse.(q .* r0 .== 0, 1.0, (3 .* j1) ./ (q .* r0))

    exp_factor = exp(-((q * (0.9/197.326963))^2) / 2)
    
    return (ratio * exp_factor)^2
end



##################################################################
##################################################################
##########---D I F F E R E N T I A L     X S E C---#########################
##################################################################
function ds(er_centers, enu_centers, params, nupar, helm)
    """
    Vectorized dSIGMA/dE_R for CEvNS over a batch of free parameter sets.

    Parameters:
    - er: array of shape (n_er, n_enu) [MeVnr]
    - enu: array of shape (n_er, n_enu) [MeV]
    - freepar_array: shape (n_samples, 4)
    - nupar: [mN, Z, N, R_n]

    Returns:
    - result: array of shape (n_samples, n_er, n_enu)
    """

    #n_er = size(er)
    #n_enu = size(enu)

    
    mN = nupar[2]
    Z = nupar[3]
    N = nupar[4]

    # Constant factor: shape (n_er, n_enu)
    C = @. (gf^2 / (4 * pi)) * ffsq(er_centers, mN, helm)

    # Broadcasted constant factor: shape (n_samples, n_er, n_enu)
    #C = np.broadcast_to(C, (n_samples, n_er, n_enu))

    # Weak nuclear charge squared
    qwsq = (N - (1 - 4 * sw2) * Z)^2

    base = stack([stack([[1., er/enu, mN * er / (2 * enu^2), (er^2) / (enu^2)] for er in er_centers]) for enu in enu_centers])

    freepar_array = [params.cevns_xsec_a, params.cevns_xsec_b, params.cevns_xsec_c, params.cevns_xsec_d]

    # Coefficients: shape (n_samples, 4)
    sm_pars = [1.0, -1.0, -1.0, 1.0]
    
    coeffs = freepar_array .+ qwsq * sm_pars

    # Expand coeffs to shape (n_samples, 4, 1, 1)
    #coeffs = coeffs[:, :, None, None]

    # Weighted sum: shape (n_samples, n_er, n_enu)

    y = coeffs .* base

    xf = dropdims(sum(y, dims=1), dims=1)
    
    # Heaviside cutoff
    heav = max.(xf, 0)

    res = @. C * mN * xf * heav
    
    return  res # shape: (n_samples, n_er, n_enu)

end




function get_diff_xsec_lar()
    function diff_xsec_lar(er_centers, enu_centers, params, nupar)
        ds(er_centers, enu_centers, params, nupar, params.cevns_xsec_helm_lar)
    end
end

function get_diff_xsec_csi()
    function diff_xsec_csi(er_centers, enu_centers, params, nupar)
        ds(er_centers, enu_centers, params, nupar, params.cevns_xsec_helm_csi)
    end
end

end