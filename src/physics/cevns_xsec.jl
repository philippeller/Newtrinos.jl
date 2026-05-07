module cevns_xsec

using LinearAlgebra
using Distributions
using SpecialFunctions
using ..Newtrinos

"""
    CevnsXsec <: Newtrinos.Physics

Configured Coherent Elastic neutrino-Nucleus Scattering (CEvNS) cross-section module.

Computes the differential cross section ``d\\sigma/dE_r`` for neutrino scattering off
atomic nuclei, including optional beyond-Standard-Model deformations parameterized by
`cevns_xsec_a` through `cevns_xsec_d`.

# Fields
- `params::NamedTuple`: nominal values for ``\\sin^2\\theta_W``, the nuclear radii
  `Rn_*` (one per isotope), and the four BSM deformation parameters.
- `priors::NamedTuple`: prior distributions for each parameter.
- `diff_xsec::Function`: closure returning the differential cross section. See
  [`get_diff_xsec`](@ref) for the call signature.
"""
@kwdef struct CevnsXsec <: Newtrinos.Physics
    params::NamedTuple
    priors::NamedTuple
    diff_xsec::Function
end

const gf=1.1663787e-11
const me = 0.510998
const mmu = 105.6
const mtau=1776.86
const mpi = 139.57
const alph = 1/137
const ep= (mpi^2-mmu^2)/(2*mpi)

"""
    configure(isotopes, er_centers, enu_centers) -> CevnsXsec

Create a [`CevnsXsec`](@ref) module from a list of isotope descriptors and energy grids.

Builds cross-section assets and assembles per-isotope parameters and priors via
[`build_params_and_priors`](@ref) and [`get_assets`](@ref).

# Arguments
- `isotopes`: collection of isotope descriptors. Each entry must have the fields:
  - `Rn_key::Symbol`: parameter name for this isotope's nuclear radius (e.g. `:Rn_Cs133`).
  - `Rn_nom::Float64`: nominal nuclear radius ``R_n`` [fm].
  - `mass::Float64`: nuclear mass [MeV/c²].
  - `Z::Int`: proton number.
  - `N::Int`: neutron number.
  - `fraction::Float64`: isotopic abundance fraction.
- `er_centers`: nuclear recoil energy grid [MeV].
- `enu_centers`: neutrino energy grid [MeV].

# Returns
A [`CevnsXsec`](@ref) instance.
"""
function configure(isotopes, er_centers, enu_centers)
    # Build assets from isotopes
    assets = get_assets(isotopes, er_centers, enu_centers)
    params, priors = build_params_and_priors(isotopes)
    CevnsXsec(
        params = params,
        priors = priors,
        diff_xsec = get_diff_xsec(assets),
    )
end

"""
    configure(params::NamedTuple, priors::NamedTuple) -> CevnsXsec

Create a [`CevnsXsec`](@ref) from externally supplied parameter and prior NamedTuples.

Use this constructor when the parameter set has already been assembled (e.g. by a previous
call to [`build_params_and_priors`](@ref)) and no isotope list is available.

# Arguments
- `params::NamedTuple`: parameter values (must include `sin2thetaW`, `cevns_xsec_a`–`d`,
  and one `Rn_*` entry per isotope).
- `priors::NamedTuple`: corresponding prior distributions.

# Returns
A [`CevnsXsec`](@ref) instance.
"""
function configure(params::NamedTuple, priors::NamedTuple)
    CevnsXsec(
        params = params,
        priors = priors,
        diff_xsec_lar = get_diff_xsec_lar(),
        diff_xsec_csi = get_diff_xsec_csi(),
    )
end

"""
    build_params_and_priors(isotopes) -> (NamedTuple, NamedTuple)

Assemble default parameter values and prior distributions for a given set of isotopes.

Includes the weak mixing angle ``\\sin^2\\theta_W`` and four BSM deformation parameters
`cevns_xsec_a` through `cevns_xsec_d` that modify the SM differential cross section.

Per-isotope nuclear radius parameters `Rn_*` are appended automatically using the
`Rn_key` field of each isotope descriptor.

# Arguments
- `isotopes`: collection of isotope descriptors; each must have `Rn_key::Symbol` and
  `Rn_nom::Float64`.

# Returns
A 2-tuple `(params::NamedTuple, priors::NamedTuple)`.
"""
function build_params_and_priors(isotopes)
    # Dynamic parameter/prior builder for isotope-specific Rn keys, using isotope list
    param_dict = Dict(
        :cevns_xsec_a => 0.0,
        :cevns_xsec_b => 0.0,
        :cevns_xsec_c => 0.0,
        :cevns_xsec_d => 0.0,
        :sin2thetaW => 0.231,
    )
    prior_dict = Dict{Symbol, Distributions.Distribution}(
        :cevns_xsec_a => Uniform(-2, 2),
        :cevns_xsec_b => Uniform(-2000.0, 2000.0),
        :cevns_xsec_c => Uniform(-3, 3),
        :cevns_xsec_d => Uniform(-1e6, 1e6),
        :sin2thetaW => truncated(Normal(0.231, 0.00013), 0.2, 0.26),
    )
    for iso in isotopes
        param_dict[iso.Rn_key] = iso.Rn_nom
        prior_dict[iso.Rn_key] = Uniform(0.0, iso.Rn_nom + 2 * 1)
    end
    return ((; param_dict...), (; prior_dict...))
end

"""
    get_assets(isotopes, er_centers, enu_centers) -> NamedTuple

Package isotope data and energy grids into an assets NamedTuple for use by
[`get_diff_xsec`](@ref).

# Arguments
- `isotopes`: collection of isotope descriptors (same format as [`configure`](@ref)).
- `er_centers`: nuclear recoil energy grid [MeV].
- `enu_centers`: neutrino energy grid [MeV].

# Returns
A `NamedTuple` with fields:
- `isotopes`: `Dict{Symbol, NamedTuple}` mapping each `Rn_key` to `(mass, Z, N, fraction, Rn_nom)`.
- `er_centers`: the recoil energy grid.
- `enu_centers`: the neutrino energy grid.
"""
function get_assets(isotopes, er_centers, enu_centers)
    # Extract isotope data into a structured format
    @info "Configuring CEvNS cross-section assets"
    isotope_data = Dict(iso.Rn_key => (
        mass = iso.mass,
        Z = iso.Z,
        N = iso.N,
        fraction = iso.fraction,
        Rn_nom = iso.Rn_nom
    ) for iso in isotopes)

    # Return assets as a NamedTuple
    return (
        isotopes = isotope_data,  # Dictionary of isotope data keyed by Rn_key
        er_centers = er_centers,
        enu_centers = enu_centers
    )
end

"""
    ffsq(er, mn, rn) -> Real

Squared Helm nuclear form factor ``F^2(q^2)``, evaluated at recoil energy `er`.

The momentum transfer is ``q = \\sqrt{2 M_N E_r}`` and the form factor is

```math
F(q) = \\frac{3\\,j_1(q R_0)}{q R_0}\\,e^{-q^2 s^2/2}
```

where ``j_1`` is the spherical Bessel function of the first kind,
``R_0 = R_n / (\\hbar c)`` with ``\\hbar c = 197.3\\,\\text{MeV\\,fm}``,
and ``s = 0.9\\,\\text{fm}`` is the nuclear skin thickness.

# Arguments
- `er`: nuclear recoil energy [MeV].
- `mn`: nuclear mass ``M_N`` [MeV/c²].
- `rn`: nuclear radius ``R_n`` [fm].

# Returns
The dimensionless form factor squared ``F^2(q^2)``.
"""
function ffsq(er, mn, rn)
    # Helm-like nuclear form factor squared, generic in type
    r0 = rn / 197.326963
    arg = 2 * mn * er
    q = sqrt(max(arg, zero(arg)))                 # typed zero
    j1 = sphericalbesselj(1, q * r0)
    denom = q * r0
    # Use short-circuiting branch to avoid evaluating the division when denom == 0
    ratio = iszero(denom) ? one(j1) : (3 * j1) / denom
    exp_factor = exp(-((q * (0.9 / 197.326963))^2) / 2)
    return (ratio * exp_factor)^2
end

"""
    ds(er, enu, params, nupar, Rn_key) -> Matrix

Vectorized differential CEvNS cross section ``d\\sigma/dE_r`` on an
``(N_{E_r} \\times N_{E_\\nu})`` grid.

Implements the BSM-extended SM formula:

```math
\\frac{d\\sigma}{dE_r} = \\frac{G_F^2}{4\\pi}\\,M_N\\,F^2(q^2)\\,\\max\\!\\Bigl(0,\\;
    Q_W^2\\bigl[(1+a) + (b-1)\\tfrac{E_r}{E_\\nu} + (c-1)\\tfrac{M_N E_r}{2 E_\\nu^2}
    + (1+d)\\bigl(\\tfrac{E_r}{E_\\nu}\\bigr)^2\\bigr]\\Bigr)
```

where ``Q_W = N - (1 - 4\\sin^2\\theta_W)\\,Z``. At the SM limit (``a = b = c = d = 0``),
the expression reduces to the standard CEvNS formula of Freedman (1974).

# Arguments
- `er`: recoil energy grid [MeV], shape `(n_er,)`.
- `enu`: neutrino energy grid [MeV], shape `(n_enu,)`.
- `params::NamedTuple`: must include `sin2thetaW`, `cevns_xsec_a`–`d`, and `params[Rn_key]`.
- `nupar`: 3-tuple `(mass, Z, N)` — nuclear mass [MeV/c²], proton number, neutron number.
- `Rn_key::Symbol`: key used to look up the nuclear radius ``R_n`` in `params`.

# Returns
Matrix of shape `(n_er, n_enu)` with units [MeV⁻¹].
"""
function ds(er, enu, params, nupar, Rn_key)
    # Vectorized differential cross section dσ/dEr (n_er × n_enu), AD-safe
    mN = nupar[1]
    Z  = nupar[2]
    N  = nupar[3]
    rn = params[Rn_key]
    sw2 = params.sin2thetaW
    # Per-Er prefactor (n_er,)
    C1d = (gf^2 / (4 * pi)) .* ffsq.(er, mN, rn)

    qwsq = (N - (1 - 4 * sw2) * Z)^2

    # SM deformation parameters as scalars; no tiny length-4 arrays needed
    c1 = qwsq * (params.cevns_xsec_a + one(params.cevns_xsec_a))
    c2 = qwsq * (params.cevns_xsec_b - one(params.cevns_xsec_b))
    c3 = qwsq * (params.cevns_xsec_c - one(params.cevns_xsec_c))
    c4 = qwsq * (params.cevns_xsec_d + one(params.cevns_xsec_d))

    # Grids
    er_grid  = reshape(er, :, 1)          # (n_er, 1)
    enu_grid = reshape(enu, 1, :)         # (1, n_enu)

    base2 = er_grid ./ enu_grid
    base3 = mN .* er_grid ./ (2 .* enu_grid.^2)
    base4 = (er_grid.^2) ./ (enu_grid.^2)

    xf = c1 .+ c2 .* base2 .+ c3 .* base3 .+ c4 .* base4
    zxf = zero(eltype(xf))
    heav = max.(xf, zxf)                  # type-stable clamp

    # Broadcast C1d over columns, keep element type generic
    res = (C1d .* mN) .* heav
    return res
end

"""
    get_diff_xsec_lar() -> Function

Return a closure that calls [`ds`](@ref) for the liquid-argon (LAr) target.

The returned function has signature:
```julia
diff_xsec_lar(er_centers, enu_centers, params, nupar, Rn_key) -> Matrix
```
"""
function get_diff_xsec_lar()
    function diff_xsec_lar(er_centers, enu_centers, params, nupar, Rn_key)
        ds(er_centers, enu_centers, params, nupar, Rn_key)
    end
end

"""
    get_diff_xsec_csi() -> Function

Return a closure that calls [`ds`](@ref) for the CsI target.

The returned function has signature:
```julia
diff_xsec_csi(er_centers, enu_centers, params, nupar, Rn_key) -> Matrix
```
"""
function get_diff_xsec_csi()
    function diff_xsec_csi(er_centers, enu_centers, params, nupar, Rn_key)
        ds(er_centers, enu_centers, params, nupar, Rn_key)
    end
end

"""
    get_diff_xsec(assets) -> Function

Build the differential cross-section closure over all isotopes from pre-computed assets.

The energy grids and isotope data are captured from `assets` at construction time.
The returned function has signature:
```julia
diff_xsec(params::NamedTuple) -> Dict{Symbol, Matrix}
```
It iterates over all isotopes, calling [`ds`](@ref) for each, and returns a `Dict`
mapping each `Rn_key` to its ``(n_{E_r} \\times n_{E_\\nu})`` cross-section matrix.

# Arguments
- `assets`: NamedTuple produced by [`get_assets`](@ref).

# Returns
A callable closure `params -> Dict{Symbol, Matrix}`.
"""
function get_diff_xsec(assets)
    # Extract assets
    er_centers = assets.er_centers
    enu_centers = assets.enu_centers
    isotopes = assets.isotopes

    # Return a callable function that computes the differential cross-section
    return function (params)
        # Determine the element type of params (e.g., from the first element)
        param_type = eltype(params[:cevns_xsec_a])

        # Compute cross-section for each isotope and store in a dictionary
        xsec_dict = Dict{Symbol, Matrix{param_type}}()
        for (Rn_key, iso) in isotopes
            mass = iso.mass
            Z = iso.Z
            N = iso.N

            # Call ds() for the current isotope
            xsec_dict[Rn_key] = ds(er_centers, enu_centers, params, (mass, Z, N), Rn_key)
        end

        return xsec_dict  # Dictionary of cross-section matrices keyed by Rn_key
    end
end
end # module cevns_xsec
