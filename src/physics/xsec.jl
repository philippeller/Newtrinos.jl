module xsec

using LinearAlgebra
using Distributions
using CSV, DataFrames
using Interpolations
using FunctionChains
using ..Newtrinos

"""
    XsecModel

Abstract type for neutrino cross-section models.

Each subtype defines how cross-section normalization factors are applied to event rates.
Subtypes:
- [`SimpleScaling`](@ref): global normalization for NC and ``\\nu_\\tau`` CC channels.
- [`Differential_H2O`](@ref): per-interaction-mode differential scaling for water
  Cherenkov detectors.
"""
abstract type XsecModel end

"""
    SimpleScaling <: XsecModel

Simple cross-section model applying global normalization factors.

Provides two nuisance parameters:
- `nc_norm`: neutral-current cross-section scale factor.
- `nutau_cc_norm`: ``\\nu_\\tau`` charged-current cross-section scale factor.

All other flavour/interaction combinations return unit scaling.
"""
struct SimpleScaling <: XsecModel end

"""
    Differential_H2O <: XsecModel

Differential cross-section model for water (H₂O) targets.

Provides per-interaction-mode normalization parameters (`cc1p1h_norm`, `cc2p2h_norm`,
`cc1pi_norm`, `ccother_norm`, `ccdis_norm`) in addition to `nc_norm` and `nutau_cc_norm`.
Energy-dependent CC interaction fractions are interpolated from digitized data
(T. Wester, Super-K PhD thesis, Figure 4.7) for ``\\nu_e`` and ``\\bar{\\nu}_e``
separately.
"""
struct Differential_H2O <: XsecModel end

"""
    Xsec <: Newtrinos.Physics

Configured cross-section physics module, returned by [`configure`](@ref).

# Fields
- `cfg::XsecModel`: the cross-section model used to build this module.
- `params::NamedTuple`: default normalization parameter values.
- `priors::NamedTuple`: prior distributions for each parameter.
- `scale::Function`: closure that computes the cross-section scale factor.
  See [`get_scale`](@ref) for the call signatures.
"""
@kwdef struct Xsec <: Newtrinos.Physics
    cfg::XsecModel
    params::NamedTuple
    priors::NamedTuple
    scale::Function
end

"""
    configure(cfg::XsecModel=SimpleScaling()) -> Xsec

Create a fully configured cross-section physics module.

Assembles default parameters, priors, and the scaling closure into an [`Xsec`](@ref)
struct ready for use in experiment forward models.

# Arguments
- `cfg::XsecModel`: cross-section model (defaults to [`SimpleScaling`](@ref)).

# Returns
An [`Xsec`](@ref) instance.

# Examples
```julia
using Newtrinos

# Default simple scaling
xsec_physics = Newtrinos.xsec.configure()

# Differential water cross-sections for Super-K
xsec_physics = Newtrinos.xsec.configure(Differential_H2O())
```
"""
function configure(cfg::XsecModel=SimpleScaling())
    Xsec(
        cfg=cfg,
        params = get_params(cfg),
        priors = get_priors(cfg),
        scale = get_scale(cfg)
        )
end

"""
    get_params(cfg::XsecModel) -> NamedTuple

Return the default cross-section normalization parameter values for the given model.

# Arguments
- `cfg::XsecModel`: a [`SimpleScaling`](@ref) or [`Differential_H2O`](@ref) instance.

# Returns
A `NamedTuple` mapping parameter names to their nominal values (all default to `1.0`).
"""
function get_params(cfg::SimpleScaling)
    (
        nc_norm = 1.,
        nutau_cc_norm = 1.,
    )
end

"""
    get_priors(cfg::XsecModel) -> NamedTuple

Return prior distributions for each cross-section normalization parameter.

All priors are truncated normal distributions centred at 1.0.

# Arguments
- `cfg::XsecModel`: a [`SimpleScaling`](@ref) or [`Differential_H2O`](@ref) instance.

# Returns
A `NamedTuple` mapping parameter names to `Distributions.Truncated{Normal}` priors.
"""
function get_priors(cfg::SimpleScaling)
    (
        nc_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        nutau_cc_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
    )
end

function get_params(cfg::Differential_H2O)
    (
        nc_norm = 1.,
        nutau_cc_norm = 1.,
        cc1p1h_norm = 1.,
        cc2p2h_norm = 1.,
        cc1pi_norm = 1.,
        ccother_norm = 1.,
        ccdis_norm = 1.,
    )
end

function get_priors(cfg::Differential_H2O)
    (
        nc_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        nutau_cc_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        cc1p1h_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        cc2p2h_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        cc1pi_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        ccother_norm = Truncated(Normal(1, 0.4), 0.2, 1.8),
        ccdis_norm = Truncated(Normal(1, 0.1), 0.7, 1.3),
    )
end

"""
    get_scale(cfg::XsecModel) -> Function

Construct the cross-section scaling closure for the given model.

**[`SimpleScaling`](@ref)** returns a function with signature:
```julia
scale(flav::Symbol, interaction::Symbol, params::NamedTuple) -> Real
```
Returns `params.nc_norm` for NC interactions, `params.nutau_cc_norm` for ``\\nu_\\tau`` CC,
and `1.0` for everything else.

**[`Differential_H2O`](@ref)** returns a function with signature:
```julia
scale(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple) -> Real or AbstractArray
```
For NC interactions returns `params.nc_norm`. For CC interactions computes an
energy-dependent weighted sum of per-mode normalizations using interpolated
interaction fractions, with an additional ``\\nu_\\tau`` CC factor when applicable.

# Arguments
- `cfg::XsecModel`: cross-section model instance.

# Returns
A closure computing the cross-section scale factor.
"""
function get_scale(cfg::SimpleScaling)
    function scale(flav::Symbol, interaction::Symbol, params::NamedTuple)
        if interaction == :NC
            return params.nc_norm
        elseif flav == :nutau
            return params.nutau_cc_norm
        else
            return one(params.nc_norm)
        end
    end
end


function get_scale(cfg::Differential_H2O)

    # digitized from T. Wester Super-K PhD thesis Figure 4.7
    df_nue = CSV.read(joinpath(@__DIR__, "xsec_nue_water.csv"), DataFrame, skipto=3);
    df_nuebar = CSV.read(joinpath(@__DIR__, "xsec_nuebar_water.csv"), DataFrame, skipto=3);

    function make_interpolation(name, df)
        idx = findfirst(==(name), names(df))
        x = collect(skipmissing(df[:,idx]))
        y = collect(skipmissing(df[:,idx+1]))
        itp = interpolate((x,), y, Gridded(Linear()))
        m(x) = max.(0, x)
        return m ∘ extrapolate(itp, Linear())
    end

    nue = (
        CC1p1h = make_interpolation("CC1p1h", df_nue),
        CC2p2h = make_interpolation("CC2p2h", df_nue),
        CC1pi = make_interpolation("CC1pi", df_nue),
        CCother = make_interpolation("CCother", df_nue),
        CCDIS = make_interpolation("CCDIS", df_nue),
        NC = make_interpolation("NC", df_nue),
    )

    nuebar = (
        CC1p1h = make_interpolation("CC1p1h", df_nuebar),
        CC2p2h = make_interpolation("CC2p2h", df_nuebar),
        CC1pi = make_interpolation("CC1pi", df_nuebar),
        CCother = make_interpolation("CCother", df_nuebar),
        CCDIS = make_interpolation("CCDIS", df_nuebar),
        NC = make_interpolation("NC", df_nuebar),
    )

    function ratios(funs, E)
        x = NamedTuple(key=>funs[key].(E) for key in keys(funs) if key != :NC)
        total_CC = sum(x)
        return NamedTuple(key=>x[key]./total_CC for key in keys(x))
    end

    function scale(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)

        if interaction == :NC
            return params.nc_norm
        else
            if anti
                rs = ratios(nuebar, E)
            else
                rs = ratios(nue, E)
            end

            s = rs.CC1p1h * params.cc1p1h_norm .+ rs.CC2p2h * params.cc2p2h_norm .+ rs.CC1pi * params.cc1pi_norm .+ rs.CCother * params.ccother_norm .+ rs.CCDIS * params.ccdis_norm 

            if flav == :nutau
                return s * params.nutau_cc_norm
            else
                return s
            end
        end
    end
end

end