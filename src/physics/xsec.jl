module xsec

using LinearAlgebra
using Distributions
using CSV, DataFrames
using Interpolations
using FunctionChains
using ..Newtrinos

abstract type XsecModel end

struct SimpleScaling <: XsecModel end

struct Differential_H2O <: XsecModel end

@kwdef struct Xsec <: Newtrinos.Physics
    cfg::XsecModel
    params::NamedTuple
    priors::NamedTuple
    scale::Function
end


function configure(cfg::XsecModel=SimpleScaling())
    Xsec(
        cfg=cfg,
        params = get_params(cfg),
        priors = get_priors(cfg),
        scale = get_scale(cfg)
        )
end

function get_params(cfg::SimpleScaling)
    (
        nc_norm = 1.,
        nutau_cc_norm = 1.,
    )
end

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
        xsec_MA_QE = 1.05,
        xsec_MA_Res = 0.95,
        xsec_I12 = 1.30,
        xsec_fsi = 1.,
    )
end

function get_priors(cfg::Differential_H2O)
    (
        nc_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        nutau_cc_norm = Truncated(Normal(1, 0.25), 0.3, 1.7),
        cc1p1h_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        cc2p2h_norm = Truncated(Normal(1, 1.0), 0, 3),
        cc1pi_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        ccother_norm = Truncated(Normal(1, 0.4), 0.2, 1.8),
        ccdis_norm = Truncated(Normal(1, 0.1), 0.7, 1.3),
        xsec_MA_QE = Normal(1.05, 0.16),
        xsec_MA_Res = Normal(0.95, 0.15),
        xsec_I12 = Normal(1.30, 0.20),
        xsec_fsi = Normal(1, 0.1),
    )
end

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


function ma_qe_ratio(E, MA)
    # Approximate CCQE σ ratio from dipole form factor
    MA_nom = 1.05
    Q2_typ = 0.5 * E
    r_nom = 1 / (1 + Q2_typ / MA_nom^2)^2
    r_new = 1 / (1 + Q2_typ / MA^2)^2
    return r_new / r_nom
end

function ma_res_ratio(E, MA)
    # Approximate resonance σ ratio from dipole form factor
    MA_nom = 0.95
    Q2_typ = 0.3 * E
    r_nom = 1 / (1 + Q2_typ / MA_nom^2)^2
    r_new = 1 / (1 + Q2_typ / MA^2)^2
    return r_new / r_nom
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
        cc_funs = (CC1p1h=funs.CC1p1h, CC2p2h=funs.CC2p2h, CC1pi=funs.CC1pi, CCother=funs.CCother, CCDIS=funs.CCDIS)
        x = map(f -> f.(E), cc_funs)
        total_CC = reduce(+, values(x))
        return map(v -> v ./ total_CC, x)
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

            ma_qe = ma_qe_ratio.(E, params.xsec_MA_QE)
            ma_res = ma_res_ratio.(E, params.xsec_MA_Res)
            fsi_1p1h = 1 .- 0.1 .* (params.xsec_fsi - 1)
            fsi_1pi = 1 .+ 0.1 .* (params.xsec_fsi - 1)

            s = rs.CC1p1h .* params.cc1p1h_norm .* ma_qe .* fsi_1p1h .+ rs.CC2p2h * params.cc2p2h_norm .+ rs.CC1pi .* params.cc1pi_norm .* ma_res .* fsi_1pi .+ rs.CCother * params.ccother_norm * params.xsec_I12 .+ rs.CCDIS * params.ccdis_norm

            if flav == :nutau
                return s * params.nutau_cc_norm
            else
                return s
            end
        end
    end
end

end