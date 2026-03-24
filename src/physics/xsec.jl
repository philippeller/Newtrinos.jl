module xsec

using LinearAlgebra
using Distributions
using CSV, DataFrames
using Interpolations
using FunctionChains
using JLD2
using ..Newtrinos

abstract type XsecModel end

struct SimpleScaling <: XsecModel end

struct Differential_H2O <: XsecModel end

struct GENIE_H2O <: XsecModel end

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
        cc_norm = 1.,
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
        cc_norm = Truncated(Normal(1, 0.15), 0.5, 1.5),
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

            s = (rs.CC1p1h .* params.cc1p1h_norm .* ma_qe .* fsi_1p1h .+ rs.CC2p2h * params.cc2p2h_norm .+ rs.CC1pi .* params.cc1pi_norm .* ma_res .* fsi_1pi .+ rs.CCother * params.ccother_norm * params.xsec_I12 .+ rs.CCDIS * params.ccdis_norm) .* params.cc_norm

            if flav == :nutau
                return s * params.nutau_cc_norm
            else
                return s
            end
        end
    end
end

function get_params(cfg::GENIE_H2O)
    (
        nc_norm = 1.,
        nutau_cc_norm = 1.,
        cc_norm = 1.,
        xsec_pc1 = 0.,
        xsec_pc2 = 0.,
    )
end

function get_priors(cfg::GENIE_H2O)
    (
        nc_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        nutau_cc_norm = Truncated(Normal(1, 0.25), 0.3, 1.7),
        cc_norm = Truncated(Normal(1, 0.15), 0.5, 1.5),
        xsec_pc1 = Normal(0, 1),
        xsec_pc2 = Normal(0, 1),
    )
end

function get_scale(cfg::GENIE_H2O)
    # Load precomputed GENIE data: nominal channel fractions + PCA components
    data = load(joinpath(@__DIR__, "xsec_genie_data.jld2"))
    E_grid = data["E_grid"]
    nominal_fractions = data["nominal_fractions"]
    pca_components = data["pca_components"]

    cc_channels = ("CC1p1h", "CC2p2h", "CC1pi", "CCDIS", "CCother")

    # Build interpolation functions for nominal fractions and PCA components
    function make_itp(vals)
        itp = interpolate((E_grid,), vals, Gridded(Linear()))
        return extrapolate(itp, Flat())
    end

    # Precompute interpolations for each flavor × channel
    flav_keys = ("nue", "nuebar", "numu", "numubar")
    nom_itps = Dict(
        fk => NamedTuple{Symbol.(cc_channels)}(
            make_itp(nominal_fractions[fk][ch]) for ch in cc_channels
        ) for fk in flav_keys
    )
    pc_itps = [
        Dict(
            fk => NamedTuple{Symbol.(cc_channels)}(
                make_itp(pca_components[k][fk][ch]) for ch in cc_channels
            ) for fk in flav_keys
        ) for k in 1:length(pca_components)
    ]

    function get_flavor_key(flav::Symbol, anti::Bool)
        if flav == :nue
            return anti ? "nuebar" : "nue"
        elseif flav == :numu || flav == :nutau
            # nutau uses numu/numubar fractions (CC threshold ~3.5 GeV, fractions converge)
            return anti ? "numubar" : "numu"
        else
            return anti ? "nuebar" : "nue"
        end
    end

    function scale(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        if interaction == :NC
            return params.nc_norm
        end

        fk = get_flavor_key(flav, anti)
        nom = nom_itps[fk]

        # Evaluate nominal fractions at energies E
        T = promote_type(eltype(E), typeof(params.cc_norm))
        f_CC1p1h = nom.CC1p1h.(E)
        f_CC2p2h = nom.CC2p2h.(E)
        f_CC1pi  = nom.CC1pi.(E)
        f_CCDIS  = nom.CCDIS.(E)
        f_CCother = nom.CCother.(E)

        # Apply PCA perturbations
        pc_weights = (params.xsec_pc1, params.xsec_pc2)
        for (k, w) in enumerate(pc_weights)
            pc = pc_itps[k][fk]
            f_CC1p1h = f_CC1p1h .+ w .* pc.CC1p1h.(E)
            f_CC2p2h = f_CC2p2h .+ w .* pc.CC2p2h.(E)
            f_CC1pi  = f_CC1pi  .+ w .* pc.CC1pi.(E)
            f_CCDIS  = f_CCDIS  .+ w .* pc.CCDIS.(E)
            f_CCother = f_CCother .+ w .* pc.CCother.(E)
        end

        # Clamp to non-negative (prevents unphysical fractions from large PCA perturbations)
        f_CC1p1h = max.(zero(T), f_CC1p1h)
        f_CC2p2h = max.(zero(T), f_CC2p2h)
        f_CC1pi  = max.(zero(T), f_CC1pi)
        f_CCDIS  = max.(zero(T), f_CCDIS)
        f_CCother = max.(zero(T), f_CCother)

        # Weighted sum (fractions sum to ~1 at nominal, PCA preserves sum)
        s = (f_CC1p1h .+ f_CC2p2h .+ f_CC1pi .+ f_CCDIS .+ f_CCother) .* params.cc_norm

        if flav == :nutau
            return s .* params.nutau_cc_norm
        else
            return s
        end
    end
end

end