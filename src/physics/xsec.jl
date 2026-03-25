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

@kwdef struct H2O_PCA <: XsecModel
    nominal::Symbol = :Wester   # which curve is the MC nominal (:Wester, :G18_10a, :G21_11a, :G18_02a)
end

const GENIE_H2O = H2O_PCA  # backward compat alias

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

function get_params(cfg::H2O_PCA)
    (
        xsec_cc1p1h_norm = 1.,
        xsec_cc2p2h_norm = 1.,
        xsec_cc1pi_norm = 1.,
        xsec_ccdis_norm = 1.,
        xsec_ccother_norm = 1.,
        xsec_nc_norm = 1.,
        xsec_nutau_cc_norm = 1.,
        xsec_cc1p1h_shape = 0.,
        xsec_cc2p2h_shape = 0.,
        xsec_cc1pi_shape = 0.,
        xsec_ccdis_shape = 0.,
        xsec_ccother_shape = 0.,
        xsec_nc_shape = 0.,
    )
end

function get_priors(cfg::H2O_PCA)
    (
        xsec_cc1p1h_norm = Truncated(Normal(1, 0.15), 0.4, 1.6),
        xsec_cc2p2h_norm = Truncated(Normal(1, 0.50), 0.0, 3.0),
        xsec_cc1pi_norm = Truncated(Normal(1, 0.15), 0.4, 1.6),
        xsec_ccdis_norm = Truncated(Normal(1, 0.10), 0.5, 1.5),
        xsec_ccother_norm = Truncated(Normal(1, 0.30), 0.2, 1.8),
        xsec_nc_norm = Truncated(Normal(1, 0.15), 0.4, 1.6),
        xsec_nutau_cc_norm = Truncated(Normal(1, 0.25), 0.3, 1.7),
        xsec_cc1p1h_shape = Normal(0, 1),
        xsec_cc2p2h_shape = Normal(0, 1),
        xsec_cc1pi_shape = Normal(0, 1),
        xsec_ccdis_shape = Normal(0, 1),
        xsec_ccother_shape = Normal(0, 1),
        xsec_nc_shape = Normal(0, 1),
    )
end

function get_scale(cfg::H2O_PCA)
    data = load(joinpath(@__DIR__, "xsec_genie_data.jld2"))
    E_grid = data["E_grid"]
    wester_xsec = data["wester_xsec"]
    all_xsec = data["all_xsec"]

    nominal_key = string(cfg.nominal)
    genie_tunes = ["G18_10a", "G21_11a", "G18_02a"]
    cc_channels = ("CC1p1h", "CC2p2h", "CC1pi", "CCDIS", "CCother")
    all_channels = (cc_channels..., "NC")

    function make_itp(vals)
        itp = interpolate((E_grid,), vals, Gridded(Linear()))
        return extrapolate(itp, Flat())
    end

    # Get nominal σ/E per channel per flavor
    function get_nominal(flav, ch)
        if nominal_key == "Wester"
            return wester_xsec[flav][ch]
        else
            return all_xsec[nominal_key][flav][ch]
        end
    end

    # Precompute nominal channel fractions per flavor (for CC reweight)
    flav_keys = ("nue", "nuebar", "numu", "numubar")
    nom_frac_itps = Dict{String, NamedTuple}()
    for fk in flav_keys
        total_cc = sum(get_nominal(fk, ch) for ch in cc_channels)
        fracs = NamedTuple{Symbol.(cc_channels)}(begin
            f = get_nominal(fk, ch) ./ total_cc
            f[isnan.(f) .| isinf.(f)] .= 0.0
            make_itp(f)
        end for ch in cc_channels)
        nom_frac_itps[fk] = fracs
    end

    # Per interaction process: compute shape PCA from tune spread
    # Combine nue + nuebar (ν + ν̄) for robust shape estimate
    # The shape PC is a fractional deviation, applied identically to all flavors

    alt_keys = [t for t in genie_tunes if t != nominal_key]
    if nominal_key != "Wester"
        push!(alt_keys, "Wester")
    end
    n_alt = length(alt_keys)

    function get_channel_curve(source, ch)
        if source == "Wester"
            # Average over nue and nuebar for combined ν+ν̄ shape
            return (wester_xsec["nue"][ch] .+ wester_xsec["nuebar"][ch]) ./ 2
        else
            return (all_xsec[source]["nue"][ch] .+ all_xsec[source]["nuebar"][ch]) ./ 2
        end
    end

    process_shape_itps = Dict{String, Any}()
    process_norm_sigmas = Dict{String, Float64}()

    for ch in all_channels
        nom = get_channel_curve(nominal_key == "Wester" ? "Wester" : nominal_key, ch)
        n_E = length(E_grid)
        nom_mean = sum(nom) / n_E

        # Collect all curve means for norm spread
        all_means = Float64[nom_mean]
        for k in alt_keys
            push!(all_means, sum(get_channel_curve(k, ch)) / n_E)
        end
        norm_sigma = std(all_means ./ nom_mean)
        process_norm_sigmas[ch] = norm_sigma

        # Shape PCA: normalize alts to same overall norm as nominal, fractional deviation
        Delta = zeros(n_E, n_alt)
        for (i, k) in enumerate(alt_keys)
            alt = get_channel_curve(k, ch)
            alt_mean = sum(alt) / n_E
            alt_rescaled = alt .* (nom_mean / alt_mean)
            frac_dev = (alt_rescaled .- nom) ./ nom
            frac_dev[isnan.(frac_dev) .| isinf.(frac_dev)] .= 0.0
            Delta[:, i] = frac_dev
        end

        U, S, _ = svd(Delta)
        process_shape_itps[ch] = make_itp(U[:, 1] .* S[1])
    end

    # NC: ν and ν̄ have different shapes — compute separate NC shape PCs
    for (label, nu_flav, nubar_flav) in [("NC_nu", "nue", "nue"), ("NC_nubar", "nuebar", "nuebar")]
        nom = nominal_key == "Wester" ? wester_xsec[nu_flav]["NC"] : all_xsec[nominal_key][nu_flav]["NC"]
        n_E = length(E_grid)
        nom_mean = sum(nom) / n_E

        Delta = zeros(n_E, n_alt)
        for (i, k) in enumerate(alt_keys)
            alt = k == "Wester" ? wester_xsec[nu_flav]["NC"] : all_xsec[k][nu_flav]["NC"]
            alt_mean = sum(alt) / n_E
            alt_rescaled = alt .* (nom_mean / alt_mean)
            frac_dev = (alt_rescaled .- nom) ./ nom
            frac_dev[isnan.(frac_dev) .| isinf.(frac_dev)] .= 0.0
            Delta[:, i] = frac_dev
        end

        U, S, _ = svd(Delta)
        process_shape_itps[label] = make_itp(U[:, 1] .* S[1])
    end

    @info "H2O_PCA xsec configured" nominal=cfg.nominal norm_sigmas=process_norm_sigmas

    # Parameter → channel mapping
    norm_syms = (
        CC1p1h = :xsec_cc1p1h_norm,
        CC2p2h = :xsec_cc2p2h_norm,
        CC1pi  = :xsec_cc1pi_norm,
        CCDIS  = :xsec_ccdis_norm,
        CCother = :xsec_ccother_norm,
    )
    shape_syms = (
        CC1p1h = :xsec_cc1p1h_shape,
        CC2p2h = :xsec_cc2p2h_shape,
        CC1pi  = :xsec_cc1pi_shape,
        CCDIS  = :xsec_ccdis_shape,
        CCother = :xsec_ccother_shape,
    )

    function get_flavor_key(flav::Symbol, anti::Bool)
        if flav == :nue
            return anti ? "nuebar" : "nue"
        elseif flav == :numu || flav == :nutau
            return anti ? "numubar" : "numu"
        else
            return anti ? "nuebar" : "nue"
        end
    end

    function scale(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        T = promote_type(eltype(E), typeof(params.xsec_nc_norm))

        if interaction == :NC
            # NC reweight: norm × (1 + ε × shape_PC(E)), clamped ≥ 0
            # shape_PC differs for ν vs ν̄
            nc_shape = anti ? process_shape_itps["NC_nubar"] : process_shape_itps["NC_nu"]
            return max.(zero(T), params.xsec_nc_norm .* (one(T) .+ params.xsec_nc_shape .* nc_shape.(E)))
        end

        fk = get_flavor_key(flav, anti)
        fracs = nom_frac_itps[fk]

        # CC reweight = Σ_ch f_ch(E) × ch_norm × (1 + ε_ch × shape_PC_ch(E))
        # Each channel contribution clamped ≥ 0 before summing
        result = zeros(T, length(E))
        for ch in cc_channels
            f_ch = getfield(fracs, Symbol(ch)).(E)
            ch_norm = getfield(params, getfield(norm_syms, Symbol(ch)))
            ch_eps = getfield(params, getfield(shape_syms, Symbol(ch)))
            ch_shape = process_shape_itps[ch]
            result .+= max.(zero(T), f_ch .* ch_norm .* (one(T) .+ ch_eps .* ch_shape.(E)))
        end

        if flav == :nutau
            return result .* params.xsec_nutau_cc_norm
        end
        return result
    end
end

end