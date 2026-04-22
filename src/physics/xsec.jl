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
    nominal::Symbol = :NEUT5_4_0   # reference curve (norm=1): :NEUT5_4_0, :G18_10a, :G21_11a, :G18_02a
    mc_nominal::Symbol = :NEUT5_4_0 # cross-section model used to generate the MC being reweighted
end

const GENIE_H2O = H2O_PCA  # backward compat alias

@kwdef struct Xsec <: Newtrinos.Physics
    cfg::XsecModel
    params::NamedTuple
    priors::NamedTuple
    scale::Function
    dσdE::Union{Function, Nothing} = nothing
    scale_event::Union{Function, Nothing} = nothing  # per-event reweighting with GENIE interaction codes
    event_weights::Union{Function, Nothing} = nothing # fast path: precompute(E,codes,flav,ia,anti) -> (params)->weights
    grid_weights::Union{Function, Nothing} = nothing  # fast path: precompute(E_grid,flav,ia,anti) -> (params)->weights
end


function configure(cfg::XsecModel=SimpleScaling())
    Xsec(
        cfg=cfg,
        params = get_params(cfg),
        priors = get_priors(cfg),
        scale = get_scale(cfg),
        dσdE = get_dσdE(cfg),
        scale_event = get_scale_event(cfg),
        event_weights = get_event_weights(cfg),
        grid_weights = get_grid_weights(cfg)
        )
end

get_dσdE(::XsecModel) = nothing
get_scale_event(::XsecModel) = nothing
get_event_weights(::XsecModel) = nothing
get_grid_weights(::XsecModel) = nothing

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
        nubar_ratio = 1.,
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
        nubar_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
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

function _load_h2o_interpolations()
    # σ/E per channel on water, digitized from T. Wester Super-K PhD thesis Figure 4.7
    df_nue = CSV.read(joinpath(@__DIR__, "xsec_nue_water.csv"), DataFrame, skipto=3);
    df_nuebar = CSV.read(joinpath(@__DIR__, "xsec_nuebar_water.csv"), DataFrame, skipto=3);

    function make_interpolation(name, df)
        idx = findfirst(==(name), names(df))
        x = collect(skipmissing(df[:,idx]))
        y = collect(skipmissing(df[:,idx+1]))
        itp = interpolate((x,), y, Gridded(Linear()))
        m(x) = max.(0, x)
        return m ∘ extrapolate(itp, Flat())
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

    return nue, nuebar
end

"""Apply MA_QE, MA_Res, FSI, channel norms, cc_norm, nubar_ratio to per-channel σ/E values."""
function _apply_h2o_systematics(E, funs, flav, anti, params)
    ma_qe = ma_qe_ratio.(E, params.xsec_MA_QE)
    ma_res = ma_res_ratio.(E, params.xsec_MA_Res)
    fsi_1p1h = 1 .- 0.1 .* (params.xsec_fsi - 1)
    fsi_1pi = 1 .+ 0.1 .* (params.xsec_fsi - 1)

    s = (funs.CC1p1h.(E) .* params.cc1p1h_norm .* ma_qe .* fsi_1p1h .+
         funs.CC2p2h.(E) .* params.cc2p2h_norm .+
         funs.CC1pi.(E) .* params.cc1pi_norm .* ma_res .* fsi_1pi .+
         funs.CCother.(E) .* params.ccother_norm .* params.xsec_I12 .+
         funs.CCDIS.(E) .* params.ccdis_norm) .* params.cc_norm

    r = params.nubar_ratio
    s = anti ? s .* (2 * r / (1 + r)) : s .* (2 / (1 + r))

    # TODO: nutau CC cross-sections are not yet handled correctly — they use numu
    # curves as proxy, but nutau has a kinematic threshold at ~3.5 GeV (tau mass)
    # and different channel fractions at low energy.
    if flav == :nutau
        return s .* params.nutau_cc_norm
    end
    return s
end

function get_scale(cfg::Differential_H2O)
    nue, nuebar = _load_h2o_interpolations()

    function ratios(funs, E)
        cc_funs = (CC1p1h=funs.CC1p1h, CC2p2h=funs.CC2p2h, CC1pi=funs.CC1pi, CCother=funs.CCother, CCDIS=funs.CCDIS)
        x = map(f -> f.(E), cc_funs)
        total_CC = reduce(+, values(x))
        return map(v -> v ./ total_CC, x)
    end

    function scale(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        if interaction == :NC
            return params.nc_norm
        end

        funs = anti ? nuebar : nue
        rs = ratios(funs, E)

        ma_qe = ma_qe_ratio.(E, params.xsec_MA_QE)
        ma_res = ma_res_ratio.(E, params.xsec_MA_Res)
        fsi_1p1h = 1 .- 0.1 .* (params.xsec_fsi - 1)
        fsi_1pi = 1 .+ 0.1 .* (params.xsec_fsi - 1)

        s = (rs.CC1p1h .* params.cc1p1h_norm .* ma_qe .* fsi_1p1h .+ rs.CC2p2h * params.cc2p2h_norm .+ rs.CC1pi .* params.cc1pi_norm .* ma_res .* fsi_1pi .+ rs.CCother * params.ccother_norm * params.xsec_I12 .+ rs.CCDIS * params.ccdis_norm) .* params.cc_norm

        r = params.nubar_ratio
        s = anti ? s .* (2 * r / (1 + r)) : s .* (2 / (1 + r))

        if flav == :nutau
            return s * params.nutau_cc_norm
        end
        return s
    end
end

function get_dσdE(cfg::Differential_H2O)
    nue, nuebar = _load_h2o_interpolations()

    function dσdE(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        funs = anti ? nuebar : nue

        if interaction == :NC
            return funs.NC.(E) .* params.nc_norm
        end

        return _apply_h2o_systematics(E, funs, flav, anti, params)
    end
end

function get_params(cfg::H2O_PCA)
    (
        xsec_cc1p1h_subgev_norm = 1.,
        xsec_cc1p1h_multigev_norm = 1.,
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
        xsec_cc1p1h_nubar_ratio = 1.,
        xsec_cc2p2h_nubar_ratio = 1.,
        xsec_cc1pi_nubar_ratio = 1.,
        xsec_ccdis_nubar_ratio = 1.,
        xsec_ccother_nubar_ratio = 1.,
        xsec_nc_nubar_ratio = 1.,
        xsec_cc1p1h_nue_numu_ratio = 1.,
    )
end

function get_priors(cfg::H2O_PCA)
    (
        xsec_cc1p1h_subgev_norm = Truncated(Normal(1, 0.05), 0.7, 1.3),
        xsec_cc1p1h_multigev_norm = Truncated(Normal(1, 0.25), 0.3, 1.7),
        xsec_cc2p2h_norm = Uniform(0, 2),
        xsec_cc1pi_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        xsec_ccdis_norm = Truncated(Normal(1, 0.10), 0.5, 1.5),
        xsec_ccother_norm = Truncated(Normal(1, 0.30), 0.2, 1.8),
        xsec_nc_norm = Truncated(Normal(1, 0.20), 0.4, 1.6),
        xsec_nutau_cc_norm = Truncated(Normal(1, 0.25), 0.3, 1.7),
        xsec_cc1p1h_shape = Normal(0, 1),
        xsec_cc2p2h_shape = Normal(0, 1),
        xsec_cc1pi_shape = Normal(0, 1),
        xsec_ccdis_shape = Normal(0, 1),
        xsec_ccother_shape = Normal(0, 1),
        xsec_nc_shape = Normal(0, 1),
        # ν̄/ν ratio priors: σ from GENIE tune spread of per-process ν̄/ν xsec ratio
        xsec_cc1p1h_nubar_ratio = Truncated(Normal(1, 0.17), 0.3, 1.7),
        xsec_cc2p2h_nubar_ratio = Truncated(Normal(1, 0.17), 0.3, 1.7),
        xsec_cc1pi_nubar_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
        xsec_ccdis_nubar_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
        xsec_ccother_nubar_ratio = Truncated(Normal(1, 0.10), 0.5, 1.5),
        xsec_nc_nubar_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
        # νe/νμ CCQE ratio: nuclear model uncertainty (RFG vs LFG)
        xsec_cc1p1h_nue_numu_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
    )
end

const _genie_data_cache = Ref{Union{Nothing, Tuple}}(nothing)
const _genie_data_lock = ReentrantLock()

function _load_genie_data()
    return lock(_genie_data_lock) do
        if !isnothing(_genie_data_cache[])
            return _genie_data_cache[]
        end
    data = load(joinpath(@__DIR__, "xsec_genie_data.jld2"))
    E_grid = data["E_grid"]
    wester_xsec = data["wester_xsec"]  # NEUT5.4.0 cross-sections (key kept for backward compat)
    all_xsec = data["all_xsec"]

    # Trim last 2 E_grid points: GENIE has boundary artifacts there
    # (CCDIS drops ~3% in the final grid step vs ~0.1%/step trend)
    n_trim = findlast(E_grid .<= 950.0)
    E_grid = E_grid[1:n_trim]
    for (_, flav_dict) in wester_xsec
        for (ch, vals) in flav_dict
            flav_dict[ch] = vals[1:n_trim]
        end
    end
    for (_, tune_dict) in all_xsec
        for (_, flav_dict) in tune_dict
            for (ch, vals) in flav_dict
                flav_dict[ch] = vals[1:n_trim]
            end
        end
    end

        result = (E_grid, wester_xsec, all_xsec)
        _genie_data_cache[] = result
        result
    end
end

function get_scale(cfg::H2O_PCA)
    E_grid, wester_xsec, all_xsec = _load_genie_data()

    nominal_key = string(cfg.nominal)
    genie_tunes = ["G18_10a", "G21_11a", "G18_02a"]
    cc_channels = ("CC1p1h", "CC2p2h", "CC1pi", "CCDIS", "CCother")
    all_channels = (cc_channels..., "NC")

    function make_itp(vals)
        itp = interpolate((E_grid,), vals, Gridded(Linear()))
        return extrapolate(itp, Flat())
    end

    # Get nominal σ/E per channel per flavor
    # Wester CSV data only valid up to ~26 GeV. Above that, extend using GENIE
    # G18_10a shape scaled to match Wester at the boundary (no discontinuity).
    E_wester_max = 26.0
    i_wester_last = findlast(E_grid .<= E_wester_max)

    function extend_wester(wester_vals, genie_vals)
        # Above E_wester_max: use GENIE shape scaled to match Wester at boundary
        extended = copy(wester_vals)
        g_at_boundary = genie_vals[i_wester_last]
        w_at_boundary = wester_vals[i_wester_last]
        if g_at_boundary > 0 && w_at_boundary > 0
            scale = w_at_boundary / g_at_boundary
            extended[i_wester_last+1:end] .= genie_vals[i_wester_last+1:end] .* scale
        end
        return extended
    end

    function get_nominal(flav, ch)
        if nominal_key == "NEUT5_4_0"
            return extend_wester(wester_xsec[flav][ch], all_xsec["G18_10a"][flav][ch])
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
    #
    # Wester CSV data only extends to ~28 GeV. Above that, extrapolated values
    # create artificial shape differences. Taper shape corrections to zero above
    # the valid range using a smooth fade-out window.
    E_valid_max = 30.0   # Wester data ends at ~28 GeV
    E_fade_start = 20.0  # start tapering shape correction here
    taper = [E_grid[i] < E_fade_start ? 1.0 :
             E_grid[i] > E_valid_max ? 0.0 :
             (1.0 - (E_grid[i] - E_fade_start) / (E_valid_max - E_fade_start))
             for i in 1:length(E_grid)]

    alt_keys = [t for t in genie_tunes if t != nominal_key]
    if nominal_key != "NEUT5_4_0"
        push!(alt_keys, "NEUT5_4_0")
    end
    n_alt = length(alt_keys)

    function get_channel_curve(source, ch)
        if source == "NEUT5_4_0"
            g_nue = all_xsec["G18_10a"]["nue"][ch]
            g_nuebar = all_xsec["G18_10a"]["nuebar"][ch]
            return (extend_wester(wester_xsec["nue"][ch], g_nue) .+ extend_wester(wester_xsec["nuebar"][ch], g_nuebar)) ./ 2
        else
            return (all_xsec[source]["nue"][ch] .+ all_xsec[source]["nuebar"][ch]) ./ 2
        end
    end

    # Only use E range where all sources have real data for shape PCA
    E_pca_mask = E_grid .<= E_valid_max

    process_shape_itps = Dict{String, Any}()
    process_norm_sigmas = Dict{String, Float64}()

    function compute_shape_pca(nom, alt_curves)
        n_E = length(E_grid)
        # Norm spread: use full E range
        nom_mean = sum(nom) / n_E
        all_means = Float64[nom_mean]
        for alt in alt_curves
            push!(all_means, sum(alt) / n_E)
        end
        norm_sigma = std(all_means ./ nom_mean)

        # Shape PCA: only use valid E range, then taper
        n_pca = sum(E_pca_mask)
        Delta = zeros(n_pca, length(alt_curves))
        nom_pca = nom[E_pca_mask]
        nom_pca_mean = sum(nom_pca) / n_pca
        for (i, alt) in enumerate(alt_curves)
            alt_pca = alt[E_pca_mask]
            alt_pca_mean = sum(alt_pca) / n_pca
            alt_rescaled = alt_pca .* (nom_pca_mean / alt_pca_mean)
            frac_dev = (alt_rescaled .- nom_pca) ./ nom_pca
            frac_dev[isnan.(frac_dev) .| isinf.(frac_dev)] .= 0.0
            Delta[:, i] = frac_dev
        end

        U, S, _ = svd(Delta)
        # Embed back into full E grid with taper
        pc_full = zeros(n_E)
        pc_full[E_pca_mask] = U[:, 1] .* S[1]
        pc_full .*= taper

        return make_itp(pc_full), norm_sigma
    end

    for ch in all_channels
        nom = get_channel_curve(nominal_key == "NEUT5_4_0" ? "NEUT5_4_0" : nominal_key, ch)
        alts = [get_channel_curve(k, ch) for k in alt_keys]
        itp, ns = compute_shape_pca(nom, alts)
        process_shape_itps[ch] = itp
        process_norm_sigmas[ch] = ns
    end

    # NC: ν and ν̄ have different shapes — compute separate NC shape PCs
    for (label, flav) in [("NC_nu", "nue"), ("NC_nubar", "nuebar")]
        nom = nominal_key == "NEUT5_4_0" ? extend_wester(wester_xsec[flav]["NC"], all_xsec["G18_10a"][flav]["NC"]) : all_xsec[nominal_key][flav]["NC"]
        alts = [(k == "NEUT5_4_0" ? extend_wester(wester_xsec[flav]["NC"], all_xsec["G18_10a"][flav]["NC"]) : all_xsec[k][flav]["NC"]) for k in alt_keys]
        itp, _ = compute_shape_pca(nom, alts)
        process_shape_itps[label] = itp
    end

    @info "H2O_PCA xsec configured" nominal=cfg.nominal mc_nominal=cfg.mc_nominal norm_sigmas=process_norm_sigmas

    # Parameter → channel mapping
    norm_syms = (
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
    nubar_ratio_syms = (
        CC1p1h = :xsec_cc1p1h_nubar_ratio,
        CC2p2h = :xsec_cc2p2h_nubar_ratio,
        CC1pi  = :xsec_cc1pi_nubar_ratio,
        CCDIS  = :xsec_ccdis_nubar_ratio,
        CCother = :xsec_ccother_nubar_ratio,
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
            # shape_PC differs for ν vs ν̄; nubar_ratio scales ν̄ relative to ν
            nc_shape = anti ? process_shape_itps["NC_nubar"] : process_shape_itps["NC_nu"]
            w = max.(zero(T), params.xsec_nc_norm .* (one(T) .+ params.xsec_nc_shape .* nc_shape.(E)))
            # Normalization-conserving nu/nubar ratio: at r=1, factor=1 for both
            r_nc = params.xsec_nc_nubar_ratio
            return anti ? w .* (2 * r_nc / (1 + r_nc)) : w .* (2 / (1 + r_nc))
        end

        fk = get_flavor_key(flav, anti)
        fracs = nom_frac_itps[fk]

        # CC reweight = Σ_ch f_ch(E) × ch_norm × [nubar_ratio if ν̄] × (1 + ε_ch × shape_PC_ch(E))
        # Each channel contribution clamped ≥ 0 before summing
        result = zeros(T, length(E))
        for ch in cc_channels
            f_ch = getfield(fracs, Symbol(ch)).(E)
            ch_norm = ch == "CC1p1h" ? ifelse.(E .< 1.33, params.xsec_cc1p1h_subgev_norm, params.xsec_cc1p1h_multigev_norm) : getfield(params, getfield(norm_syms, Symbol(ch)))
            ch_eps = getfield(params, getfield(shape_syms, Symbol(ch)))
            ch_shape = process_shape_itps[ch]
            ch_w = max.(zero(T), f_ch .* ch_norm .* (one(T) .+ ch_eps .* ch_shape.(E)))
            # Normalization-conserving nu/nubar ratio: scale both sides symmetrically
            r = getfield(params, getfield(nubar_ratio_syms, Symbol(ch)))
            if anti
                ch_w = ch_w .* (2 * r / (1 + r))
            else
                ch_w = ch_w .* (2 / (1 + r))
            end
            # Normalization-conserving νe/νμ ratio for CCQE
            if ch == "CC1p1h"
                r_fl = params.xsec_cc1p1h_nue_numu_ratio
                if flav == :nue
                    ch_w = ch_w .* (2 * r_fl / (1 + r_fl))
                else
                    ch_w = ch_w .* (2 / (1 + r_fl))
                end
            end
            result .+= ch_w
        end

        if flav == :nutau
            return result .* params.xsec_nutau_cc_norm
        end
        return result
    end
end

function get_dσdE(cfg::H2O_PCA)
    E_grid, wester_xsec, all_xsec = _load_genie_data()

    nominal_key = string(cfg.nominal)
    cc_channels = ("CC1p1h", "CC2p2h", "CC1pi", "CCDIS", "CCother")

    function make_itp(vals)
        itp = interpolate((E_grid,), vals, Gridded(Linear()))
        return extrapolate(itp, Flat())
    end

    # Extend Wester above valid range using GENIE shape (same as get_scale)
    E_wester_max = 26.0
    i_wester_last = findlast(E_grid .<= E_wester_max)

    function extend_wester(wester_vals, genie_vals)
        extended = copy(wester_vals)
        g_at_boundary = genie_vals[i_wester_last]
        w_at_boundary = wester_vals[i_wester_last]
        if g_at_boundary > 0 && w_at_boundary > 0
            scale = w_at_boundary / g_at_boundary
            extended[i_wester_last+1:end] .= genie_vals[i_wester_last+1:end] .* scale
        end
        return extended
    end

    function get_nominal(flav, ch)
        if nominal_key == "NEUT5_4_0"
            return extend_wester(wester_xsec[flav][ch], all_xsec["G18_10a"][flav][ch])
        else
            return all_xsec[nominal_key][flav][ch]
        end
    end

    # Precompute absolute σ/E interpolations per channel per flavor
    flav_keys = ("nue", "nuebar", "numu", "numubar")
    nom_xsec_itps = Dict{String, NamedTuple}()
    for fk in flav_keys
        xsecs = NamedTuple{Symbol.(cc_channels)}(make_itp(get_nominal(fk, ch)) for ch in cc_channels)
        nom_xsec_itps[fk] = xsecs
    end

    # NC σ/E per flavor
    nc_itps = Dict{String, Any}()
    for fk in flav_keys
        nc_itps[fk] = make_itp(get_nominal(fk, "NC"))
    end

    # Reuse shape PCA and sym mappings from get_scale (computed at configure time)
    # We need to recompute here since get_scale's closures aren't accessible
    E_valid_max = 30.0
    E_fade_start = 20.0
    taper = [E_grid[i] < E_fade_start ? 1.0 :
             E_grid[i] > E_valid_max ? 0.0 :
             (1.0 - (E_grid[i] - E_fade_start) / (E_valid_max - E_fade_start))
             for i in 1:length(E_grid)]

    genie_tunes = ["G18_10a", "G21_11a", "G18_02a"]
    alt_keys = [t for t in genie_tunes if t != nominal_key]
    if nominal_key != "NEUT5_4_0"; push!(alt_keys, "NEUT5_4_0"); end

    function get_channel_curve(source, ch)
        if source == "NEUT5_4_0"
            g_nue = all_xsec["G18_10a"]["nue"][ch]
            g_nuebar = all_xsec["G18_10a"]["nuebar"][ch]
            return (extend_wester(wester_xsec["nue"][ch], g_nue) .+ extend_wester(wester_xsec["nuebar"][ch], g_nuebar)) ./ 2
        else
            return (all_xsec[source]["nue"][ch] .+ all_xsec[source]["nuebar"][ch]) ./ 2
        end
    end

    E_pca_mask = E_grid .<= E_valid_max

    function compute_shape_pca(nom, alt_curves)
        n_E = length(E_grid)
        n_pca = sum(E_pca_mask)
        Delta = zeros(n_pca, length(alt_curves))
        nom_pca = nom[E_pca_mask]
        nom_pca_mean = sum(nom_pca) / n_pca
        for (i, alt) in enumerate(alt_curves)
            alt_pca = alt[E_pca_mask]
            alt_pca_mean = sum(alt_pca) / n_pca
            alt_rescaled = alt_pca .* (nom_pca_mean / alt_pca_mean)
            frac_dev = (alt_rescaled .- nom_pca) ./ nom_pca
            frac_dev[isnan.(frac_dev) .| isinf.(frac_dev)] .= 0.0
            Delta[:, i] = frac_dev
        end
        U, S, _ = svd(Delta)
        pc_full = zeros(n_E)
        pc_full[E_pca_mask] = U[:, 1] .* S[1]
        pc_full .*= taper
        return make_itp(pc_full)
    end

    process_shape_itps = Dict{String, Any}()
    all_channels = (cc_channels..., "NC")
    for ch in all_channels
        nom = get_channel_curve(nominal_key == "NEUT5_4_0" ? "NEUT5_4_0" : nominal_key, ch)
        alts = [get_channel_curve(k, ch) for k in alt_keys]
        process_shape_itps[ch] = compute_shape_pca(nom, alts)
    end
    for (label, flav) in [("NC_nu", "nue"), ("NC_nubar", "nuebar")]
        nom = nominal_key == "NEUT5_4_0" ? extend_wester(wester_xsec[flav]["NC"], all_xsec["G18_10a"][flav]["NC"]) : all_xsec[nominal_key][flav]["NC"]
        alts = [(k == "NEUT5_4_0" ? extend_wester(wester_xsec[flav]["NC"], all_xsec["G18_10a"][flav]["NC"]) : all_xsec[k][flav]["NC"]) for k in alt_keys]
        process_shape_itps[label] = compute_shape_pca(nom, alts)
    end

    norm_syms = (CC2p2h=:xsec_cc2p2h_norm, CC1pi=:xsec_cc1pi_norm, CCDIS=:xsec_ccdis_norm, CCother=:xsec_ccother_norm)
    shape_syms = (CC1p1h=:xsec_cc1p1h_shape, CC2p2h=:xsec_cc2p2h_shape, CC1pi=:xsec_cc1pi_shape, CCDIS=:xsec_ccdis_shape, CCother=:xsec_ccother_shape)
    nubar_ratio_syms = (CC1p1h=:xsec_cc1p1h_nubar_ratio, CC2p2h=:xsec_cc2p2h_nubar_ratio, CC1pi=:xsec_cc1pi_nubar_ratio, CCDIS=:xsec_ccdis_nubar_ratio, CCother=:xsec_ccother_nubar_ratio)

    function get_flavor_key(flav::Symbol, anti::Bool)
        if flav == :nue
            return anti ? "nuebar" : "nue"
        elseif flav == :numu || flav == :nutau
            return anti ? "numubar" : "numu"
        else
            return anti ? "nuebar" : "nue"
        end
    end

    function dσdE(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        T = promote_type(eltype(E), typeof(params.xsec_nc_norm))

        if interaction == :NC
            fk = get_flavor_key(flav, anti)
            nc_shape = anti ? process_shape_itps["NC_nubar"] : process_shape_itps["NC_nu"]
            w = max.(zero(T), nc_itps[fk].(E) .* params.xsec_nc_norm .* (one(T) .+ params.xsec_nc_shape .* nc_shape.(E)))
            r_nc = params.xsec_nc_nubar_ratio
            return anti ? w .* (2 * r_nc / (1 + r_nc)) : w .* (2 / (1 + r_nc))
        end

        fk = get_flavor_key(flav, anti)
        xsecs = nom_xsec_itps[fk]

        result = zeros(T, length(E))
        for ch in cc_channels
            σ_ch = getfield(xsecs, Symbol(ch)).(E)
            ch_norm = ch == "CC1p1h" ? ifelse.(E .< 1.33, params.xsec_cc1p1h_subgev_norm, params.xsec_cc1p1h_multigev_norm) : getfield(params, getfield(norm_syms, Symbol(ch)))
            ch_eps = getfield(params, getfield(shape_syms, Symbol(ch)))
            ch_shape = process_shape_itps[ch]
            ch_w = max.(zero(T), σ_ch .* ch_norm .* (one(T) .+ ch_eps .* ch_shape.(E)))
            r = getfield(params, getfield(nubar_ratio_syms, Symbol(ch)))
            if anti
                ch_w = ch_w .* (2 * r / (1 + r))
            else
                ch_w = ch_w .* (2 / (1 + r))
            end
            if ch == "CC1p1h"
                r_fl = params.xsec_cc1p1h_nue_numu_ratio
                if flav == :nue
                    ch_w = ch_w .* (2 * r_fl / (1 + r_fl))
                else
                    ch_w = ch_w .* (2 / (1 + r_fl))
                end
            end
            result .+= ch_w
        end

        if flav == :nutau
            return result .* params.xsec_nutau_cc_norm
        end
        return result
    end
end

function get_scale_event(cfg::H2O_PCA)
    # Per-event reweighting using GENIE interaction codes.
    # Returns a weight w_i = σ_nominal(E_i, ch_i) / σ_mc_nominal(E_i, ch_i) × norm × shape × nubar_ratio × ...
    # At nominal parameters (all norms=1, shapes=0, ratios=1), weight = σ_nominal / σ_mc_nominal,
    # which reweights the MC from mc_nominal to nominal — ensuring consistent cross-section
    # treatment across experiments in a global fit (e.g. DeepCore G00_00a → NEUT5_4_0).
    #
    # GENIE interaction code mapping (DeepCore data release convention):
    #   0 → CC1p1h (QE)
    #   1 → CC1pi  (resonance)
    #   2 → CCDIS
    #   3 or -1 → CCother (coherent or other)
    #   NC: passed via interaction=:NC, genie_codes ignored
    E_grid, wester_xsec, all_xsec = _load_genie_data()

    nominal_key    = string(cfg.nominal)
    mc_nominal_key = string(cfg.mc_nominal)

    cc_channels = ("CC1p1h", "CC1pi", "CCDIS", "CCother")
    flav_keys   = ("nue", "nuebar", "numu", "numubar")

    function make_itp(vals)
        itp = interpolate((E_grid,), vals, Gridded(Linear()))
        return extrapolate(itp, Flat())
    end

    E_neut_max   = 26.0
    i_neut_last  = findlast(E_grid .<= E_neut_max)

    function extend_neut(neut_vals, genie_vals)
        extended = copy(neut_vals)
        g_at_boundary = genie_vals[i_neut_last]
        n_at_boundary = neut_vals[i_neut_last]
        if g_at_boundary > 0 && n_at_boundary > 0
            sc = n_at_boundary / g_at_boundary
            extended[i_neut_last+1:end] .= genie_vals[i_neut_last+1:end] .* sc
        end
        return extended
    end

    function get_curve(key, flav, ch)
        if key == "NEUT5_4_0"
            return extend_neut(wester_xsec[flav][ch], all_xsec["G18_10a"][flav][ch])
        else
            return all_xsec[key][flav][ch]
        end
    end

    # Shape PCA (same as get_scale — recomputed here for independent closure)
    E_valid_max  = 30.0
    E_fade_start = 20.0
    taper = [E_grid[i] < E_fade_start ? 1.0 :
             E_grid[i] > E_valid_max  ? 0.0 :
             (1.0 - (E_grid[i] - E_fade_start) / (E_valid_max - E_fade_start))
             for i in 1:length(E_grid)]
    E_pca_mask = E_grid .<= E_valid_max

    genie_tunes = ["G18_10a", "G21_11a", "G18_02a"]
    alt_keys = [t for t in genie_tunes if t != nominal_key]
    if nominal_key != "NEUT5_4_0"; push!(alt_keys, "NEUT5_4_0"); end

    function get_ch_curve(source, ch)
        if source == "NEUT5_4_0"
            gn = all_xsec["G18_10a"]["nue"][ch]
            gnb = all_xsec["G18_10a"]["nuebar"][ch]
            return (extend_neut(wester_xsec["nue"][ch], gn) .+ extend_neut(wester_xsec["nuebar"][ch], gnb)) ./ 2
        else
            return (all_xsec[source]["nue"][ch] .+ all_xsec[source]["nuebar"][ch]) ./ 2
        end
    end

    function compute_shape_pca(nom, alt_curves)
        n_E   = length(E_grid)
        n_pca = sum(E_pca_mask)
        Delta = zeros(n_pca, length(alt_curves))
        nom_pca      = nom[E_pca_mask]
        nom_pca_mean = sum(nom_pca) / n_pca
        for (i, alt) in enumerate(alt_curves)
            alt_pca      = alt[E_pca_mask]
            alt_pca_mean = sum(alt_pca) / n_pca
            alt_rescaled = alt_pca .* (nom_pca_mean / alt_pca_mean)
            frac_dev     = (alt_rescaled .- nom_pca) ./ nom_pca
            frac_dev[isnan.(frac_dev) .| isinf.(frac_dev)] .= 0.0
            Delta[:, i] = frac_dev
        end
        U, S, _ = svd(Delta)
        pc_full = zeros(n_E)
        pc_full[E_pca_mask] = U[:, 1] .* S[1]
        pc_full .*= taper
        return make_itp(pc_full)
    end

    shape_itps = Dict{String, Any}()
    for ch in (cc_channels..., "NC")
        nom  = get_ch_curve(nominal_key == "NEUT5_4_0" ? "NEUT5_4_0" : nominal_key, ch)
        alts = [get_ch_curve(k, ch) for k in alt_keys]
        shape_itps[ch] = compute_shape_pca(nom, alts)
    end
    for (label, flav) in [("NC_nu", "nue"), ("NC_nubar", "nuebar")]
        nom  = get_curve(nominal_key, flav, "NC")
        alts = [get_curve(k, flav, "NC") for k in alt_keys]
        shape_itps[label] = compute_shape_pca(nom, alts)
    end

    # Precompute σ_nominal / σ_mc_nominal ratio interpolations per channel per flavor.
    # At nominal parameters this ratio is the baseline reweight from mc_nominal to nominal.
    # When nominal == mc_nominal all ratios are 1.0 (no baseline shift).
    ratio_itps = Dict{String, Dict{String, Any}}()
    for ch in (cc_channels..., "NC")
        ratio_itps[ch] = Dict{String, Any}()
        for fk in flav_keys
            nom    = get_curve(nominal_key,    fk, ch)
            mc_nom = get_curve(mc_nominal_key, fk, ch)
            ratio  = similar(nom)
            for i in eachindex(nom)
                if mc_nom[i] > 0
                    ratio[i] = nom[i] / mc_nom[i]
                else
                    ratio[i] = 1.0  # both zero or mc zero → fallback to no reweight
                end
            end
            ratio_itps[ch][fk] = make_itp(ratio)
        end
    end

    @info "H2O_PCA scale_event configured" nominal=cfg.nominal mc_nominal=cfg.mc_nominal

    # Parameter → channel symbol mappings (same as get_scale)
    norm_syms = (
        CC1pi   = :xsec_cc1pi_norm,
        CCDIS   = :xsec_ccdis_norm,
        CCother = :xsec_ccother_norm,
    )
    shape_syms = (
        CC1p1h  = :xsec_cc1p1h_shape,
        CC1pi   = :xsec_cc1pi_shape,
        CCDIS   = :xsec_ccdis_shape,
        CCother = :xsec_ccother_shape,
    )
    nubar_syms = (
        CC1p1h  = :xsec_cc1p1h_nubar_ratio,
        CC1pi   = :xsec_cc1pi_nubar_ratio,
        CCDIS   = :xsec_ccdis_nubar_ratio,
        CCother = :xsec_ccother_nubar_ratio,
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

    function scale_event(E::AbstractArray, genie_codes, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        T  = promote_type(eltype(E), typeof(params.xsec_nc_norm))
        fk = get_flavor_key(flav, anti)
        result = zeros(T, length(E))

        if interaction == :NC
            nc_shape_itp = anti ? shape_itps["NC_nubar"] : shape_itps["NC_nu"]
            ratio_nc     = ratio_itps["NC"][fk]
            r_nc = params.xsec_nc_nubar_ratio
            for i in eachindex(E)
                e       = E[i]
                baseline = ratio_nc(e)
                w = max(zero(T), baseline * params.xsec_nc_norm * (one(T) + params.xsec_nc_shape * nc_shape_itp(e)))
                result[i] = anti ? w * (2 * r_nc / (1 + r_nc)) : w * (2 / (1 + r_nc))
            end
        else  # CC
            r_fl = params.xsec_cc1p1h_nue_numu_ratio
            for i in eachindex(E)
                e    = E[i]
                code = genie_codes[i]

                local w::T
                if code == 0  # QE → CC1p1h
                    ch_norm  = e < 1.33 ? params.xsec_cc1p1h_subgev_norm : params.xsec_cc1p1h_multigev_norm
                    baseline = ratio_itps["CC1p1h"][fk](e)
                    w = max(zero(T), baseline * ch_norm * (one(T) + params.xsec_cc1p1h_shape * shape_itps["CC1p1h"](e)))
                    r = params.xsec_cc1p1h_nubar_ratio
                    w = anti ? w * (2 * r / (1 + r)) : w * (2 / (1 + r))
                    w = flav == :nue ? w * (2 * r_fl / (1 + r_fl)) : w * (2 / (1 + r_fl))
                elseif code == 1  # RES → CC1pi
                    baseline = ratio_itps["CC1pi"][fk](e)
                    w = max(zero(T), baseline * params.xsec_cc1pi_norm * (one(T) + params.xsec_cc1pi_shape * shape_itps["CC1pi"](e)))
                    r = params.xsec_cc1pi_nubar_ratio
                    w = anti ? w * (2 * r / (1 + r)) : w * (2 / (1 + r))
                elseif code == 2  # DIS → CCDIS
                    baseline = ratio_itps["CCDIS"][fk](e)
                    w = max(zero(T), baseline * params.xsec_ccdis_norm * (one(T) + params.xsec_ccdis_shape * shape_itps["CCDIS"](e)))
                    r = params.xsec_ccdis_nubar_ratio
                    w = anti ? w * (2 * r / (1 + r)) : w * (2 / (1 + r))
                else  # coherent (3) or other (-1) → CCother
                    baseline = ratio_itps["CCother"][fk](e)
                    w = max(zero(T), baseline * params.xsec_ccother_norm * (one(T) + params.xsec_ccother_shape * shape_itps["CCother"](e)))
                    r = params.xsec_ccother_nubar_ratio
                    w = anti ? w * (2 * r / (1 + r)) : w * (2 / (1 + r))
                end

                if flav == :nutau
                    w = w * params.xsec_nutau_cc_norm
                end
                result[i] = w
            end
        end

        return result
    end
end

# Fast precomputed path for H2O_PCA per-event reweighting.
# Called once at configure time; returns (E, codes, flav, interaction, anti) -> (params -> weights).
# All interpolations happen during precomputation — eval time is purely vectorized arithmetic.
function get_event_weights(cfg::H2O_PCA)
    E_grid, wester_xsec, all_xsec = _load_genie_data()

    nominal_key    = string(cfg.nominal)
    mc_nominal_key = string(cfg.mc_nominal)

    cc_channels = ("CC1p1h", "CC1pi", "CCDIS", "CCother")
    flav_keys   = ("nue", "nuebar", "numu", "numubar")

    function make_itp(vals)
        itp = interpolate((E_grid,), vals, Gridded(Linear()))
        return extrapolate(itp, Flat())
    end

    E_neut_max  = 26.0
    i_neut_last = findlast(E_grid .<= E_neut_max)
    function extend_neut(neut_vals, genie_vals)
        extended = copy(neut_vals)
        g = genie_vals[i_neut_last]; n = neut_vals[i_neut_last]
        if g > 0 && n > 0
            sc = n / g
            extended[i_neut_last+1:end] .= genie_vals[i_neut_last+1:end] .* sc
        end
        return extended
    end

    function get_curve(key, flav, ch)
        key == "NEUT5_4_0" ? extend_neut(wester_xsec[flav][ch], all_xsec["G18_10a"][flav][ch]) : all_xsec[key][flav][ch]
    end

    # Shape PCA (same setup as get_scale_event)
    E_valid_max  = 30.0
    E_fade_start = 20.0
    taper = [E_grid[i] < E_fade_start ? 1.0 :
             E_grid[i] > E_valid_max  ? 0.0 :
             (1.0 - (E_grid[i] - E_fade_start) / (E_valid_max - E_fade_start))
             for i in 1:length(E_grid)]
    E_pca_mask = E_grid .<= E_valid_max

    genie_tunes = ["G18_10a", "G21_11a", "G18_02a"]
    alt_keys = [t for t in genie_tunes if t != nominal_key]
    if nominal_key != "NEUT5_4_0"; push!(alt_keys, "NEUT5_4_0"); end

    function get_ch_curve(source, ch)
        if source == "NEUT5_4_0"
            gn = all_xsec["G18_10a"]["nue"][ch]; gnb = all_xsec["G18_10a"]["nuebar"][ch]
            return (extend_neut(wester_xsec["nue"][ch], gn) .+ extend_neut(wester_xsec["nuebar"][ch], gnb)) ./ 2
        else
            return (all_xsec[source]["nue"][ch] .+ all_xsec[source]["nuebar"][ch]) ./ 2
        end
    end

    function compute_shape_pca(nom, alt_curves)
        n_E = length(E_grid); n_pca = sum(E_pca_mask)
        Delta = zeros(n_pca, length(alt_curves))
        nom_pca = nom[E_pca_mask]; nom_mean = sum(nom_pca) / n_pca
        for (i, alt) in enumerate(alt_curves)
            alt_pca = alt[E_pca_mask]; alt_mean = sum(alt_pca) / n_pca
            frac = (alt_pca .* (nom_mean / alt_mean) .- nom_pca) ./ nom_pca
            frac[isnan.(frac) .| isinf.(frac)] .= 0.0
            Delta[:, i] = frac
        end
        U, S, _ = svd(Delta)
        pc = zeros(n_E); pc[E_pca_mask] = U[:, 1] .* S[1]; pc .*= taper
        return make_itp(pc)
    end

    shape_itps = Dict{String, Any}()
    for ch in (cc_channels..., "NC")
        nom  = get_ch_curve(nominal_key == "NEUT5_4_0" ? "NEUT5_4_0" : nominal_key, ch)
        alts = [get_ch_curve(k, ch) for k in alt_keys]
        shape_itps[ch] = compute_shape_pca(nom, alts)
    end
    for (label, flav) in [("NC_nu", "nue"), ("NC_nubar", "nuebar")]
        nom  = get_curve(nominal_key, flav, "NC")
        alts = [get_curve(k, flav, "NC") for k in alt_keys]
        shape_itps[label] = compute_shape_pca(nom, alts)
    end

    # σ_nominal / σ_mc_nominal ratio interpolations
    ratio_itps = Dict{String, Dict{String, Any}}()
    for ch in (cc_channels..., "NC")
        ratio_itps[ch] = Dict{String, Any}()
        for fk in flav_keys
            nom = get_curve(nominal_key, fk, ch); mc_nom = get_curve(mc_nominal_key, fk, ch)
            ratio = map(i -> mc_nom[i] > 0 ? nom[i] / mc_nom[i] : 1.0, eachindex(nom))
            ratio_itps[ch][fk] = make_itp(ratio)
        end
    end

    function get_flavor_key(flav::Symbol, anti::Bool)
        flav == :nue ? (anti ? "nuebar" : "nue") :
        (flav == :numu || flav == :nutau) ? (anti ? "numubar" : "numu") :
        (anti ? "nuebar" : "nue")
    end

    # Returns fast eval closure (params) -> weight vector for a given MC component.
    function event_weights(E, genie_codes, flav::Symbol, interaction::Symbol, anti::Bool)
        fk   = get_flavor_key(flav, anti)
        n    = length(E)
        Evec = vec(E); cvec = vec(genie_codes)

        if interaction == :NC
            baseline  = ratio_itps["NC"][fk].(Evec)
            nc_sh_itp = anti ? shape_itps["NC_nubar"] : shape_itps["NC_nu"]
            shape_val = nc_sh_itp.(Evec)
            _anti     = anti
            return function eval_nc(params)
                T  = typeof(params.xsec_nc_norm)
                r  = params.xsec_nc_nubar_ratio
                nf = _anti ? 2r/(1+r) : 2/(1+r)
                return max.(zero(T), baseline .* params.xsec_nc_norm .*
                       (one(T) .+ params.xsec_nc_shape .* shape_val)) .* nf
            end
        else
            # Precompute per-event, per-channel arrays (non-channel events stay 0.0)
            b_subgev   = zeros(n); b_multigev = zeros(n); s_1p1h = zeros(n)
            b_1pi      = zeros(n); s_1pi      = zeros(n)
            b_dis      = zeros(n); s_dis      = zeros(n)
            b_other    = zeros(n); s_other    = zeros(n)

            for i in 1:n
                e = Evec[i]; code = cvec[i]
                if code == 0
                    b = ratio_itps["CC1p1h"][fk](e); s = shape_itps["CC1p1h"](e)
                    if e < 1.33; b_subgev[i] = b; else; b_multigev[i] = b; end
                    s_1p1h[i] = s
                elseif code == 1
                    b_1pi[i] = ratio_itps["CC1pi"][fk](e); s_1pi[i] = shape_itps["CC1pi"](e)
                elseif code == 2
                    b_dis[i] = ratio_itps["CCDIS"][fk](e);  s_dis[i] = shape_itps["CCDIS"](e)
                else
                    b_other[i] = ratio_itps["CCother"][fk](e); s_other[i] = shape_itps["CCother"](e)
                end
            end

            _anti = anti; _flav = flav
            return function eval_cc(params)
                T   = typeof(params.xsec_cc1p1h_subgev_norm)
                r1  = params.xsec_cc1p1h_nubar_ratio;  r_fl = params.xsec_cc1p1h_nue_numu_ratio
                r2  = params.xsec_cc1pi_nubar_ratio;   r3   = params.xsec_ccdis_nubar_ratio
                r4  = params.xsec_ccother_nubar_ratio

                nf1  = _anti ? 2r1/(1+r1)  : 2/(1+r1)
                nf_fl = _flav == :nue ? 2r_fl/(1+r_fl) : 2/(1+r_fl)
                nf2  = _anti ? 2r2/(1+r2)  : 2/(1+r2)
                nf3  = _anti ? 2r3/(1+r3)  : 2/(1+r3)
                nf4  = _anti ? 2r4/(1+r4)  : 2/(1+r4)

                w1 = max.(zero(T),
                    (b_subgev .* params.xsec_cc1p1h_subgev_norm .+
                     b_multigev .* params.xsec_cc1p1h_multigev_norm) .*
                    (one(T) .+ params.xsec_cc1p1h_shape .* s_1p1h)) .* (nf1 * nf_fl)

                w2 = max.(zero(T),
                    b_1pi .* params.xsec_cc1pi_norm .*
                    (one(T) .+ params.xsec_cc1pi_shape .* s_1pi)) .* nf2

                w3 = max.(zero(T),
                    b_dis .* params.xsec_ccdis_norm .*
                    (one(T) .+ params.xsec_ccdis_shape .* s_dis)) .* nf3

                w4 = max.(zero(T),
                    b_other .* params.xsec_ccother_norm .*
                    (one(T) .+ params.xsec_ccother_shape .* s_other)) .* nf4

                result = w1 .+ w2 .+ w3 .+ w4
                return _flav == :nutau ? result .* params.xsec_nutau_cc_norm : result
            end
        end
    end

    return event_weights
end

# Grid-based precomputed path for H2O_PCA reweighting on a fixed true-energy grid.
# Used by experiments (e.g. ORCA) that only have CC/NC labels per event, not sub-channel codes.
# Called once at configure time; returns (E_grid, flav, interaction, anti) -> (params -> weights).
# The weight at energy bin k is:
#   CC: [Σ_ch σ_nominal_ch(E_k) × norm_ch × shape_ch] / σ_mc_nominal_total_CC(E_k)
#   NC: σ_nominal_NC(E_k) × norm_nc × shape_nc / σ_mc_nominal_NC(E_k)
# In get_expected, events are weighted by indexing into this array with E_true_bin.
function get_grid_weights(cfg::H2O_PCA)
    E_grid, wester_xsec, all_xsec = _load_genie_data()

    nominal_key    = string(cfg.nominal)
    mc_nominal_key = string(cfg.mc_nominal)

    # All CC channels in the nominal model (NEUT has CC2p2h; G00_00a does not)
    cc_channels    = ("CC1p1h", "CC2p2h", "CC1pi", "CCDIS", "CCother")
    # Only channels present in the MC generator (denominator)
    mc_cc_channels = ("CC1p1h", "CC1pi", "CCDIS", "CCother")
    flav_keys      = ("nue", "nuebar", "numu", "numubar")

    function make_itp(vals)
        itp = interpolate((E_grid,), vals, Gridded(Linear()))
        return extrapolate(itp, Flat())
    end

    E_neut_max  = 26.0
    i_neut_last = findlast(E_grid .<= E_neut_max)
    function extend_neut(neut_vals, genie_vals)
        extended = copy(neut_vals)
        g = genie_vals[i_neut_last]; n = neut_vals[i_neut_last]
        if g > 0 && n > 0
            sc = n / g
            extended[i_neut_last+1:end] .= genie_vals[i_neut_last+1:end] .* sc
        end
        return extended
    end

    function get_curve(key, flav, ch)
        key == "NEUT5_4_0" ? extend_neut(wester_xsec[flav][ch], all_xsec["G18_10a"][flav][ch]) : all_xsec[key][flav][ch]
    end

    # Shape PCA (same setup as get_event_weights)
    E_valid_max  = 30.0
    E_fade_start = 20.0
    taper = [E_grid[i] < E_fade_start ? 1.0 :
             E_grid[i] > E_valid_max  ? 0.0 :
             (1.0 - (E_grid[i] - E_fade_start) / (E_valid_max - E_fade_start))
             for i in 1:length(E_grid)]
    E_pca_mask = E_grid .<= E_valid_max

    genie_tunes = ["G18_10a", "G21_11a", "G18_02a"]
    alt_keys = [t for t in genie_tunes if t != nominal_key]
    if nominal_key != "NEUT5_4_0"; push!(alt_keys, "NEUT5_4_0"); end

    function get_ch_curve(source, ch)
        if source == "NEUT5_4_0"
            gn = all_xsec["G18_10a"]["nue"][ch]; gnb = all_xsec["G18_10a"]["nuebar"][ch]
            return (extend_neut(wester_xsec["nue"][ch], gn) .+ extend_neut(wester_xsec["nuebar"][ch], gnb)) ./ 2
        else
            return (all_xsec[source]["nue"][ch] .+ all_xsec[source]["nuebar"][ch]) ./ 2
        end
    end

    function compute_shape_pca(nom, alt_curves)
        n_E = length(E_grid); n_pca = sum(E_pca_mask)
        Delta = zeros(n_pca, length(alt_curves))
        nom_pca = nom[E_pca_mask]; nom_mean = sum(nom_pca) / n_pca
        for (i, alt) in enumerate(alt_curves)
            alt_pca = alt[E_pca_mask]; alt_mean = sum(alt_pca) / n_pca
            frac = (alt_pca .* (nom_mean / alt_mean) .- nom_pca) ./ nom_pca
            frac[isnan.(frac) .| isinf.(frac)] .= 0.0
            Delta[:, i] = frac
        end
        U, S, _ = svd(Delta)
        pc = zeros(n_E); pc[E_pca_mask] = U[:, 1] .* S[1]; pc .*= taper
        return make_itp(pc)
    end

    shape_itps = Dict{String, Any}()
    for ch in (cc_channels..., "NC")
        nom  = get_ch_curve(nominal_key == "NEUT5_4_0" ? "NEUT5_4_0" : nominal_key, ch)
        alts = [get_ch_curve(k, ch) for k in alt_keys]
        shape_itps[ch] = compute_shape_pca(nom, alts)
    end
    for (label, flav) in [("NC_nu", "nue"), ("NC_nubar", "nuebar")]
        nom  = get_curve(nominal_key, flav, "NC")
        alts = [get_curve(k, flav, "NC") for k in alt_keys]
        shape_itps[label] = compute_shape_pca(nom, alts)
    end

    # Absolute σ_nominal per channel per flavor
    nom_xsec_itps = Dict{String, Dict{String, Any}}()
    for fk in flav_keys
        nom_xsec_itps[fk] = Dict{String, Any}()
        for ch in (cc_channels..., "NC")
            nom_xsec_itps[fk][ch] = make_itp(get_curve(nominal_key, fk, ch))
        end
    end

    # MC total CC denominator per flavor (only channels present in mc_nominal)
    mc_total_cc_itps = Dict{String, Any}()
    for fk in flav_keys
        total = sum(get_curve(mc_nominal_key, fk, ch) for ch in mc_cc_channels)
        mc_total_cc_itps[fk] = make_itp(max.(total, 1e-30))
    end

    # NC ratio: σ_nominal_NC / σ_mc_nominal_NC per flavor
    nc_ratio_itps = Dict{String, Any}()
    for fk in flav_keys
        nom_nc = get_curve(nominal_key,    fk, "NC")
        mc_nc  = get_curve(mc_nominal_key, fk, "NC")
        ratio  = map(i -> mc_nc[i] > 0 ? nom_nc[i] / mc_nc[i] : 1.0, eachindex(nom_nc))
        nc_ratio_itps[fk] = make_itp(ratio)
    end

    function get_flavor_key(flav::Symbol, anti::Bool)
        flav == :nue ? (anti ? "nuebar" : "nue") :
        (flav == :numu || flav == :nutau) ? (anti ? "numubar" : "numu") :
        (anti ? "nuebar" : "nue")
    end

    # Returns a closure (params) -> weight_vector of length(E_eval).
    # All interpolations are evaluated at configure/asset-load time; eval is pure arithmetic.
    function grid_weights(E_eval, flav::Symbol, interaction::Symbol, anti::Bool)
        fk   = get_flavor_key(flav, anti)
        n    = length(E_eval)
        Evec = vec(E_eval)

        if interaction == :NC
            baseline  = nc_ratio_itps[fk].(Evec)
            nc_sh_itp = anti ? shape_itps["NC_nubar"] : shape_itps["NC_nu"]
            shape_val = nc_sh_itp.(Evec)
            _anti = anti
            return function eval_nc_grid(params)
                T  = typeof(params.xsec_nc_norm)
                r  = params.xsec_nc_nubar_ratio
                nf = _anti ? 2r/(1+r) : 2/(1+r)
                return max.(zero(T), baseline .* params.xsec_nc_norm .*
                       (one(T) .+ params.xsec_nc_shape .* shape_val)) .* nf
            end
        else  # CC: weight = Σ_ch σ_nominal_ch(E) × norm_ch × shape_ch / σ_mc_total_cc(E)
            mc_denom = mc_total_cc_itps[fk].(Evec)

            b_subgev = zeros(n); b_multigev = zeros(n); s_1p1h  = zeros(n)
            b_2p2h   = zeros(n);                        s_2p2h  = zeros(n)
            b_1pi    = zeros(n);                        s_1pi   = zeros(n)
            b_dis    = zeros(n);                        s_dis   = zeros(n)
            b_other  = zeros(n);                        s_other = zeros(n)

            for i in 1:n
                e = Evec[i]; d = mc_denom[i]
                σ1 = nom_xsec_itps[fk]["CC1p1h"](e) / d
                if e < 1.33; b_subgev[i] = σ1; else; b_multigev[i] = σ1; end
                s_1p1h[i]  = shape_itps["CC1p1h"](e)
                b_2p2h[i]  = nom_xsec_itps[fk]["CC2p2h"](e) / d
                s_2p2h[i]  = shape_itps["CC2p2h"](e)
                b_1pi[i]   = nom_xsec_itps[fk]["CC1pi"](e)   / d
                s_1pi[i]   = shape_itps["CC1pi"](e)
                b_dis[i]   = nom_xsec_itps[fk]["CCDIS"](e)   / d
                s_dis[i]   = shape_itps["CCDIS"](e)
                b_other[i] = nom_xsec_itps[fk]["CCother"](e) / d
                s_other[i] = shape_itps["CCother"](e)
            end

            _anti = anti; _flav = flav
            return function eval_cc_grid(params)
                T    = typeof(params.xsec_cc1p1h_subgev_norm)
                r1   = params.xsec_cc1p1h_nubar_ratio;  r_fl = params.xsec_cc1p1h_nue_numu_ratio
                r2   = params.xsec_cc2p2h_nubar_ratio;  r3   = params.xsec_cc1pi_nubar_ratio
                r4   = params.xsec_ccdis_nubar_ratio;   r5   = params.xsec_ccother_nubar_ratio

                nf1  = _anti ? 2r1/(1+r1)  : 2/(1+r1)
                nf_fl = _flav == :nue ? 2r_fl/(1+r_fl) : 2/(1+r_fl)
                nf2  = _anti ? 2r2/(1+r2)  : 2/(1+r2)
                nf3  = _anti ? 2r3/(1+r3)  : 2/(1+r3)
                nf4  = _anti ? 2r4/(1+r4)  : 2/(1+r4)
                nf5  = _anti ? 2r5/(1+r5)  : 2/(1+r5)

                w1 = max.(zero(T),
                    (b_subgev .* params.xsec_cc1p1h_subgev_norm .+
                     b_multigev .* params.xsec_cc1p1h_multigev_norm) .*
                    (one(T) .+ params.xsec_cc1p1h_shape .* s_1p1h)) .* (nf1 * nf_fl)
                w2 = max.(zero(T),
                    b_2p2h .* params.xsec_cc2p2h_norm .*
                    (one(T) .+ params.xsec_cc2p2h_shape .* s_2p2h)) .* nf2
                w3 = max.(zero(T),
                    b_1pi .* params.xsec_cc1pi_norm .*
                    (one(T) .+ params.xsec_cc1pi_shape .* s_1pi)) .* nf3
                w4 = max.(zero(T),
                    b_dis .* params.xsec_ccdis_norm .*
                    (one(T) .+ params.xsec_ccdis_shape .* s_dis)) .* nf4
                w5 = max.(zero(T),
                    b_other .* params.xsec_ccother_norm .*
                    (one(T) .+ params.xsec_ccother_shape .* s_other)) .* nf5

                result = w1 .+ w2 .+ w3 .+ w4 .+ w5
                return _flav == :nutau ? result .* params.xsec_nutau_cc_norm : result
            end
        end
    end

    return grid_weights
end

end
