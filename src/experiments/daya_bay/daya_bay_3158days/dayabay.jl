module dayabay
using DataFrames
using CSV
using LinearAlgebra
using Distributions
using DataStructures
using CairoMakie
using Accessors
using Logging
import ..Newtrinos

@kwdef struct DayaBay <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function configure(physics)
    physics = (;physics.osc)
    assets = get_assets(physics)
    return DayaBay(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)

    @info "Loading dayabay data"
    # Experimental Halls
    EH_list = [1, 2, 3]
    
    # Data-taking Periods
    period_list = ["Six", "Eight", "Seven"]
    period_idx = [20, 49, 78]
    periods_dict = Dict("Six" => 6, "Eight" => 8, "Seven" => 7)
    
    # Reactors
    reactor_list = ["D1", "D2", "L1", "L2", "L3", "L4"]
    
    # Correlation matrix from Fig. 29 in https://arxiv.org/pdf/1607.05378.pdf
    corr_mat_df = CSV.read(joinpath(datadir, "DayaBay_CorrMat_arXiv_1607.05378.txt"), DataFrame, delim=' ')
    corr_mat = Symmetric(Matrix(corr_mat_df[:, 5:end]))
    
    # Fractional systematic uncertainty on the expected IBD energy spectra
    rel_unc_diag = corr_mat_df.diag_sys_unc_relative_to_spectrum;
    
    # Reactors / Baselines, etc
    exp_dict = Dict("Detector" => ["AD1", "AD2", "AD3", "AD8", "AD4", "AD5", "AD6", "AD7"],
                "EH" => ["EH1", "EH1", "EH2", "EH2", "EH3", "EH3", "EH3", "EH3"],
                "Target [kg]" => [19941, 19967, 19891, 19944, 19917, 19989, 19892, 19931],
                "Efficiency" => [0.7743, 0.7716, 0.8127, 0.8105, 0.9513, 0.9514, 0.9512, 0.9513],
                "Six-AD Period" => [true, true, true, false, true, true, true, false],
                "Eight-AD Period" => [true, true, true, true, true, true, true, true],
                "Seven-AD Period" => [true, false, true, true, true, true, true, true],
                "D1" => [362.38, 357.94, 1332.48, 1337.43, 1919.63, 1917.52, 1925.26, 1923.15],
                "D2" => [371.76, 368.41, 1358.15, 1362.88, 1894.34, 1891.98, 1899.86, 1897.51],
                "L1" => [903.47, 903.35, 467.57, 472.97, 1533.18, 1534.92, 1538.93, 1540.67],
                "L2" => [817.16, 816.90, 489.58, 495.35, 1533.63, 1535.03, 1539.47, 1540.87],
                "L3" => [1353.62, 1354.23, 557.58, 558.71, 1551.38, 1554.77, 1556.34, 1559.72],
                "L4" => [1265.32, 1265.89, 499.21, 501.07, 1524.94, 1528.05, 1530.08, 1533.18])
    df_exp = DataFrame(exp_dict);
    
    function parse(fname, idx, len)
        # Parsing the DayaBay format CSV files
        header = Array(CSV.read(fname, DataFrame, delim=' ', skipto=idx-1, limit=1, ignorerepeated=true, header=false));
        header = header[2:end];
        df = CSV.read(fname, DataFrame, delim=' ', skipto=idx, limit=len, ignorerepeated=true, header=false);
        rename!(df, header, makeunique=false);
        return df
    end
    
    # Dicts to fill
    dfBKG_dict = Dict()
    dfIBD_dict = Dict()
    
    for EH in EH_list
        fileBKG = joinpath(datadir,  "DayaBay_BackgroundSpectrum_EH$(EH)_3158days.txt")
        fileIBD = joinpath(datadir,  "DayaBay_IBDPromptSpectrum_EH$(EH)_3158days.txt")
        dfIBD_dict["dfIBD_EH$EH"] = parse(fileIBD, 11, size(corr_mat_df, 1))
        for i in 1:length(period_list)
            period = period_list[i]
            idx = period_idx[i]
            dfBKG_dict["dfBKG_$(period)_EH$(EH)"] = parse(fileBKG, idx, size(corr_mat_df, 1))
        end
        # Sum over data-taking periods
        dfIBD_dict["dfIBD_EH$EH"][!, "Nobs"] = sum(eachcol(dfIBD_dict["dfIBD_EH$EH"][!, ["Nobs_6AD", "Nobs_8AD", "Nobs_7AD"]]))
        dfIBD_dict["dfIBD_EH$EH"][!, "Npred"] = sum(eachcol(dfIBD_dict["dfIBD_EH$EH"][!, ["Npred_6AD", "Npred_8AD", "Npred_7AD"]]))
        BKG = zeros(size(dfIBD_dict["dfIBD_EH$EH"], 1))
        for period in period_list
            BKG .+= dfBKG_dict["dfBKG_$(period)_EH$(EH)"].Nbkg
        end
        dfIBD_dict["dfIBD_EH$EH"][!, "BKG"] = BKG
        dfIBD_dict["dfIBD_EH$EH"][!, "N"] = dfIBD_dict["dfIBD_EH$EH"].Nobs .- BKG
    end
    
    # Setup for DayaBay bestfit to recalculate unoscillated spectrum
    bestift_osc = Newtrinos.osc.configure()
    @reset bestift_osc.params.θ₁₂ = asin(sqrt(0.307))
    @reset bestift_osc.params.θ₁₃ = asin(sqrt(0.0851)) * 0.5
    @reset bestift_osc.params.θ₂₃ = asin(sqrt(0.57))
    @reset bestift_osc.params.δCP = 0.
    @reset bestift_osc.params.Δm²₂₁ = 7.53e-5
    @reset bestift_osc.params.Δm²₃₁ = 2.466e-3 + bestift_osc.params[:Δm²₂₁]
    
    ad_contribs_to_far_hall = Float64[]
    E_arrs = Vector{Vector{Float64}}()
    L_arrs = Vector{Vector{Float64}}()
    Npred_EH3_nooscs = Vector{Vector{Float64}}()
    
    for period in period_list
    
        Nobs1 = dfIBD_dict["dfIBD_EH1"][:, "Nobs_$(periods_dict[period])AD"] .- dfBKG_dict["dfBKG_$(period)_EH1"].Nbkg
        Nobs2 = dfIBD_dict["dfIBD_EH2"][:, "Nobs_$(periods_dict[period])AD"] .- dfBKG_dict["dfBKG_$(period)_EH2"].Nbkg
        Nobs3 = dfIBD_dict["dfIBD_EH3"][:, "Nobs_$(periods_dict[period])AD"] .- dfBKG_dict["dfBKG_$(period)_EH3"].Nbkg
        df_period = filter(row -> row["$(period)-AD Period"], df_exp)
        df_period = filter(row -> row["EH"] == "EH3", df_period)
    
        E_arr = dfIBD_dict["dfIBD_EH1"].Ec .+ 0.78
        L_matrix = df_period[:, ["D1", "D2", "L1", "L2", "L3", "L4"]]
        L_arr = vec(Matrix(L_matrix))
    
        Npred_EH3_after_best_fit_osc = dfIBD_dict["dfIBD_EH3"][:, "Npred_$(periods_dict[period])AD"]
        best_fit_prob_arr = bestift_osc.osc_prob(E_arr, L_arr, bestift_osc.params, anti=true)[:, :, 1, 1]'
        baseline_average_best_fit_prob_arr = vec(sum(best_fit_prob_arr ./ (L_arr .^ 2), dims=1) ./ sum(1 ./(L_arr .^ 2)))
        # unoscillated N predicted EH3:
        Npred_EH3_noosc = Npred_EH3_after_best_fit_osc ./ baseline_average_best_fit_prob_arr
    
        push!(E_arrs, E_arr)
        push!(L_arrs, L_arr)
        push!(Npred_EH3_nooscs, Npred_EH3_noosc)
        
        for reactor in reactor_list
            ad_contrib_this_period_this_reactor = (1 ./ df_period[:, reactor] .^ 2) .* df_period[:, "Target [kg]"] .* df_period[:, "Efficiency"]
            append!(ad_contribs_to_far_hall, ad_contrib_this_period_this_reactor)
        end
    end
    
    normalized_ad_contribs_to_far_hall = ad_contribs_to_far_hall ./ sum(ad_contribs_to_far_hall)
    covmat_prefactor = sum(normalized_ad_contribs_to_far_hall .^ 2)
    
    energy_bins = copy(dfBKG_dict["dfBKG_Six_EH3"].Emin)
    energy = copy(dfBKG_dict["dfBKG_Six_EH3"].Ec)
    push!(energy_bins, dfBKG_dict["dfBKG_Six_EH3"].Emax[end])

    observed = convert(Vector{Float64}, dfIBD_dict["dfIBD_EH3"].N)
    
    assets = (;
        E_arrs, 
        L_arrs, 
        Npred_EH3_nooscs,
        rel_unc_diag,
        corr_mat,
        covmat_prefactor,
        observed,
        period_list,
        energy_bins,
        energy
    )

end


function get_expected_per_period(params, period, physics, assets)
    E = assets.E_arrs[period]
    L = assets.L_arrs[period]
    prob_arr = physics.osc.osc_prob(E, L, params, anti=true)[:, :, 1, 1]'
    L2 = L .^ 2
    prob = vec(sum(prob_arr./L2, dims=1) ./ sum(1 ./L2))
    Npred_EH3_with_osc = assets.Npred_EH3_nooscs[period] .* prob
end

# Define function to give the expected events at the far hall (EH3)
function get_expected(params, physics, assets)
    sum([get_expected_per_period(params, period, physics, assets) for period in 1:length(assets.period_list)])
end

function get_forward_model(physics, assets)
    function forward_model(params)
        exp_events = get_expected(params, physics, assets)
        cov = Symmetric(Diagonal(exp_events .* assets.rel_unc_diag) * assets.corr_mat * Diagonal(exp_events .* assets.rel_unc_diag)) * assets.covmat_prefactor + Diagonal(exp_events)
        Distributions.MvNormal(exp_events, cov)
    end
end

function get_plot(physics, assets)

    function plot(params, data=assets.observed)
        
        m = mean(get_forward_model(physics, assets)(params))
        v = var(get_forward_model(physics, assets)(params))
    
        f = Figure()
        ax = Axis(f[1,1])
        
        plot!(ax, assets.energy, data, color=:black, label="Observed")
        stephist!(ax, assets.energy, weights=m, bins=assets.energy_bins, label="Expected")
        barplot!(ax, assets.energy, m .+ sqrt.(v), width=diff(assets.energy_bins), gap=0, fillto= m .- sqrt.(v), alpha=0.5, label="Standard Deviation")
        
        ax.ylabel="Counts"
        ax.title="Daya Bay"
        axislegend(ax, framevisible = false)
        
        
        ax2 = Axis(f[2,1])
        plot!(ax2, assets.energy, data ./ m, color=:black, label="Observed")
        hlines!(ax2, 1, label="Expected")
        barplot!(ax2, assets.energy, 1 .+ sqrt.(v) ./ m, width=diff(assets.energy_bins), gap=0, fillto= 1 .- sqrt.(v)./m, alpha=0.5, label="Standard Deviation")
        ylims!(ax2, 0.9, 1.1)
        
        ax.xticksvisible = false
        ax.xticklabelsvisible = false
        
        rowsize!(f.layout, 1, Relative(3/4))
        rowgap!(f.layout, 1, 0)
        
        ax2.xlabel="Eₚ (MeV)"
        ax2.ylabel="Counts/Expected"
    
        xlims!(ax, minimum(assets.energy_bins), maximum(assets.energy_bins))
        xlims!(ax2, minimum(assets.energy_bins), maximum(assets.energy_bins))
        
        ylims!(ax, 0, 60000)
        
        f
    
    end
end


function get_multiplot(physics, assets)

    function multiplot(params, data=assets.observed)
        
        p=params
        p_5 = merge(p, (N = ftype(5),))
        p_10 = merge(p, (N = ftype(10),))
        p_20 = merge(p, (N = ftype(20),))
        p_50 = merge(p, (N = ftype(50),))

        m_5= mean(get_forward_model(physics, assets)(p_5))
        m_10 = mean(get_forward_model(physics, assets)(p_10))
        m_20 = mean(get_forward_model(physics, assets)(p_20))
        m_50= mean(get_forward_model(physics, assets)(p_50))
        v_5 = var(get_forward_model(physics, assets)(p_5))
        v_10 = var(get_forward_model(physics, assets)(p_10))
        v_20 = var(get_forward_model(physics, assets)(p_20))
        v_50 = var(get_forward_model(physics, assets)(p_50))
    
        f = Figure()
        ax = Axis(f[1,1])
        
        plot!(ax, assets.energy, data, color=:black, label="Observed")
        stephist!(ax, assets.energy, weights=m, bins=assets.energy_bins, label="Expected")
        barplot!(ax, assets.energy, m .+ sqrt.(v), width=diff(assets.energy_bins), gap=0, fillto= m .- sqrt.(v), alpha=0.5, label="Standard Deviation")
        
        ax.ylabel="Counts"
        ax.title="Daya Bay"
        axislegend(ax, framevisible = false)
        
        
        ax2 = Axis(f[2,1])
        plot!(ax2, assets.energy, data ./ m, color=:black, label="Observed")
        hlines!(ax2, 1, label="Expected")
        barplot!(ax2, assets.energy, 1 .+ sqrt.(v) ./ m, width=diff(assets.energy_bins), gap=0, fillto= 1 .- sqrt.(v)./m, alpha=0.5, label="Standard Deviation")
        ylims!(ax2, 0.9, 1.1)
        
        ax.xticksvisible = false
        ax.xticklabelsvisible = false
        
        rowsize!(f.layout, 1, Relative(3/4))
        rowgap!(f.layout, 1, 0)
        
        ax2.xlabel="Eₚ (MeV)"
        ax2.ylabel="Counts/Expected"
    
        xlims!(ax, minimum(assets.energy_bins), maximum(assets.energy_bins))
        xlims!(ax2, minimum(assets.energy_bins), maximum(assets.energy_bins))
        
        ylims!(ax, 0, 60000)
        
        f
    
    end
end

end
