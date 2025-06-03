module kamland

using DataFrames
using CSV
using Distributions
using LinearAlgebra
using Statistics
using DataStructures
using DataStructures
using BAT
using CairoMakie
using Logging
import ..Newtrinos

@kwdef struct KamLAND <: Newtrinos.Experiment
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
    return KamLAND(
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
        kamland_energy_scale = 0.,
        kamland_geonu_scale = 0.,
        kamland_flux_scale = 0.,
        )
end

function get_priors()
    priors = (
        kamland_energy_scale = Truncated(Normal(0, 1.), -3, 3),
        kamland_geonu_scale = Uniform(-0.5, 0.5),
        kamland_flux_scale = Truncated(Normal(0, 1.), -3, 3),
        )
end

function get_assets(physics, datadir = @__DIR__)
    @info "Loading kamland data"

    # Important global constants
    ENERGY_RESOL_1MEV = 0.065 # Reduces as 1/sqrt(visible energy in MeV)
    RATE_UNC = 0.041 # Fully correlated event rate uncertainty
    ENERGY_SCALE_UNC = 0.019 # Uncertainty in the deposited prompt energy conversion
    ELECTRON_DENSITY_ROCK = 0.5 * 2.7 # g/cm3: approximate electron density
    NEUTRINO_POSITRON_ENERGY_SHIFT = 0.782 # MeV: energy shift between true neutrino energy and deposited positron energy
    
    no_osc = CSV.read(joinpath(datadir, "KamLAND_no_osc.csv"), DataFrame; header=false)[:, 2]
    bestfit_osc = CSV.read(joinpath(datadir, "KamLAND_best_fit_osc.csv"), DataFrame; header=false)[:, 2]
    bestfit_BG = CSV.read(joinpath(datadir, "KamLAND_best_fit_withBG.csv"), DataFrame; header=false)[:, 2]
    allBG = bestfit_BG .- bestfit_osc
    geonu = CSV.read(joinpath(datadir, "KamLAND_best_fit_geonu.csv"), DataFrame; header=false)[:, 2]
    df_reactors = CSV.read(joinpath(datadir, "Reactors.csv"), DataFrame)
    
    L = df_reactors.L
    fluxes = df_reactors.N_reactors ./ (L .^ 2)
    flux_factor = fluxes ./ sum(fluxes)
    Ep_bins = LinRange(0.9,8.5,19)[1:end-1]
    Ep = 0.5 * (Ep_bins[1:end-1] .+ Ep_bins[2:end])
    upsampling = 10
    Ep_bins_fine = LinRange(0.9,8.5,(length(Ep)+1) * upsampling + 1)[1:end-upsampling]
    Ep_fine = 0.5 * (Ep_bins_fine[1:end-1] .+ Ep_bins_fine[2:end])
    # these weights are according to very approximately a reactor nergy spectrum
    weights = pdf(LogNormal(log(3.25),0.38), Ep_fine);

    observed = round.(CSV.read(joinpath(datadir, "KamLAND_data.csv"), DataFrame; header=false)[:, 2])
    assets = (;
        observed,
        Ep,
        Ep_bins,
        Ep_fine,
        Ep_bins_fine,
        ENERGY_RESOL_1MEV,
        RATE_UNC,
        ENERGY_SCALE_UNC,
        ELECTRON_DENSITY_ROCK,
        NEUTRINO_POSITRON_ENERGY_SHIFT,
        L,
        upsampling,
        no_osc,
        weights,
        allBG,
        geonu,
        flux_factor,        
        )
end

function smear(E, p, sigma, weights; width=10, E_scale=1.0, E_bias=0.0)
    l = length(p)
    out = similar(p, l)
    for i in 1:l
        out[i] = 0.
        e = E[i] * E_scale + E_bias
        norm = 0.0
        for j in max(1, i - width):min(l, i + width)
            coeff = 1 / sigma[j] * exp(-0.5 * ((e - E[j]) / sigma[j])^2)
            norm += coeff * weights[j]
            out[i] += coeff * weights[j] * p[j]
        end
        out[i] /= norm
    end
    return copy(out)
end

function get_expected(params, physics, assets)
    E_fine = (assets.Ep_fine .+ assets.NEUTRINO_POSITRON_ENERGY_SHIFT) .* (1 .+ assets.ENERGY_SCALE_UNC .* params.kamland_energy_scale)

    BG = assets.allBG .+ assets.geonu .* (params.kamland_geonu_scale)

    prob_outer_fine = physics.osc.osc_prob(E_fine .* 1e-3, assets.L, params, anti=true)[:, :, 1, 1]'

    prob_fine = dropdims(sum(assets.flux_factor .* prob_outer_fine, dims=1), dims=1)
    Ep_resolutions = assets.ENERGY_RESOL_1MEV .* sqrt.(assets.Ep_fine)

    prob_smeared = smear(assets.Ep_fine, prob_fine, Ep_resolutions, assets.weights, width=10, E_scale=1.0, E_bias=0.0)
    
    # use uniform averages:
    #prob = dropdims(mean(reshape(prob_smeared, upsampling, :), dims=1), dims=1)

    # use some approximate flux weights to do a non uniform bin average
    #prob_rs = reshape(prob_smeared, upsampling, :)
    #weights_rs = reshape(pdf(LogNormal(log(3.25),0.38), Ep_fine), upsampling, :)
    prob = dropdims(mean(reshape(prob_smeared, assets.upsampling, :), dims=1), dims=1)

    exp_events_noBG = assets.no_osc .* prob .* (1 .+ params.kamland_flux_scale .* assets.RATE_UNC)
    return exp_events_noBG .+ BG
end

function get_forward_model(physics, assets)
    function forward_model(params)
        exp_events = get_expected(params, physics, assets)
        distprod(Poisson.(exp_events))
    end
end

function get_plot_old(physics, assets)

    function plot_old(params, data=assets.observed)
    
        m = mean(get_forward_model(physics, assets)(params))
        v = var(get_forward_model(physics, assets)(params))
        
        energy = assets.Ep
        energy_bins = assets.Ep_bins
    
        f = Figure()
        ax = Axis(f[1,1])
        
        plot!(ax, energy, data, color=:black, label="Observed")
        stephist!(ax, energy, weights=m, bins=energy_bins, label="Expected")
        barplot!(ax, energy, m .+ sqrt.(v), width=diff(energy_bins), gap=0, fillto= m .- sqrt.(v), alpha=0.5, label="Standard Deviation")
        
        ax.ylabel="Counts"
        ax.title="KamLAND"
        axislegend(ax, framevisible = false)
        
        
        ax2 = Axis(f[2,1])
        plot!(ax2, energy, data ./ m, color=:black, label="Observed")
        hlines!(ax2, 1, label="Expected")
        barplot!(ax2, energy, 1 .+ sqrt.(v) ./ m, width=diff(energy_bins), gap=0, fillto= 1 .- sqrt.(v)./m, alpha=0.5, label="Standard Deviation")
        ylims!(ax2, 0.3, 1.7)
        
        ax.xticksvisible = false
        ax.xticklabelsvisible = false
        
        rowsize!(f.layout, 1, Relative(3/4))
        rowgap!(f.layout, 1, 0)
        
        ax2.xlabel="Eₚ (MeV)"
        ax2.ylabel="Counts/Expected"
        
        xlims!(ax, minimum(energy_bins), maximum(energy_bins))
        xlims!(ax2, minimum(energy_bins), maximum(energy_bins))
        
        ylims!(ax, 0, 300)
        
        f
    end
end

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        # Define parameter values to compare (adjust these based on your physics)
        # For KamLAND, these might be different oscillation parameters
        param_values = [5, 10, 20, 50]  # N parameter values
        param_name = "N"  # Parameter name
        colors = [:red, :blue, :green, :orange]  # Different colors for each parameter value
        
        # Calculate all means and variances first
        all_means = []
        all_variances = []
        
        for val in param_values
            # Merge parameter - adjust the parameter name as needed
            p_val = merge(params, (Symbol(param_name) => Float64(val),))
            m = mean(get_forward_model(physics, assets)(p_val))
            v = var(get_forward_model(physics, assets)(p_val))
            push!(all_means, m)
            push!(all_variances, v)
        end
        
        # Generate individual plots for each parameter value
        for (i, val) in enumerate(param_values)
            m = all_means[i]
            v = all_variances[i]
            
            f = Figure()
            ax = Axis(f[1,1])
            
            plot!(ax, assets.Ep, data, color=:black, label="Observed")
            stephist!(ax, assets.Ep, weights=m, bins=assets.Ep_bins, label="Expected")
            barplot!(ax, assets.Ep, m .+ sqrt.(v), width=diff(assets.Ep_bins), gap=0, fillto= m .- sqrt.(v), alpha=0.5, label="Standard Deviation")
            
            ax.ylabel = "Counts"
            ax.title = "KamLAND ($param_name = $val)"
            axislegend(ax, framevisible = false)
            
            ax2 = Axis(f[2,1])
            plot!(ax2, assets.Ep, data ./ m, color=:black, label="Observed")
            hlines!(ax2, 1, label="Expected")
            barplot!(ax2, assets.Ep, 1 .+ sqrt.(v) ./ m, width=diff(assets.Ep_bins), gap=0, fillto= 1 .- sqrt.(v)./m, alpha=0.5, label="Standard Deviation")
            ylims!(ax2, 0.3, 1.7)
            
            ax.xticksvisible = false
            ax.xticklabelsvisible = false
            
            rowsize!(f.layout, 1, Relative(3/4))
            rowgap!(f.layout, 1, 0)
            
            ax2.xlabel = "Eₚ (MeV)"
            ax2.ylabel = "Counts/Expected"
        
            xlims!(ax, minimum(assets.Ep_bins), maximum(assets.Ep_bins))
            xlims!(ax2, minimum(assets.Ep_bins), maximum(assets.Ep_bins))
            
            ylims!(ax, 0, 300)  # Adjusted for KamLAND count range
            
            display(f)
            #save("/home/sofialon/Newtrinos.jl/natural plot/kamland/kamland_data_NNM_$(param_name)_$val.png", f)
        end
        
        # Generate comparison plot with all parameter values
        f_comp = Figure()
        ax_comp = Axis(f_comp[1,1])
        
        # Plot observed data
        scatter!(ax_comp, assets.Ep, data, color=:black, label="Observed")
        
        # Plot all expected values
        for (i, val) in enumerate(param_values)
            m = all_means[i]
            v = all_variances[i]
            stephist!(ax_comp, assets.Ep, weights=m, bins=assets.Ep_bins, 
                    color=colors[i], label="Expected $param_name=$val")
            # Add uncertainty bands
            barplot!(ax_comp, assets.Ep, m .+ sqrt.(v), width=diff(assets.Ep_bins), 
                    gap=0, fillto= m .- sqrt.(v), alpha=0.2, color=colors[i])
        end
        
        ax_comp.ylabel = "Counts"
        ax_comp.xlabel = "Eₚ (MeV)"
        ax_comp.title = "KamLAND - Comparison of All $param_name Values"
        axislegend(ax_comp, framevisible = false, position = :rt)
        
        xlims!(ax_comp, minimum(assets.Ep_bins), maximum(assets.Ep_bins))
        ylims!(ax_comp, 0, 300)  # Adjusted for KamLAND count range
        
        display(f_comp)
        #save("/home/sofialon/Newtrinos.jl/natural plot/kamland/kamland_data_NNM_$(param_name)_comp.png", f_comp)
        
        # Generate ratio comparison plot
        f_ratio = Figure()
        ax_ratio = Axis(f_ratio[1,1])
        
        # Plot ratios for all parameter values
        for (i, val) in enumerate(param_values)
            m = all_means[i]
            v = all_variances[i]
            lines!(ax_ratio, assets.Ep, data ./ m, color=colors[i], label="Data/Expected $param_name=$val")
            # Add uncertainty bands for ratios
            barplot!(ax_ratio, assets.Ep, 1 .+ sqrt.(v) ./ m, width=diff(assets.Ep_bins), 
                    gap=0, fillto= 1 .- sqrt.(v)./m, alpha=0.2, color=colors[i])
        end
        
        hlines!(ax_ratio, 1, color=:black, linestyle=:dash, label="Unity")
        
        ax_ratio.ylabel = "Data/Expected"
        ax_ratio.xlabel = "Eₚ (MeV)"
        ax_ratio.title = "KamLAND - Ratio Comparison of All $param_name Values"
        axislegend(ax_ratio, framevisible = false, position = :rt)
        
        xlims!(ax_ratio, minimum(assets.Ep_bins), maximum(assets.Ep_bins))
        ylims!(ax_ratio, 0.3, 1.7)  # Kept the wider range from original KamLAND code
        
        display(f_ratio)
        #save("/home/sofialon/Newtrinos.jl/natural plot/kamland/kamland_data_NNM_$(param_name)_ratio.png", f_ratio)
    end
    
end
end



