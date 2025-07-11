module minos

using LinearAlgebra
using Distributions
using HDF5
using BAT
using DataStructures
using CairoMakie
using Logging
using Printf
import ..Newtrinos


@kwdef struct Minos <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function configure(physics)
    physics = (;physics.osc, physics.xsec)
    assets = get_assets(physics)
    return Minos(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)
    @info "Loading minos data"

    h5file = h5open(joinpath(datadir, "dataRelease.h5"), "r")

    channels = ["FDCC", "FDNC", "NDCC", "NDNC"]
    experiments = ["minos", "minosPlus"]

    L=735.0

    ch_data = Dict([
        channel => (
            observed = sum([read(h5file["data$(channel)_$(ex)_hist"]) for ex in experiments]),
            bin_edges = read(h5file["data$(channel)_minosPlus_bins"]),
            smearings = (
                E=(x -> L ./((x[1:end-1] .+ x[2:end]) ./ 2))(read(h5file["hRecoToTrue$(channel)SelectedNuMu_minos_bins1"])),
                energy_edges=read(h5file["hRecoToTrue$(channel)SelectedNuMu_minos_bins2"]),
                NuMu=sum([read(h5file["hRecoToTrue$(channel)SelectedNuMu_$(ex)_hist"]) for ex in experiments]),
                TrueNC=sum([read(h5file["hRecoToTrue$(channel)SelectedTrueNC_$(ex)_hist"]) for ex in experiments]),
                BeamNue=sum([read(h5file["hRecoToTrue$(channel)SelectedBeamNue_$(ex)_hist"]) for ex in experiments]),
                AppNue=sum([read(h5file["hRecoToTrue$(channel)SelectedAppNue_$(ex)_hist"]) for ex in experiments]),
                AppNuTau=sum([read(h5file["hRecoToTrue$(channel)SelectedAppNuTau_$(ex)_hist"]) for ex in experiments])
            ),
            L=L,
        )
        for channel in channels
    ])

   assets = (
        ch_data = ch_data,
        TotalCCCovar = (x->reshape(x, fill(Int(sqrt(length(x))), 2)...))(read(h5file["TotalCCCovar"])),
        TotalNCCovar = (x->reshape(x, fill(Int(sqrt(length(x))), 2)...))(read(h5file["TotalNCCovar"])),
        observed = (
            CC = ch_data["FDCC"].observed,
            NC = ch_data["FDNC"].observed,
        ),
    )

end


function get_expected_per_channel(params, physics, assets)
    # Minos baseline:
    s = assets.smearings
    p = physics.osc.osc_prob(s.E, [assets.L], params)
    NuMu = s.NuMu * p[:,[1],2,2]
    TrueNC = s.TrueNC * dropdims(sum(p[:,[1],2,1:3], dims=3), dims=3)
    BeamNue = s.BeamNue * p[:,[1],1,1]
    AppNue = s.AppNue * p[:,[1],2,1]
    AppNuTau = s.AppNuTau * p[:,[1],2,3]
    dropdims(NuMu + TrueNC + BeamNue + AppNue + AppNuTau, dims=2)
end

function forward_model_per_channel(params, physics, channel, assets)
   
    observed_far = assets.ch_data["FD"*channel].observed
    expected_far = get_expected_per_channel(params, physics, assets.ch_data["FD"*channel]) * physics.xsec.scale(:any, Symbol(channel), params)
    observed_near = assets.ch_data["ND"*channel].observed
    expected_near = get_expected_per_channel(params, physics, assets.ch_data["ND"*channel]) * physics.xsec.scale(:any, Symbol(channel), params)
    
    cov = channel == "CC" ? assets.TotalCCCovar : assets.TotalNCCovar

    tot = vcat(expected_far, expected_near)
    cov = (tot * tot') .* cov + diagm(tot)
        
    cov11 = cov[1:length(observed_far), 1:length(observed_far)]
    cov12 = cov[1:length(observed_far), end-length(observed_near)+1:end]
    cov21 = cov[end-length(observed_near)+1:end, 1:length(observed_far)]
    cov22 = cov[end-length(observed_near)+1:end, end-length(observed_near)+1:end]

    cov22inv = inv(cov22)

    x = cov12 * (cov22inv * (observed_near - expected_near))
    expected = expected_far + x

    cov = cov11 - cov12 * cov22inv * cov21

    Distributions.MvNormal(expected, (cov+cov')/2)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        distprod(
            CC = forward_model_per_channel(params, physics, "CC", assets),
            NC = forward_model_per_channel(params, physics, "NC", assets),
        )
    end
end

function get_plot_old(physics, assets)

    function plot_old(params, d=assets.observed)
        f = Figure()

        m = mean(get_forward_model(physics, assets)(params))
        v = var(get_forward_model(physics, assets)(params))
        
        # Calculate chi-square
        chisq_total = 0.0
        dof_total = 0
        
        for (i, ch) in enumerate([:CC, :NC])
        
            ax = Axis(f[1,i])
            energy_bins = assets.ch_data["FD"*String(ch)].bin_edges
            energy = 0.5 .* (energy_bins[1:end-1] .+ energy_bins[2:end])
            
            # Chi-square calculation for this channel
            # Using variance as uncertainty (Poisson-like assumption)
            chisq_ch = sum((d[ch] .- m[ch]).^2 ./ max.(v[ch], 1e-10))  # Avoid division by zero
            dof_ch = length(d[ch]) - length(params)

            chisq_total += chisq_ch
            dof_total += dof_ch
            

            plot!(ax, energy, d[ch] ./ diff(energy_bins), color=:black, label="Observed")
            stephist!(ax, energy, weights=m[ch] ./ diff(energy_bins), bins=energy_bins, label="Expected")
            barplot!(ax, energy, (m[ch] .+ sqrt.(v[ch])) ./ diff(energy_bins), width=diff(energy_bins), gap=0, fillto= (m[ch] .- sqrt.(v[ch])) ./ diff(energy_bins), alpha=0.5, label="Standard Deviation")
            
            ax.ylabel="Counts / GeV"
            ax.title="MINOS/MINOS+ Far Detector "*String(ch)
            axislegend(ax, framevisible = false)
            
            
            ax2 = Axis(f[2,i])
            plot!(ax2, energy, d[ch] ./ m[ch], color=:black, label="Observed")
            hlines!(ax2, 1, label="Expected")
            barplot!(ax2, energy, 1 .+ sqrt.(v[ch]) ./ m[ch], width=diff(energy_bins), gap=0, fillto= 1 .- sqrt.(v[ch])./m[ch], alpha=0.5, label="Standard Deviation")
            ylims!(ax2, 0.7, 1.3)
            
            ax.xticksvisible = false
            ax.xticklabelsvisible = false
            
            rowsize!(f.layout, 1, Relative(3/4))
            rowgap!(f.layout, 1, 0)
            
            ax2.xlabel="Reconstructed Energy (GeV)"
            ax2.ylabel="Counts/Expected"
            
            xlims!(ax, minimum(energy_bins), maximum(energy_bins))
            xlims!(ax2, minimum(energy_bins), maximum(energy_bins))
        
        end
        
        ylims!(f.content[1], 0, 800)
        ylims!(f.content[4], 0, 600)
        
        # Add chi-square text to the bottom right of the figure
        dof_total=dof_total-length(params)
        chisq_per_dof = chisq_total / dof_total
        chisq_text = @sprintf("χ² = %.2f\nNDF = %d\nχ²/NDF = %.2f", chisq_total, dof_total, chisq_per_dof)

        # Add text annotation to the bottom right corner
        text!(f.content[4], 0.97, 0.06, text=chisq_text, 
            align=(:right, :bottom), 
            space=:relative,
            fontsize=12,
            color=:black)
        
        f
    end

end


function get_plot( physics, assets)
    function plot(params, d=assets.observed)
        N_values = [5, 10, 20, 100]
        colors = [:red, :blue, :green, :orange]  # Different colors for each N
        
        # Calculate all means and variances first
        all_means = []
        all_variances = []
        
        for N in N_values
            p_N = merge(params, (N = Float64(N),))
            m = mean(get_forward_model(physics, assets)(p_N))
            v = var(get_forward_model(physics, assets)(p_N))
            push!(all_means, m)
            push!(all_variances, v)
        end
        
        # Generate individual plots for each N value
        for (i, N) in enumerate(N_values)
            m = all_means[i]
            v = all_variances[i]
            
            f = Figure()
            
            for (j, ch) in enumerate([:CC, :NC])
                energy_bins = assets.ch_data["FD"*String(ch)].bin_edges
                energy = 0.5 .* (energy_bins[1:end-1] .+ energy_bins[2:end])
                
                ax = Axis(f[1,j])
                plot!(ax, energy, d[ch] ./ diff(energy_bins), color=:black, label="Observed")
                stephist!(ax, energy, weights=m[ch] ./ diff(energy_bins), bins=energy_bins, label="Expected")
                barplot!(ax, energy, (m[ch] .+ sqrt.(v[ch])) ./ diff(energy_bins), width=diff(energy_bins), gap=0, fillto= (m[ch] .- sqrt.(v[ch])) ./ diff(energy_bins), alpha=0.5, label="Standard Deviation")
                
                ax.ylabel="Counts / GeV"
                ax.title="MINOS/MINOS+ Far Detector "*String(ch)*" (N = $N)"
                axislegend(ax, framevisible = false)
                
                ax2 = Axis(f[2,j])
                plot!(ax2, energy, d[ch] ./ m[ch], color=:black, label="Observed")
                hlines!(ax2, 1, label="Expected")
                barplot!(ax2, energy, 1 .+ sqrt.(v[ch]) ./ m[ch], width=diff(energy_bins), gap=0, fillto= 1 .- sqrt.(v[ch])./m[ch], alpha=0.5, label="Standard Deviation")
                ylims!(ax2, 0.7, 1.3)
                
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
                
                rowsize!(f.layout, 1, Relative(3/4))
                rowgap!(f.layout, 1, 0)
                
                ax2.xlabel="Reconstructed Energy (GeV)"
                ax2.ylabel="Counts/Expected"
                
                xlims!(ax, minimum(energy_bins), maximum(energy_bins))
                xlims!(ax2, minimum(energy_bins), maximum(energy_bins))
            end
            
            ylims!(f.content[1], 0, 800)
            ylims!(f.content[4], 0, 600)
            
            display(f)
            #save("/home/sofialon/Newtrinos.jl/natural_plot/minos_data_N_$N.png", f)
        end
        
        # Generate comparison plots with all N values for each channel
        for (ch_idx, ch) in enumerate([:CC, :NC])
            energy_bins = assets.ch_data["FD"*String(ch)].bin_edges
            energy = 0.5 .* (energy_bins[1:end-1] .+ energy_bins[2:end])
            
            f_comp = Figure()
            ax_comp = Axis(f_comp[1,1])
            
            # Plot observed data
            scatter!(ax_comp, energy, d[ch] ./ diff(energy_bins), color=:black, label="Observed")
            
            # Plot all expected values
            for (i, N) in enumerate(N_values)
                m = all_means[i]
                v = all_variances[i]
                stephist!(ax_comp, energy, weights=m[ch] ./ diff(energy_bins), bins=energy_bins, 
                        color=colors[i], label="Expected N=$N")
                # Add uncertainty bands
                barplot!(ax_comp, energy, (m[ch] .+ sqrt.(v[ch])) ./ diff(energy_bins), width=diff(energy_bins), 
                        gap=0, fillto= (m[ch] .- sqrt.(v[ch])) ./ diff(energy_bins), alpha=0.2, color=colors[i])
            end
            
            ax_comp.ylabel = "Counts / GeV"
            ax_comp.xlabel = "Reconstructed Energy (GeV)"
            ax_comp.title = "MINOS/MINOS+ Far Detector "*String(ch)*" - Comparison of All N Values"
            axislegend(ax_comp, framevisible = false, position = :rt)
            
            xlims!(ax_comp, minimum(energy_bins), maximum(energy_bins))
            ylims!(ax_comp, 0, ch == :CC ? 800 : 600)
            
            display(f_comp)
            save("/home/sofialon/Newtrinos.jl/natural_plot/minos_data_"*String(ch)*"_N_comp.png", f_comp)
            
            # Generate ratio comparison plot for this channel
            f_ratio = Figure()
            ax_ratio = Axis(f_ratio[1,1])
            
            # Plot ratios for all N values
            for (i, N) in enumerate(N_values)
                m = all_means[i]
                v = all_variances[i]
                lines!(ax_ratio, energy, d[ch] ./ m[ch], color=colors[i], label="Data/Expected N=$N")
                # Add uncertainty bands for ratios
                barplot!(ax_ratio, energy, 1 .+ sqrt.(v[ch]) ./ m[ch], width=diff(energy_bins), 
                        gap=0, fillto= 1 .- sqrt.(v[ch])./m[ch], alpha=0.2, color=colors[i])
            end
            
            hlines!(ax_ratio, 1, color=:black, linestyle=:dash, label="Unity")
            
            ax_ratio.ylabel = "Data/Expected"
            ax_ratio.xlabel = "Reconstructed Energy (GeV)"
            ax_ratio.title = "MINOS/MINOS+ Far Detector "*String(ch)*" - Ratio Comparison of All N Values"
            axislegend(ax_ratio, framevisible = false, position = :rt)
            
            xlims!(ax_ratio, minimum(energy_bins), maximum(energy_bins))
            ylims!(ax_ratio, 0.7, 1.3)
            
            display(f_ratio)
            save("/home/sofialon/Newtrinos.jl/profiled_plot/minos_data_"*String(ch)*"_N_ratio.png", f_ratio)
        end
        
        # Generate combined comparison plot (both CC and NC channels together)
        f_combined = Figure()
        
        for (j, ch) in enumerate([:CC, :NC])
            energy_bins = assets.ch_data["FD"*String(ch)].bin_edges
            energy = 0.5 .* (energy_bins[1:end-1] .+ energy_bins[2:end])
            
            ax_comp = Axis(f_combined[1,j])
            
            # Plot observed data
            scatter!(ax_comp, energy, d[ch] ./ diff(energy_bins), color=:black, label="Observed")
            
            # Plot all expected values
            for (i, N) in enumerate(N_values)
                m = all_means[i]
                v = all_variances[i]
                stephist!(ax_comp, energy, weights=m[ch] ./ diff(energy_bins), bins=energy_bins, 
                        color=colors[i], label="Expected N=$N")
                # Add uncertainty bands
                barplot!(ax_comp, energy, (m[ch] .+ sqrt.(v[ch])) ./ diff(energy_bins), width=diff(energy_bins), 
                        gap=0, fillto= (m[ch] .- sqrt.(v[ch])) ./ diff(energy_bins), alpha=0.2, color=colors[i])
            end
            
            ax_comp.ylabel = "Counts / GeV"
            ax_comp.xlabel = "Reconstructed Energy (GeV)"
            ax_comp.title = "MINOS/MINOS+ "*String(ch)*" - All N Values"
            if j == 2  # Only show legend on right panel
                axislegend(ax_comp, framevisible = false, position = :rt)
            end
            
            xlims!(ax_comp, minimum(energy_bins), maximum(energy_bins))
            ylims!(ax_comp, 0, ch == :CC ? 800 : 600)
            
            # Add ratio plots below
            ax_ratio = Axis(f_combined[2,j])
            
            for (i, N) in enumerate(N_values)
                m = all_means[i]
                v = all_variances[i]
                lines!(ax_ratio, energy, d[ch] ./ m[ch], color=colors[i], label="Data/Expected N=$N")
                barplot!(ax_ratio, energy, 1 .+ sqrt.(v[ch]) ./ m[ch], width=diff(energy_bins), 
                        gap=0, fillto= 1 .- sqrt.(v[ch])./m[ch], alpha=0.2, color=colors[i])
            end
            
            hlines!(ax_ratio, 1, color=:black, linestyle=:dash, label="Unity")
            
            ax_ratio.ylabel = "Data/Expected"
            ax_ratio.xlabel = "Reconstructed Energy (GeV)"
            
            xlims!(ax_ratio, minimum(energy_bins), maximum(energy_bins))
            ylims!(ax_ratio, 0.7, 1.3)
            
            # Format layout
            ax_comp.xticksvisible = false
            ax_comp.xticklabelsvisible = false
            rowsize!(f_combined.layout, 1, Relative(3/4))
            rowgap!(f_combined.layout, 1, 0)
        end
        
        display(f_combined)
       # save("/home/sofialon/Newtrinos.jl/natural_plot/minos_data_combined_N_comp.png", f_combined)
        
        return f_combined  # Return the final combined plot
    end
end

end  # module minos