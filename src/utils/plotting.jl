module plotting
using StatsBase
using LinearAlgebra
using Distributions
using DataStructures
using Makie
import Makie: plot, plot!
using Makie
using BAT
using ValueShapes
using ArraysOfArrays
using ColorSchemes
using PairPlots

import Newtrinos.NewtrinosResult

"""
    plot!(ax, result::NewtrinosResult;
          max_llh=maximum(result.values.log_posterior),
          levels=1 .- 2*ccdf(Normal(), 1:3),
          label=nothing, color=:blue, linestyle=:solid,
          cmap=:Blues, filled=false, edge=true,
          transform_x=identity, transform_y=identity) -> Axis

Plot a [`NewtrinosResult`](@ref) into an existing `Makie` `Axis`.

Dispatches on the dimensionality of `result.axes`:
- **1D**: plots `-2Δllh` as a line with horizontal dotted lines at the
  requested confidence levels.
- **2D**: plots confidence contours via `contour!` and optionally a filled
  version via `contourf!`.

Confidence levels are converted to `-2Δllh` thresholds using the appropriate
chi-squared distribution (`Chisq(1)` for 1D, `Chisq(2)` for 2D).

# Arguments
- `ax::Axis`: the Makie axis to plot into.
- `result::NewtrinosResult`: scan or profile result to visualize.
- `max_llh`: reference log-posterior value for computing `-2Δllh`; defaults
  to the maximum in `result.values.log_posterior`.
- `levels`: confidence levels to draw contours at; defaults to 1σ, 2σ, 3σ
  Gaussian intervals.
- `label`: legend label attached to the line or contour; default `nothing`.
- `color`: line/contour color; default `:blue`.
- `linestyle`: line style; default `:solid`.
- `cmap`: colormap used for the filled contours (2D only); default `:Blues`.
- `filled::Bool`: if `true`, draw filled contours via `contourf!` (2D only);
  default `false`.
- `edge::Bool`: if `true`, draw contour edges via `contour!` (2D only);
  default `true`.
- `transform_x`: function applied to the first axis before plotting;
  default `identity`.
- `transform_y`: function applied to the second axis before plotting;
  default `identity`.

# Returns
The modified `Axis` `ax`.
"""
function plot!(ax, result::NewtrinosResult; max_llh=maximum(result.values.log_posterior), levels=1 .- 2*ccdf(Normal(), 1:3), label=nothing, color=:blue, linestyle=:solid, cmap=:Blues, filled=false, edge=true, transform_x=identity, transform_y=identity)
    neg2dllh = 2*(max_llh .- result.values.log_posterior)

    if length(result.axes) == 1
        x = transform_x(result.axes[1])

        hlines!(ax, quantile(Chisq(1), levels), color=:black, linestyle=:dot)
        lines!(ax, x, neg2dllh, linewidth=2,
            color=color,
            linestyle=linestyle,
            label=label)
        
    elseif length(result.axes) == 2
        x = transform_x(result.axes[1])
        y = transform_y(result.axes[2])
        
        if filled
        contourf!(ax, x, y,
                neg2dllh,
                levels=quantile(Chisq(2), levels),
                colormap=cmap)
        end
        if edge
            contour!(ax, x, y, 
                neg2dllh, 
                levels=quantile(Chisq(2), levels),
                linewidth=2,
                color=color,
                linestyle=linestyle)
            lines!(ax, [NaN], [NaN], color = color, linestyle=linestyle, label = label)
        end
    else
        DimensionMismatch("Cannot plot contours in $(length(result.axes)) dimesions")
    end
        
    ax
end

"""
    plot(result::NewtrinosResult; kwargs...) -> Figure

Convenience wrapper that creates a new [`Figure`] and [`Axis`] and calls
[`plot!`](@ref) with all `kwargs`. Returns the `Figure`.
"""
function plot(result::NewtrinosResult; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, result; kwargs...)
    fig
end
function CairoMakie.plot(result::NewtrinosResult; title="Parameter Estimation Results", log=0, mass=0, values_to_plot=nothing, log_colormap=false)
    dLLH = 2 * (maximum(result.values.log_posterior) .- result.values.log_posterior)
    
    # Find best fit values
    best_idx = argmin(dLLH)
    best_fit = []
    for i in 1:length(result.axes)
        push!(best_fit, result.axes[i][best_idx[i]])
    end
    f = Figure(size=(800, 600))
    
    kwargs = (
        xlabel = String(keys(result.axes)[1]),
        ylabel = String(keys(result.axes)[2]),
        title = title,
        xminorticksvisible = true, 
        xminorgridvisible = true, 
        yminorticksvisible = true, 
        yminorgridvisible = true,
        titlesize = 20,           # Title font size
        xlabelsize = 18,          # X-axis label font size
        ylabelsize = 18,          # Y-axis label font size
        xticklabelsize = 18,      # X-axis tick label font size
        yticklabelsize = 18,      # Y-axis tick label font size
    )
    if log == 1
        kwargs = merge(kwargs, (xscale = log10,))
    end
    if mass == 1 
       kwargs = merge(kwargs, (xlabel = String(keys(result.axes)[1]) * " (eV)",))
    end    
    if log == 2
        kwargs = merge(kwargs, (yscale = log10,))
    end
     if log == 3
        kwargs = merge(kwargs, (xscale = log10, yscale = log10,))
    end
    if mass == 2
       kwargs = merge(kwargs, (ylabel = String(keys(result.axes)[2]) * " (eV)",))
    end    
    ax = Axis(f[1,1]; kwargs...)
    
    is_likelihood = values_to_plot === nothing
    if values_to_plot === nothing
        values_to_plot = dLLH
    end
    
    # Apply log to colormap if requested
    plot_values = values_to_plot
    colorbar_label = ""
    if is_likelihood
        if log_colormap
            plot_values = log10.(values_to_plot)
            colorbar_label = "log₁₀(dLLH)"
        else
            colorbar_label = "dLLH"
        end
    else
        if log_colormap
            plot_values = log10.(values_to_plot)
            colorbar_label = "log₁₀(ΛH)  (GeV)"
        else
            colorbar_label = "ΛH  (GeV)"
        end
    end
    
    # Create heatmap
    hm = heatmap!(ax, result.axes[1], result.axes[2], plot_values, 
        colormap = Reverse(:Greens))
    Colorbar(f[1, 2], hm, label=colorbar_label, width=15, labelsize=18, spinewidth=0, ticklabelsize=14)
    
    # If plotting likelihood, overlay confidence level contours
    if is_likelihood
        levels = quantile(Chisq(2), 1 .- 2*ccdf(Normal(), 1:3))
        
        # Plot contours with different line styles for each sigma level
        contour!(ax, result.axes[1], result.axes[2], dLLH, 
            levels=[levels[1]], 
            color=:black, linewidth=3)
        
        contour!(ax, result.axes[1], result.axes[2], dLLH, 
            levels=[levels[2]], 
            color=:black, linewidth=2)
        
        contour!(ax, result.axes[1], result.axes[2], dLLH, 
            levels=[levels[3]], 
            color=:black, linewidth=2, linestyle=:dot)
        
        # Add legend in column 3 (to the right of colorbar)
        Legend(f[1, 3],
            [LineElement(color=:black, linewidth=3),
             LineElement(color=:black, linewidth=2),
             LineElement(color=:black, linewidth=2, linestyle=:dot)],
            ["1σ", "2σ", "3σ"],
            framevisible=true,
            labelsize=18,
            rowgap=5)
    else
        # For custom values, add contour line at lambda = 10^4
        contour!(ax, result.axes[1], result.axes[2], values_to_plot, 
            levels=[1e4], 
            color=:red, linewidth=2.5, linestyle=:dash)
    end
    
    f
end

function corner(samples::DensitySampleVector; variables=nothing)

    println("WEIGHTS ARE IGNORED!!!")

    println(kwargs)
    
    if isnothing(variables)
        variables = keys(samples.v[1])
    end
    
    x = NamedTuple(Dict(var=>[x[var] for x in flatview(samples.v)] for var in variables));
    pairplot(x)


end
    

function plot!(ax, result::NewtrinosResult; max_llh=maximum(result.values.log_posterior), levels=1 .- 2*ccdf(Normal(), 1:3), label=["68%", "90%", "95%"], color=:blue, linestyle=:solid, cmap=:Blues, filled=false, edge=true, transform_x=identity, transform_y=identity)
    neg2dllh = 2*(max_llh .- result.values.log_posterior)

    if length(result.axes) == 1
        x = transform_x(result.axes[1])

        hlines!(ax, quantile(Chisq(1), levels), color=:black, linestyle=linestyle,
            label=label, linewidth=2)
        lines!(ax, x, neg2dllh, linewidth=2,
            color=color,
            linestyle=linestyle,
            label=label)
        
    elseif length(result.axes) == 2
        x = transform_x(result.axes[1])
        y = transform_y(result.axes[2])
        
        if filled
        contourf!(ax, x, y,
                neg2dllh,
                levels=quantile(Chisq(2), levels),
                colormap=cmap)
        end
        if edge
            contour!(ax, x, y, 
                neg2dllh, 
                levels=quantile(Chisq(2), levels),
                linewidth=2,
                color=color,
                linestyle=linestyle)
            lines!(ax, [NaN], [NaN], color = color, linestyle=linestyle, label = label)
        end
    else
        DimensionMismatch("Cannot plot contours in $(length(result.axes)) dimesions")
    end
        
    ax
end

    
end
