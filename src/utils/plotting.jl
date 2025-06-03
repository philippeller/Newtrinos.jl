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

function plot!(ax, result::NewtrinosResult; max_llh=maximum(result.values.log_posterior), levels=1 .- 2*ccdf(Normal(), 1:3), label=nothing, color=:blue, linestyle=:solid, cmap=:Blues, filled=false, edge=true, transform_x=identity, transform_y=identity)
    neg2dllh = 2*(max_llh .- result.values.log_posterior)

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
    ax
end

function plot(result::NewtrinosResult; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, result; kwargs...)
    fig
end
    
end