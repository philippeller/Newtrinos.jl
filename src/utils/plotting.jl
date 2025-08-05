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
import CairoMakie: plot

using StatsBase
using LinearAlgebra
using Distributions
using DataStructures
using CairoMakie
using BAT
using Newtrinos
using ArraysOfArrays
using ColorSchemes
using PairPlots

export plot

function quantile_plot(ax, vals, weight, levels; nbins=10, cmap=:heat, rev=true)
    edges = LinRange(minimum(vals), maximum(vals), nbins)
    h = fit(Histogram, vals, weights(weight), edges)
    p = h.weights ./ sum(h.weights)
    idx = reverse(sortperm(p))
    c = cumsum(p[idx])
    #pal = palette(cmap, length(levels)+1, rev=rev)
    pal = get(colorschemes[cmap], LinRange(rev ? 1 : 0,rev ? 0 : 1,length(levels)+1))
    colors = pal[searchsortedfirst.(Ref(levels), c)][sortperm(idx)]
    barplot!(ax, h.edges[1][1:end-1], p, width=diff(h.edges[1]), color=colors, gap=0, strokecolor = :black, strokewidth = 0.5)
end

function CairoMakie.plot(samples::DensitySampleVector; variables=nothing, fig=nothing)

    if isnothing(variables)
        variables = keys(samples.v[1])
    end

    N = length(variables)

    if isnothing(fig)
        fig = Figure(size=(N*200,N*200), fontsize=12)
    end
    
    for i in 1:N
        var = variables[i]
        ax = Axis(fig[i, i], xlabel=String(var), aspect=1, ylabelvisible=false, yticklabelsvisible=false, yticksvisible=false)
        if i < N
            ax.xticklabelsvisible=false
            ax.xlabelvisible=false
        end
        #hist!(ax, [x[var] for x in flatview(samples.v)], weights=samples.weight, normalization=:pdf)
        #CairoMakie.density!(ax, [x[var] for x in flatview(samples.v)], weights=samples.weight)
        quantile_plot(ax, [x[var] for x in flatview(samples.v)], samples.weight,  1 .- 2*ccdf(Normal(), 0:3), nbins=30, cmap=:Blues, rev=true)
        for j in i+1:N
            var2 = variables[j]
            ax2 = Axis(fig[j, i], xlabel=String(var), ylabel=String(var2), aspect=1)
            if j < N
                ax2.xticklabelsvisible=false
                ax2.xlabelvisible=false
            end
            if i > 1
                ax2.yticklabelsvisible=false
                ax2.ylabelvisible=false
            end
            Makie.hexbin!(ax2, [x[var] for x in flatview(samples.v)], [x[var2] for x in flatview(samples.v)], weights=samples.weight, colormap=:Blues, bins=30)
        end
    end

    fig

end

#=
function CairoMakie.plot_old(result::NewtrinosResult)

    dLLH = 2 * (maximum(result.values.log_posterior) .- result.values.log_posterior);
    #dLLH = 2 * (maximum(result.values.llh) .- result.values.llh);
    f = Figure()
    ax = Axis(f[1, 1], xlabel = String(keys(result.axes)[1]), ylabel = String(keys(result.axes)[2]), xminorticksvisible = true, xminorgridvisible = true, yminorticksvisible = true, yminorgridvisible = true)
    contourf!(ax, result.axes[1], result.axes[2], dLLH, levels=quantile(Chisq(2), 1 .- 2*ccdf(Normal(), 0:3)), colormap=(Reverse(:Blues), 0.7))
    contour!(ax, result.axes[1], result.axes[2], dLLH, levels=quantile(Chisq(2), 1 .- 2*ccdf(Normal(), 0:3)), color=:black)
    f
end =#

function CairoMakie.plot(result::NewtrinosResult; title="Parameter Estimation Results")

    dLLH = 2 * (maximum(result.values.log_posterior) .- result.values.log_posterior);
    
    # Find best fit values (indices of maximum log_posterior)
    best_idx = argmin(dLLH)  # Minimum dLLH corresponds to maximum log_posterior
    best_fit_x = result.axes[1][best_idx[1]]
    best_fit_y = result.axes[2][best_idx[2]]
    
    f = Figure()
    ax = Axis(f[1, 1], 
        xlabel = String(keys(result.axes)[1]), 
        ylabel = String(keys(result.axes)[2]), 
        title = title,
        xminorticksvisible = true, 
        xminorgridvisible = true, 
        yminorticksvisible = true, 
        yminorgridvisible = true
    )
    
    contourf!(ax, result.axes[1], result.axes[2], dLLH, 
        levels=quantile(Chisq(2), 1 .- 2*ccdf(Normal(), 0:3)), 
        colormap=(Reverse(:Blues), 0.7))
    contour!(ax, result.axes[1], result.axes[2], dLLH, 
        levels=quantile(Chisq(2), 1 .- 2*ccdf(Normal(), 0:3)), 
        color=:black)
    
    # Add best fit point
    scatter!(ax, [best_fit_x], [best_fit_y], color=:red, markersize=8, marker=:star5)
    
 
    # Create text with best fit values using absolute positioning
    var1_name = String(keys(result.axes)[1])
    var2_name = String(keys(result.axes)[2])
    
    text_content = "Best Fit Values:\n$(var1_name) = $(round(best_fit_x, digits=8))\n$(var2_name) = $(round(best_fit_y, digits=8))"
    
    # Get axis ranges for absolute positioning
    x_min, x_max = extrema(result.axes[1])
    y_min, y_max = extrema(result.axes[2])
    
    # Position text in data coordinates (top-right corner)
    text_x = x_min + 0.8 * (x_max - x_min)  # 2% from left
    text_y = y_max - 0.02 * (y_max - y_min)  # 2% from top
    
    text!(ax, text_content,
        position = (text_x, text_y),
        align = (:left, :top),
        fontsize = 12,
        color = :black
    )
    
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

function plot_old(result::NewtrinosResult; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, result; kwargs...)
    fig
end

    
end