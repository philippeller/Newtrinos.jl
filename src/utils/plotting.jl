module plotting
using StatsBase
using LinearAlgebra
using Distributions
using DataStructures
using Makie
import Makie
using CairoMakie
using BAT
using ArraysOfArrays
using ColorSchemes
using PairPlots

import Newtrinos.NewtrinosResult




@recipe(CLPlot, NewtrinosResult) do scene
    Attributes(
        test_statistic = :log_posterior,
        filled = false,
        color = :black,
        colormap = (Reverse(:Blues), 0.7),
        alpha=0.7,
        levels=1 .- 2*ccdf(Normal(), 1:3),
    )
end


function Makie.plot!(p::CLPlot{<:Tuple{<:NewtrinosResult}})
    result = p[:NewtrinosResult][]
    ts = x = getproperty(result.values, p[:test_statistic][])
    dLLH = 2 * (maximum(ts) .- ts);
    if p[:filled][]
        contourf!(p, result.axes[1], result.axes[2], dLLH, levels=quantile(Chisq(2), p[:levels][]), colormap=p[:colormap][])
    end
    contour!(p, result.axes[1], result.axes[2], dLLH, levels=quantile(Chisq(2), p[:levels][]), color=p[:color][])
    return p

end





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


function CairoMakie.plot(result::NewtrinosResult; levels=1 .- 2*ccdf(Normal(), 0:3))

    dLLH = 2 * (maximum(result.values.log_posterior) .- result.values.log_posterior);
    #dLLH = 2 * (maximum(result.values.llh) .- result.values.llh);
    f = Figure()
    ax = Axis(f[1, 1], xlabel = String(keys(result.axes)[1]), ylabel = String(keys(result.axes)[2]), xminorticksvisible = true, xminorgridvisible = true, yminorticksvisible = true, yminorgridvisible = true)
    contourf!(ax, result.axes[1], result.axes[2], dLLH, levels=quantile(Chisq(2), levels), colormap=(Reverse(:Blues), 0.7))
    contour!(ax, result.axes[1], result.axes[2], dLLH, levels=quantile(Chisq(2), levels), color=:black)
    f
end

function corner(samples::DensitySampleVector; variables=nothing)

    peinrln("WEIGHTS ARE IGNORED!!!")

    println(kwargs)
    
    if isnothing(variables)
        variables = keys(samples.v[1])
    end
    
    x = NamedTuple(Dict(var=>[x[var] for x in flatview(samples.v)] for var in variables));
    pairplot(x)


end
    
end