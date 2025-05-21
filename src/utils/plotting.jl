module plotting
using StatsBase
using LinearAlgebra
using Distributions
using DataStructures
using Makie
import Makie
using CairoMakie
using BAT
using ValueShapes
using ArraysOfArrays
using ColorSchemes
using PairPlots

import Newtrinos.NewtrinosResult

function flatten(samples::DensitySampleVector)
    variables = keys(varshape(samples))
    ls = [length(varshape(samples)[var]) for var in variables]
    
    x = OrderedDict{String, AbstractArray}()
    
    for i in 1:length(variables)
        var = variables[i]
        col = flatview(getproperty(samples.v, var))
        for k in 1:ls[i]
            if ls[i] > 1
                xlabel = String(var) * "[$k]"
                x[xlabel] = col[k, :]
            else
                xlabel=String(var)
                x[xlabel] = col
            end
        end
    end
    return x
end


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


function nice_scatter(ax, x, y; weight=1)
    # 2D data matrix: each row is a sample, each column is a variable
    data = hcat(x, y)  # 100 × 2
    
    # Mean and covariance
    μ = mean(data, ProbabilityWeights(weight), dims=1) |> vec  # Make it a vector of length 2
    Σ = cov(data, ProbabilityWeights(weight))
    
    # Eigen-decomposition
    eigvals, eigvecs = eigen(Σ)
    stds = sqrt.(clamp.(eigvals, 0, Inf))  # protect against negative values
    
    # Unit circle points
    θ = range(0, 2π, length=200)
    circle = [cos.(θ)'; sin.(θ)']  # 2 × 200 matrix
    
    # Transform unit circle into ellipse
    ellipse = eigvecs * Diagonal(stds) * circle .+ μ  # 2 × 200 matrix + 2-vector
    
    scatter!(ax, x, y, markersize=weight)
    lines!(ax, ellipse[1, :], ellipse[2, :], color=:red)
    
    # Mean cross
    hlines!(ax, [μ[2]], color=:black, linestyle=:dash)
    vlines!(ax, [μ[1]], color=:black, linestyle=:dash)
    
    for i in 1:2
        direction = eigvecs[:, i] * stds[i] * 3  # 3σ scaling for visibility
        p1 = μ .- direction
        p2 = μ .+ direction
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]],
               color=:red, linewidth=2, linestyle=:dot)
    end

    ax
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

    x = flatten(samples)

    if variables isa AbstractDict
        x = OrderedDict(variables[var]=>x[var] for var in keys(variables))
        variables = nothing
    end
    
    if isnothing(variables)
        variables = collect(keys(x))
    end

    ranges = OrderedDict()
    for var in variables
        m = mean(x[var], weights(samples.weight))
        s = std(x[var], weights(samples.weight))
        ranges[var] = (max(m-4s, minimum(x[var])), min(m+4s, maximum(x[var])))
    end
    
    N = length(variables)

    if isnothing(fig)
        fig = Figure(size=(N*200,N*200), fontsize=12)
    end

    grid = fig[1:N, 2:N+1] = GridLayout()

    for i in 1:N
        var = variables[i]
        ax = Axis(grid[i, i], aspect=1, ylabelvisible=false, xlabelvisible=false, yticklabelsvisible=false, yticksvisible=false)
        xlims!(ax, ranges[var])
        ax.xticklabelrotation = pi/2
        if i < N
            ax.xticklabelsvisible=false
            ax.xlabelvisible=false
        end
        quantile_plot(ax, x[var], samples.weight,  1 .- 2*ccdf(Normal(), 0:3), nbins=50, cmap=:Blues, rev=true)

        ylims!(ax, 0, nothing)
        ax.ygridvisible = false
        hidespines!(ax, :t, :l, :r)
        
        for j in i+1:N
            var2 = variables[j]
            ax2 = Axis(grid[j, i], aspect=1)
            ax2.xticklabelrotation = pi/2
            ax3 = Axis(grid[i, j], aspect=1)
            ax3.xticklabelsvisible=false
            ax3.xlabelvisible=false
            ax3.yticklabelsvisible=false
            ax3.ylabelvisible=false
            if j < N
                ax2.xticklabelsvisible=false
                ax2.xlabelvisible=false
            end
            if i > 1
                ax2.yticklabelsvisible=false
                ax2.ylabelvisible=false
            end
            Makie.hexbin!(ax2, x[var], x[var2], weights=samples.weight, colormap=:Blues, bins=50)

            nice_scatter(ax3, x[var2], x[var], weight=samples.weight*2)
            xlims!(ax2, ranges[var])
            xlims!(ax3, ranges[var2])
            ylims!(ax2, ranges[var2])
            ylims!(ax3, ranges[var])
        end

        Label(fig[i, 1], var, fontsize=12, rotation=π/2, tellheight=false)
        Label(fig[N+1, i + 1], var, fontsize=12, tellwidth=false)

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

    println("WEIGHTS ARE IGNORED!!!")
    
    if isnothing(variables)
        variables = keys(samples.v[1])
    end
    
    x = NamedTuple(Dict(var=>[x[var] for x in flatview(samples.v)] for var in variables));
    pairplot(x)


end
    
end