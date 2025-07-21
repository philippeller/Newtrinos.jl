
# PIPELINE FOR CONSTRAINING N AND r

using Newtrinos
using CairoMakie
using JLD2


function pipeline_plot(JLD_file::String,experiment::String,NNtype::String, Nmax::Int, Nfixed::Int)

    JLD2.@load JLD_file result

    #plot and save the image
    img = CairoMakie.plot(result)
    display("image/png", img)
    save("/home/sofialon/Newtrinos.jl/profiled plot/$(experiment)/$(experiment)_rN_m0=0.1_$(NNtype).png", img)

    #plot the logposterior value vs r

    Nindex = round(Int, 31/Nmax* Nfixed)
    println("Nindex: ", Nindex)

    #calsculate the best fit
    bf = Newtrinos.bestfit(result)

    # Calculate confidence intervals
    sigma_1_threshold = maximum(result.values.log_posterior[:, Nindex ]) - 0.5
    sigma_2_threshold = maximum(result.values.log_posterior[:, Nindex ]) - 2.0

    
    # find the N at which the log posterior reach the 2-sigma treshold

    N_values= result.axes.N
    log_post_values = result.values.log_posterior[31, :]
    N_2sigma = N_values[log_post_values .< sigma_2_threshold]

    Nfixed=round(Int, maximum(N_2sigma)*100/31)
    println("Nfixed: ", Nfixed)

    # Create the plot
    fig = Figure(resolution = (800, 600))
    ax = Axis(fig[1, 1],
        xlabel = "r",
        ylabel = "Log Posterior",
        title = "r vs Log posterior - N = $Nfixed - $experiment $NNtype",
        titlesize = 16,
        xlabelsize = 14,
        ylabelsize = 14
    )

    # Plot the main curve
    lines!(ax, result.axes.r, result.values.log_posterior[:, Nindex],
        color = :blue,
        linewidth = 2,
        label = "Log Posterior"
    )

    # Add confidence level lines
    hlines!(ax, [sigma_1_threshold], 
        color = :red, 
        linestyle = :dash, 
        linewidth = 2,
        label = "1σ"
    )

    hlines!(ax, [sigma_2_threshold], 
        color = :orange, 
        linestyle = :dash, 
        linewidth = 2,
        label = "2σ"
    )

    # Add legend
    axislegend(ax, position = :rb)  # right bottom

    display("image/png", fig)
    save("/home/sofialon/Newtrinos.jl/profiled plot/$(experiment)/$(experiment)_rLogpost_m0=0.1_$(NNtype)_N=$(Nfixed).png", fig)


end