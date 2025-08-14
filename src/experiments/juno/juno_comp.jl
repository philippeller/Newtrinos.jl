# %%
using LinearAlgebra
using Distributions
using LaTeXStrings
using Printf
using FileIO
import JLD2

# %%
using DataFrames
using CSV

# %%
using Revise
using Newtrinos
using Newtrinos.osc

# %%
# for NNaturalness simulated data

osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NNM(),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg)


atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()

physics = (; osc, atm_flux, earth_layers);

experiments = (

   juno= Newtrinos.juno.configure(physics),
);

# %%
p = Newtrinos.get_params(experiments)

# %%
toy_data_NN = Newtrinos.generate_toy_data(experiments.juno, p)
poisson_err_NN=sqrt.(toy_data_NN)
energy= experiments.juno.assets.E_bins_visible


# %%
# for SM simulated data

osc_cfg_SM = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.ThreeFlavour(),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg_SM)

physics_SM = (; osc);

experiments_SM = (

   juno= Newtrinos.juno.configure(physics_SM),
);

# %%
p_SM= Newtrinos.get_params(experiments_SM)

# %%
toy_data_SM = Newtrinos.generate_toy_data(experiments_SM.juno, p_SM)
poisson_err_SM=sqrt.(toy_data_SM)

# %%
using CairoMakie

# Create the figure
fig = Figure(size = (1200, 600))
ax = Axis(fig[1, 1], 
    xlabel = "Energy (GeV)",
    ylabel = "Event Counts", 
    title = "NNaturalness N=100, r=1 vs Standard Model Comparison")

# Plot both datasets with error bars
errorbars!(ax, energy, toy_data_NN, poisson_err_NN, 
          color = :blue, linewidth = 2, whiskerwidth = 10)
scatter!(ax, energy, toy_data_NN, 
        color = :blue, markersize = 8, label = "NNM")
lines!(ax, energy, toy_data_NN, 
       color = :blue, linewidth = 1.5, alpha = 0.7)

errorbars!(ax, energy, toy_data_SM, poisson_err_SM, 
          color = :red, linewidth = 2, whiskerwidth = 10)
scatter!(ax, energy, toy_data_SM, 
        color = :red, marker = :rect, markersize = 8, label = "SM")
lines!(ax, energy, toy_data_SM, 
       color = :red, linewidth = 1.5, alpha = 0.7)

# Add legend
axislegend(ax, position = :rt)

# Add grid
ax.xgridvisible = true
ax.ygridvisible = true
ax.xgridcolor = (:black, 0.1)
ax.ygridcolor = (:black, 0.1)

# Show the plot
#display(fig)

#save("/home/sofialon/Newtrinos.jl/aug_plots/juno/juno_comp_N=100_r=1_NND_6.png", fig)

# %%
using CairoMakie

# Create the figure
fig = Figure(size = (1200, 600))
ax = Axis(fig[1, 1], 
    xlabel = "Energy ",
    ylabel = "Event Counts", 
    title = "NNaturalness N=100, r=0 vs Standard Model Comparison")

# Plot New Model (NN) with transparent band
band!(ax, energy, toy_data_NN .- poisson_err_NN, toy_data_NN .+ poisson_err_NN, 
      color = (:blue, 0.4), label = "NND Uncertainty")
#lines!(ax, energy, toy_data_NN, 
       #color = :blue, linewidth = 2.5, label = "New Model (NN)")
scatter!(ax, energy, toy_data_NN, 
        color = :blue, markersize = 6, label = "NNM")

# Plot Standard Model (SM) with transparent band
band!(ax, energy, toy_data_SM .- poisson_err_SM, toy_data_SM .+ poisson_err_SM, 
      color = (:red, 0.4), label = "SM Uncertainty")
#lines!(ax, energy, toy_data_SM, 
       #color = :red, linewidth = 2.5, label = "Standard Model (SM)")
scatter!(ax, energy, toy_data_SM, 
        color = :red, marker = :rect, markersize = 6,label = "SM")

# Add legend
axislegend(ax, position = :rt)

# Add grid
ax.xgridvisible = true
ax.ygridvisible = true
ax.xgridcolor = (:black, 0.1)
ax.ygridcolor = (:black, 0.1)

# Show the plot
#display(fig)
save("/home/sofialon/Newtrinos.jl/aug_plots/juno/juno_comp_N=100_r=1_NNM_6.png", fig)

# %%
difference = toy_data_NN .- toy_data_SM

# %%
error_difference= sqrt.(poisson_err_NN.^2 .+ poisson_err_SM.^2)

# Create the figure
fig = Figure(size = (1200, 600))
ax = Axis(fig[1, 1], 
    xlabel = "Energy (GeV)",
    ylabel = "Residues (NNM - SM)", 
    title = "Difference NNaturalness N=100, r=1 vs Standard Model")

# Plot the difference with error bars

      
errorbars!(ax, energy, difference, error_difference, 
          color = :purple, linewidth = 2, whiskerwidth = 10, label = "Residues Uncertainty")

scatter!(ax, energy, difference, 
        color = :purple, markersize = 8)
#lines!(ax, energy, difference, 
       #color = :purple, linewidth = 1.5, alpha = 0.7)

# Add a horizontal line at zero for reference
hlines!(ax, [0], color = :gray, linestyle = :dash, linewidth = 2)

# Add grid
ax.xgridvisible = true
ax.ygridvisible = true
ax.xgridcolor = (:black, 0.1)
ax.ygridcolor = (:black, 0.1)

# Show the plot
#display(fig)

# Optional: save the plot
# save("neutrino_difference.png", fig)

# %%

# Check which differences are compatible with zero (within error bars)
compatible_with_zero = abs.(difference) .<= error_difference
not_compatible_with_zero = .!compatible_with_zero

# Create the figure
fig = Figure(size = (1200, 600))
ax = Axis(fig[1, 1], 
    xlabel = "Energy (GeV)",
    ylabel = "Event Count Difference (NNM- SM)", 
    title = " Difference NNaturalness N=100, r=1 vs Standard Model")

# Plot error bars for points compatible with zero (purple/gray)
if any(compatible_with_zero)
    errorbars!(ax, energy[compatible_with_zero], difference[compatible_with_zero], 
              error_difference[compatible_with_zero],
              color = :gray, linewidth = 2, whiskerwidth = 10)
    scatter!(ax, energy[compatible_with_zero], difference[compatible_with_zero], 
            color = :gray, markersize = 8, label = "Compatible with zero")
end

# Plot error bars for points NOT compatible with zero (red)
if any(not_compatible_with_zero)
    errorbars!(ax, energy[not_compatible_with_zero], difference[not_compatible_with_zero], 
              error_difference[not_compatible_with_zero],
              color = :purple, linewidth = 2, whiskerwidth = 10)
    scatter!(ax, energy[not_compatible_with_zero], difference[not_compatible_with_zero], 
            color = :purple, markersize = 10, label = "Significant difference")
end

# Connect all points with lines
#lines!(ax, energy, difference, 
       #color = :gray, linewidth = 1.5, alpha = 0.5)

# Add a horizontal line at zero for reference
hlines!(ax, [0], color = :gray, linestyle = :dash, linewidth = 2)


axislegend(ax, position = :rt)

# Add grid
ax.xgridvisible = true
ax.ygridvisible = true
ax.xgridcolor = (:black, 0.1)
ax.ygridcolor = (:black, 0.1)

# Calculate summary statistics
n_significant = sum(not_compatible_with_zero)
n_total = length(difference)

# Get axis limits to position text box
xlims = ax.finallimits[].widths[1]
ylims = ax.finallimits[].widths[2]



summary_text = "Significant differences: $n_significant/$n_total\nCompatible with zero: $(n_total - n_significant)/$n_total"

# Re-add the text on top of the rectangle
text!(ax, 0.98, 0.05, text = summary_text,
      align = (:right, :bottom),
      space = :relative,
      fontsize = 11,
      color = :black)

# Add grid
ax.xgridvisible = true
ax.ygridvisible = true
ax.xgridcolor = (:black, 0.1)
ax.ygridcolor = (:black, 0.1)



#display(fig)

save("/home/sofialon/Newtrinos.jl/aug_plots/juno/juno_comp_res_N=100_r=1_NNM_6.png", fig)


