using Plots, Statistics, Random, LinearAlgebra, StatsBase
using Printf
using Newtrinos


# setting up the experiment to generate the asimov data

osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NND(),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg)

atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()

physics = (; osc, atm_flux, earth_layers);

experiments = (
 
    dayabay = Newtrinos.dayabay.configure(physics),
);

p = Newtrinos.get_params(experiments)



"""
Generate Latin Hypercube samples for efficient parameter space sampling
"""
function latin_hypercube_sampling(n_samples, n_dimensions; seed=42)
    Random.seed!(seed)
    
    samples = zeros(n_samples, n_dimensions)
    
    for dim in 1:n_dimensions
        # Create equally spaced intervals [0, 1/n, 2/n, ..., (n-1)/n]
        intervals = (0:(n_samples-1)) / n_samples
        # Add random jitter within each interval
        jittered = intervals .+ rand(n_samples) / n_samples  
        # Randomly permute
        samples[:, dim] = shuffle(jittered)
    end
    
    return samples
end

"""
Comprehensive Asimov sensitivity analysis
"""
function asimov_sensitivity_analysis(experiments, p , r_range, N_range; 
                                   sampling_method=:lhs, n_samples=20, grid_size=31)
    
    println("ASIMOV SENSITIVITY ANALYSIS")
    println("="^60)
    println("Parameter r range: $(r_range[1]) to $(r_range[2])")
    println("Parameter N range: $(N_range[1]) to $(N_range[2])")
    println("Sampling method: $(sampling_method)")
    
    if sampling_method == :lhs
        println("Number of samples: $(n_samples)")
        
        # Latin Hypercube Sampling
        lhs_samples = latin_hypercube_sampling(n_samples, 2)
        
        # Scale to parameter ranges
        r_vals = lhs_samples[:, 1] .* (r_range[2] - r_range[1]) .+ r_range[1]
        N_vals = lhs_samples[:, 2] .* (N_range[2] - N_range[1]) .+ N_range[1]
        N_vals = round.(Int, N_vals)  # N should be integer
        
    elseif sampling_method == :grid
        println("Grid size: $(grid_size) × $(grid_size)")
        
        # Grid sampling
        r_grid = range(r_range[1], r_range[2], length=grid_size)
        N_grid = range(N_range[1], N_range[2], length=grid_size)
        N_grid = round.(Int, N_grid)
        
        r_vals = Float64[]
        N_vals = Int[]
        
        for r in r_grid
            for N in N_grid
                push!(r_vals, r)
                push!(N_vals, N)
            end
        end
    end
    
    println("Generating Asimov dataset...")
    println("Running $(length(r_vals)) simulations...")
    
    # Generate Asimov measurements
    measurements = Float64[]
    simulation_data = []
    
    progress_interval = max(1, length(r_vals) ÷ 10)
    
    for (i, (r, N)) in enumerate(zip(r_vals, N_vals))
        # Create modified parameters for this iteration
        p_new = merge(p, (r=r, N=N))
        
        # Generate Asimov data using your actual physics model
        asimov_data_array = Newtrinos.generate_asimov_data(experiments.dayabay, p_new)
        
        # Extract both the minimum count and which energy bin it occurs in
        measurement = minimum(asimov_data_array)
        min_bin_index = argmin(asimov_data_array)
        
        push!(measurements, measurement)
        push!(simulation_data, (r=r, N=N, measurement=measurement, min_bin=min_bin_index))
        
        if i % progress_interval == 0
            progress = round(100 * i / length(r_vals), digits=1)
            println("  Progress: $(progress)%")
        end
    end
    
    println("Simulation complete!")
    
    # Calculate sensitivity metrics
    correlation_r = cor(r_vals, measurements)
    correlation_N = cor(N_vals, measurements)
    
    # Spearman rank correlation (captures non-linear relationships)
    spearman_r = corspearman(r_vals, measurements)
    spearman_N = corspearman(N_vals, measurements)
    
    # Calculate variance metrics
    total_variance = var(measurements)
    
    
    # This helps identify independent vs coupled effects
    residuals_r = measurements .- mean(measurements) .- correlation_N * std.(measurements) / std(N_vals) .* (N_vals .- mean(N_vals))
    partial_corr_r = cor(r_vals, residuals_r)
    
    residuals_N = measurements .- mean(measurements) .- correlation_r * std.(measurements) / std(r_vals) .* (r_vals .- mean(r_vals))
    partial_corr_N = cor(N_vals, residuals_N)
    
    # Which energy bins have the minimum most often?
    min_bins = [data.min_bin for data in simulation_data]
    min_bin_frequencies = StatsBase.countmap(min_bins)
    most_common_min_bin = argmax(min_bin_frequencies)
    
    # Calculate sensitivity ranges
    measurement_range = maximum(measurements) .- minimum(measurements)
    
    results = (
        # Data
        r_vals = r_vals,
        N_vals = N_vals,
        measurements = measurements,
        min_bins = min_bins,
        simulation_data = simulation_data,
        
        # Linear correlations
        correlation_r = correlation_r,
        correlation_N = correlation_N,
        
        # Non-linear correlations
        spearman_r = spearman_r,
        spearman_N = spearman_N,
        
        # Partial correlations
        partial_corr_r = partial_corr_r,
        partial_corr_N = partial_corr_N,
        
        # Enhanced analysis
        min_bin_frequencies = min_bin_frequencies,
        most_common_min_bin = most_common_min_bin,
        
        # Variance metrics
        total_variance = total_variance,
        measurement_range = measurement_range,
        
        # Statistics
        mean_measurement = mean(measurements),
        std_measurement = std(measurements),
        n_simulations = length(measurements),
        
        # Parameter statistics
        r_mean = mean(r_vals),
        r_std = std(r_vals),
        N_mean = mean(N_vals), 
        N_std = std(N_vals)
    )
    
    return results
end
function create_sensitivity_plots(results)
    println("Creating visualization plots...")
    
    # Plot 1: Parameter r vs Measurement scatter plot
    p1 = scatter(results.r_vals, results.measurements,
                alpha=0.6, markersize=4,
                xlabel="Parameter r",
                ylabel="Minimum Bin Count",
                title="r scan",
                legend=false,
                color=:blue,
                left_margin=8Plots.mm,
                bottom_margin=6Plots.mm)
    
    # Add correlation text in bottom right
    annotate!(p1, maximum(results.r_vals) * 0.95, minimum(results.measurements) * 1.05, 
             text("Pearson: $(round(results.correlation_r, digits=3))\nSpearman: $(round(results.spearman_r, digits=3))", 
                  :left, :bottom, 10, :black))
    
    # Plot 2: Parameter N vs Measurement scatter plot
    p2 = scatter(results.N_vals, results.measurements,
                alpha=0.6, markersize=4,
                xlabel="Parameter N", 
                ylabel="Minimum Bin Count",
                title="N scan",
                legend=false,
                color=:red,
                left_margin=8Plots.mm,
                bottom_margin=6Plots.mm)
    
    # Add correlation text in bottom right
    annotate!(p2, maximum(results.N_vals) * 0.95, minimum(results.measurements) * 1.05, 
             text("Pearson: $(round(results.correlation_N, digits=3))\nSpearman: $(round(results.spearman_N, digits=3))", 
                  :left, :bottom, 10, :black))
    
    # Plot 3: 2D Parameter space comparison (r vs N, colored by minimum bin count)
    # Create interpolated surface for smooth visualization
    r_grid = range(minimum(results.r_vals), maximum(results.r_vals), length=30)
    N_grid = range(minimum(results.N_vals), maximum(results.N_vals), length=30)
    
    # Simple gridded interpolation for visualization
    measurement_grid = zeros(length(N_grid), length(r_grid))
    
    for (i, N_val) in enumerate(N_grid)
        for (j, r_val) in enumerate(r_grid)
            # Find nearest simulation points and average
            distances = sqrt.((results.r_vals .- r_val).^2 + ((results.N_vals .- N_val)/50).^2)  # Scale N for distance calc
            nearest_idx = sortperm(distances)[1:min(5, length(distances))]  # 5 nearest points
            measurement_grid[i, j] = mean(results.measurements[nearest_idx])
        end
    end
    
    p3 = heatmap(r_grid, N_grid, measurement_grid,
                xlabel="Parameter r",
                ylabel="Parameter N",
                title="r vs N Parameter Space",
                color=:viridis,
                left_margin=8Plots.mm,
                bottom_margin=6Plots.mm)
    
    # Add scatter points on top showing actual simulation points
    scatter!(p3, results.r_vals, results.N_vals,
            marker_z=results.measurements,
            markersize=3, alpha=0.8,
            markerstrokewidth=0.5,
            markerstrokecolor=:white,
            colorbar_title="Min Bin Count",
            legend=false)
    
    # Add correlation info as annotation
    annotate!(p3, maximum(results.r_vals) * 0.95, maximum(results.N_vals) * 0.95, 
             text("r sensitivity: $(round(abs(results.correlation_r), digits=3))\nN sensitivity: $(round(abs(results.correlation_N), digits=3))", 
                  :right, :top, 9, :white))
    
    # Combine all plots
    combined_plot = plot(p1, p2, p3,
                        layout=(1,3), 
                        size=(1800, 600),
                        plot_title="Dayabay Asimov Sensitivity Analysis: Minimum Bin Count",
                        left_margin=10Plots.mm,
                        bottom_margin=8Plots.mm)
    
    return combined_plot
end


"""
Print comprehensive sensitivity analysis summary 
"""
function print_sensitivity_summary(results)
    println("\n" * "="^60)
    println("ENHANCED SENSITIVITY ANALYSIS RESULTS")
    println("="^60)
    println("Observable: Minimum Energy Bin Count")
    
    @printf("Total simulations run: %d\n", results.n_simulations)
    @printf("Min bin count range: %.3f to %.3f (span: %.3f)\n", 
            minimum(results.measurements), maximum(results.measurements), results.measurement_range)
    @printf("Min bin count variance: %.3f\n", results.total_variance)
    
    println("\n" * "-"^40)
    println("CORRELATION ANALYSIS")
    println("-"^40)
    
    @printf("Parameter r (minimum bin count sensitivity):\n")
    @printf("  • Pearson correlation:  %+.3f\n", results.correlation_r)
    @printf("  • Spearman correlation: %+.3f\n", results.spearman_r)
    @printf("  • Partial correlation:  %+.3f\n", results.partial_corr_r)
    
    @printf("Parameter N (minimum bin count sensitivity):\n")
    @printf("  • Pearson correlation:  %+.3f\n", results.correlation_N)
    @printf("  • Spearman correlation: %+.3f\n", results.spearman_N)
    @printf("  • Partial correlation:  %+.3f\n", results.partial_corr_N)
    
    println("\n" * "-"^40)
    println("ENERGY BIN ANALYSIS")
    println("-"^40)
    
    println("Energy bins that had minimum count:")
    sorted_bins = sort(collect(results.min_bin_frequencies), by=x->x[2], rev=true)
    for (bin_idx, frequency) in sorted_bins[1:min(5, length(sorted_bins))]
        percentage = round(100 * frequency / results.n_simulations, digits=1)
        @printf("  • Bin %2d: %3d times (%.1f%%)\n", bin_idx, frequency, percentage)
    end
    
    @printf("\nMost common minimum bin: Bin %d (%.1f%% of simulations)\n", 
            results.most_common_min_bin, 
            round(100 * results.min_bin_frequencies[results.most_common_min_bin] / results.n_simulations, digits=1))
    
    # Check if minimum bin location is stable
    n_different_bins = length(results.min_bin_frequencies)
    if n_different_bins == 1
        println("Minimum always occurs in the same energy bin - very stable!")
    elseif n_different_bins <= 3
        println("Minimum occurs in only $(n_different_bins) different bins - fairly stable")
    else
        println("Minimum occurs in $(n_different_bins) different bins - location varies significantly")
    end
    
    println("\n" * "-"^40)
    println("SENSITIVITY INTERPRETATION")
    println("-"^40)
    
    # Interpret r sensitivity
    r_sensitivity = max(abs(results.correlation_r), abs(results.spearman_r))
    if r_sensitivity > 0.7
        r_level = "HIGHLY SENSITIVE"
        r_advice = "Small changes in r will significantly affect minimum bin count"
    elseif r_sensitivity > 0.4
        r_level = "MODERATELY SENSITIVE"
        r_advice = "Parameter r has noticeable impact on minimum bin count"
    else
        r_level = "LOW SENSITIVITY"
        r_advice = "Parameter r has little impact on minimum bin count"
    end
    
    # Interpret N sensitivity  
    N_sensitivity = max(abs(results.correlation_N), abs(results.spearman_N))
    if N_sensitivity > 0.7
        N_level = "HIGHLY SENSITIVE"
        N_advice = "Small changes in N will significantly affect minimum bin count"
    elseif N_sensitivity > 0.4
        N_level = "MODERATELY SENSITIVE"  
        N_advice = "Parameter N has noticeable impact on minimum bin count"
    else
        N_level = "LOW SENSITIVITY"
        N_advice = "Parameter N has little impact on minimum bin count"
    end
    
    println("Parameter r: $(r_level)")
    println("  └── $(r_advice)")
    println("Parameter N: $(N_level)")  
    println("  └── $(N_advice)")
    
    
end

"""
Save enhanced results to file
"""
function save_results(results, filename="asimov_sensitivity_dayabay.txt")
    open(filename, "w") do file
        println(file, "Enhanced Asimov Sensitivity Analysis Results")
        println(file, "Observable: Minimum Energy Bin Count")
        println(file, "="^60)
        println(file, "Parameter r correlation: $(results.correlation_r)")
        println(file, "Parameter N correlation: $(results.correlation_N)")
        println(file, "Parameter r Spearman: $(results.spearman_r)")
        println(file, "Parameter N Spearman: $(results.spearman_N)")
        println(file, "Total variance: $(results.total_variance)")
        println(file, "Number of simulations: $(results.n_simulations)")
        
        println(file, "\nEnergy Bin Analysis:")
        println(file, "Most common minimum bin: $(results.most_common_min_bin)")
        println(file, "Bin frequencies:")
        for (bin_idx, freq) in sort(collect(results.min_bin_frequencies), by=x->x[1])
            percentage = round(100 * freq / results.n_simulations, digits=1)
            println(file, "  Bin $(bin_idx): $(freq) times ($(percentage)%)")
        end
        
    end
    println("Enhanced results saved to: $(filename)")
end

# ============================================================================
# MAIN ANALYSIS EXECUTION
# ============================================================================



# Run the enhanced analysis
results = asimov_sensitivity_analysis(
    experiments,                # Your experiments NamedTuple
    p,                         # Your parameter NamedTuple  
    (0.0, 1.0),               # r range: 0 to 1
    (1, 100),                 # N range: 1 to 100  
    sampling_method=:grid,      # Use Latin Hypercube Sampling
    n_samples=200             # Number of parameter combinations to test
)

# Print comprehensive summary with enhanced analysis
print_sensitivity_summary(results)

# Create and display enhanced plots
sensitivity_plots = create_sensitivity_plots(results)
display(sensitivity_plots)

# Save enhanced results
save_results(results)

# Optional: Save the plot
savefig(sensitivity_plots, "dayabay_NND_asimov_sensitivity_plots.png")

