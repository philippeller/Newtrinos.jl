"""
Scanning script that reads from config.txt and performs scans
for selected experiments with specified model and ordering.

Configuration:
  - Edit config.txt to specify experiments, model, and ordering
  - Run: julia scan.jl
  - Results saved to plots/<experiment>/

Supported experiments: gerda, katrin, legend, juno, kamland, dayabay, minos, nova, tao
"""

using Revise
using Newtrinos
using Distributions
using CairoMakie
using FileIO
using ColorSchemes
using ImageFiltering
import JLD2

# ==================== CONFIG PARSING ====================

"""
    parse_config(filename::String) -> Dict

Parse configuration file and return parameters as dictionary.
"""
function parse_config(filename::String)
    config = Dict()
    
    open(filename) do f
        for line in eachline(f)
            # Skip empty lines and comments
            stripped = strip(line)
            if isempty(stripped) || startswith(stripped, "#")
                continue
            end
            
            # Parse key = value
            if contains(stripped, "=")
                parts = split(stripped, "=")
                key = strip(parts[1])
                value = strip(parts[2])
                
                # Try to parse as different types
                if value in ["true", "false"]
                    config[key] = parse(Bool, value)
                elseif tryparse(Int64, value) !== nothing
                    config[key] = parse(Int64, value)
                elseif tryparse(Float64, value) !== nothing
                    config[key] = parse(Float64, value)
                else
                    config[key] = value  # Keep as string
                end
            end
        end
    end
    
    return config
end

# ==================== SETUP ====================

println("\n" * "="^70)
println("  NEUTRINO PHYSICS SCANNING SYSTEM")
println("="^70)
println("\nReading configuration from N-Naturalness/config.txt...")
config = parse_config("N-Naturalness/config.txt")

model = config["model"]
ordering = Symbol(config["ordering"])
scan_resolution = config["scan_resolution"]
output_dir = config["output_dir"]
# Smoothing sigma for plotting (default 0.8)
sigma_val = get(config, "sigma", 0.8)
if isa(sigma_val, AbstractString)
    sigma = parse(Float64, String(sigma_val))
else
    sigma = Float64(sigma_val)
end

# Parse experiments (can be comma-separated)
experiments_str = config["experiments"]
experiments_list = [String(strip(e)) for e in split(experiments_str, ",")]

println("\nConfiguration:")
println("  Model: $model")
println("  Ordering: $ordering")
println("  Experiments: $(join(experiments_list, ", "))")
println("  Scan resolution: $scan_resolution")
println("  Output directory: $output_dir")
println("  Plot smoothing sigma: $sigma")

# Map experiment name to Newtrinos module name
experiment_modules = Dict(
    "gerda" => :gerda,
    "katrin" => :katrin,
    "legend" => :legend,
    "juno" => :juno,
    "kamland" => :kamland,
    "dayabay" => :dayabay,
    "minos" => :minos,
    "nova" => :nova,
    "tao" => :tao,
)

# ==================== SETUP OSCILLATION CONFIG ====================

function setup_oscillation()
    if model == "NNM"
        flavour_config = Newtrinos.osc.NNM(three_flavour=Newtrinos.osc.ThreeFlavour(ordering=ordering))
    elseif model == "NND"
        flavour_config = Newtrinos.osc.NND(three_flavour=Newtrinos.osc.ThreeFlavour(ordering=ordering))
    else
        error("Unknown model: $model. Choose NND or NNM.")
    end

    osc_cfg = Newtrinos.osc.OscillationConfig(
        flavour=flavour_config,
        propagation=Newtrinos.osc.Basic(),
        states=Newtrinos.osc.All(),
        interaction=Newtrinos.osc.SI()
    )

    osc = Newtrinos.osc.configure(osc_cfg)
    return osc_cfg, osc
end

# ==================== PLOTTING FUNCTION ====================

"""
    create_contour_plot(result, exp_name, scan_type, params_dict, axes_names; sigma=0.8)

Create a contour plot in the style of plots_final.ipynb with 2σ regions.
"""

function gaussian_smooth(Z; σ=1.0)
    # Check for all-NaN or all-Inf data
    finite_vals = Z[isfinite.(Z)]
    if isempty(finite_vals)
        @warn "gaussian_smooth: No finite values in data, skipping smoothing"
        return Z
    end
    
    # Clean data: replace NaN and Inf with max finite value
    Z_clean = copy(Z)
    max_finite = maximum(finite_vals)
    Z_clean[isnan.(Z_clean)] .= max_finite
    Z_clean[isinf.(Z_clean)] .= max_finite
    
    # Skip smoothing if sigma is 0 or very small
    if σ ≤ 0.01
        return Z_clean
    end
    
    kernel = Kernel.gaussian((σ, σ))
    try
        result = imfilter(Z_clean, kernel, Pad(:replicate))
        return result
    catch e
        @warn "gaussian_smooth imfilter failed: $e. Returning cleaned data"
        return Z_clean
    end
end

function create_contour_plot(result, exp_name, scan_type, params_dict, axes_names; sigma=0.8)
    try
        # Calculate likelihood difference
        dLLH = 2 * (maximum(result.values.log_posterior) .- result.values.log_posterior)
        
        # Check for invalid data
        if any(isnan.(dLLH)) || any(isinf.(dLLH))
            @warn "Skipping plot for $exp_name - $scan_type: likelihood surface contains NaN or Inf"
            return Figure()  # Return empty figure
        end
        
        # Calculate 2σ level
        sigma_2_level = quantile(Chisq(2), 1 - 2*ccdf(Normal(), 2))
    
    # Determine scales based on axis names
    xscale_func = identity
    yscale_func = identity
    
    axis_1_str = String(axes_names[1])
    axis_2_str = String(axes_names[2])
    
    if contains(axis_1_str, "r") || contains(axis_1_str, "m₀") || contains(axis_1_str, "m0")
        xscale_func = log10
    end
    if contains(axis_2_str, "r") || contains(axis_2_str, "m₀") || contains(axis_2_str, "m0")
        yscale_func = log10
    end
    
    # Create figure
    xlabel_text = (contains(axis_1_str, "m₀") || contains(axis_1_str, "m0")) ? "m0 (eV)" : String(axes_names[1])
    ylabel_text = (contains(axis_2_str, "m₀") || contains(axis_2_str, "m0")) ? "m0 (eV)" : String(axes_names[2])
    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1];
        xlabel = xlabel_text,
        ylabel = ylabel_text,
        title = "$(uppercase(exp_name)) - 90% C.L. excluded region $scan_type ($(model) $(ordering))",
        xminorticksvisible = true,
        xminorgridvisible = true,
        yminorticksvisible = true,
        yminorgridvisible = true,
        titlesize = 20,
        xlabelsize = 18,
        ylabelsize = 18,
        xticklabelsize = 14,
        yticklabelsize = 14,
        xscale = xscale_func,
        yscale = yscale_func
    )
    
    # Use Dark2_8 colorscheme
    color = ColorSchemes.Dark2_8.colors[1]
    color_excluded = RGBAf(color.r, color.g, color.b, 0.3)
    
    # Apply Gaussian smoothing to the likelihood surface and plot excluded region
    z_smooth = gaussian_smooth(dLLH; σ=sigma)
    max_dLLH = maximum(z_smooth)
    
    # Ensure axes are sorted for contourf!
    x_axis = sort(result.axes[1])
    y_axis = sort(result.axes[2])
    
    # Debug info
    @debug "Plot axes info:" size(z_smooth) length(x_axis) length(y_axis) issorted(x_axis) issorted(y_axis)
    
    # If axes were reordered, transpose z_smooth accordingly
    if issorted(result.axes[1]) && issorted(result.axes[2])
        z_plot = z_smooth
    elseif issorted(result.axes[1])
        # Only y needs reordering
        perm_y = sortperm(result.axes[2])
        z_plot = z_smooth[:, perm_y]
    elseif issorted(result.axes[2])
        # Only x needs reordering
        perm_x = sortperm(result.axes[1])
        z_plot = z_smooth[perm_x, :]
    else
        # Both need reordering
        perm_x = sortperm(result.axes[1])
        perm_y = sortperm(result.axes[2])
        z_plot = z_smooth[perm_x, :][:, perm_y]
    end
    
    # Verify data is valid before plotting
    if any(isnan.(z_plot))
        z_plot[isnan.(z_plot)] .= maximum(z_plot[isfinite.(z_plot)])
    end
    if any(isinf.(z_plot))
        z_plot[isinf.(z_plot)] .= maximum(z_plot[isfinite.(z_plot)])
    end
    
    contourf!(ax, x_axis, y_axis, z_plot,
        levels = [sigma_2_level, max_dLLH],
        colormap = cgrad([color_excluded, color]))

    contour!(ax, x_axis, y_axis, z_plot,
        levels = [sigma_2_level],
        color = color,
        linewidth = 2.5)
    
    # Legend removed by user request
    
    return fig
    
    catch e
        println("\n  ✗ Plotting error in create_contour_plot:")
        println("    $e")
        showerror(stdout, e, catch_backtrace())
        # Return an empty figure if plotting fails
        fig_empty = Figure()
        return fig_empty
    end
end

# ==================== SCANNING FUNCTION ====================

"""
    run_scan_for_experiment(exp_name::String) -> Bool

Run all three scans for a given experiment.
Returns true if successful, false otherwise.
"""
function run_scan_for_experiment(exp_name::String)
    
    println("\n" * "-"^70)
    println("EXPERIMENT: $(uppercase(exp_name))")
    println("-"^70)
    
    # Get module name
    if !haskey(experiment_modules, exp_name)
        println("✗ Unknown experiment: $exp_name")
        println("  Supported: $(join(keys(experiment_modules), ", "))")
        return nothing
    end
    
    exp_module = experiment_modules[exp_name]
    exp_module_obj = getfield(Newtrinos, exp_module)
    
    try
        # Setup oscillation configuration
        osc_cfg, osc = setup_oscillation()
        physics = (; osc)
        
        # Setup experiment
        experiments = (;)
        experiments = merge(experiments, (Symbol(exp_name) => exp_module_obj.configure(physics),))
        
        par = Newtrinos.get_params(experiments)
        all_priors = Newtrinos.get_priors(experiments)
        
        # Create output directory
        output_path_exp = joinpath(output_dir, exp_name)
        mkpath(output_path_exp)
        
        # ==================== SCAN 1: r vs N ====================
        println("\n  Scan 1/3: r vs N")
        
        m0_val = config["m0_scan1"]
        p1 = merge(par, (m₀=m0_val,))
        
        vars_to_scan_1 = (r=scan_resolution, N=scan_resolution)
        
        modified_priors_1 = (
            N = Uniform(2, 200),
            m₀ = all_priors.m₀,
            r = LogUniform(1e-8, 1),
            Δm²₂₁ = par.Δm²₂₁,
            Δm²₃₁ = all_priors.Δm²₃₁,
            δCP = par.δCP,
            θ₁₂ = par.θ₁₂,
            θ₁₃ = all_priors.θ₁₃,
            θ₂₃ = par.θ₂₃
        )
        
        likelihood_1 = Newtrinos.generate_likelihood(experiments)
        result_1 = Newtrinos.scan(likelihood_1, modified_priors_1, vars_to_scan_1, p1)
        
        filename_1 = joinpath(output_path_exp, "scan_$(exp_name)_rN_m0=$(m0_val)_$(model)_$(ordering)_1000new.jld2")
        JLD2.@save filename_1 result_1
        
        axes_1 = collect(keys(result_1.axes))
        fig_1 = create_contour_plot(result_1, exp_name, "r vs N", 
            Dict("m₀" => m0_val), axes_1; sigma=sigma)
        
        plot_filename_1 = joinpath(output_path_exp, "scan_$(exp_name)_rN_m0=$(m0_val)_$(model)_$(ordering)_1000new.png")
        save(plot_filename_1, fig_1)
        println("    ✓ Saved: $(basename(plot_filename_1))")
        
       # ==================== SCAN 2: r vs m₀ ====================
       println("\n  Scan 2/3: r vs m₀")
        
        N_val = config["N_scan2"]
        p2 = merge(par, (N=N_val,))
        
        vars_to_scan_2 = (r=scan_resolution, m₀=scan_resolution)
        
        modified_priors_2 = (
            N = p2.N,
            m₀ = LogUniform(1e-6, 0.1),
            r = LogUniform(1e-8, 1),
            Δm²₂₁ = par.Δm²₂₁,
            Δm²₃₁ = all_priors.Δm²₃₁,
            δCP = par.δCP,
            θ₁₂ = par.θ₁₂,
            θ₁₃ = all_priors.θ₁₃,
            θ₂₃ = par.θ₂₃
        )
        
        likelihood_2 = Newtrinos.generate_likelihood(experiments)
        result_2 = Newtrinos.scan(likelihood_2, modified_priors_2, vars_to_scan_2, p2)
        
        filename_2 = joinpath(output_path_exp, "scan_$(exp_name)_rm0_N=$(N_val)_$(model)_$(ordering)_1000new.jld2")
        JLD2.@save filename_2 result_2
        
        axes_2 = collect(keys(result_2.axes))
        fig_2 = create_contour_plot(result_2, exp_name, "r vs m₀", 
            Dict("N" => N_val), axes_2; sigma=sigma)
        
        plot_filename_2 = joinpath(output_path_exp, "scan_$(exp_name)_rm0_N=$(N_val)_$(model)_$(ordering)_1000new.png")
        save(plot_filename_2, fig_2)
        println("    ✓ Saved: $(basename(plot_filename_2))")
        
        # ==================== SCAN 3: m₀ vs N ====================
        println("\n  Scan 3/3: m₀ vs N")
        
        r_val = config["r_scan3"]
        p3 = merge(par, (r=r_val,))
        
        vars_to_scan_3 = (m₀=scan_resolution, N=scan_resolution)
        
        modified_priors_3 = (
            N = DiscreteUniform(2, 200),
            m₀ = all_priors.m₀,
            r = p3.r,
            Δm²₂₁ = par.Δm²₂₁,
            Δm²₃₁ = all_priors.Δm²₃₁,
            δCP = par.δCP,
            θ₁₂ = par.θ₁₂,
            θ₁₃ = all_priors.θ₁₃,
            θ₂₃ = par.θ₂₃
        )
        
        likelihood_3 = Newtrinos.generate_likelihood(experiments)
        result_3 = Newtrinos.scan(likelihood_3, modified_priors_3, vars_to_scan_3, p3)
        
        filename_3 = joinpath(output_path_exp, "scan_$(exp_name)_m0N_r=$(r_val)_$(model)_$(ordering)_1000new.jld2")
        JLD2.@save filename_3 result_3
        
        axes_3 = collect(keys(result_3.axes))
        fig_3 = create_contour_plot(result_3, exp_name, "m₀ vs N", 
            Dict("r" => r_val), axes_3; sigma=sigma)
        
        plot_filename_3 = joinpath(output_path_exp, "scan_$(exp_name)_m0N_r=$(r_val)_$(model)_$(ordering)_1000new.png")
        save(plot_filename_3, fig_3)
        println("    ✓ Saved: $(basename(plot_filename_3))")
        
        println("\n  ✓ $(uppercase(exp_name)) scans completed!")
        return nothing
        
    catch e
        println("\n  ✗ Error scanning $exp_name:")
        println("    $e")
        return nothing
    end 
end

# ==================== MAIN EXECUTION ====================

println("\nSetup complete!\n")

for exp_name in experiments_list
    run_scan_for_experiment(exp_name)
end

# ==================== SUMMARY ====================

println("\n" * "="^70)
println("  SCANNING COMPLETE")
println("="^70)

println("\nOutput directory: $output_dir/")
for exp_name in experiments_list
    exp_path = joinpath(output_dir, exp_name)
    if isdir(exp_path)
        println("  - $exp_name/")
    end
end

println("\n" * "="^70 * "\n")
