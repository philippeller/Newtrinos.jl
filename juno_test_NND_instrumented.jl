using LinearAlgebra
using Distributions
using LaTeXStrings
using Printf
using FileIO
import JLD2
using DataFrames
using Setfield
using Accessors

using Revise
using Newtrinos
using CairoMakie

# For memory tracking
using Base: GC

osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NNM(three_flavour=Newtrinos.osc.ThreeFlavour(ordering=:NO)),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )

osc = Newtrinos.osc.configure(osc_cfg)

atm_flux = Newtrinos.atm_flux.configure()
erath_layers = Newtrinos.earth_layers.configure()
xsec=Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec);

experiments = (
 
   juno= Newtrinos.juno.configure(physics;livetime_years = 6.0),
   tao=Newtrinos.tao.configure(physics;livetime_years = 6.0),
);

p = Newtrinos.get_params(experiments)

all_priors = Newtrinos.get_priors(experiments)

p_complete = merge(p, (
        tao_detection_epsilon = 1.0,
        tao_res_a = 0.015,
        tao_res_b = 0.0,
        tao_res_c = 0.0,
        tao_accidental_norm = 1.0,
        tao_fast_neutron_norm = 1.0,
        tao_lihe_norm = 1.0,
    
))

m0_values=[1e-2]

for i in 1:length(m0_values)

    m0 =m0_values[i]
    p_complete_new= merge(p_complete, (m₀=m0,))
    
    vars_to_scan = (r=31, N=31)  
    
    modified_priors = merge(p_complete_new,(N=DiscreteUniform(2,200), r =LogUniform(1e-8,1), 
        junotao_flux_scale = Truncated(Normal(1.0, 0.02), 0.0, Inf), 
        junotao_energy_scale = Truncated(Normal(1.0, 0.005), 0.0, Inf),
        juno_detection_epsilon = Truncated(Normal(1.0, 0.01), 0.0, Inf),

        juno_res_a = Truncated(Normal(0.0261, 0.0002), 0.0, Inf),
        juno_res_b = Truncated(Normal(0.0082, 0.0001), 0.0, Inf),
        juno_res_c = Truncated(Normal(0.0123, 0.0004), 0.0, Inf),
        
        junotao_shape_eps = Normal(0,1),
        juno_geo_shape_eps = Normal(0,1),
        
        juno_geo_rate_norm = Truncated(Normal(1.0, 0.30), 0.0, Inf),
        juno_accidental_norm = Truncated(Normal(1.0, 0.01), 0.0, Inf),     
        juno_world_reactor_norm = Truncated(Normal(1.0, 0.02), 0.0, Inf),  
        juno_lihe_norm = Truncated(Normal(1.0, 0.20), 0.0, Inf),      
        juno_co_norm = Truncated(Normal(1.0, 0.50), 0.0, Inf),         
        juno_atmnc_norm = Truncated(Normal(1.0, 0.50), 0.0, Inf),   
        juno_fast_neutron_norm = Truncated(Normal(1.0, 1.0), 0.0, Inf), 
        
        tao_detection_epsilon = Truncated(Normal(1.0, 0.005), 0.0, Inf),
        tao_res_a = Truncated(Normal(0.015, 0.015 * 0.05), 0.0, Inf),
        tao_res_b = Truncated(Normal(0.0, 0.001), 0.0, Inf),
        tao_res_c = Truncated(Normal(0.0, 0.001), 0.0, Inf),
        
        tao_accidental_norm = Truncated(Normal(1.0, 0.20), 0.0, Inf),
        tao_fast_neutron_norm = Truncated(Normal(1.0, 0.30), 0.0, Inf),
        tao_lihe_norm = Truncated(Normal(1.0, 0.30), 0.0, Inf),  )
        )
    
    
    likelihood = Newtrinos.generate_likelihood(experiments);
    
    cache_dir = "cache_test"
    mkpath(cache_dir)

    # Custom profiling wrapper
    using Base.Threads
    
    # Get the scan points to understand the grid
    using Newtrinos: generate_scanpoints
    values, scanpoints = generate_scanpoints(vars_to_scan, modified_priors)
    
    println("\n=== PROFILE SCAN CONFIGURATION ===")
    println("Grid size: $(size(scanpoints))")
    println("Total points: $(length(scanpoints))")
    println("Parameters scanned: $(keys(vars_to_scan))")
    
    # Create arrays to store timing and memory info
    point_times = zeros(length(scanpoints))
    point_mem_allocs = zeros(length(scanpoints))
    point_N_values = [scanpoints[i].N.val for i in eachindex(scanpoints)]
    point_r_values = [scanpoints[i].r.val for i in eachindex(scanpoints)]
    
    # Measure baseline memory
    GC.gc()
    baseline_mem = Base.summarysize(likelihood) / 1024 / 1024  # MB
    println("\nBaseline likelihood memory: ~$(round(baseline_mem, digits=2)) MB")
    
    # Warm up
    println("\nWarming up first point...")
    @time opt_result_test = Newtrinos.find_mle_cached(likelihood, scanpoints[1], p_complete_new, cache_dir)
    
    println("\nStarting timed profile scan...")
    start_total = time()
    
    @threads for i in eachindex(scanpoints)
        GC.gc()
        mem_before = Sys.memory_allocation() / 1024 / 1024  # MB
        t_start = time()
        
        opt_result = Newtrinos.find_mle_cached(likelihood, scanpoints[i], p_complete_new, cache_dir)
        
        t_end = time()
        mem_after = Sys.memory_allocation() / 1024 / 1024
        
        point_times[i] = t_end - t_start
        point_mem_allocs[i] = mem_after - mem_before
        
        if i % 50 == 0 || i == 1
            N_val = scanpoints[i].N.val
            r_val = scanpoints[i].r.val
            println("Point $i/$(length(scanpoints)): N=$N_val, r=$(round(r_val, digits=4)), time=$(round(point_times[i], digits=2))s, mem=$(round(point_mem_allocs[i], digits=1))MB")
        end
    end
    
    total_time = time() - start_total
    
    println("\n=== PROFILE SCAN RESULTS ===")
    println("Total time: $(round(total_time, digits=2)) seconds ($(round(total_time/60, digits=2)) minutes)")
    println("Mean time per point: $(round(mean(point_times), digits=2)) seconds")
    println("Median time per point: $(round(median(point_times), digits=2)) seconds")
    println("Min time per point: $(round(minimum(point_times), digits=2)) seconds")
    println("Max time per point: $(round(maximum(point_times), digits=2)) seconds")
    println("\nMemory allocation:")
    println("Mean per point: $(round(mean(point_mem_allocs), digits=1)) MB")
    println("Median per point: $(round(median(point_mem_allocs), digits=1)) MB")
    println("Max per point: $(round(maximum(point_mem_allocs), digits=1)) MB")
    
    # Check if higher N points are slower
    println("\n=== ANALYSIS: N vs Time ===")
    using Statistics
    # Group by N value and compute mean time
    unique_Ns = unique(point_N_values)
    for N_val in sort(unique_Ns)
        idxs = point_N_values .== N_val
        mean_t = mean(point_times[idxs])
        count = sum(idxs)
        println("N=$N_val: mean_time=$(round(mean_t, digits=2))s, n_points=$count")
    end
    
    # Save timing data
    timing_data = Dict(
        "point_times" => point_times,
        "point_mem_allocs" => point_mem_allocs,
        "point_N_values" => point_N_values,
        "point_r_values" => point_r_values,
        "total_time" => total_time
    )
    JLD2.@save "juno_profile_timing_data.jld2" timing_data
    
    println("\nTiming data saved to juno_profile_timing_data.jld2")
    
    # Also run the original profile for comparison
    println("\nRunning original profile function for comparison...")
    @time result = Newtrinos.profile(likelihood, modified_priors, vars_to_scan, p_complete_new; cache_dir=cache_dir)
    
    JLD2.@save "./plots_17_05/junotao_full6_rN_NNM_NO_prof_log_instrumented.jld2" result
    
    img = CairoMakie.plot(result; title="Juno-Tao NNM NO - LogLikelihood r vs N, mo=$m0, profiled", log=0, mass=0)
    save("./plots_17_05/junotao_full6_rN_NNM_NO_prof_log_instrumented.png", img)
    
end
