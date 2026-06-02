using LinearAlgebra
using Distributions
using Printf
using FileIO
import JLD2

using Revise
using Newtrinos

# Configure physics and experiments
osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.NNM(three_flavour=Newtrinos.osc.ThreeFlavour(ordering=:NO)),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
)

osc = Newtrinos.osc.configure(osc_cfg)
atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec = Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec)

experiments = (
    juno = Newtrinos.juno.configure(physics; livetime_years=6.0),
    tao = Newtrinos.tao.configure(physics; livetime_years=6.0),
)

p = Newtrinos.get_params(experiments)
all_priors = Newtrinos.get_priors(experiments)

# Add TAO-specific parameters
p_complete = merge(p, (
    tao_detection_epsilon=1.0,
    tao_res_a=0.015,
    tao_res_b=0.0,
    tao_res_c=0.0,
    tao_accidental_norm=1.0,
    tao_fast_neutron_norm=1.0,
    tao_lihe_norm=1.0,
))

m0 = 1e-2
p_complete_new = merge(p_complete, (m₀=m0,))

vars_to_scan = (r=31, N=31)

modified_priors = merge(p_complete_new, (
    N=DiscreteUniform(2,200),
    r=LogUniform(1e-8,1),
    junotao_flux_scale=Truncated(Normal(1.0, 0.02), 0.0, Inf),
    junotao_energy_scale=Truncated(Normal(1.0, 0.005), 0.0, Inf),
    juno_detection_epsilon=Truncated(Normal(1.0, 0.01), 0.0, Inf),
    juno_res_a=Truncated(Normal(0.0261, 0.0002), 0.0, Inf),
    juno_res_b=Truncated(Normal(0.0082, 0.0001), 0.0, Inf),
    juno_res_c=Truncated(Normal(0.0123, 0.0004), 0.0, Inf),
    junotao_shape_eps=Normal(0,1),
    juno_geo_shape_eps=Normal(0,1),
    juno_geo_rate_norm=Truncated(Normal(1.0, 0.30), 0.0, Inf),
    juno_accidental_norm=Truncated(Normal(1.0, 0.01), 0.0, Inf),
    juno_world_reactor_norm=Truncated(Normal(1.0, 0.02), 0.0, Inf),
    juno_lihe_norm=Truncated(Normal(1.0, 0.20), 0.0, Inf),
    juno_co_norm=Truncated(Normal(1.0, 0.50), 0.0, Inf),
    juno_atmnc_norm=Truncated(Normal(1.0, 0.50), 0.0, Inf),
    juno_fast_neutron_norm=Truncated(Normal(1.0, 1.0), 0.0, Inf),
    tao_detection_epsilon=Truncated(Normal(1.0, 0.005), 0.0, Inf),
    tao_res_a=Truncated(Normal(0.015, 0.015 * 0.05), 0.0, Inf),
    tao_res_b=Truncated(Normal(0.0, 0.001), 0.0, Inf),
    tao_res_c=Truncated(Normal(0.0, 0.001), 0.0, Inf),
    tao_accidental_norm=Truncated(Normal(1.0, 0.20), 0.0, Inf),
    tao_fast_neutron_norm=Truncated(Normal(1.0, 0.30), 0.0, Inf),
    tao_lihe_norm=Truncated(Normal(1.0, 0.30), 0.0, Inf),
))

likelihood = Newtrinos.generate_likelihood(experiments)

cache_dir = "cache_test_simple"
mkpath(cache_dir)

println("\n=== STARTING PROFILE SCAN ===")
println("Grid: r=31, N=31 (961 points total)")
println("Experiments: JUNO + TAO")
println("Parameters to profile over: r, N")

# Test a single point first to estimate time
println("\n--- Testing first point ---")
using Newtrinos: generate_scanpoints
values, scanpoints = generate_scanpoints(vars_to_scan, modified_priors)

println("First point: N=$(scanpoints[1].N.val), r=$(scanpoints[1].r.val)")
@time begin
    GC.gc()
    mem_before = Base.summarysize(likelihood) / 1024 / 1024
    opt_result = Newtrinos.find_mle_cached(likelihood, scanpoints[1], p_complete_new, cache_dir)
    mem_after = Base.summarysize(likelihood) / 1024 / 1024
    println("  Memory delta: $(round(mem_after - mem_before, digits=2)) MB")
end

# Test a high-N point
println("\n--- Testing high-N point (N=200) ---")
high_N_idx = findfirst(x -> x.N.val == 200, scanpoints)
if !isnothing(high_N_idx)
    println("Point: N=$(scanpoints[high_N_idx].N.val), r=$(scanpoints[high_N_idx].r.val)")
    @time begin
        GC.gc()
        opt_result = Newtrinos.find_mle_cached(likelihood, scanpoints[high_N_idx], p_complete_new, cache_dir)
    end
end

# Test a mid-N point
println("\n--- Testing mid-N point (N=100) ---")
mid_N_idx = findfirst(x -> x.N.val >= 100, scanpoints)
if !isnothing(mid_N_idx)
    println("Point: N=$(scanpoints[mid_N_idx].N.val), r=$(scanpoints[mid_N_idx].r.val)")
    @time begin
        GC.gc()
        opt_result = Newtrinos.find_mle_cached(likelihood, scanpoints[mid_N_idx], p_complete_new, cache_dir)
    end
end

# Test a low-N point
println("\n--- Testing low-N point (N=2) ---")
low_N_idx = findfirst(x -> x.N.val == 2, scanpoints)
if !isnothing(low_N_idx)
    println("Point: N=$(scanpoints[low_N_idx].N.val), r=$(scanpoints[low_N_idx].r.val)")
    @time begin
        GC.gc()
        opt_result = Newtrinos.find_mle_cached(likelihood, scanpoints[low_N_idx], p_complete_new, cache_dir)
    end
end

println("\n=== Estimating full scan time ===")
println("Based on single point timing, 961 points will take:")
println("Estimate: ~$(round(961 * mean([15.0, 20.0, 25.0]) / 3600, digits=2)) hours (rough estimate)")

println("\nNote: Actual time depends on N value distribution.")
println("Higher N points may be slower. Run with @threads for parallelization.")
