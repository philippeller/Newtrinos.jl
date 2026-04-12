using Test
using Newtrinos
using DensityInterface
using Printf

# All reference likelihood values at default parameters.
# If a value changes beyond rtol=1e-6, it indicates a regression.
# To update: run the experiment and replace the reference value.
const ALL_REFERENCE_VALUES = (
    dayabay      = -168.90003273508322,
    kamland      = -63.111403860037875,
    minos        = -268.4046154382636,
    deepcore     = -950.625021522011,
    super_k      = -3712.2559541498677,
    orca         = -1164.3664813530054,
    coherent_csi = -574.3416032522271,
    coherent_lAr = -1754.9946939034153,
)

# Parse --experiments from ARGS if present, otherwise run all
function select_experiments()
    idx = findfirst(==("--experiments"), ARGS)
    if idx === nothing
        return ALL_REFERENCE_VALUES
    end
    names = Symbol[]
    for i in (idx+1):length(ARGS)
        startswith(ARGS[i], "-") && break
        push!(names, Symbol(lowercase(ARGS[i])))
    end
    isempty(names) && error("--experiments requires at least one experiment name")
    for n in names
        haskey(ALL_REFERENCE_VALUES, n) || error("Unknown experiment: $n. Available: $(join(keys(ALL_REFERENCE_VALUES), ", "))")
    end
    return (; (n => ALL_REFERENCE_VALUES[n] for n in names)...)
end

selected = select_experiments()

@testset "Likelihood Regression" begin
    # Run all experiments in parallel using threads, then check results
    exp_names = collect(keys(selected))
    n_exps = length(exp_names)

    # Storage for results (thread-safe: each index written by one task)
    actual_llh = Vector{Float64}(undef, n_exps)
    ref_llh = Vector{Float64}(undef, n_exps)

    # Configure, evaluate likelihood for each experiment in parallel
    t_start = time()
    timings = Vector{Float64}(undef, n_exps)
    @sync for idx in 1:n_exps
        Threads.@spawn begin
            t0 = time()
            name = exp_names[idx]
            ref_llh[idx] = selected[name]
            mod = getproperty(Newtrinos, name)
            exp = mod.configure()
            experiments = (; name => exp)
            params = Newtrinos.get_params(experiments)
            likelihood = Newtrinos.generate_likelihood(experiments)
            actual_llh[idx] = logdensityof(likelihood, params)
            timings[idx] = time() - t0
            @info "  $(name) done on thread $(Threads.threadid()) in $(round(timings[idx], digits=1))s"
        end
    end
    t_total = time() - t_start
    t_sequential = sum(timings)
    @info @sprintf("Regression tests: %.1fs wall time (%.1fs sequential, %.1fx speedup from %d threads)",
                   t_total, t_sequential, t_sequential / t_total, Threads.nthreads())

    # Now run the assertions on the main thread
    results = NamedTuple{(:name, :ref, :actual, :rdiff, :status), Tuple{Symbol, Float64, Float64, Float64, Symbol}}[]

    for idx in 1:n_exps
        name = exp_names[idx]
        llh = actual_llh[idx]
        ref = ref_llh[idx]

        @testset "$name" begin
            diff = llh - ref
            rdiff = abs(diff) / abs(ref)

            if rdiff > 1e-6
                if llh > ref
                    push!(results, (name=name, ref=ref, actual=llh, rdiff=rdiff, status=:improved))
                    @warn "$name likelihood improved: $ref → $llh (rdiff=$rdiff). Update reference value."
                    @test_broken llh ≈ ref rtol=1e-6
                else
                    push!(results, (name=name, ref=ref, actual=llh, rdiff=rdiff, status=:regressed))
                    @test llh ≈ ref rtol=1e-6
                end
            else
                push!(results, (name=name, ref=ref, actual=llh, rdiff=rdiff, status=:ok))
                @test llh ≈ ref rtol=1e-6
            end
        end
    end

    # Print summary table
    green  = "\e[32m"
    orange = "\e[33m"
    red    = "\e[31m"
    bold   = "\e[1m"
    reset  = "\e[0m"

    println()
    println("$(bold)╔══════════════════════════════════════════════════════════════════════════════════════╗$(reset)")
    println("$(bold)║                         Likelihood Regression Summary                              ║$(reset)")
    println("$(bold)╠══════════════╦══════════════════╦══════════════════╦══════════════╦═════════════════╣$(reset)")
    println("$(bold)║ Experiment   ║ Reference        ║ Actual           ║ Rel. Diff    ║ Status          ║$(reset)")
    println("$(bold)╠══════════════╬══════════════════╬══════════════════╬══════════════╬═════════════════╣$(reset)")

    for r in results
        color = r.status == :ok ? green : r.status == :improved ? orange : red
        status_str = r.status == :ok ? "$(green)OK$(reset)" :
                     r.status == :improved ? "$(orange)IMPROVED$(reset)" :
                     "$(red)REGRESSED$(reset)"

        @printf("║ %-12s ║ %16.6f ║ %16.6f ║ %12.2e ║ %-24s║\n",
                r.name, r.ref, r.actual, r.rdiff, status_str)
    end

    println("$(bold)╚══════════════╩══════════════════╩══════════════════╩══════════════╩═════════════════╝$(reset)")
    println()
end
