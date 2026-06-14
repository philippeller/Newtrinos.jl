#!/usr/bin/env julia
# Memory test: runs single_point computation in-process with memory tracking
# Usage: julia --project=@. test_memory.jl [POINT_INDEX] [TOTAL_POINTS]

using Printf

const POINT_INDEX = get(ARGS, 1, "1") |> x -> parse(Int, x)
const TOTAL_POINTS = get(ARGS, 2, "10") |> x -> parse(Int, x)

println("Memory profiling for point $POINT_INDEX / $TOTAL_POINTS")

# Track memory before
mem_before = Base.gc_bytes() / 1024 / 1024  # MB
@printf("Memory before: %.2f MB\n", mem_before)

# Track GC stats
gc_stats_before = Base.gc_num()

try
    # Include and run the single point computation
    include("single_point.jl")
    
    mem_after = Base.gc_bytes() / 1024 / 1024
    gc_stats_after = Base.gc_num()
    
    @printf("Memory after: %.2f MB\n", mem_after)
    @printf("Memory delta: %.2f MB\n", mem_after - mem_before)
    @printf("GC allocations: %d\n", gc_stats_after.allocs - gc_stats_before.allocs)
    
    println("\n✓ SUCCESS")
    
catch e
    mem_after = Base.gc_bytes() / 1024 / 1024
    @printf("Memory at failure: %.2f MB\n", mem_after)
    @printf("Memory delta: %.2f MB\n", mem_after - mem_before)
    
    println("\n✗ FAILED with exception:")
    showerror(stdout, e, catch_backtrace())
    
    # Check if it looks like OOM
    if isa(e, OutOfMemoryError) || occursin("Out of memory", string(e))
        println("\n*** OUT OF MEMORY ERROR ***")
    end
end
