#!/usr/bin/env julia
# Diagnostic script to test why profiling fails after point 5
# Tests individual points with memory monitoring to identify OOM or other errors

using Printf
using FileIO
import JLD2

# Parse range from command line or test all
const START_POINT = get(ENV, "START", 1) |> x -> parse(Int, x)
const END_POINT = get(ENV, "END", 10) |> x -> parse(Int, x)

println("Testing points $START_POINT to $END_POINT")

for i in START_POINT:END_POINT
    println("\n" * "="^60)
    println("Testing point $i...")
    
    # Run the single point computation in a separate process to catch OOM
    cmd = `julia --project=@. single_point.jl $i $(END_POINT)`
    
    # Redirect output and capture exit code
    output_file = "diagnose_output_$i.txt"
    error_file = "diagnose_error_$i.txt"
    
    result = run(pipeline(cmd, stdout=output_file, stderr=error_file))
    
    if result.exitcode == 0
        println("✓ Point $i: SUCCESS")
        # Check if result file was created
        result_file = "results/point_$i.jld2"
        if isfile(result_file)
            file_size = filesize(result_file) / 1024 / 1024  # MB
            @printf("  Result file: %.2f MB\n", file_size)
        else
            println("  WARNING: No result file created")
        end
    else
        println("✗ Point $i: FAILED (exit code: $(result.exitcode))")
        println("  Check $output_file and $error_file for details")
        
        # Print last lines of error file
        if isfile(error_file)
            error_lines = readlines(error_file)
            last_n = min(10, length(error_lines))
            println("  Last $last_n lines of stderr:")
            for line in error_lines[end-last_n+1:end]
                println("    $line")
            end
            
            # Check for common OOM indicators
            if any(occursin.("Out of memory", error_lines)) || 
               any(occursin.("OOM", error_lines)) ||
               any(occursin.("Allocation failed", error_lines))
                println("  *** OUT OF MEMORY DETECTED ***")
            end
        end
    end
    
    # Clean up diagnostic files
    isfile(output_file) && rm(output_file)
    isfile(error_file) && rm(error_file)
end

println("\n" * "="^60)
println("Diagnostic complete")
