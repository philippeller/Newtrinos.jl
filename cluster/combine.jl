#!/usr/bin/env julia
# Combine all single-point results into a final profile result
# Usage: julia combine.jl

using JLD2
using Newtrinos

const TOTAL_POINTS = 31 * 31  # 961 points
const RESULTS_DIR = "cluster/results"
const OUTPUT_FILE = "juno_NND_profile_combined.jld2"

# Load all results
println("Loading results from $RESULTS_DIR...")
results = Vector{Any}(undef, TOTAL_POINTS)
for i in 1:TOTAL_POINTS
    file = joinpath(RESULTS_DIR, "point_$i.jld2")
    if !isfile(file)
        error("Missing result file: $file")
    end
    results[i] = JLD2.load(file, "opt_result")
end

# Extract components (matching _profile() output format)
llhs = [r[1] for r in results]
log_posteriors = [r[2] for r in results]
result_tuples = [r[3] for r in results]

# Build the same structure as _profile() returns
s = OrderedDict(key => [x[key] for x in result_tuples] for key in keys(first(result_tuples)))
s[:llh] = llhs
s[:log_posterior] = log_posteriors
combined = NamedTuple(s)

# Generate scanpoint values for the axes
# We need to reconstruct the grid from the first few results
# Since all points were computed with the same vars_to_scan=(r=31, N=31)
r_values = unique([result_tuples[i].r for i in 1:TOTAL_POINTS])
N_values = unique([result_tuples[i].N for i in 1:TOTAL_POINTS])

# Sort them (they should be in order from generate_scanpoints)
sort!(r_values)
sort!(N_values)

axes = (r=r_values, N=N_values)
final_result = Newtrinos.NewtrinosResult(axes=axes, values=combined)

# Save combined result
println("Saving combined result to $OUTPUT_FILE...")
JLD2.@save OUTPUT_FILE final_result
println("Done! Combined result has:")
println("  - $(length(r_values)) r values")
println("  - $(length(N_values)) N values")
println("  - $(TOTAL_POINTS) total points")
