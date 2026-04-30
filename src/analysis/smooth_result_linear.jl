"""
Apply a linear fit over the scan grid to all nuisance parameter arrays in a
NewtrinosResult. Replaces each field (except llh and log_posterior) with the
least-squares linear fit in the scan-axis coordinates, producing smooth tracks
suitable for seeding a refine_profile run.

Usage:
  julia --project src/analysis/smooth_result_linear.jl INPUT.jld2 OUTPUT.jld2
"""

using Newtrinos
using FileIO
using ArgParse
using Statistics

function parse_args_()
    s = ArgParseSettings()
    @add_arg_table s begin
        "input"
        help = "Input JLD2 file containing a NewtrinosResult under key 'result'"
        arg_type = String
        required = true

        "output"
        help = "Output JLD2 file for the linearly-smoothed result"
        arg_type = String
        required = true
    end
    parse_args(s)
end

args = parse_args_()
input_file  = args["input"]
output_file = args["output"]

@info "Loading $input_file"
result = FileIO.load(input_file)["result"]

axis_keys = keys(result.axes)
axis_vecs = [collect(Float64, result.axes[k]) for k in axis_keys]
N         = length(axis_vecs)
grid_size = Tuple(length(v) for v in axis_vecs)
n_total   = prod(grid_size)

@info "Grid: $(join(["$k=$(length(v))" for (k,v) in zip(axis_keys, axis_vecs)], " × ")), $n_total points total"

# Build full-grid coordinate arrays via broadcasting, one per axis
# Each grid[i] has shape grid_size, holding the i-th coordinate at every point
coord_grids = [reshape(v, ntuple(d -> d == i ? length(v) : 1, N)...) .*
               ones(Float64, grid_size)
               for (i, v) in enumerate(axis_vecs)]

# Design matrix: [1, x1, x2, ...] — shape (n_total, 1+N)
X = hcat(ones(Float64, n_total), [vec(g) for g in coord_grids]...)

# Fields to skip — leave llh and log_posterior untouched
const SKIP = (:llh, :log_posterior)

fitted_arrays = map(keys(result.values)) do k
    arr = result.values[k]
    if k in SKIP
        @info "  $k — kept as-is"
        return arr
    end
    y = vec(Float64.(arr))
    β = X \ y
    fitted = reshape(X * β, grid_size)
    residual_rms = sqrt(mean((y .- X*β).^2))
    @info "  $k — linear fit done  (residual RMS = $(round(residual_rms, sigdigits=3)))"
    fitted
end

new_meta = merge(result.meta, Dict("smoothed" => "linear_fit",
                                    "smooth_input" => input_file))
smoothed = Newtrinos.NewtrinosResult(axes=result.axes,
                                     values=NamedTuple{keys(result.values)}(fitted_arrays),
                                     meta=new_meta)

FileIO.save(output_file, Dict("result" => smoothed))
@info "Saved linearly-smoothed result to $output_file"
