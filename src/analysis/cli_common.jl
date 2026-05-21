using Newtrinos
using FileIO

"""
    configure_experiments(experiment_list) -> NamedTuple

Configure a list of experiments using each experiment's built-in defaults.

Looks up each experiment by name in the `Newtrinos` module (case-insensitive),
calls its `configure()` method with no arguments, and collects the results into
a NamedTuple keyed by the lower-cased experiment name.

# Arguments
- `experiment_list`: iterable of experiment name strings (e.g. `["deepcore",
  "dayabay"]`).

# Returns
A `NamedTuple` mapping the experiment name to the configured
[`Newtrinos.Experiment`](@ref) object.

# Examples
```julia
experiments = configure_experiments(["deepcore", "dayabay"])
# (deepcore = ..., dayabay = ...)
```
"""
function configure_experiments(experiment_list)
    pairs = (Symbol(lowercase(exp)) => getproperty(getproperty(Newtrinos, Symbol(lowercase(exp))), :configure)() for exp in experiment_list)
    return (; pairs...)
end

"""
    configure_experiments(experiment_list, physics) -> NamedTuple

Configure a list of experiments with a shared physics module override.

Like the single-argument form but passes `physics` to each experiment's
`configure(physics)` method, allowing a custom oscillation or cross-section
configuration to be shared across all experiments.

# Arguments
- `experiment_list`: iterable of experiment name strings.
- `physics`: a physics object (or NamedTuple of physics objects) to pass to
  each experiment's `configure` method.

# Returns
A `NamedTuple` mapping the experiment name to the configured
[`Newtrinos.Experiment`](@ref) object with specified physics module.
"""
function configure_experiments(experiment_list, physics)
    pairs = (Symbol(lowercase(exp)) => getproperty(getproperty(Newtrinos, Symbol(lowercase(exp))), :configure)(physics) for exp in experiment_list)
    return (; pairs...)
end

"""
    save_result(result, name)

Save a [`NewtrinosResult`](@ref) to a JLD2 file.

Writes `result` under the key `"result"` to `"\$(name).jld2"` in the
current directory.

# Arguments
- `result::NewtrinosResult`: the scan or profile result to persist.
- `name::String`: base filename (without extension).

# Returns
`nothing`.
"""
function save_result(result, name)
    FileIO.save(name * ".jld2", Dict("result" => result))
end

"""
    plot_result(result, name, vars_to_scan; title=nothing)

Plot a scan or profile result and save it as a PNG file.

Renders the result using `plot!` (dispatched on [`NewtrinosResult`](@ref)),
labels the axes from `vars_to_scan`, and saves the figure to `"\$(name).png"`.
For a 1-D scan the y-axis is labelled `"-2ΔLLH"`; for a 2-D scan it shows
the name of the second scanned parameter.

!!! note
    `CairoMakie` (or another Makie backend) must be loaded in the caller's
    scope before calling this function.

# Arguments
- `result::NewtrinosResult`: output of [`scan`](@ref) or [`profile`](@ref).
- `name::String`: base filename for the output PNG (without extension).
- `vars_to_scan`: ordered collection whose keys are the scanned parameter names.
- `title::Union{String,Nothing}`: optional axis title. Defaults to `nothing`
  (no title).

# Returns
`nothing`.
"""
function plot_result(result, name, vars_to_scan; title=nothing)
    fig = Figure()
    ax = Axis(fig[1,1])
    plot!(ax, result)
    ax.xlabel = String(collect(keys(vars_to_scan))[1])
    if length(vars_to_scan) == 1
        ax.ylabel = "-2ΔLLH"
    else
        ax.ylabel = String(collect(keys(vars_to_scan))[2])
    end
    if !isnothing(title)
        ax.title = title
    end
    save(name * ".png", fig)
end
