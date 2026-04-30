using LinearAlgebra
using Distributions
using DensityInterface
using InverseFunctions
import ForwardDiff
import PolyesterForwardDiff
using BAT
using Optimization
using IterTools
using DataStructures
using ADTypes
using AutoDiffOperators
using ContentHashes
using ValueShapes
using FileIO
using FillArrays
import JLD2
using MeasureBase
using FunctionChains
using Accessors
using Logging
using ProgressMeter
using Dates
using LibGit2
import Mooncake
import Random
using Interpolations
using Statistics
using ..Newtrinos

const AD_BACKEND = Ref{Symbol}(:auto)

"""
    set_ad_backend(backend::Symbol)

Set the global AD backend. Valid values: `:auto`, `:forwarddiff`, `:polyester`, `:mooncake`.
"""
function set_ad_backend(backend::Symbol)
    backend in (:auto, :forwarddiff, :polyester, :mooncake) || error("Unknown AD backend: $backend. Choose from: auto, forwarddiff, polyester, mooncake")
    AD_BACKEND[] = backend
end

"""
    select_ad(n_params; threshold=12)

Choose AD backend for gradient-based optimization based on the global setting
(see [`set_ad_backend`](@ref)).

- `:auto` — ForwardDiff for ≤ threshold params, PolyesterForwardDiff otherwise
- `:forwarddiff` — standard chunked ForwardDiff
- `:polyester` — threaded chunked ForwardDiff (PolyesterForwardDiff)
- `:mooncake` — reverse-mode AD via Mooncake (constant overhead, lower memory)
"""
function select_ad(n_params; threshold=12)
    backend = AD_BACKEND[]
    if backend == :auto
        return n_params > threshold ? AutoPolyesterForwardDiff() : AutoForwardDiff()
    elseif backend == :forwarddiff
        return AutoForwardDiff()
    elseif backend == :polyester
        return AutoPolyesterForwardDiff()
    elseif backend == :mooncake
        return AutoMooncake(Mooncake.Config())
    end
end

# AD backend is set via set_ad_backend() + set_batcontext(ad = select_ad(n))
# Do NOT set a hardcoded default here — it overrides user selection

# ── Experiment Configuration ───────────────────────────────────────

"""
    configure_experiments(experiment_list)
    configure_experiments(experiment_list, physics)

Construct a NamedTuple of configured experiments from a list of experiment name strings.
Optionally pass a shared `physics` override.
"""
function configure_experiments(experiment_list)
    pairs = (Symbol(lowercase(exp)) => getproperty(getproperty(Newtrinos, Symbol(lowercase(exp))), :configure)() for exp in experiment_list)
    return (; pairs...)
end

function configure_experiments(experiment_list, physics)
    pairs = (Symbol(lowercase(exp)) => getproperty(getproperty(Newtrinos, Symbol(lowercase(exp))), :configure)(physics) for exp in experiment_list)
    return (; pairs...)
end

# ── Core Types ──────────────────────────────────────────────────────

"""
    NewtrinosResult(; axes, values, meta=Dict())

Container for scan/profile results. `axes` holds the scan grid coordinates,
`values` holds likelihood values and optimized parameters, `meta` holds execution metadata.
"""
@kwdef struct NewtrinosResult
    axes::NamedTuple
    values::NamedTuple
    meta::Dict = Dict()
end

# ── Utility Functions ───────────────────────────────────────────────

function sort_nt(nt::NamedTuple)
    keys_sorted = sort(collect(keys(nt)))
    values_sorted = getindex.(Ref(nt), keys_sorted)
    return NamedTuple{Tuple(keys_sorted)}(values_sorted)
end

"""
    safe_merge(nt_list::NamedTuple...)

Merge NamedTuples, checking that duplicate keys have equal values. Result is sorted by key.
"""
function safe_merge(nt_list::NamedTuple...)
    merged = NamedTuple()
    for nt in nt_list
        for (k, v) in pairs(nt)
            if haskey(merged, k)
                if merged[k] != v
                    error("Conflict on key '$k': $(merged[k]) ≠ $v")
                end
            end
        end
        merged = merge(merged, nt)
    end
    sort_nt(merged)
end

# ── Wrapper Type ─────────────────────────────────────────────────────
# Defined before accessors since get_params/get_priors dispatch on it

"""
    Wrapper <: Experiment

Wraps an experiment with parameter name aliases. The forward model and plot functions
automatically translate between aliased and original parameter names.
"""
struct Wrapper <: Newtrinos.Experiment
    x::Newtrinos.Experiment
    aliases::Dict{Symbol, Symbol}  # actual -> alias
    translated_keys::Vector{Symbol}
    reverse_lookup::Dict{Symbol, Symbol} # alias -> actual
end

function Base.getproperty(wrapper::Wrapper, name::Symbol)
    if name ∈ (:x, :aliases, :translated_keys, :reverse_lookup)
        return getfield(wrapper, name)
    end
    if name == :forward_model
        function forward_model(params)
            orig_param_names = Tuple([get(wrapper.reverse_lookup, k, k) for k in keys(params)])
            orig_params = NamedTuple{orig_param_names}(values(params))
            return wrapper.x.forward_model(orig_params)
        end
        return forward_model
    end
    if name == :plot
        function plot(params, data=wrapper.x.assets.observed)
            orig_param_names = Tuple([get(wrapper.reverse_lookup, k, k) for k in keys(params)])
            orig_params = NamedTuple{orig_param_names}(values(params))
            return wrapper.x.plot(orig_params, data)
        end
        return plot
    end
    return getfield(wrapper.x, name)
end

# ── Parameter & Prior Accessors ─────────────────────────────────────

"""
    get_params(x)

Extract nominal parameter values as a sorted NamedTuple. Works on Physics, Experiment,
Wrapper, or a NamedTuple of modules (merging all with conflict checking).
"""
function get_params(x::Newtrinos.Physics)
    sort_nt(x.params)
end

function get_params(x::Newtrinos.Experiment)
    safe_merge(x.params, get_params(x.physics))
end

function get_params(w::Newtrinos.Wrapper)
    NamedTuple{Tuple(w.translated_keys)}(values(get_params(w.x)))
end

function get_params(modules::NamedTuple)
    all_params = [get_params(m) for m in modules]
    safe_merge(all_params...)
end

"""
    get_priors(x)

Extract prior distributions as a sorted NamedTuple. Works on Physics, Experiment,
Wrapper, or a NamedTuple of modules (merging all with conflict checking).
"""
function get_priors(x::Newtrinos.Physics)
    sort_nt(x.priors)
end

function get_priors(x::Newtrinos.Experiment)
    safe_merge(x.priors, get_priors(x.physics))
end

function get_priors(w::Newtrinos.Wrapper)
    NamedTuple{Tuple(w.translated_keys)}(values(get_priors(w.x)))
end

function get_priors(modules::NamedTuple)
    all_priors = [get_priors(m) for m in modules]
    safe_merge(all_priors...)
end

"""
    condition(priors, conditional_vars, params)

Fix parameters to specific values by replacing their priors with constants.
`conditional_vars` can be an Array of symbols (uses values from `params`) or a Dict mapping symbols to values.
"""
function condition(priors::NamedTuple, conditional_vars::AbstractArray, p)
    for var in conditional_vars
        @reset priors[var] = p[var]
    end
    priors
end

function condition(priors::NamedTuple, conditional_vars::AbstractDict, p)
    for var in keys(conditional_vars)
        if isnothing(conditional_vars[var])
            @reset priors[var] = p[var]
        else
            @reset priors[var] = conditional_vars[var]
        end
    end
    priors
end

# Wrapper constructor (needs get_params defined above)
function Wrapper(x::Newtrinos.Experiment, aliases::Dict{Symbol, Symbol})
    original_keys = keys(get_params(x))
    translated_keys = [get(aliases, k, k) for k in original_keys]
    reverse_lookup = Dict(value => key for (key, value) in aliases)
    return Wrapper(x, aliases, translated_keys, reverse_lookup)
end

# ── Experiment Utilities ────────────────────────────────────────────

function get_observed(experiments::NamedTuple)
    NamedTuple{keys(experiments)}(e.assets.observed for e in experiments)
end

function get_fwd_model(experiments::NamedTuple)
    fwd_models = NamedTuple{keys(experiments)}(e.forward_model for e in experiments)
    distprod ∘ ffanout(fwd_models)
end

"""
    generate_likelihood(experiments[, observed])

Construct a joint likelihood from a NamedTuple of configured experiments.
Combines all forward models and observed data into a single likelihood object
compatible with `DensityInterface.logdensityof`.
"""
function generate_likelihood(experiments::NamedTuple, observed=get_observed(experiments))
    likelihoodof(get_fwd_model(experiments), observed)
end

"""
    correlated_priors_vars(priors, vars, dist)

Replace independent priors for `vars` with a correlated multivariate distribution `dist`.
"""
function correlated_priors_vars(priors::NamedTuple, vars::Union{AbstractArray, Tuple}, dist::Distribution)
    named_shapes = NamedTuple(var => ValueShapes.ScalarShape{Real}() for var in vars)
    corr_prior = Returns(ValueShapes.ReshapedDist(dist, ValueShapes.NamedTupleShape(named_shapes)))
    keys_to_keep = Tuple(k for k in keys(priors) if k ∉ vars)
    other_prior = distprod(;NamedTuple{keys_to_keep}(priors)...)
    return corr_prior, other_prior
end

"""
    generate_toy_data(experiment, params)
    generate_toy_data(experiments::NamedTuple, params)

Generate random toy data by sampling from the forward model distribution.
"""
function generate_toy_data(experiment::Newtrinos.Experiment, params::NamedTuple)
    dist_obj = experiment.forward_model(params)
    rand(dist_obj)
end

function generate_toy_data(experiments::NamedTuple, params::NamedTuple)
    map(experiments) do experiment
        dist_obj = experiment.forward_model(params)
        rand(dist_obj)
    end
end

"""
    generate_asimov_data(experiment, params)
    generate_asimov_data(experiments::NamedTuple, params)

Generate Asimov (expected) data from the forward model at the given parameters.
Poisson-distributed observables are rounded to integers.
"""
function generate_asimov_data(experiment::Newtrinos.Experiment, params::NamedTuple)
    dist_obj = experiment.forward_model(params)
    asimov_data_flt = mean(dist_obj)
    check_dist(d) = (d isa Distributions.Poisson) |
        (d isa Distributions.ProductDistribution && !isempty(d.dists) && first(d.dists) isa Distributions.Poisson) |
        (d isa Distributions.Product && !isempty(d.v) && first(d.v) isa Distributions.Poisson)

    if dist_obj isa ValueShapes.NamedTupleDist
        for key in keys(dist_obj)
            if check_dist(dist_obj[key])
                @info "Poisson-based model for $(key). Rounding Asimov data to nearest integer."
                @reset asimov_data_flt[key] = round.(Int, asimov_data_flt[key])
            end
        end
        return asimov_data_flt
    end
    if check_dist(dist_obj)
        @info "Poisson-based model. Rounding Asimov data to nearest integer."
        return round.(Int, asimov_data_flt)
    end

    @info "Not Poisson-based model. Returning std floating-point Asimov data."
    return asimov_data_flt
end

function generate_asimov_data(experiments::NamedTuple, params::NamedTuple)
    map(experiments) do experiment
        generate_asimov_data(experiment, params)
    end
end

# ── Optimization ────────────────────────────────────────────────────

"""
    find_mle(likelihood, prior, params; adsel=select_ad(length(params)))

Find the Maximum Likelihood Estimator using LBFGS optimization via BAT.
Returns `(llh, log_posterior, optimized_params)`. Parameters with `ConstValueDist`
priors are held fixed.
"""
function find_mle(likelihood, prior, params; adsel = select_ad(length(params)))
    try
        set_batcontext(ad = adsel)
        posterior = PosteriorMeasure(likelihood, prior)

        msg = "Running Optimization for point "
        for key in keys(prior)
            if prior[key] isa ValueShapes.ConstValueDist
                value = prior[key].value
                @reset params[key] = value
                msg *= " $(key): $(value)"
            end
        end

        @info msg
        res = bat_findmode(posterior, OptimizationAlg(optalg=Optimization.LBFGS(), init = ExplicitInit([params]), kwargs = (reltol=1e-7, maxiters=1000)))

        return logdensityof(likelihood, res.result), logdensityof(posterior, res.result), res.result
    catch e
        if e isa ArgumentError
            return NaN, NaN, (; (k => NaN for k in keys(params))... )
        else
            rethrow(e)
        end
    end
end

"""
    find_mle_cached(likelihood, prior, params, cache_dir)

Like [`find_mle`](@ref) but caches results to `cache_dir` using content hashing.
Subsequent calls with the same prior and params skip the optimization.
"""
function find_mle_cached(likelihood, prior, params, cache_dir)
    opt_result = nothing

    h = ContentHashes.hash([prior, params])

    if !isnothing(cache_dir)
        fname = joinpath(cache_dir, "$h.jld2")
        if isfile(fname)
            @info "using cached file $fname"
            cached = FileIO.load(fname)
            opt_result = (cached["llh"], cached["log_posterior"], cached["result"])
        end
    end

    if isnothing(opt_result)
        opt_result = find_mle(likelihood, prior, params)
    end

    if !isnothing(cache_dir)
        fname = joinpath(cache_dir, "$h.jld2")
        FileIO.save(fname, OrderedDict("llh"=>opt_result[1], "log_posterior"=>opt_result[2], "result"=>opt_result[3]))
    end

    opt_result
end

# ── Scanning & Profiling ────────────────────────────────────────────

function _generate_grid(vars_to_scan, priors)
    vars = collect(keys(vars_to_scan))
    values = [quantile(priors[var], collect(range(0,1,vars_to_scan[var]))) for var in vars]
    mesh = collect(IterTools.product(values...))
    vars, values, mesh
end

"""
    generate_scanpoints(vars_to_scan, priors)

Create a grid of prior distributions for scanning. `vars_to_scan` is an OrderedDict
mapping parameter symbols to grid sizes. Grid points are placed at quantiles of the priors.
Returns `(values, scanpoints)`.
"""
function generate_scanpoints(vars_to_scan, priors)
    vars, values, mesh = _generate_grid(vars_to_scan, priors)
    scanpoints = Array{Any}(undef, size(mesh))

    function make_prior(vals)
        p = deepcopy(priors)
        for i in 1:length(vars)
            @reset p[vars[i]] = vals[i]
        end
        distprod(;p...)
    end

    for i in eachindex(mesh)
        scanpoints[i] = make_prior(mesh[i])
    end

    values, scanpoints, vec(mesh)
end

"""Assemble optimization results into a NamedTuple of arrays."""
function assemble_profile_results(opt_results, result_size)
    results = Array{Any}(undef, result_size)
    llhs = Array{Float64}(undef, result_size)
    log_posteriors = Array{Float64}(undef, result_size)
    for (i, opt_result) in enumerate(opt_results)
        llhs[i] = opt_result[1]
        log_posteriors[i] = opt_result[2]
        results[i] = opt_result[3]
    end
    s = OrderedDict(key=>[x[key] for x in results] for key in keys(first(results)))
    s[:llh] = llhs
    s[:log_posterior] = log_posteriors
    NamedTuple(s)
end

"""
    randomize_params(rng, params, prior)

Sample free parameters from their prior distributions to create randomized starting points.
Parameters fixed via `ConstValueDist` are left unchanged.
"""
function randomize_params(rng, params, prior)
    p = deepcopy(params)
    for key in keys(p)
        d = prior[key]
        if !(d isa ValueShapes.ConstValueDist)
            @reset p[key] = rand(rng, d)
        end
    end
    return p
end

"""
    _sequential_order(grid_shape, center_linear)

Compute a BFS traversal order of grid points (as linear indices) starting from `center_linear`,
walking to Manhattan-adjacent neighbours. Returns `(order, parent)`:
- `order`: vector of linear indices in visit order
- `parent`: for each linear index, the linear index of its BFS parent (-1 for the start)
"""
function _sequential_order(grid_shape, center_linear::Int)
    li = LinearIndices(grid_shape)
    ci = CartesianIndices(grid_shape)
    n  = prod(grid_shape)
    nd = length(grid_shape)

    visited = falses(n)
    order   = Int[]
    sizehint!(order, n)
    parent  = fill(-1, n)

    visited[center_linear] = true
    push!(order, center_linear)
    queue = [center_linear]
    head  = 1

    while head <= length(queue)
        curr    = queue[head]; head += 1
        curr_ci = ci[curr]
        for d in 1:nd, delta in (-1, 1)
            coords = ntuple(i -> i == d ? curr_ci[i] + delta : curr_ci[i], nd)
            all(1 .<= coords .<= grid_shape) || continue
            nb = li[CartesianIndex(coords)]
            visited[nb] && continue
            visited[nb]  = true
            parent[nb]   = curr
            push!(order, nb)
            push!(queue, nb)
        end
    end

    order, parent
end

function _profile(likelihood, scanpoints, params, cache_dir; map_func=nothing, nseeds=1, seed_params=nothing, sequential=false, center_linear=1)
    get_p(i) = isnothing(seed_params) ? params : seed_params[i]

    if sequential
        # Sequential warm-starting: BFS order from center_linear, each point seeded by parent
        grid_shape = size(scanpoints)
        flat_sp    = vec(scanpoints)
        order, parent = _sequential_order(grid_shape, center_linear)
        opt_results_flat = Vector{Any}(undef, prod(grid_shape))

        @info "Sequential profile: $(prod(grid_shape)) scan points, BFS from linear index $center_linear"
        p_idx = ProgressMeter.Progress(prod(grid_shape))
        for (step, i) in enumerate(order)
            seed = parent[i] < 0 ? deepcopy(get_p(i)) : deepcopy(opt_results_flat[parent[i]][3])
            opt_results_flat[i] = find_mle_cached(likelihood, flat_sp[i], seed, cache_dir)
            ProgressMeter.next!(p_idx)
        end
        return assemble_profile_results(opt_results_flat, grid_shape)
    end

    if nseeds <= 1
        # Original single-seed path: no randomization
        if isnothing(map_func)
            opt_results = Array{Any}(undef, size(scanpoints))
            @showprogress Threads.@threads for i in eachindex(scanpoints)
                opt_results[i] = find_mle_cached(likelihood, scanpoints[i], deepcopy(get_p(i)), cache_dir)
            end
        else
            work = collect(eachindex(scanpoints))
            opt_results_flat = map_func(work, scanpoints, get_p.(eachindex(scanpoints)), cache_dir)
            opt_results = reshape(opt_results_flat, size(scanpoints))
        end
        return assemble_profile_results(opt_results, size(scanpoints))
    end

    # Multi-seed path: expand each scan point into nseeds independent jobs
    n_points = length(scanpoints)
    flat_scanpoints = vec(scanpoints)
    n_jobs = n_points * nseeds

    expanded_scanpoints = Vector{eltype(flat_scanpoints)}(undef, n_jobs)
    expanded_params = Vector{typeof(params)}(undef, n_jobs)
    rng = Random.Xoshiro(42)
    for i in 1:n_points
        for s in 1:nseeds
            j = (s - 1) * n_points + i
            expanded_scanpoints[j] = flat_scanpoints[i]
            expanded_params[j] = randomize_params(rng, get_p(i), flat_scanpoints[i])
        end
    end

    @info "Running $n_jobs jobs ($n_points scan points × $nseeds seeds)"

    if isnothing(map_func)
        opt_results = Vector{Any}(undef, n_jobs)
        @showprogress Threads.@threads for j in 1:n_jobs
            opt_results[j] = find_mle_cached(likelihood, expanded_scanpoints[j], deepcopy(expanded_params[j]), cache_dir)
        end
    else
        work = collect(1:n_jobs)
        opt_results = collect(map_func(work, expanded_scanpoints, expanded_params, cache_dir))
    end

    # Select best fit per scan point (highest log_posterior)
    best_results = Vector{Any}(undef, n_points)
    for i in 1:n_points
        seed_indices = [(s - 1) * n_points + i for s in 1:nseeds]
        posteriors = [let p = opt_results[j][2]; isnan(p) ? -Inf : p end for j in seed_indices]
        best_results[i] = opt_results[seed_indices[argmax(posteriors)]]
    end
    assemble_profile_results(reshape(best_results, size(scanpoints)), size(scanpoints))
end

"""
    profile(likelihood, priors, vars_to_scan, params; cache_dir=nothing, map_func=nothing, nseeds=1, seed_result=nothing, sequential=false, start_from=nothing)

Run a profile likelihood scan. At each grid point defined by `vars_to_scan`,
optimizes over all other parameters. Use `cache_dir` to cache and resume results.
Use `map_func=pmap` for distributed parallelism (default: `Threads.@threads`).
Use `nseeds > 1` to run multiple fits per point from randomized starting values and keep the best.

Set `sequential=true` to compute scan points in BFS order starting from `start_from`
(a NamedTuple of axis values, e.g. the best-fit of a previous run). Each point is seeded
by its BFS parent's optimised result, ensuring smooth profiles. Incompatible with `map_func`.
If `start_from` is not given with `sequential=true`, starts from the grid's central point.
"""
function profile(likelihood, priors, vars_to_scan, params; cache_dir=nothing, map_func=nothing, nseeds=1, seed_result=nothing, seed_results=nothing, sequential=false, start_from=nothing)
    t1 = time()
    # check if there is actually any variable to be profiled over, or if they are all just Numbers
    if all([isa(priors[var], Number) for var in setdiff(keys(priors), keys(vars_to_scan))])
        return scan(likelihood, priors, vars_to_scan, params)
    end

    values, scanpoints, mesh_flat = generate_scanpoints(vars_to_scan, priors)
    if !isnothing(cache_dir)
        if isdir(cache_dir)
            @info "Reusing cache dir `$(cache_dir)`"
        else
            mkdir(cache_dir)
        end
    end

    seed_params = nothing
    if !isnothing(seed_result)
        itps = make_seed_interpolants(seed_result)
        param_keys = collect(keys(itps))
        axis_keys  = keys(seed_result.axes)
        n_axes = length(axis_keys)
        seed_params = [let coords = Tuple(Float64(mesh_flat[i][d]) for d in 1:n_axes)
            merge(params, NamedTuple{Tuple(param_keys)}(itps[k](coords...) for k in param_keys))
        end for i in eachindex(mesh_flat)]
        @info "Using per-point seeds from smoothed result ($(length(param_keys)) params interpolated)"
    elseif !isnothing(seed_results)
        merged_itps = make_merged_seed_interpolants(seed_results, keys(vars_to_scan))
        merged_keys = collect(keys(merged_itps))
        n_axes = length(keys(vars_to_scan))
        seed_params = [let coords = Tuple(Float64(mesh_flat[i][d]) for d in 1:n_axes)
            nt_vals = [merged_itps[k](coords...) for k in merged_keys]
            merge(params, NamedTuple{Tuple(merged_keys)}(nt_vals))
        end for i in eachindex(mesh_flat)]
        @info "Using merged per-point seeds from $(length(seed_results)) files ($(length(merged_keys)) params interpolated)"
    end

    # Determine BFS center for sequential mode
    center_linear = div(prod(size(scanpoints)), 2) + 1  # default: near center
    if sequential
        if !isnothing(start_from)
            axis_keys = collect(keys(vars_to_scan))
            indices = [begin
                ax = values[findfirst(==(k), collect(keys(vars_to_scan)))]
                argmin(abs.(ax .- Float64(start_from[k])))
            end for k in axis_keys]
            center_linear = LinearIndices(size(scanpoints))[CartesianIndex(Tuple(indices))]
            @info "Sequential profile: starting from grid indices $indices (nearest to start_from)"
        else
            center_linear = div(prod(size(scanpoints)), 2) + 1
            @info "Sequential profile: starting from central grid point (linear index $center_linear)"
        end
        isnothing(map_func) || @warn "sequential=true ignores map_func (distributed workers); running locally"
    end

    res = _profile(likelihood, scanpoints, params, cache_dir; map_func=map_func, nseeds=nseeds, seed_params=seed_params, sequential=sequential, center_linear=center_linear)
    t2 = time()
    meta = Dict("task"=> "profile", "priors"=>priors, "vars_to_scan"=>vars_to_scan, "params"=>params, "exec_time"=>t2-t1, "cache_dir"=>cache_dir, "nseeds"=>nseeds, "sequential"=>sequential)
    add_meta!(meta)
    axes = NamedTuple{tuple(keys(vars_to_scan)...)}(values)
    result = NewtrinosResult(axes=axes, values=res, meta=meta)
end

"""
    scan(likelihood, priors, vars_to_scan, params; gradient_map=false)

Run a simple likelihood scan on a grid (no optimization over nuisance parameters).
Faster than [`profile`](@ref) but does not account for nuisance parameter variations.
Set `gradient_map=true` to also compute gradients at each point.
"""
function scan(likelihood, priors, vars_to_scan, params; gradient_map=false)
    t1 = time()
    vars, values, mesh = _generate_grid(vars_to_scan, priors)
    scanpoints = Array{Any}(undef, size(mesh))

    function make_params(vals)
        p = deepcopy(params)
        for i in 1:length(vars)
            @reset p[vars[i]] = vals[i]
        end
        return p
    end

    for i in eachindex(mesh)
        scanpoints[i] = make_params(mesh[i])
    end

    llhs = Array{Float64}(undef, size(scanpoints))
    if gradient_map
        grads = Array{Any}(undef, size(scanpoints))
    end

    @showprogress Threads.@threads for i in eachindex(scanpoints)
        p = scanpoints[i]
        llhs[i] = logdensityof(likelihood, p)
        if gradient_map
            grads[i] = ForwardDiff.gradient(x -> logdensityof(likelihood, x), p)
        end
    end

    s = OrderedDict{Symbol, Array}(key=>Fill(params[key], size(mesh)) for key in setdiff(keys(params), keys(vars_to_scan)))
    if gradient_map
        g = OrderedDict(Symbol(key, "_grad")=>[x[key] for x in grads] for key in keys(first(grads)))
        s = merge(s, g)
    end
    s[:llh] = llhs
    s[:log_posterior] = llhs
    res = NamedTuple(s)
    t2 = time()
    meta = Dict("task"=>"scan", "priors"=>priors, "vars_to_scan"=>vars_to_scan, "params"=>params, "exec_time"=>t2-t1,)
    add_meta!(meta)
    axes = NamedTuple{tuple(keys(vars_to_scan)...)}(values)
    result = NewtrinosResult(axes=axes, values=res, meta=meta)
end

# ── Results ─────────────────────────────────────────────────────────

"""
    bestfit(result::NewtrinosResult)

Extract the best-fit parameter values from a scan/profile result (maximum log_posterior point).
"""
function bestfit(result::NewtrinosResult)
    idx = argmax(result.values.log_posterior)
    bf = OrderedDict(var=>result.values[var][idx] for var in keys(result.values))
    for i in 1:length(result.axes)
        bf[keys(result.axes)[i]] = result.axes[i][idx[i]]
    end
    NamedTuple(bf)
end

"""
    add_meta!(meta::Dict)

Populate a metadata dictionary with hostname, username, date, git repo path, commit hash, and repo cleanliness.
"""
function add_meta!(meta)
    meta["hostname"] = gethostname()
    meta["username"] = get(ENV, "USER", get(ENV, "USERNAME", "unknown"))
    meta["date"] = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    repo = dirname(dirname(pathof(Newtrinos)))
    meta["repo"] = repo
    meta["commit_hash"] = LibGit2.head(repo)
    meta["repo_clean"] = !LibGit2.isdirty(LibGit2.GitRepo(repo))
end

# ── Result smoothing for refine_profile ────────────────────────────────────────

"""
    _gaussian_blur(arr, sigma)

Apply a separable Gaussian blur of width `sigma` (in grid units) to an N-dimensional array.
Boundary effects are handled by renormalising the kernel weights at edges.
"""
function _gaussian_blur(arr::Array, sigma::Float64)
    sigma <= 0 && return float.(arr)
    result = float.(arr)
    for d in 1:ndims(arr)
        n = size(result, d)
        radius = min(ceil(Int, 3 * sigma), n - 1)
        k_range = -radius:radius
        kernel = [exp(-k^2 / (2 * sigma^2)) for k in k_range]
        old = copy(result)
        for idx in CartesianIndices(size(old))
            s = 0.0; w = 0.0
            for (ki, delta) in enumerate(k_range)
                ni = idx[d] + delta
                1 <= ni <= n || continue
                nidx = CartesianIndex(ntuple(i -> i == d ? ni : idx[i], ndims(arr)))
                s += kernel[ki] * old[nidx]
                w += kernel[ki]
            end
            result[idx] = s / w
        end
    end
    result
end

"""
    smooth_result(result; outlier_threshold=10.0, gaussian_sigma=1.0)

Prepare a `NewtrinosResult` for use as per-point seeds in `refine_profile`:

1. Detect outlier scan points where `log_posterior` is more than `outlier_threshold` below
   the maximum of its direct grid neighbours (indicating a failed optimisation).
2. Replace each outlier's parameter values with the mean of its valid (non-outlier) neighbours.
3. Apply a Gaussian blur of width `gaussian_sigma` (in grid units) to all parameter arrays.

Returns a new `NewtrinosResult` with smoothed values.
"""
function smooth_result(result::NewtrinosResult; outlier_threshold=10.0, gaussian_sigma=1.0)
    lp = result.values.log_posterior        # already grid-shaped
    grid_shape = size(lp)
    ndim = ndims(lp)
    param_keys = [k for k in keys(result.values) if k ∉ (:llh, :log_posterior)]

    # Helper: direct grid neighbours of idx (in-bounds only)
    function neighbor_indices(idx)
        nbrs = CartesianIndex[]
        for d in 1:ndim, delta in (-1, 1)
            nidx = CartesianIndex(ntuple(i -> i == d ? idx[i] + delta : idx[i], ndim))
            checkbounds(Bool, lp, nidx) && push!(nbrs, nidx)
        end
        nbrs
    end

    # 1. Detect outliers
    is_outlier = falses(grid_shape)
    for idx in CartesianIndices(grid_shape)
        nbrs = neighbor_indices(idx)
        isempty(nbrs) && continue
        lp[idx] < maximum(lp[n] for n in nbrs) - outlier_threshold && (is_outlier[idx] = true)
    end
    @info "smooth_result: $(count(is_outlier)) outlier(s) detected in $(prod(grid_shape)) scan points"

    # 2. Replace outlier param values with mean of valid neighbours, then Gaussian-blur
    new_params = Dict{Symbol, Vector{Float64}}()
    for k in param_keys
        arr = reshape(copy(float.(result.values[k])), grid_shape)
        for idx in CartesianIndices(grid_shape)
            is_outlier[idx] || continue
            valid_vals = [arr[n] for n in neighbor_indices(idx) if !is_outlier[n]]
            isempty(valid_vals) || (arr[idx] = mean(valid_vals))
        end
        new_params[k] = vec(_gaussian_blur(arr, gaussian_sigma))
    end

    # Also repair log_posterior at outlier points (for reference only, not used as seeds)
    new_lp = copy(float.(lp))
    for idx in CartesianIndices(grid_shape)
        is_outlier[idx] || continue
        valid_vals = [lp[n] for n in neighbor_indices(idx) if !is_outlier[n]]
        isempty(valid_vals) || (new_lp[idx] = mean(valid_vals))
    end

    new_values = NamedTuple{keys(result.values)}(
        [k == :log_posterior ? new_lp :
         k == :llh           ? result.values.llh :
         new_params[k]
         for k in keys(result.values)])
    meta = merge(result.meta, Dict("smoothed"=>true,
                                   "outlier_threshold"=>outlier_threshold,
                                   "gaussian_sigma"=>gaussian_sigma))
    NewtrinosResult(axes=result.axes, values=new_values, meta=meta)
end

"""
    make_seed_interpolants(result)

Build a linear interpolant (with linear extrapolation outside the grid) for each
optimised parameter in `result`, indexed by the scan axis values.
Returns a `Dict{Symbol, AbstractExtrapolation}`.
"""
function make_seed_interpolants(result::NewtrinosResult)
    axis_vals = Tuple(collect(Float64, v) for v in values(result.axes))
    grid_shape = size(result.values.log_posterior)
    param_keys = [k for k in keys(result.values) if k ∉ (:llh, :log_posterior)]
    itps = Dict{Symbol, Any}()
    for k in param_keys
        arr = Float64.(reshape(result.values[k], grid_shape))
        itp = interpolate(axis_vals, arr, Gridded(Linear()))
        itps[k] = extrapolate(itp, Linear())
    end
    itps
end

"""
    make_merged_seed_interpolants(seed_results, joint_axis_keys)

Build merged per-parameter seed evaluators from multiple `NewtrinosResult` files.

For each parameter:
- Present in one file: use that file's interpolant.
- Present in multiple files: evaluate each and take the mean.

`joint_axis_keys`: the scan axis symbols of the new joint scan (e.g. `keys(vars_to_scan)`).
Source files scanned over a subset of these axes are handled via coordinate projection.

Returns a `Dict{Symbol, Function}` where each function takes `(coords...)` positional
coordinates aligned to `joint_axis_keys`.
"""
function make_merged_seed_interpolants(
    seed_results::AbstractVector,
    joint_axis_keys
)
    joint_keys = collect(joint_axis_keys)
    source_groups = map(seed_results) do r
        src_keys = collect(keys(r.axes))
        idxs = [findfirst(==(ak), joint_keys) for ak in src_keys]
        bad = findfirst(isnothing, idxs)
        isnothing(bad) || error("Seed file axis $(src_keys[bad]) not in joint axes $joint_keys")
        (idxs, make_seed_interpolants(r))
    end

    all_param_keys = unique(reduce(vcat, [collect(keys(sg[2])) for sg in source_groups]))

    merged = Dict{Symbol, Function}()
    for k in all_param_keys
        sources_k = [(idxs, itps[k]) for (idxs, itps) in source_groups if haskey(itps, k)]
        if length(sources_k) == 1
            (idxs, itp) = sources_k[1]
            merged[k] = (coords...) -> itp(coords[idxs]...)
        else
            merged[k] = (coords...) -> mean(itp(coords[idxs]...) for (idxs, itp) in sources_k)
        end
    end
    merged
end
