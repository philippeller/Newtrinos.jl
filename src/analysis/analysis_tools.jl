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

    values, scanpoints
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

function _profile(likelihood, scanpoints, params, cache_dir; map_func=nothing)
    do_work(i) = find_mle_cached(likelihood, scanpoints[i], deepcopy(params), cache_dir)

    if isnothing(map_func)
        # Default: threaded execution
        opt_results = Array{Any}(undef, size(scanpoints))
        @showprogress Threads.@threads for i in eachindex(scanpoints)
            opt_results[i] = do_work(i)
        end
    else
        # Custom map (e.g., pmap for distributed)
        work = collect(eachindex(scanpoints))
        opt_results_flat = map_func(do_work, work)
        opt_results = reshape(opt_results_flat, size(scanpoints))
    end
    assemble_profile_results(opt_results, size(scanpoints))
end

"""
    profile(likelihood, priors, vars_to_scan, params; cache_dir=nothing, map_func=nothing)

Run a profile likelihood scan. At each grid point defined by `vars_to_scan`,
optimizes over all other parameters. Use `cache_dir` to cache and resume results.
Use `map_func=pmap` for distributed parallelism (default: `Threads.@threads`).
"""
function profile(likelihood, priors, vars_to_scan, params; cache_dir=nothing, map_func=nothing)
    t1 = time()
    # check if there is actually any variable to be profiled over, or if they are all just Numbers
    if all([isa(priors[var], Number) for var in setdiff(keys(priors), keys(vars_to_scan))])
        return scan(likelihood, priors, vars_to_scan, params)
    end

    values, scanpoints = generate_scanpoints(vars_to_scan, priors)
    if !isnothing(cache_dir)
        if isdir(cache_dir)
            @info "Reusing cache dir `$(cache_dir)`"
        else
            mkdir(cache_dir)
        end
    end
    res = _profile(likelihood, scanpoints, params, cache_dir; map_func=map_func)
    t2 = time()
    meta = Dict("task"=> "profile", "priors"=>priors, "vars_to_scan"=>vars_to_scan, "params"=>params, "exec_time"=>t2-t1, "cache_dir"=>cache_dir)
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
