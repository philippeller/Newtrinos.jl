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
using ..Newtrinos

adsel = AutoForwardDiff()
set_batcontext(ad = adsel)

# ── Core Types ──────────────────────────────────────────────────────

"""
    NewtrinosResult(; axes, values, meta=Dict())

Container for the results of a [`scan`](@ref) or [`profile`](@ref) run.

# Fields
- `axes::NamedTuple`: scan grid coordinates, one entry per scanned parameter.
  Each entry is a vector of grid-point values.
- `values::NamedTuple`: per-grid-point outputs — likelihood values (`llh`,
  `log_posterior`) and optimized nuisance parameter arrays.
- `meta::Dict`: execution metadata (hostname, date, git commit, timing).
  Populated by [`add_meta!`](@ref).
"""
@kwdef struct NewtrinosResult
    axes::NamedTuple
    values::NamedTuple
    meta::Dict = Dict()
end

# ── Utility Functions ───────────────────────────────────────────────

"""
    sort_nt(nt::NamedTuple) -> NamedTuple

Return a copy of a named tuple `nt` with keys sorted alphabetically.

Used internally by [`safe_merge`](@ref), [`get_params`](@ref), and
[`get_priors`](@ref) to ensure a canonical key ordering across modules.

# Arguments
- `nt::NamedTuple`: input NamedTuple.

# Returns
A NamedTuple with the same entries as `nt`, sorted alphabetically by key name.
"""
function sort_nt(nt::NamedTuple)
    keys_sorted = sort(collect(keys(nt)))
    values_sorted = getindex.(Ref(nt), keys_sorted)
    return NamedTuple{Tuple(keys_sorted)}(values_sorted)
end

"""
    safe_merge(nt_list::NamedTuple...) -> NamedTuple

Merge NamedTuples, asserting that any key shared by two or more inputs has the
same value in all of them.

Raises an `error` if a key conflict is detected. The result is sorted
alphabetically by key.

# Arguments
- `nt_list::NamedTuple...`: one or more NamedTuples to merge.

# Returns
A sorted NamedTuple containing all key-value pairs from the inputs.

# Examples
```julia
a = (x = 1.0, y = 2.0)
b = (y = 2.0, z = 3.0)
safe_merge(a, b)   # (x = 1.0, y = 2.0, z = 3.0)
```
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
    Wrapper <: Newtrinos.Experiment

Wraps a [`Newtrinos.Experiment`](@ref) with parameter name aliases.

When two experiments share a physics parameter under different names (e.g.
`theta13` vs `sin2theta13`), a `Wrapper` renames parameters seen by the
outer analysis while transparently translating back to the original names
before calling the inner experiment's `forward_model` and `plot` functions.

Construct with `Wrapper(experiment, aliases)` where `aliases` is a
`Dict{Symbol,Symbol}` mapping original parameter names to their aliases.

# Fields
- `x::Newtrinos.Experiment`: the wrapped experiment.
- `aliases::Dict{Symbol,Symbol}`: map from original name → alias.
- `translated_keys::Vector{Symbol}`: all parameter keys after alias substitution.
- `reverse_lookup::Dict{Symbol,Symbol}`: map from alias → original name.

# Examples
```julia
aliases = Dict(:dm21_sq => :Dm21)
wrapped = Wrapper(dayabay_experiment, aliases)
```
"""
struct Wrapper <: Newtrinos.Experiment
    x::Newtrinos.Experiment
    aliases::Dict{Symbol, Symbol}  # actual -> alias
    translated_keys::Vector{Symbol}
    reverse_lookup::Dict{Symbol, Symbol} # alias -> actual
end

"""
    Base.getproperty(wrapper::Wrapper, name::Symbol)

Property accessor for [`Wrapper`](@ref) that intercepts `forward_model` and
`plot` to inject parameter name translation.

When `name` is `:forward_model` or `:plot`, returns a closure that translates
aliased parameter names back to their original names before delegating to the
inner experiment. All other property accesses are forwarded directly to the
wrapped experiment `wrapper.x`.

# Arguments
- `wrapper::Wrapper`: the wrapping experiment.
- `name::Symbol`: the property name to access.

# Returns
The requested property value of `name`, with translation logic applied for
`:forward_model` and `:plot`.
"""
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
    get_params(x) -> NamedTuple

Extract nominal parameter values as a sorted NamedTuple.

Dispatches on the type of `x`:
- `Physics`: returns `x.params` sorted by key.
- `Experiment`: merges `x.params` with the parameters of its `physics` module.
- `Wrapper`: applies the alias translation from [`Wrapper`](@ref) to the keys.
- `NamedTuple` of modules: merges parameters from all modules via
  [`safe_merge`](@ref), raising an error on conflicts.

# Arguments
- `x`: a [`Newtrinos.Physics`](@ref), [`Newtrinos.Experiment`](@ref),
  [`Wrapper`](@ref), or `NamedTuple` of such objects.

# Returns
A sorted `NamedTuple` of parameter name → nominal value.
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
    get_priors(x) -> NamedTuple

Extract prior distributions as a sorted NamedTuple.

Dispatches on the type of `x`:
- `Physics`: returns `x.priors` sorted by key.
- `Experiment`: merges `x.priors` with priors from its `physics` module.
- `Wrapper`: applies the alias translation from [`Wrapper`](@ref) to the keys.
- `NamedTuple` of modules: merges priors from all modules via
  [`safe_merge`](@ref), raising an error on conflicts.

# Arguments
- `x`: a [`Newtrinos.Physics`](@ref), [`Newtrinos.Experiment`](@ref),
  [`Wrapper`](@ref), or `NamedTuple` of such objects.

# Returns
A sorted `NamedTuple` of parameter name → prior distribution.
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
    condition(priors, conditional_vars, params) -> NamedTuple

Fix a subset of parameters to specific values by replacing their prior
distributions with constants (`ConstValueDist`).

Two dispatch methods:

**Array form** (conditional_vars is an array) — fixes each symbol in `conditional_vars` to its value in `params`:
```julia
condition(priors, [:theta23, :dm31_sq], params)
```

**Dict form** (conditional_vars is a dict) — fixes each key to the paired value, or to `params[key]` when the
value is `nothing`:
```julia
condition(priors, Dict(:theta23 => pi/4, :dm31_sq => nothing), params)
```

# Arguments
- `priors::NamedTuple`: prior distributions to modify.
- `conditional_vars`: `AbstractArray{Symbol}` or `AbstractDict{Symbol}` of
  parameters to fix.
- `params::NamedTuple`: fallback parameter values used when a dict entry is
  `nothing` or when using the array form.

# Returns
The modified `priors` NamedTuple with the specified parameters frozen.
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

"""
    Wrapper(x::Newtrinos.Experiment, aliases::Dict{Symbol,Symbol}) -> Wrapper

Construct a [`Wrapper`](@ref) around `x` with the given parameter name aliases.

Derives `translated_keys` by applying `aliases` to the full key list returned
by `get_params(x)`, and builds the `reverse_lookup` for inverse translation.

# Arguments
- `x::Newtrinos.Experiment`: experiment to wrap.
- `aliases::Dict{Symbol,Symbol}`: map from original parameter name → alias.

# Returns
A [`Wrapper`](@ref) instance.
"""
function Wrapper(x::Newtrinos.Experiment, aliases::Dict{Symbol, Symbol})
    original_keys = keys(get_params(x))
    translated_keys = [get(aliases, k, k) for k in original_keys]
    reverse_lookup = Dict(value => key for (key, value) in aliases)
    return Wrapper(x, aliases, translated_keys, reverse_lookup)
end

# ── Experiment Utilities ────────────────────────────────────────────

"""
    get_observed(experiments::NamedTuple) -> NamedTuple

Extract the observed data from each experiment in a NamedTuple of experiments.

# Arguments
- `experiments::NamedTuple`: named collection of [`Newtrinos.Experiment`](@ref) objects.

# Returns
A NamedTuple with the same keys as `experiments`, each entry holding the
corresponding `experiment.assets.observed` data.
"""
function get_observed(experiments::NamedTuple)
    NamedTuple{keys(experiments)}(e.assets.observed for e in experiments)
end

"""
    get_fwd_model(experiments::NamedTuple) -> Function

Compose the forward models of all experiments into a single joint model.

Builds a `NamedTuple` of per-experiment forward models and composes them
with `ffanout` and `distprod` so that calling the result with a parameter
NamedTuple returns a joint product distribution over all experiments.

# Arguments
- `experiments::NamedTuple`: named collection of [`Newtrinos.Experiment`](@ref) objects.

# Returns
A callable `params -> joint_distribution`.
"""
function get_fwd_model(experiments::NamedTuple)
    fwd_models = NamedTuple{keys(experiments)}(e.forward_model for e in experiments)
    distprod ∘ ffanout(fwd_models)
end

"""
    generate_likelihood(experiments[, observed]) -> likelihood

Construct a joint likelihood from a NamedTuple of configured experiments.

Combines all experiment forward models into a product distribution and
pairs it with the observed data. The returned object supports
`DensityInterface.logdensityof(likelihood, params)`.

# Arguments
- `experiments::NamedTuple`: named collection of configured
  [`Newtrinos.Experiment`](@ref) objects.
- `observed` (optional): NamedTuple of observed data arrays matching
  `experiments`. Defaults to the observed data stored in each experiment's
  `assets.observed` field.

# Returns
A likelihood object compatible with `DensityInterface.logdensityof`.

# Examples
```julia
experiments = (deepcore = Newtrinos.deepcore.configure(),
               dayabay  = Newtrinos.dayabay.configure())
llh     = generate_likelihood(experiments)
logdensityof(llh, params)
```
"""
function generate_likelihood(experiments::NamedTuple, observed=get_observed(experiments))
    likelihoodof(get_fwd_model(experiments), observed)
end

"""
    correlated_priors_vars(priors, vars, dist) -> (corr_prior, other_prior)

Replace the independent priors for `vars` with a single correlated
multivariate distribution, splitting the prior NamedTuple into two parts.

Useful when a set of parameters has a known covariance structure (e.g. from
an external fit) that cannot be captured by independent marginals.

# Arguments
- `priors::NamedTuple`: the full prior NamedTuple.
- `vars::Union{AbstractArray, Tuple}`: parameter names to correlate.
- `dist::Distributions.Distribution`: a multivariate distribution over `vars`
  (must match the length of `vars`).

# Returns
A 2-tuple `(corr_prior, other_prior)` where:
- `corr_prior`: a callable returning a `ReshapedDist` over `vars`.
- `other_prior`: a `distprod` of the remaining (uncorrelated) priors.
"""
function correlated_priors_vars(priors::NamedTuple, vars::Union{AbstractArray, Tuple}, dist::Distribution)
    named_shapes = NamedTuple(var => ValueShapes.ScalarShape{Real}() for var in vars)
    corr_prior = Returns(ValueShapes.ReshapedDist(dist, ValueShapes.NamedTupleShape(named_shapes)))
    keys_to_keep = Tuple(k for k in keys(priors) if k ∉ vars)
    other_prior = distprod(;NamedTuple{keys_to_keep}(priors)...)
    return corr_prior, other_prior
end

"""
    generate_toy_data(experiment, params) -> data
    generate_toy_data(experiments::NamedTuple, params) -> NamedTuple

Generate random toy data by sampling from the experiment forward model.

Calls `experiment.forward_model(params)` to obtain the predictive
distribution and draws one sample from it.

# Arguments
- `experiment`: a single [`Newtrinos.Experiment`](@ref) or a NamedTuple of them.
- `params::NamedTuple`: parameter values at which to evaluate the forward model.

# Returns
- Single-experiment form: one sample from the forward model distribution.
- Multi-experiment form: a NamedTuple with the same keys as `experiments`,
  each entry holding one sample.
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
    generate_asimov_data(experiment, params) -> data
    generate_asimov_data(experiments::NamedTuple, params) -> NamedTuple

Generate Asimov (expected-value) data from the experiment forward model.

Evaluates `mean(forward_model(params))`. For Poisson-distributed observables
the mean is rounded to the nearest integer so that the data type matches what
a real run would produce.

# Arguments
- `experiment`: a single [`Newtrinos.Experiment`](@ref) or a NamedTuple of them.
- `params::NamedTuple`: parameter values at which to evaluate the forward model.

# Returns
- Single-experiment form: the expected-value data array (or NamedTuple of arrays).
- Multi-experiment form: a NamedTuple with the same keys as `experiments`.
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
    find_mle(likelihood, prior, params; adsel=AutoPolyesterForwardDiff())
        -> (llh, log_posterior, result)

Find the Maximum Likelihood Estimator (MLE) using L-BFGS optimization via BAT.jl.

Parameters whose prior is a `ConstValueDist` are held fixed during optimization.
If the optimizer raises an `ArgumentError` (e.g. due to a degenerate starting
point), returns `(NaN, NaN, NaN-filled NamedTuple)` instead of propagating the
error.

# Arguments
- `likelihood`: a likelihood object supporting `DensityInterface.logdensityof`.
- `prior`: a `distprod`-style prior NamedTuple or distribution.
- `params::NamedTuple`: starting parameter values for the optimizer.
- `adsel`: AD backend selector (default: `AutoPolyesterForwardDiff()`).

# Returns
A 3-tuple `(llh, log_posterior, result)` where:
- `llh::Float64`: log-likelihood at the optimum.
- `log_posterior::Float64`: log-posterior at the optimum.
- `result::NamedTuple`: optimized parameter values.
"""
function find_mle(likelihood, prior, params; adsel = AutoPolyesterForwardDiff())
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
        -> (llh, log_posterior, result)

Like [`find_mle`](@ref) but caches results to disk using content hashing.

A hash of `(prior, params)` is computed with `ContentHashes.hash`; if a
matching `.jld2` file exists in `cache_dir` the optimization is skipped and
the cached result is returned. Otherwise the result is computed and saved.

# Arguments
- `likelihood`: a likelihood object supporting `DensityInterface.logdensityof`.
- `prior`: prior distribution (same as [`find_mle`](@ref)).
- `params::NamedTuple`: starting parameter values.
- `cache_dir::Union{String,Nothing}`: directory for cached `.jld2` files.
  Pass `nothing` to disable caching.

# Returns
A 3-tuple `(llh, log_posterior, result)` — see [`find_mle`](@ref).
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

"""
    _generate_grid(vars_to_scan, priors) -> (vars, values, mesh)

Construct the Cartesian grid used by [`generate_scanpoints`](@ref) and
[`scan`](@ref).

Grid points for each variable are placed at evenly-spaced quantiles of its
prior distribution. All combinations are formed via `IterTools.product`.

# Arguments
- `vars_to_scan::OrderedDict{Symbol,Int}`: maps parameter names to grid sizes.
- `priors::NamedTuple`: prior distributions used to compute quantiles.

# Returns
A 3-tuple `(vars, values, mesh)` where:
- `vars`: `Vector{Symbol}` of scanned parameter names.
- `values`: vector of grid-point vectors (one per variable).
- `mesh`: Cartesian product array of grid-point tuples.

# Examples
```julia
    priors = (a=Uniform(0.0, 2.0), b=Uniform(-1.0, 1.0))
    vars_to_scan = OrderedDict(:a => 3, :b => 4)
    vars, values, mesh = Newtrinos._generate_grid(vars_to_scan, priors)
    # => mesh is a 3x4 grid of evenly spaced points from within the prior range of a and b 
```
"""
function _generate_grid(vars_to_scan, priors)
    vars = collect(keys(vars_to_scan))
    values = [quantile(priors[var], collect(range(0,1,vars_to_scan[var]))) for var in vars]
    mesh = collect(IterTools.product(values...))
    vars, values, mesh
end

"""
    generate_scanpoints(vars_to_scan, priors) -> (values, scanpoints)

Build a grid of fixed-parameter priors for a profile likelihood scan.

For each variable in `vars_to_scan`, grid points are placed at evenly-spaced
quantiles of its prior distribution. All combinations are formed via a
Cartesian product. At each grid point, the prior for the scanned variables
is replaced with a `ConstValueDist` fixing them to the grid values.

# Arguments
- `vars_to_scan::OrderedDict{Symbol,Int}`: maps each parameter name to the
  desired number of evenly spaced grid points within the prior range.
- `priors::NamedTuple`: prior distributions for all parameters.

# Returns
A 2-tuple `(values, scanpoints)` where:
- `values`: vector of grid-point vectors (one per scanned parameter).
- `scanpoints`: array of `distprod` priors, one per grid point.
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

"""
    assemble_profile_results(opt_results, result_size) -> NamedTuple

Collect the 'raw' per-grid-point optimization results and structure them into flat arrays within a NamedTuple.

Stacks the `(llh, log_posterior, params)` 3-tuples returned by
[`find_mle_cached`](@ref) into a NamedTuple of arrays with shape
`result_size`.

# Arguments
- `opt_results`: iterable of `(llh, log_posterior, NamedTuple)` 3-tuples.
- `result_size`: shape of the output arrays (matching the scan grid).

# Returns
A `NamedTuple` with one array-entry per parameter (values across the grid) plus
`:llh` and `:log_posterior` arrays.

# Example
```julia
opt_results = [
    (-10.0, -12.0, (a=1.0, b=2.0)), # llh, log_posterior, params output of find_mle
    (-5.0,  -7.0,  (a=3.0, b=4.0))
]
res = Newtrinos.assemble_profile_results(opt_results, (2,))
# returns res = (a = [1.0, 3.0], b = [2.0, 4.0], llh = [-10.0, -5.0], log_posterior = [-12.0, -7.0])
```
"""
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
    _profile(likelihood, scanpoints, params, cache_dir; map_func=nothing)
        -> NamedTuple

Execute [`find_mle_cached`](@ref) at every point in `scanpoints`, collect
the results, and assemble them into a NamedTuple of result arrays with [`assemble_profile_results`](@ref).

Uses `Threads.@threads` by default. Pass a custom `map_func` (e.g. `pmap`)
to override the parallelism strategy.

# Arguments
- `likelihood`: a likelihood object supporting `DensityInterface.logdensityof`.
- `scanpoints`: array of prior objects, one per grid point.
- `params::NamedTuple`: starting values for nuisance optimization.
- `cache_dir::Union{String,Nothing}`: passed to [`find_mle_cached`](@ref).
- `map_func`: optional custom mapping function; default=`nothing` uses threaded loops.

# Returns
A NamedTuple of result arrays as produced by [`assemble_profile_results`](@ref).
"""
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
    profile(likelihood, priors, vars_to_scan, params;
            cache_dir=nothing, map_func=nothing) -> NewtrinosResult

Run a profile likelihood scan over a parameter grid.

Creates grid points defined by `vars_to_scan` with [`generate_scanpoints`](@ref).
At each grid point, all nuisance parameters are optimized with [`_profile`](@ref) , i.e. via [`find_mle_cached`](@ref). 
If all non-scanned parameters have fixed (`Number`) priors, falls back to [`scan`](@ref) automatically.
Also collects meta data of the profile process and attaches it the returned object.

# Arguments
- `likelihood`: a likelihood object supporting `DensityInterface.logdensityof`.
- `priors::NamedTuple`: prior distributions for all parameters.
- `vars_to_scan::OrderedDict{Symbol,Int}`: parameters to scan and grid sizes.
- `params::NamedTuple`: starting values for nuisance parameter optimization.
- `cache_dir::Union{String,Nothing}`: directory for caching MLE results
  (created if absent). Pass `nothing` to disable (this is the default).
- `map_func`: custom mapping function for parallelism (e.g. `pmap` for
  distributed workers). Defaults to `Threads.@threads`.

# Returns
A [`NewtrinosResult`](@ref) NewtrinosResult(axes, values, meta) 
with the scan grid axes, per-point profiling results, and meta data.
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
    scan(likelihood, priors, vars_to_scan, params;
         gradient_map=false) -> NewtrinosResult

Run a simple likelihood scan on a parameter grid (NO nuisance optimization).

Generates grid with [`_generate_grid`](@ref) via `vars_to_scan` and then 
evaluates `logdensityof(likelihood, params)` at each grid point without
optimizing over nuisance parameters. Faster than [`profile`](@ref) but
does not account for nuisance parameter variations. 
Also collects meta data of the scan process and attaches it the returned object.

# Arguments
- `likelihood`: a likelihood object supporting `DensityInterface.logdensityof`.
- `priors::NamedTuple`: prior distributions used to place grid points at
  quantiles.
- `vars_to_scan::OrderedDict{Symbol,Int}`: parameters to scan and grid sizes.
- `params::NamedTuple`: nominal values; non-scanned parameters are held fixed.
- `gradient_map::Bool`: default=`false`; if `true`, also evaluates `ForwardDiff.gradient` at
  each grid point and includes per-parameter gradient arrays in the result.

# Returns
A [`NewtrinosResult`](@ref) NewtrinosResult(axes, values, meta) 
with the scan grid axes, per-point scan results, and meta data.
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
    bestfit(result::NewtrinosResult) -> NamedTuple

Extract the best-fit parameter values from a scan or profile result.

Finds the grid point with the highest `log_posterior` and returns both
the scanned parameter values (from `result.axes`) and all optimized
nuisance parameter values at that point.

# Arguments
- `result::NewtrinosResult`: output of [`scan`](@ref) or [`profile`](@ref).

# Returns
A `NamedTuple` containing all parameter values at the best-fit point,
including the scanned axis values, nuisance parameters, `llh`, and
`log_posterior`.
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

Populate a metadata dictionary with execution environment information.

Adds the following keys in-place:
- `"hostname"`: result of `gethostname()`.
- `"username"`: from `ENV["USER"]` or `ENV["USERNAME"]`.
- `"date"`: current date-time formatted as `"yyyy-mm-dd HH:MM:SS"`.
- `"repo"`: path to the Newtrinos.jl repository root.
- `"commit_hash"`: current HEAD commit hash.
- `"repo_clean"`: `true` if the repository has no uncommitted changes.

Called automatically by [`profile`](@ref) and [`scan`](@ref) before
returning their [`NewtrinosResult`](@ref). 

# Arguments
- `meta::Dict`: dictionary to populate in-place.

# Returns
`nothing` (mutates `meta`).
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
