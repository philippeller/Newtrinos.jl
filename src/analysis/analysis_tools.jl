using LinearAlgebra
using Distributions
using DensityInterface
using InverseFunctions
using Base
using ForwardDiff
using BAT
using Optimization
using Optim
using IterTools
using DataStructures
using ADTypes
using AutoDiffOperators
using ContentHashes
import ValueShapes
using FileIO
using FillArrays
import JLD2
using MeasureBase
using FunctionChains
using Accessors
using Logging
using ProgressMeter
using ..Newtrinos

adsel = AutoForwardDiff()
set_batcontext(ad = adsel)

@kwdef struct NewtrinosResult
    axes::NamedTuple
    values::NamedTuple
end

function build_optimizationfunction(f, adsel::AutoDiffOperators.ADSelector)
    adm = convert(ADTypes.AbstractADType, reverse_ad_selector(adsel))
    optimization_function = Optimization.OptimizationFunction(f, adm)
    return optimization_function
end

function build_optimizationfunction(f, adsel::BAT._NoADSelected)
    optimization_function = Optimization.OptimizationFunction(f)
    return optimization_function
end

"Find Maximum Likelihood Estimator (MLE)"
function find_mle(likelihood, prior, params)

    adsel = AutoForwardDiff()
    set_batcontext(ad = adsel)
    
    posterior = PosteriorMeasure(likelihood, prior)

    #res = bat_findmode(posterior, OptimizationAlg(optalg=Optimization.LBFGS()))

    msg = "Running Optimization for point "
    
    for key in keys(prior)
    	if prior[key] isa ValueShapes.ConstValueDist
            value = prior[key].value
    	    @reset params[key] = value
            msg *= " $(key): $(value)"
    	end
    end

    #@info msg
    # THIS ONE WORKS:
    res = bat_findmode(posterior, OptimizationAlg(optalg=Optimization.LBFGS(), init = ExplicitInit([params]), kwargs = (reltol=1e-7, maxiters=10000)))

    # This one also works, and IS thread safe:

    #res = bat_findmode(posterior, OptimAlg(optalg=Optim.LBFGS(), init = ExplicitInit([v_init]), kwargs = (g_tol=1e-5, iterations=100)))

    
    #res = bat_findmode(posterior, OptimizationAlg(optalg=Optimization.LBFGS(), kwargs = ()))#reltol=1e-7, maxiters=10000)))

    # target = posterior
    # context = get_batcontext()
    # transformed_m, f_pretransform = BAT.transform_and_unshape(PriorToUniform(), target, context)
    # target_uneval = BAT.unevaluated(target)
    # inv_trafo = inverse(f_pretransform)
    # initalg = BAT.apply_trafo_to_init(f_pretransform, InitFromTarget())
    # x_init = collect(bat_initval(transformed_m, initalg, context).result)
    # # Maximize density of original target, but run in transformed space, don't apply LADJ:
    # f = BAT.fchain(inv_trafo, logdensityof(target_uneval), -)
    # target_f = (x, p) -> f(x)
    # adsel = BAT.get_adselector(context)
    # optimization_function = build_optimizationfunction(target_f, adsel)
    # optimization_problem = Optimization.OptimizationProblem(optimization_function, x_init, (), lb=zeros(size(x_init)), ub=ones(size(x_init)))
    # #algopts = (maxiters = algorithm.maxiters, maxtime = algorithm.maxtime, abstol = algorithm.abstol, reltol = algorithm.reltol)
    # # Not all algorithms support abstol, just filter all NaN-valued opts out:
    # #filtered_algopts = NamedTuple(filter(p -> !isnan(p[2]), ))
    # optimization_result = Optimization.solve(optimization_problem, Evolutionary.CMAES(μ = 40, λ = 100)) #NLopt.GN_CRS2_LM()) 
    # transformed_mode =  optimization_result.u
    # result_mode = inv_trafo(transformed_mode)
    # res = (result = result_mode, result_trafo = transformed_mode, f_pretransform = f_pretransform, info = optimization_result)

    #println(res)

    #res = bat_findmode(posterior, OptimizationAlg(optalg=Optimization.LBFGS(), init = ExplicitInit([v_init]), kwargs = (reltol=1e-7, maxiters=10000)))
    #res = bat_findmode(posterior, OptimAlg(optalg=Optim.LBFGS(), init = ExplicitInit([v_init]), kwargs = (f_tol=1e-7, iterations=10000)))
    #res = bat_findmode(posterior, OptimAlg(optalg=Optim.LBFGS(), init = ExplicitInit([v_init]), kwargs = (f_tol=1e-7, iterations=10000)))
    #res = bat_findmode(posterior, OptimizationAlg(optalg=NLopt.GN_CRS2_LM()))

    return logdensityof(likelihood, res.result), logdensityof(posterior, res.result), res.result

    # posterior = PosteriorMeasure(llh, prior)

    # tr_pstr, f_trafo = bat_transform(PriorToGaussian(), posterior)

    # v0 = mean(posterior.prior.dist)
    # x0 = f_trafo(v0)

    # tr_neg_log_likelihood = (-) ∘ logdensityof(posterior.likelihood) ∘ inverse(f_trafo)

    # tr_neg_log_likelihood(x0)

    # r = Optim.optimize(tr_neg_log_likelihood, x0, Optim.LBFGS(), Optim.Options(f_tol=1e-13), autodiff = :forward)

    # x_opt = Optim.minimizer(r)
    # v_opt = inverse(f_trafo)(x_opt)
    # f_opt = -Optim.minimum(r)
    # return f_opt, v_opt
end


function generate_scanpoints(vars_to_scan, priors)
    vars = collect(keys(vars_to_scan))
    values = [quantile(priors[var], collect(range(0,1,vars_to_scan[var]))) for var in vars]
    mesh = collect(IterTools.product(values...))
    scanpoints = Array{Any}(undef, size(mesh))

    function make_prior(vals)
        p = deepcopy(priors)
        for i in 1:length(vars_to_scan)
            @reset p[vars[i]] = vals[i]
        end
        distprod(;p...)
    end

    for i in eachindex(mesh)
        scanpoints[i] = make_prior(mesh[i])
    end

    values, scanpoints
end

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
        opt_result = find_mle(likelihood, prior, deepcopy(params))
    end

    if !isnothing(cache_dir)
        fname = joinpath(cache_dir, "$h.jld2")
        FileIO.save(fname, OrderedDict("llh"=>opt_result[1], "log_posterior"=>opt_result[2], "result"=>opt_result[3]))
    end

    opt_result
end

function _profile(likelihood, scanpoints, params, cache_dir)
    results = Array{Any}(undef, size(scanpoints))
    llhs = Array{Any}(undef, size(scanpoints))
    log_posteriors = Array{Any}(undef, size(scanpoints))

    @showprogress Threads.@threads for i in eachindex(scanpoints)
        opt_result = find_mle_cached(likelihood, scanpoints[i], params, cache_dir)
        llhs[i] = opt_result[1]
        log_posteriors[i] = opt_result[2]
        results[i] = opt_result[3]
    end
    s = OrderedDict(key=>[x[key] for x in results] for key in keys(first(results)))
    s[:llh] = llhs
    s[:log_posterior] = log_posteriors
    NamedTuple(s)
end

"Run Profile llh scan"
function profile(likelihood, priors, vars_to_scan, params; cache_dir=nothing)

    #check if there is actually any variable to be profiled over, or if they all or just Numbers
    if all([isa(priors[var], Number) for var in setdiff(keys(priors), keys(vars_to_scan))])
        # so all variables are just numbers and it reduces to a simple scan
        return scan(likelihood, priors, vars_to_scan, params)
    end
    
    values, scanpoints = generate_scanpoints(vars_to_scan, priors)
    if !isnothing(cache_dir)
        if !isdir(cache_dir)
            mkdir(cache_dir)
        end
    end
    res = _profile(likelihood, scanpoints, params, cache_dir)

    axes = NamedTuple{tuple(keys(vars_to_scan)...)}(values)
    result = NewtrinosResult(axes=axes, values=res)

end

"Run simple llh scan"
function scan(likelihood, priors, vars_to_scan, params; gradient_map=false)
    vars = collect(keys(vars_to_scan))
    values = [quantile(priors[var], collect(range(0,1,vars_to_scan[var]))) for var in vars]
    mesh = collect(IterTools.product(values...))
    scanpoints = Array{Any}(undef, size(mesh))

    function make_params(vals)
        p = deepcopy(params)
        for i in 1:length(vars_to_scan)
            @reset p[vars[i]] = vals[i]
        end
        return p
    end

    for i in eachindex(mesh)
        scanpoints[i] = make_params(mesh[i])
    end

    llhs = Array{Any}(undef, size(scanpoints))
    if gradient_map
        grads = Array{Any}(undef, size(scanpoints))
    end

    @showprogress Threads.@threads for i in eachindex(scanpoints)
        p = scanpoints[i]
        llhs[i] = logdensityof(likelihood, p)
        if gradient_map
            grads[i] = ForwardDiff.gradient(x -> logdensityof(likelihood, x),  p)
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

    axes = NamedTuple{tuple(keys(vars_to_scan)...)}(values)
    result = NewtrinosResult(axes=axes, values=res)
    
end

function bestfit(result::NewtrinosResult)
    idx = argmax(result.values.log_posterior)
    bf = OrderedDict(var=>result.values[var][idx] for var in keys(result.values))
    for i in 1:length(result.axes)
        bf[keys(result.axes)[i]] = result.axes[i][idx[i]]
    end
    NamedTuple(bf)
end

function sort_nt(nt::NamedTuple)
    keys_sorted = sort(collect(keys(nt)))
    values_sorted = getindex.(Ref(nt), keys_sorted)
    return NamedTuple{Tuple(keys_sorted)}(values_sorted)
end

function safe_merge(nt_list::NamedTuple...)
    """ Merge namedtuples such that duplicates are checked for consistentcy
    """
    merged = NamedTuple()  # start with an empty NamedTuple
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

function get_priors(x::Newtrinos.Experiment)
    safe_merge(x.priors, get_priors(x.physics))
end

function get_priors(x::Newtrinos.Physics)
    sort_nt(x.priors)
end

function get_params(x::Newtrinos.Experiment)
    safe_merge(x.params, get_params(x.physics))
end

function get_params(x::Newtrinos.Physics)
    sort_nt(x.params)
end

function get_priors(modules::NamedTuple)
    all_priors = [get_priors(m) for m in modules]
    safe_merge(all_priors...)
end

function get_params(modules::NamedTuple)
    all_params = [get_params(m) for m in modules]
    safe_merge(all_params...)
end

function get_observed(experiments::NamedTuple)
    NamedTuple{keys(experiments)}(e.assets.observed for e in experiments)
end

function get_fwd_model(experiments::NamedTuple)
    fwd_models = NamedTuple{keys(experiments)}(e.forward_model for e in experiments)
    distprod ∘ ffanout(fwd_models)
end

function generate_likelihood(experiments::NamedTuple, observed=get_observed(experiments))
    likelihoodof(get_fwd_model(experiments), observed)
end
