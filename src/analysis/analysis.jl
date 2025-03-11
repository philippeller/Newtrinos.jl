
using LinearAlgebra
using Distributions
using DensityInterface
using InverseFunctions
using Base
using Zygote
using BAT
using Optim
using Optimization
using OptimizationNLopt
using IterTools
using DataStructures
using ADTypes
using OptimizationEvolutionary
using AutoDiffOperators
import ValueShapes
using FileIO
import JLD2

adsel = AutoZygote()

#set_batcontext(ad = AutoForwardDiff())
set_batcontext(ad = adsel)

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
function find_mle(llh, prior, v_init_dict)

    posterior = PosteriorMeasure(llh, prior)

    println(prior)

    #res = bat_findmode(posterior, OptimizationAlg(optalg=Optimization.LBFGS()))
    
    for key in keys(prior)
    	if prior[key] isa ValueShapes.ConstValueDist
    	    v_init_dict[key] = prior[key].value
    	end
    end

    v_init = NamedTuple(v_init_dict)
    println(v_init)

    # THIS ONE WORKS:
    res = bat_findmode(posterior, OptimizationAlg(optalg=Optimization.LBFGS(), init = ExplicitInit([v_init])))#, kwargs = (reltol=1e-7, maxiters=10000)))
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

    println(res)

    #res = bat_findmode(posterior, OptimizationAlg(optalg=Optimization.LBFGS(), init = ExplicitInit([v_init]), kwargs = (reltol=1e-7, maxiters=10000)))
    #res = bat_findmode(posterior, OptimAlg(optalg=Optim.LBFGS(), init = ExplicitInit([v_init]), kwargs = (f_tol=1e-7, iterations=10000)))
    #res = bat_findmode(posterior, OptimAlg(optalg=Optim.LBFGS(), init = ExplicitInit([v_init]), kwargs = (f_tol=1e-7, iterations=10000)))
    #res = bat_findmode(posterior, OptimizationAlg(optalg=NLopt.GN_CRS2_LM()))

    return logdensityof(llh, res.result), res.result

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


function generate_scanpoints(vars_to_scan, prior_dict)
    vars = collect(keys(vars_to_scan))
    values = [quantile(prior_dict[var], collect(range(0,1,vars_to_scan[var]))) for var in vars]
    mesh = collect(IterTools.product(values...))
    scanpoints = Array{Any}(undef, size(mesh))

    function make_prior(vals)
        p = copy(prior_dict)
        for i in 1:length(vars_to_scan)
            p[vars[i]] = vals[i]
        end
        distprod(;p...)
    end

    for i in eachindex(mesh)
        scanpoints[i] = make_prior(mesh[i])
    end

    values, scanpoints
end

function _profile(llh, scanpoints, v_init_dict, cache_dir)
    results = Array{Any}(undef, size(scanpoints))
    llhs = Array{Any}(undef, size(scanpoints))

    #Threads.@threads
    for i in eachindex(scanpoints)

        result = nothing

        if !isnothing(cache_dir)
            fname = joinpath(cache_dir, "idx_$i.jld2")
            if isfile(fname)
                cached = FileIO.load(fname)
                result = (cached["llh"], cached["result"])
            end
        end
        
        if isnothing(result)
            prior = scanpoints[i]
	        result = find_mle(llh, prior, copy(v_init_dict))
        end

        if !isnothing(cache_dir)
            fname = joinpath(cache_dir, "idx_$i.jld2")
            FileIO.save(fname, Dict("llh"=>result[1], "result"=>result[2]))
        end

        llhs[i] = result[1]
        results[i] = result[2]
    end
    s = Dict(key=>[x[key] for x in results] for key in keys(first(results)))
    s[:llh] = llhs
    NamedTuple(s)
end

"Run Profile llh scan"
function profile(llh, prior_dict, vars_to_scan, v_init_dict; cache_dir=nothing)
    values, scanpoints = generate_scanpoints(vars_to_scan, prior_dict)
    if !isnothing(cache_dir)
        println(cache_dir)
        if !isdir(cache_dir)
            mkdir(cache_dir)
        end
    end
    res = _profile(llh, scanpoints, v_init_dict, cache_dir)
    return values, res
end

"Run simple llh scan"
function scan(llh, param_dict, prior_dict, vars_to_scan)
    vars = collect(keys(vars_to_scan))
    values = [quantile(prior_dict[var], collect(range(0,1,vars_to_scan[var]))) for var in vars]
    mesh = collect(IterTools.product(values...))
    scanpoints = Array{Any}(undef, size(mesh))

    function make_params(vals)
        p = copy(param_dict)
        for i in 1:length(vars_to_scan)
            p[vars[i]] = vals[i]
        end
        NamedTuple(p)
    end

    for i in eachindex(mesh)
        scanpoints[i] = make_params(mesh[i])
    end

    llhs = Array{Any}(undef, size(scanpoints))

    Threads.@threads for i in eachindex(scanpoints)
        params = scanpoints[i]
        llhs[i] = logdensityof(llh, params)
    end

    values, (llh = llhs,)
end
