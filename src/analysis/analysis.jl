using Distributions
using Distributed
using DensityInterface
using BAT
using DataStructures
using MeasureBase
using ADTypes
using Newtrinos
using FileIO
using Accessors
using ArgParse

include("cli_common.jl")

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--experiments"
        help = "List of experiments to run"
        nargs = '+'
        required = true

        "--name"
        help = "Name for outputs"
        arg_type = String
        required = true

        "--task"
        help = "Task to perform: Choice of NestedSampling, ImportanceSampling, Profile, Scan"
        arg_type = String
        required = true

        "--plot"
        help = "Enable plotting"
        action = :store_true

        "--workers"
        help = "Number of distributed workers (default: 1, no distributed)"
        arg_type = Int
        default = 1

        "--threads"
        help = "Number of threads per worker (only used when --workers > 1)"
        arg_type = Int
        default = 1

        "--ad"
        help = "AD backend: auto, forwarddiff, polyester, mooncake"
        arg_type = String
        default = "auto"
    end

    return parse_args(s)
end

args = parse_command_line()

name = args["name"]
n_workers = args["workers"]
n_threads = args["threads"]
use_distributed = n_workers > 1

"""
    _distributed_profile(args, conditional_vars, prior_overrides, vars_to_scan, cache_dir, n_workers, n_threads)

Run profiling on distributed workers. Each worker builds its own likelihood, scanpoints,
and AD context locally — nothing is serialized except integer indices and scalar results.
`prior_overrides` is a Dict{Symbol,Distribution} of prior replacements to apply after conditioning.
"""
function _distributed_profile(args, conditional_vars, prior_overrides, vars_to_scan, cache_dir, n_workers, n_threads)
    t1 = time()

    addprocs(n_workers; exeflags="--threads=$n_threads")

    # Send only plain data to workers (strings, numbers, dicts, simple distributions)
    @everywhere args = $args
    @everywhere conditional_vars = $conditional_vars
    @everywhere prior_overrides = $prior_overrides
    @everywhere vars_to_scan = $vars_to_scan
    @everywhere cache_dir = $cache_dir

    @everywhere begin
        using Distributions
        using DensityInterface
        using BAT
        using DataStructures
        using MeasureBase
        using ADTypes
        using Newtrinos
        using Accessors

        include(joinpath(@__DIR__, "cli_common.jl"))

        # Each worker builds everything from scratch — no serialization needed
        experiments = configure_experiments(args["experiments"])
        p = Newtrinos.get_params(experiments)
        priors = Newtrinos.get_priors(experiments)
        likelihood = Newtrinos.generate_likelihood(experiments)

        ad_backend = Symbol(args["ad"])
        Newtrinos.set_ad_backend(ad_backend)
        set_batcontext(ad = Newtrinos.select_ad(length(p)))

        priors = Newtrinos.condition(priors, conditional_vars, p)
        for (k, v) in prior_overrides
            @reset priors[k] = v
        end

        _, scanpoints = Newtrinos.generate_scanpoints(vars_to_scan, priors)
    end

    # Workers have everything — just send integer indices
    @everywhere function _do_work(i)
        Newtrinos.find_mle_cached(likelihood, scanpoints[i], deepcopy(p), cache_dir)
    end

    n_points = prod(values(vars_to_scan))
    work = collect(1:n_points)

    if !isnothing(cache_dir)
        if isdir(cache_dir)
            @info "Reusing cache dir `$(cache_dir)`"
        else
            mkdir(cache_dir)
        end
    end

    opt_results_flat = pmap(_do_work, work)

    # Collect scanpoint grid shape and assemble results
    grid_shape = Tuple(values(vars_to_scan))
    opt_results = reshape(opt_results_flat, grid_shape)
    res = Newtrinos.assemble_profile_results(opt_results, grid_shape)

    # Build axes from main-process priors (same conditioning as workers)
    values_grid, _ = Newtrinos.generate_scanpoints(vars_to_scan, priors)
    axes = NamedTuple{tuple(keys(vars_to_scan)...)}(values_grid)

    t2 = time()
    meta = Dict("task"=> "profile", "priors"=>priors, "vars_to_scan"=>vars_to_scan, "params"=>p, "exec_time"=>t2-t1, "cache_dir"=>cache_dir)
    Newtrinos.add_meta!(meta)

    rmprocs(workers())

    Newtrinos.NewtrinosResult(axes=axes, values=res, meta=meta)
end

##### PHYSICS CONFIG #####
experiments = configure_experiments(args["experiments"])

# To override physics (e.g. for IO, sterile models, custom flux, etc.), uncomment and modify:
# osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(
#     flavour=Newtrinos.osc.ThreeFlavour(ordering=:IO),
#     interaction=Newtrinos.osc.SI(),
# ))
# atm_flux = Newtrinos.atm_flux.configure()
# earth_layers = Newtrinos.earth_layers.configure()
# xsec = Newtrinos.xsec.configure()
# physics = (; osc, atm_flux, earth_layers, xsec)
# experiments = configure_experiments(args["experiments"], physics)

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)

ad_backend = Symbol(args["ad"])
Newtrinos.set_ad_backend(ad_backend)
set_batcontext(ad = Newtrinos.select_ad(length(p)))

# Variables to condition on (=fix)
conditional_vars = Dict(:θ₁₂=>p.θ₁₂, :δCP=>-1.89, :Δm²₂₁=>p.Δm²₂₁)

# For profile / scan task only: choose scan grid
vars_to_scan = OrderedDict()
vars_to_scan[:θ₂₃] = 11
vars_to_scan[:Δm²₃₁] = 11

# Prior overrides (applied after conditioning)
prior_overrides = Dict{Symbol,Distribution}(
    :Δm²₃₁ => Uniform(0.002, 0.003),
    :θ₂₃ => Uniform(pi/4-0.2, pi/4+0.2),
)

###### END CONFIG ######

priors = Newtrinos.condition(priors, conditional_vars, p)
for (k, v) in prior_overrides
    @reset priors[k] = v
end

if lowercase(args["task"]) == "nestedsampling"
    import UltraNest
    prior = distprod(;priors...)
    posterior = PosteriorMeasure(likelihood, prior)
    samples = bat_sample(posterior, ReactiveNestedSampling()).result
    FileIO.save(name * ".jld2", Dict("samples" => samples))

elseif lowercase(args["task"]) == "importancesampling"
    prior = distprod(;priors...)
    posterior = PosteriorMeasure(likelihood, prior)

    #seed_points = load("darkdim_seeds.jld2")["df"]
    #seed_points = seed_points[seed_points.ca3 .< 0, :]
    #init_samples = make_init_samples(posterior, seed_points[1:10, :], 10_000)
    init_samples = make_init_samples(posterior, 10, 50_000)

    FileIO.save(name * "_init_samples.jld2", Dict(String(a)=>init_samples[a] for a in keys(init_samples)))
    whack_samples = whack_many_moles(posterior, init_samples, target_samplesize=10_000, cache_dir=name)
    FileIO.save(name * ".jld2", Dict(String(a)=>whack_samples[a] for a in keys(whack_samples)))
else
    if lowercase(args["task"]) == "profile"
        if use_distributed
            result = _distributed_profile(args, conditional_vars, prior_overrides, vars_to_scan, name, n_workers, n_threads)
        else
            result = Newtrinos.profile(likelihood, priors, vars_to_scan, p; cache_dir=name)
        end
    elseif lowercase(args["task"]) == "scan"
        result = Newtrinos.scan(likelihood, priors, vars_to_scan, p)
    end

    save_result(result, name)

    if args["plot"]
        using CairoMakie
        title = args["task"] * ": " * join(args["experiments"], " + ")
        plot_result(result, name, vars_to_scan; title=title)
    end
end
