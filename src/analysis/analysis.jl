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

        "--profile-vars"
        help = "Variables to profile/scan over (e.g. theta23 dm231). Uses default grid size unless specified as var:N"
        nargs = '+'
        default = String[]

        "--seed"
        help = "JLD2 file containing a NewtrinosResult (under key 'result') to seed initial parameter values from its best-fit point"
        arg_type = String
        default = nothing

        "--nseeds"
        help = "Number of fits per scan point from randomized starting values (best fit is kept)"
        arg_type = Int
        default = 1
    end

    return parse_args(s)
end

# Map CLI-friendly names to Julia symbols for oscillation parameters
const PARAM_NAME_MAP = Dict(
    "theta12" => :θ₁₂, "th12" => :θ₁₂, "θ₁₂" => :θ₁₂,
    "theta13" => :θ₁₃, "th13" => :θ₁₃, "θ₁₃" => :θ₁₃,
    "theta23" => :θ₂₃, "th23" => :θ₂₃, "θ₂₃" => :θ₂₃,
    "dm221" => :Δm²₂₁, "dm21" => :Δm²₂₁, "Δm²₂₁" => :Δm²₂₁,
    "dm231" => :Δm²₃₁, "dm31" => :Δm²₃₁, "Δm²₃₁" => :Δm²₃₁,
    "dcp" => :δCP, "deltacp" => :δCP, "δCP" => :δCP,
)

"""
    parse_profile_var(s, available_params)

Parse a profile variable specification like "theta23" or "dm231:21".
Returns (symbol, n_points). Falls back to direct Symbol match against available params.
"""
function parse_profile_var(s::String, available_params)
    parts = split(s, ":")
    name = parts[1]
    n = length(parts) > 1 ? parse(Int, parts[2]) : 11

    # Try the alias map first
    sym = get(PARAM_NAME_MAP, name, nothing)
    if sym !== nothing
        return (sym, n)
    end

    # Try direct symbol match against available parameters
    sym = Symbol(name)
    if sym in keys(available_params)
        return (sym, n)
    end

    error("Unknown parameter '$name'. Available: $(join(sort(collect(string.(keys(available_params)))), ", "))")
end

args = parse_command_line()

name = args["name"]
n_workers = args["workers"]
n_threads = args["threads"]
use_distributed = n_workers > 1

# Set up distributed workers if requested
map_func = nothing
if use_distributed
    addprocs(n_workers; exeflags="--threads=$n_threads")

    @everywhere args = $args

    @everywhere begin
        using Distributions
        using DensityInterface
        using BAT
        using MeasureBase
        using ADTypes
        using Newtrinos

        experiments = Newtrinos.configure_experiments(args["experiments"])
        likelihood = Newtrinos.generate_likelihood(experiments)

        ad_backend = Symbol(args["ad"])
        Newtrinos.set_ad_backend(ad_backend)
        p = Newtrinos.get_params(experiments)
        set_batcontext(ad = Newtrinos.select_ad(length(p)))
    end

    # Define work function on workers that uses their local likelihood
    @everywhere function _do_work(scanpoint, params, cache_dir)
        Newtrinos.find_mle_cached(likelihood, scanpoint, deepcopy(params), cache_dir)
    end

    map_func = (work, scanpoints, params_list, cache_dir) -> pmap(work) do i
        _do_work(scanpoints[i], params_list[i], cache_dir)
    end
end

##### PHYSICS CONFIG #####
# To use defaults:
if !use_distributed
    experiments = configure_experiments(args["experiments"])
end

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

# Seed initial parameters from a previous result file
if args["seed"] !== nothing
    seed_data = FileIO.load(args["seed"])
    seed_bf = Newtrinos.bestfit(seed_data["result"])
    matched = [k for k in keys(p) if haskey(seed_bf, k)]
    seed_vals = NamedTuple{Tuple(matched)}(seed_bf[k] for k in matched)
    p = merge(p, seed_vals)
    @info "Seeded $(length(matched)) parameters from $(args["seed"])"
end

ad_backend = Symbol(args["ad"])
Newtrinos.set_ad_backend(ad_backend)
set_batcontext(ad = Newtrinos.select_ad(length(p)))

# Variables to condition on (=fix)
conditional_vars = Dict(:θ₁₂=>p.θ₁₂, :Δm²₂₁=>p.Δm²₂₁)

# For profile / scan task only: choose scan grid
vars_to_scan = OrderedDict()
if !isempty(args["profile-vars"])
    for v in args["profile-vars"]
        sym, n = parse_profile_var(v, p)
        vars_to_scan[sym] = n
    end
else
    # Default: scan over θ₂₃ and Δm²₃₁
    vars_to_scan[:θ₂₃] = 8
    vars_to_scan[:Δm²₃₁] = 8
end

###### END CONFIG ######

if !use_distributed
    likelihood = Newtrinos.generate_likelihood(experiments);
end

priors = Newtrinos.condition(priors, conditional_vars, p)

@reset priors.Δm²₃₁ = Uniform(0.002, 0.003)
@reset priors.θ₂₃ = Uniform(pi/4-0.2, pi/4+0.2)

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
        result = Newtrinos.profile(likelihood, priors, vars_to_scan, p; cache_dir=name, map_func=map_func, nseeds=args["nseeds"])
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
