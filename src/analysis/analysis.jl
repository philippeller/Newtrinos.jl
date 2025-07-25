using LinearAlgebra
using Distributions
using DensityInterface
using Base
using ForwardDiff
using BAT
using IterTools
using DataStructures
using MeasureBase
using ADTypes
using Newtrinos
using FileIO
using Accessors
using ArgParse

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--experiments"
        help = "List of experiments to run"
        nargs = '+'
        required = true

        "--ordering"
        help = "NMO: either NO or IO"
        arg_type = String
        default = "NO"      

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

        "--lambda"
        arg_type = Float64
        default = 1.

        #"--ca"
        #arg_type = Float64
        #default = 1.

        #"--scan"
        #arg_type = String
        #required = true

    end

    return parse_args(s)
end

# dynamically construct named tuple from experiment names
function configure_experiments(experiment_list, physics)
    pairs = (Symbol(lowercase(exp)) => getproperty(getproperty(Newtrinos, Symbol(lowercase(exp))), :configure)(physics) for exp in experiment_list)
    return (; pairs...)
end


args = parse_command_line()

adsel = AutoForwardDiff()
context = set_batcontext(ad = adsel)

name = args["name"]

osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.Darkdim_Masses(three_flavour=Newtrinos.osc.ThreeFlavour(ordering=Symbol(args["ordering"]))),
    #propagation=Newtrinos.osc.Basic(),
    propagation=Newtrinos.osc.Damping(σₑ=0.05),
    #states=Newtrinos.osc.All(),
    states=Newtrinos.osc.Cut(cutoff=10.),
    interaction=Newtrinos.osc.SI()
    )
osc = Newtrinos.osc.configure(osc_cfg)
atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec = Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec);

# Choose experiments to include
#experiments = (
#    deepcore = Newtrinos.deepcore.configure(physics),
#    dayabay = Newtrinos.dayabay.configure(physics),
#    kamland = Newtrinos.kamland.configure(physics),
#    minos = Newtrinos.minos.configure(physics),
#    orca = Newtrinos.orca.configure(physics),
#);

experiments = configure_experiments(args["experiments"], physics)

# Variables to condition on (=fix)
#conditional_vars = [:θ₁₂, :θ₁₃, :δCP, :Δm²₂₁, :nutau_cc_norm]
#conditional_vars = [:Darkdim_radius, :δCP, :λ₁, :λ₂, :λ₃]

#conditional_vars = Dict(:δCP=>0., :ca1=>args["ca"], :ca2=>args["ca"], :ca3=>args["ca"], :nutau_cc_norm=>1., :nc_norm=>1.)
conditional_vars = Dict(:δCP=>0., :λ₁=>args["lambda"], :λ₂=>args["lambda"], :λ₃=>args["lambda"], :nutau_cc_norm=>1., :nc_norm=>1.)
#conditional_vars = Dict(:δCP=>0., :nutau_cc_norm=>1., :nc_norm=>1.)
#conditional_vars = []

# For profile / scan task only: choose scan grid
vars_to_scan = OrderedDict()
#vars_to_scan[:θ₂₃] = 31
#vars_to_scan[:Δm²₃₁] = 31


###### END CONFIG ######

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)


if args["ordering"] == "NO"
    #@reset p.ca1 = 0.95
    #@reset priors.ca1 = LogUniform(1e-3, 10.)
    #@reset p.ca2 = 7.36
    #@reset priors.ca2 = LogUniform(1e-3, 10.)
    ##@reset p.ca3 = 0.2
    ##@reset priors.ca3 = LogUniform(1e-3, 10.)
    #@reset p.ca3 = -1.
    #@reset priors.ca3 = -LogUniform(1e-3, 10.)
   
    # informed choices:
    #@reset p.ca1 = 8.
    #@reset priors.ca1 = Uniform(0.1, 10.)
    #@reset p.ca2 = 0.952
    #@reset priors.ca2 = Uniform(0.01, 1.1)
    #@reset p.ca3 = 0.203
    #@reset priors.ca3 = Uniform(0.001, 0.3)

    #@reset priors.θ₁₂ = Uniform(0., pi/2)
    #@reset priors.θ₁₃ = Uniform(0., pi/2)
    #@reset priors.θ₂₃ = Uniform(0., pi/2)

    # informed choices 2:
    #@reset p.ca1 = 2.
    #@reset priors.ca1 = Uniform(0.5, 10.)
    #@reset p.ca2 = 0.9
    #@reset priors.ca2 = Uniform(0.1, 2.)
    #@reset p.ca3 = -0.1
    #@reset priors.ca3 = Uniform(-5, -0.01)
    #
    #@reset p.λ₁ = 0.044
    #@reset p.λ₂ = 0.031
    #@reset p.λ₃ = 0.058


elseif args["ordering"] == "IO"
    #@reset p.λ₁ = 0.021
    #@reset p.λ₂ = 0.700
    #@reset p.λ₃ = 0.396
    #@reset p.ca1 = -4.9977
    #@reset priors.ca1 = Uniform(-5., -0.1)
    #@reset p.ca2 = -5.
    #@reset priors.ca2 = Uniform(-5., -0.1)
    #@reset p.ca3 = -4.92
    #@reset priors.ca3 = Uniform(-5., -0.1)

    #@reset priors.θ₁₂ = Uniform(0., pi/2)
    #@reset priors.θ₁₃ = Uniform(0., pi/2)
    #@reset priors.θ₂₃ = Uniform(0., pi/2)

    #@reset p.ca3 = -0.1.
    #@reset priors.ca3 = -LogUniform(1e-3, 10.)
end

@reset priors.Darkdim_radius = LogUniform(1., 100.)
@reset priors.m₀ = LogUniform(0.001, 1)

vars_to_scan[:Darkdim_radius] = 11
vars_to_scan[:m₀] = 16

#if args["scan"] == "ca1"
#    vars_to_scan[:ca1] = 11
#elseif args["scan"] == "ca2"
#    vars_to_scan[:ca2] = 11
#elseif args["scan"] == "ca3"
#    vars_to_scan[:ca3] = 11
#end

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

priors = condition(priors, conditional_vars, p)
    
if lowercase(args["task"]) == "nestedsampling"
    #import NestedSamplers
    import UltraNest
    prior = distprod(;priors...)
    posterior = PosteriorMeasure(likelihood, prior)
    samples = bat_sample(posterior, ReactiveNestedSampling()).result
    #samples = bat_sample(posterior, EllipsoidalNestedSampling()).result
    #samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^4, nchains = 4)).result
    FileIO.save(name * ".jld2", Dict("samples" => samples))
    
elseif lowercase(args["task"]) == "importancesampling"
    prior = distprod(;priors...)
    posterior = PosteriorMeasure(likelihood, prior)
    #init_samples =  make_prior_samples(posterior, 1_000)

    seed_points = load("darkdim_seeds.jld2")["df"]
    seed_points = seed_points[seed_points.ca3 .< 0, :]
    init_samples =  make_init_samples(posterior, seed_points[1:10, :], 10_000)
    
    #init_samples =  make_init_samples(posterior, 10, 100_000)
    FileIO.save(name * "_init_samples.jld2", Dict(String(a)=>init_samples[a] for a in keys(init_samples)))
    whack_samples = whack_many_moles(posterior, init_samples, target_samplesize=100_000, cache_dir=name)
    FileIO.save(name * ".jld2", Dict(String(a)=>whack_samples[a] for a in keys(whack_samples)))
else
    if lowercase(args["task"]) == "profile"
        result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir=name)

    elseif lowercase(args["task"]) == "scan"
        result = Newtrinos.scan(likelihood, priors, vars_to_scan, p)
    end 
    FileIO.save(name * ".jld2", Dict("result" => result))

    if args["plot"]
        using CairoMakie
        fig = Figure()
        ax = Axis(fig[1,1])
        plot!(ax, result)
        ax.xlabel = String(collect(keys(vars_to_scan))[1])

        if length(vars_to_scan) == 1
            ax.ylabel = "-2ΔLLH"
        else
            ax.ylabel = String(collect(keys(vars_to_scan))[2])
        end
        ax.title = args["task"] * ": " * join(args["experiments"], " + ")
        save(name * ".png", fig)
    end
end
