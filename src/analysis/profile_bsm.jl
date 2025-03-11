using LinearAlgebra
using Distributions
using DensityInterface
using Base
using Zygote
using Revise
using BAT
using Optim
using IterTools
using DataStructures
using MeasureBase
using ADTypes
using FileIO
import JLD2

adsel = AutoZygote()
context = set_batcontext(ad = adsel)

includet("../newtrinos/theory/osc.jl")
includet("../newtrinos/experiments/Daya_Bay/DayaBay_AZ/dayabay.jl")
includet("../newtrinos/experiments/minos/minos.jl")
includet("../newtrinos/analysis/analysis.jl")

minos_llh = let osc_prob = osc.osc_prob_Darkdim_L, observed = minos.data.observed
    params -> logpdf(minos.forward_model(osc_prob)(params), observed)
end 
dayabay_llh = let osc_prob = osc.osc_prob_Darkdim_L, observed = dayabay.observed
    params -> logpdf(dayabay.forward_model(params, osc_prob), observed)
end 

llh = logfuncdensity(params -> dayabay_llh(params) + minos_llh(params))

vars_to_scan = OrderedDict()
vars_to_scan[:Darkdim_radius] = 10

v_init_dict = copy(osc.Darkdim_params_L)

name = "profile_Darkdim_test_L_bfgs_myseed_2"

println(name)

vals, results = profile(llh, osc.Darkdim_priors_L, vars_to_scan, v_init_dict, cache_dir=name)

FileIO.save(name * ".jld2", Dict("vals" => vals, "results" => results))
