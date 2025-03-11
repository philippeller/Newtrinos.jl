using LinearAlgebra
using Distributions
using DensityInterface
using Base
using Zygote
using BAT
using IterTools
using DataStructures
using MeasureBase
using ADTypes
using Newtrinos
using FileIO
import JLD2

adsel = AutoZygote()
context = set_batcontext(ad = adsel)

osc = Newtrinos.osc.Darkdim_L

minos_llh = let osc_prob = osc.osc_prob, observed = Newtrinos.minos.data.observed
    params -> logpdf(Newtrinos.minos.forward_model(osc_prob)(params), observed)
end 
dayabay_llh = let osc_prob = osc.osc_prob, observed = Newtrinos.dayabay.observed
    params -> logpdf(Newtrinos.dayabay.forward_model(params, osc_prob), observed)
end 

llh = logfuncdensity(params -> dayabay_llh(params) + minos_llh(params))

vars_to_scan = OrderedDict()
vars_to_scan[:Darkdim_radius] = 10

v_init_dict = copy(osc.params)

name = "julia_test"

println(name)

vals, results = Newtrinos.profile(llh, osc.priors, vars_to_scan, v_init_dict, cache_dir=name)

FileIO.save(name * ".jld2", Dict("vals" => vals, "results" => results))
