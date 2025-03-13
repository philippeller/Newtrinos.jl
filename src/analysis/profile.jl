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

#osc = Newtrinos.osc.Darkdim_L
osc = Newtrinos.osc.standard

minos_llh = let osc_prob = osc.osc_prob, observed = Newtrinos.minos.data.observed
    params -> logpdf(Newtrinos.minos.forward_model(osc_prob)(params), observed)
end 
dayabay_llh = let osc_prob = osc.osc_prob, observed = Newtrinos.dayabay.observed
    params -> logpdf(Newtrinos.dayabay.forward_model(params, osc_prob), observed)
end 
kamland_llh = let osc_prob = osc.osc_prob, observed = Newtrinos.kamland.observed
    params -> logpdf(Newtrinos.kamland.forward_model(params, osc_prob), observed)
end 

llh = logfuncdensity(params -> kamland_llh(params)) # dayabay_llh(params) + minos_llh(params))

priors_dict = merge(osc.priors, Newtrinos.kamland.priors)
params_dict = merge(osc.params, Newtrinos.kamland.params)

params_dict[:θ₁₃] = 0.

vars_to_scan = OrderedDict()
#vars_to_scan[:Darkdim_radius] = 10
vars_to_scan[:θ₁₂] = 30
vars_to_scan[:Δm²₂₁] = 30

vars_not_to_fit = [:θ₂₃, :δCP, :Δm²₃₁, :θ₁₃]

for var in vars_not_to_fit
    priors_dict[var] = params_dict[var]
end

name = "julia_test_kamland"

vals, results = Newtrinos.profile(llh, priors_dict, vars_to_scan, params_dict, cache_dir=name)
#vals, results = Newtrinos.scan(llh, priors_dict, vars_to_scan, params_dict)

FileIO.save(name * ".jld2", Dict("vals" => vals, "results" => results))
