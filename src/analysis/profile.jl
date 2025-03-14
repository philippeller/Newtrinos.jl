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

modules = [Newtrinos.kamland]#, Newtrinos.dayabay, Newtrinos.minos]
#modules = [Newtrinos.dayabay, Newtrinos.minos]

llhs = [let osc_prob = osc.osc_prob, observed = m.observed 
        params -> logpdf(m.forward_model(osc_prob)(params), observed)
        end
        for m in modules]

llh = logfuncdensity(params -> sum([f(params) for f in llhs]))

params_dict = merge(osc.params, [m.params for m in modules]...)
priors_dict = merge(osc.priors, [m.priors for m in modules]...)


#### Config

params_dict[:θ₁₃] = 0.

vars_to_scan = OrderedDict()
#vars_to_scan[:Darkdim_radius] = 10
vars_to_scan[:θ₁₂] = 10
vars_to_scan[:Δm²₂₁] = 10

vars_not_to_fit = [:θ₂₃, :δCP, :Δm²₃₁, :θ₁₃]

name = "julia_test_kamland"

for var in vars_not_to_fit
    priors_dict[var] = params_dict[var]
end

@kwdef struct NewtrinosResult
    axes::NamedTuple
    values::NamedTuple
end

ax, values = Newtrinos.profile(llh, priors_dict, vars_to_scan, params_dict, cache_dir=name)
#ax, values = Newtrinos.scan(llh, priors_dict, vars_to_scan, params_dict)

axes = NamedTuple{tuple(keys(vars_to_scan)...)}(ax)

result = NewtrinosResult(axes=axes, values=values)

FileIO.save(name * ".jld2", Dict("result" => result))
