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
import JLD2

adsel = AutoForwardDiff()
context = set_batcontext(ad = adsel)

###### CONFIG ######

# Name for output files etc
name = "deepcore_test_new_flux"

# Choice of MCMC, Profile, Scan
task = "Profile"

# Choose oscillation model
osc = Newtrinos.osc.standard

# Choose experiments to include
modules = [Newtrinos.deepcore]#, Newtrinos.kamland, Newtrinos.dayabay, Newtrinos.minos]

# Variables to condition on (=fix)
conditional_vars = [:θ₁₂, :θ₁₃, :δCP, :Δm²₂₁, :nutau_cc_norm]

# For profile / scan task only: choose scan grid
vars_to_scan = OrderedDict()
vars_to_scan[:θ₂₃] = 31
vars_to_scan[:Δm²₃₁] = 31

###### END CONFIG ######

llhs = [let osc_prob = osc.osc_prob, observed = m.observed 
        params -> logpdf(m.forward_model(osc_prob)(params), observed)
        end
        for m in modules]

llh = logfuncdensity(params -> sum([f(params) for f in llhs]))

params_dict = merge(osc.params, [m.params for m in modules]...)
priors_dict = merge(osc.priors, [m.priors for m in modules]...)

for var in conditional_vars
    priors_dict[var] = params_dict[var]
end

if task == "MCMC"
    prior = distprod(;priors_dict...)
    posterior = PosteriorMeasure(llh, prior)
    samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^4, nchains = 4)).result
    FileIO.save(name * ".jld2", Dict("samples" => samples))

else
    if task == "Profile"
        result = Newtrinos.profile(llh, priors_dict, vars_to_scan, params_dict, cache_dir=name)

    elseif task == "Scan"
        result = Newtrinos.scan(llh, priors_dict, vars_to_scan, params_dict)

    end 
    FileIO.save(name * ".jld2", Dict("result" => result))

end
