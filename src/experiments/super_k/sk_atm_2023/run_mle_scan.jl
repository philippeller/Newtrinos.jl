#!/usr/bin/env julia
# Run MLE at two fixed θ₂₃ values to measure the actual Δχ² between them
# Also remove sk_total_norm to test if it helps

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../../.."))

using Newtrinos, Accessors, Printf, JLD2
using BAT, DensityInterface, Distributions, ValueShapes

sk = Newtrinos.super_k.configure()
experiments = (super_k=sk,)
params = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)

# Starting point from best previous fit (with sk_total_norm)
@reset params.Δm²₃₁ = 0.002471892652340831
@reset params.θ₁₃ = 0.14627305077324582
@reset params.δCP = 3.7699111843077517
@reset params.sk_total_norm = 1.0859840692424074
@reset params.xsec_cc1p1h_shape = 0.5200493562001464
@reset params.xsec_cc1pi_shape = 1.459204145906447
@reset params.sk_flux_norm_high = 0.6638253292609961
@reset params.sk_flux_norm_low = -0.6476531169410142
@reset params.xsec_cc1p1h_norm = 0.8803763874976104
@reset params.xsec_nc_norm = 1.1349243033945697
@reset params.sk_fc_multigev_rel_norm = 1.0150464973591837

results = Dict{String, Any}()

# Test 1: Free fit (with sk_total_norm)
println("=" ^ 60)
println("Test 1: Free fit with sk_total_norm")
println("=" ^ 60)
@reset params.θ₂₃ = asin(sqrt(0.484))  # start near previous best
prior1 = distprod(;priors...)
llh1, lp1, r1 = Newtrinos.find_mle(likelihood, prior1, params)
results["free_with_total_norm"] = (llh=llh1, lp=lp1, params=r1)
@printf "sin²θ₂₃=%.4f  llh=%.4f  lp=%.4f  total_norm=%.4f\n" sin(r1.θ₂₃)^2 llh1 lp1 r1.sk_total_norm
jldsave(joinpath(@__DIR__, "mle_scan_results.jld2"); results=results)
flush(stdout)

# Test 2: Fix θ₂₃ = 0.45, free everything else (with sk_total_norm)
println("\n" * "=" ^ 60)
println("Test 2: Fixed sin²θ₂₃=0.45 with sk_total_norm")
println("=" ^ 60)
priors2 = deepcopy(priors)
@reset priors2.θ₂₃ = ValueShapes.ConstValueDist(asin(sqrt(0.45)))
prior2 = distprod(;priors2...)
params2 = deepcopy(r1)  # start from best-fit of test 1
llh2, lp2, r2 = Newtrinos.find_mle(likelihood, prior2, params2)
results["fixed_045_with_total_norm"] = (llh=llh2, lp=lp2, params=r2)
@printf "sin²θ₂₃=%.4f  llh=%.4f  lp=%.4f  total_norm=%.4f\n" sin(r2.θ₂₃)^2 llh2 lp2 r2.sk_total_norm
@printf "Δllh(free - 0.45) = %.4f\n" llh1 - llh2
jldsave(joinpath(@__DIR__, "mle_scan_results.jld2"); results=results)
flush(stdout)

# Test 3: Remove sk_total_norm (fix to 1.0), free θ₂₃
println("\n" * "=" ^ 60)
println("Test 3: Free fit WITHOUT sk_total_norm")
println("=" ^ 60)
priors3 = deepcopy(priors)
@reset priors3.sk_total_norm = ValueShapes.ConstValueDist(1.0)
prior3 = distprod(;priors3...)
params3 = deepcopy(r1)
@reset params3.sk_total_norm = 1.0
llh3, lp3, r3 = Newtrinos.find_mle(likelihood, prior3, params3)
results["free_no_total_norm"] = (llh=llh3, lp=lp3, params=r3)
@printf "sin²θ₂₃=%.4f  llh=%.4f  lp=%.4f\n" sin(r3.θ₂₃)^2 llh3 lp3
jldsave(joinpath(@__DIR__, "mle_scan_results.jld2"); results=results)
flush(stdout)

# Test 4: Remove sk_total_norm AND fix θ₂₃ = 0.45
println("\n" * "=" ^ 60)
println("Test 4: Fixed sin²θ₂₃=0.45 WITHOUT sk_total_norm")
println("=" ^ 60)
priors4 = deepcopy(priors)
@reset priors4.sk_total_norm = ValueShapes.ConstValueDist(1.0)
@reset priors4.θ₂₃ = ValueShapes.ConstValueDist(asin(sqrt(0.45)))
prior4 = distprod(;priors4...)
params4 = deepcopy(r3)
llh4, lp4, r4 = Newtrinos.find_mle(likelihood, prior4, params4)
results["fixed_045_no_total_norm"] = (llh=llh4, lp=lp4, params=r4)
@printf "sin²θ₂₃=%.4f  llh=%.4f  lp=%.4f\n" sin(r4.θ₂₃)^2 llh4 lp4
@printf "Δllh(free - 0.45) without total_norm = %.4f\n" llh3 - llh4
jldsave(joinpath(@__DIR__, "mle_scan_results.jld2"); results=results)
flush(stdout)

# Summary
println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
@printf "With sk_total_norm:    free sin²θ₂₃=%.4f (llh=%.2f)  fixed 0.45 (llh=%.2f)  Δllh=%.2f\n" sin(r1.θ₂₃)^2 llh1 llh2 llh1-llh2
@printf "Without sk_total_norm: free sin²θ₂₃=%.4f (llh=%.2f)  fixed 0.45 (llh=%.2f)  Δllh=%.2f\n" sin(r3.θ₂₃)^2 llh3 llh4 llh3-llh4
