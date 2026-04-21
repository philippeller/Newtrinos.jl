#!/usr/bin/env julia
# Test: remove bathtub flux norm (fix sk_flux_norm_high, sk_flux_norm_low, and
# atm_flux_norm_ratio_hi, atm_flux_norm_ratio_lo to their nominal values)
# Keep sk_total_norm as the only overall normalization freedom

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../../.."))

using Newtrinos, Accessors, Printf, JLD2
using BAT, DensityInterface, Distributions, ValueShapes

sk = Newtrinos.super_k.configure()
experiments = (super_k=sk,)
params = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)

# Start from previous best-fit
@reset params.Δm²₃₁ = 0.002471892652340831
@reset params.θ₂₃ = 0.7715663639056108
@reset params.θ₁₃ = 0.14627305077324582
@reset params.δCP = 3.7699111843077517
@reset params.sk_total_norm = 1.0859840692424074
@reset params.xsec_cc1p1h_shape = 0.5200493562001464
@reset params.xsec_cc1pi_shape = 1.459204145906447
@reset params.sk_flux_norm_high = 0.0
@reset params.sk_flux_norm_low = 0.0
@reset params.xsec_cc1p1h_subgev_norm = 0.8798531573440069
@reset params.xsec_cc1p1h_multigev_norm = 0.8798531573440069
@reset params.xsec_nc_norm = 1.1196641134274956
@reset params.sk_fc_multigev_rel_norm = 1.0118968918009859
@reset params.sk_multigev_ring_counting = 0.9677503494043178
@reset params.sk_i_v_bdt_1 = 0.9162341049294918
@reset params.atm_flux_norm_ratio_hi = 1.0
@reset params.atm_flux_norm_ratio_lo = 1.0

# Fix the energy-dependent flux norms
priors_mod = deepcopy(priors)
@reset priors_mod.sk_flux_norm_high = ValueShapes.ConstValueDist(0.0)
@reset priors_mod.sk_flux_norm_low = ValueShapes.ConstValueDist(0.0)
@reset priors_mod.atm_flux_norm_ratio_hi = ValueShapes.ConstValueDist(1.0)
@reset priors_mod.atm_flux_norm_ratio_lo = ValueShapes.ConstValueDist(1.0)

prior = distprod(;priors_mod...)

println("Test: No bathtub flux norm, only sk_total_norm for overall normalization")
println(@sprintf "Start: sin²θ₂₃=%.4f" sin(params.θ₂₃)^2)
flush(stdout)

llh, log_post, opt_params = Newtrinos.find_mle(likelihood, prior, params)

outpath = joinpath(@__DIR__, "mle_result_nofluxnorm.jld2")
jldsave(outpath; llh=llh, log_posterior=log_post, opt_params=opt_params)
println("Saved to $outpath")

println("\n=== Result: No bathtub flux norm ===")
@printf "sin²θ₂₃ = %.4f\n" sin(opt_params.θ₂₃)^2
@printf "Δm²₃₁ = %.6e\n" opt_params.Δm²₃₁
@printf "δCP = %.4f\n" opt_params.δCP
@printf "sin²θ₁₃ = %.4f\n" sin(opt_params.θ₁₃)^2
@printf "sk_total_norm = %.4f\n" opt_params.sk_total_norm
@printf "xsec_cc1p1h_shape = %.4f\n" opt_params.xsec_cc1p1h_shape
@printf "xsec_cc1pi_shape = %.4f\n" opt_params.xsec_cc1pi_shape
@printf "xsec_cc1p1h_subgev_norm = %.4f\n" opt_params.xsec_cc1p1h_subgev_norm
@printf "xsec_cc1p1h_multigev_norm = %.4f\n" opt_params.xsec_cc1p1h_multigev_norm
@printf "xsec_nc_norm = %.4f\n" opt_params.xsec_nc_norm
@printf "sk_fiducial_norm = %.4f\n" opt_params.sk_fiducial_norm
@printf "sk_fc_norm = %.4f\n" opt_params.sk_fc_norm
@printf "llh = %.4f\n" llh
@printf "log_posterior = %.4f\n" log_post
