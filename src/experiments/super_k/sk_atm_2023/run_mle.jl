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
@reset params.xsec_cc1p1h_shape = 0.435466316028621
@reset params.xsec_cc1pi_shape = 1.435466316028621
@reset params.sk_flux_norm_high = 0.3994660286467456
@reset params.sk_flux_norm_low = -0.7448957257833595
@reset params.xsec_cc1p1h_subgev_norm = 0.8798531573440069
@reset params.xsec_cc1p1h_multigev_norm = 0.8798531573440069
@reset params.xsec_nc_norm = 1.1196641134274956

prior = distprod(;priors...)

println(@sprintf "Start: sin²θ₂₃=%.4f" sin(params.θ₂₃)^2)
flush(stdout)

llh, log_posterior, opt_params = Newtrinos.find_mle(likelihood, prior, params)

# Save full result to JLD2
outpath = joinpath(@__DIR__, "mle_result_tight_shape.jld2")
jldsave(outpath; llh=llh, log_posterior=log_posterior, opt_params=opt_params)
println("Saved to $outpath")

# Print key results
println("\n=== MLE Result ===")
@printf "llh = %.4f\n" llh
@printf "log_posterior = %.4f\n" log_posterior
@printf "sin²θ₂₃ = %.4f  (θ₂₃ = %.4f)\n" sin(opt_params.θ₂₃)^2 opt_params.θ₂₃
@printf "Δm²₃₁ = %.6e\n" opt_params.Δm²₃₁
@printf "sin²θ₁₃ = %.4f\n" sin(opt_params.θ₁₃)^2
@printf "δCP = %.4f\n" opt_params.δCP
@printf "sk_total_norm = %.4f\n" opt_params.sk_total_norm
@printf "xsec_cc1p1h_shape = %.4f\n" opt_params.xsec_cc1p1h_shape
@printf "xsec_cc1pi_shape = %.4f\n" opt_params.xsec_cc1pi_shape
@printf "sk_flux_norm_high = %.4f\n" opt_params.sk_flux_norm_high
@printf "sk_flux_norm_low = %.4f\n" opt_params.sk_flux_norm_low
@printf "xsec_nc_norm = %.4f\n" opt_params.xsec_nc_norm
@printf "xsec_cc1p1h_subgev_norm = %.4f\n" opt_params.xsec_cc1p1h_subgev_norm
@printf "xsec_cc1p1h_multigev_norm = %.4f\n" opt_params.xsec_cc1p1h_multigev_norm
