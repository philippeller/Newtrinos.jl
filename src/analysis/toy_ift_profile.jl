"""
Toy validation for ift_profile: 2D correlated Gaussian posterior.

For a quadratic log-posterior, the IFT predictor is exact — one Newton step
reaches the true optimum. This verifies the MGVI covariance + gradient computation
before running on real experiments.

Setup:
  - Scan variable: θ, with Uniform(-2, 2) prior (determines scan range)
  - Nuisance variable: p, with Normal(0, σ_prior) broad prior
  - Likelihood: (θ, p) ~ MvNormal([0, 0], Σ_true)
    modeled as y=[0,0] ~ Normal(U * [θ, p], I) where U'U = Λ_true
  - Analytic profile optimum: p*(θ) ≈ (Σ_true[2,1]/Σ_true[1,1]) · θ

Usage:
  julia --project src/analysis/toy_ift_profile.jl
"""

using Newtrinos
using BAT
using Distributions
using LinearAlgebra
using DataStructures
using Printf

# ── Toy posterior setup ────────────────────────────────────────────────────────

σ_prior = 10.0   # broad prior for p: posterior dominated by likelihood
σ_θ     = 1.0
σ_p     = 0.5
ρ       = 0.7
Σ_true  = [σ_θ^2        ρ*σ_θ*σ_p;
            ρ*σ_θ*σ_p   σ_p^2]
Λ_true  = inv(Σ_true)   # precision matrix

# Exact analytic profile optimum (penalized likelihood with Normal(0, σ_prior) prior on p):
# In z-space (z_p = p/σ_prior): log-posterior = -½z_p² - ½[θ,σ_prior*z_p]'Λ[θ,σ_prior*z_p]
# Mode: z_p*(θ) = -σ_prior*Λ[2,1]*θ / (1 + σ_prior²*Λ[2,2]) → p*(θ) = σ_prior*z_p*(θ)
p_star_coeff = -σ_prior^2 * Λ_true[2, 1] / (1 + σ_prior^2 * Λ_true[2, 2])
p_star_of_theta = θ -> p_star_coeff * θ
@info "Exact analytic profile optimum: p*(θ) = $(round(p_star_coeff, digits=6)) · θ  (asymptotic: $(round(Σ_true[2,1]/Σ_true[1,1], digits=4)) for σ_prior→∞)"

# MGVI-compatible likelihood: observe y=[0,0] from Normal(U * [θ, p], I)
# where U = chol(Λ_true).U so U'U = Λ_true →
# log P(y=0|θ,p) = -½ ||U*[θ,p]||² = -½ [θ,p]' Λ_true [θ,p]
U = cholesky(Λ_true).U
y_obs = [0.0, 0.0]

fwd_model = params -> Newtrinos.distprod(Normal.(U * [params.θ, params.p], 1.0))
likelihood = BAT.likelihoodof(fwd_model, y_obs)

# Priors: Uniform for θ (finite scan range ±2σ_θ); broad Normal for p
priors_full = (
    θ = Uniform(-2.0, 2.0),
    p = Normal(0.0, σ_prior),
)

# Starting params
params_0 = (θ = 0.0, p = 0.0)

# Scan grid: θ from -2 to 2 in 11 steps (via quantiles of Uniform(-2, 2))
vars_to_scan = OrderedDict(:θ => 11)

# ── Run ift_profile ────────────────────────────────────────────────────────────

@info "Running ift_profile (no polish, pure IFT prediction)..."
result_nopol = Newtrinos.ift_profile(likelihood, priors_full, vars_to_scan, params_0;
                                      start_from=(θ=0.0,), polish=false)

@info "Running ift_profile (with LBFGS polish)..."
result_pol = Newtrinos.ift_profile(likelihood, priors_full, vars_to_scan, params_0;
                                    start_from=(θ=0.0,), polish=true)

# ── Compare to analytic result ─────────────────────────────────────────────────

θ_grid = result_nopol.axes.θ
@printf "\n%-8s  %-12s  %-12s  %-12s  %-12s\n" "θ" "p*(θ) true" "IFT pred" "IFT polished" "IFT err"
@printf "%-8s  %-12s  %-12s  %-12s  %-12s\n" "─"^8 "─"^12 "─"^12 "─"^12 "─"^12

max_err_vs_analytic = 0.0
max_err_pred_vs_pol = 0.0
for (i, θ) in enumerate(θ_grid)
    global max_err_vs_analytic, max_err_pred_vs_pol
    p_true  = p_star_of_theta(θ)
    p_nopol = result_nopol.values.p[i]
    p_pol   = result_pol.values.p[i]
    max_err_vs_analytic = max(max_err_vs_analytic, abs(p_nopol - p_true))
    max_err_pred_vs_pol = max(max_err_pred_vs_pol, abs(p_nopol - p_pol))
    @printf "%-8.3f  %-12.6f  %-12.6f  %-12.6f  %-12.2e\n" θ p_true p_nopol p_pol abs(p_nopol - p_true)
end

println()
@info "Max |IFT pred - analytic exact| = $max_err_vs_analytic  (should be ~0)"
@info "Max |IFT pred - polished|        = $max_err_pred_vs_pol  (key test: should be ~0)"

if max_err_vs_analytic < 1e-4
    @info "✓ IFT predictor matches exact analytic optimum for quadratic posterior"
else
    @warn "IFT vs analytic error is $max_err_vs_analytic — check implementation"
end

if max_err_pred_vs_pol < 1e-6
    @info "✓ IFT is exact (polishing adds nothing): one Newton step = true optimum"
else
    @warn "IFT pred differs from polished by $max_err_pred_vs_pol — not exact for quadratic posterior"
end
