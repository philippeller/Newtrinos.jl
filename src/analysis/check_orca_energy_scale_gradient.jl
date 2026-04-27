"""
Diagnostic: check likelihood landscape and gradient quality for orca_energy_scale.

Runs three checks:
  1. 1D likelihood profile over orca_energy_scale (fixed all other params) — is the LLH smooth?
  2. Gradient via ForwardDiff at each point — smooth / does it agree with finite differences?
  3. Finite-difference gradient (central differences) for cross-check

Usage:
  julia --project src/analysis/check_orca_energy_scale_gradient.jl
"""

using Newtrinos
using DensityInterface
import ForwardDiff
using CairoMakie
using Accessors

@info "Configuring ORCA..."
exp = Newtrinos.orca.configure()
experiments = (orca = exp,)
p = Newtrinos.get_params(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)

llh0 = logdensityof(likelihood, p)
@info "Default LLH = $llh0"

# 1D sweep over orca_energy_scale
scale_range = range(0.88, 1.12, length=51)

llh_vals  = Float64[]
grad_vals = Float64[]  # ForwardDiff gradient wrt orca_energy_scale

@info "Sweeping orca_energy_scale from $(first(scale_range)) to $(last(scale_range))..."

for s in scale_range
    p_s = @set p.orca_energy_scale = s

    # Likelihood value
    push!(llh_vals, logdensityof(likelihood, p_s))

    # ForwardDiff gradient wrt orca_energy_scale only (scalar AD)
    g = ForwardDiff.derivative(scale -> logdensityof(likelihood, @set p.orca_energy_scale = scale), s)
    push!(grad_vals, g)
end

# Finite-difference gradient for cross-check (central differences, h=0.002)
h = 0.002
fd_grad = Float64[]
for (i, s) in enumerate(scale_range)
    s_lo = max(first(scale_range), s - h)
    s_hi = min(last(scale_range),  s + h)
    # find nearest precomputed values
    i_lo = max(1, i-1)
    i_hi = min(length(scale_range), i+1)
    ds = collect(scale_range)[i_hi] - collect(scale_range)[i_lo]
    push!(fd_grad, (llh_vals[i_hi] - llh_vals[i_lo]) / ds)
end

@info "Max |ForwardDiff - FD central| = $(maximum(abs.(grad_vals .- fd_grad)))"
@info "Likelihood range: $(minimum(llh_vals)) to $(maximum(llh_vals))"

# Plot
fig = Figure(size=(900, 600))

ax1 = Axis(fig[1,1], xlabel="orca_energy_scale", ylabel="LLH", title="Likelihood vs energy scale (all other params fixed)")
lines!(ax1, collect(scale_range), llh_vals, color=:blue)
scatter!(ax1, [1.0], [llh0], color=:red, markersize=10, label="default")
axislegend(ax1)

ax2 = Axis(fig[2,1], xlabel="orca_energy_scale", ylabel="dLLH/d(scale)", title="Gradient: ForwardDiff (blue) vs finite differences (orange dashed)")
lines!(ax2, collect(scale_range), grad_vals, color=:blue, label="ForwardDiff")
lines!(ax2, collect(scale_range), fd_grad, color=:orange, linestyle=:dash, label="finite diff (central)")
axislegend(ax2)

outfile = joinpath(@__DIR__, "orca_energy_scale_gradient_check.png")
save(outfile, fig)
@info "Saved plot to $outfile"
