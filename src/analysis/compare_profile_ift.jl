"""
Compare profile (sequential warm-start) vs ift_profile on ORCA θ₂₃ scan.

Run:
  julia --project --threads=auto src/analysis/compare_profile_ift.jl
"""

using Newtrinos
using BAT
using Distributions
using DensityInterface
using DataStructures
using Accessors
using CairoMakie

# ── Setup ──────────────────────────────────────────────────────────────────────

exp = Newtrinos.orca.configure()
experiments = (orca = exp,)
p      = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

Newtrinos.set_ad_backend(:auto)
set_batcontext(ad = Newtrinos.select_ad(length(p)))

likelihood = Newtrinos.generate_likelihood(experiments)

# Fix parameters ORCA is insensitive to
cond_vars = Dict(:θ₁₂ => p.θ₁₂, :Δm²₂₁ => p.Δm²₂₁, :θ₁₃ => p.θ₁₃, :δCP => p.δCP)
priors_cond = Newtrinos.condition(priors, cond_vars, p)

# Scan grid
vars_to_scan = OrderedDict(:θ₂₃ => 21)

nuisance_keys = setdiff(collect(keys(p)), [:θ₂₃, :θ₁₂, :Δm²₂₁, :θ₁₃, :δCP])
@info "Free nuisance parameters ($(length(nuisance_keys))): $(nuisance_keys)"
@info "Scan: θ₂₃ over $(vars_to_scan[:θ₂₃]) points"

# ── Old method: profile with sequential warm-starting ──────────────────────────

@info "Running profile (sequential warm-start)..."
t0 = time()
result_old = Newtrinos.profile(likelihood, priors_cond, vars_to_scan, p;
    sequential = true,
    start_from = (θ₂₃ = p.θ₂₃,))
t_old = time() - t0
@info "profile done in $(round(t_old, digits=1))s"

# ── New method: ift_profile ────────────────────────────────────────────────────

@info "Running ift_profile..."
t0 = time()
result_ift = Newtrinos.ift_profile(likelihood, priors_cond, vars_to_scan, p;
    start_from = (θ₂₃ = p.θ₂₃,),
    polish = true)
t_ift = time() - t0
@info "ift_profile done in $(round(t_ift, digits=1))s"

# ── Plot ───────────────────────────────────────────────────────────────────────

θ_grid = result_old.axes.θ₂₃

llh_old = result_old.values.llh
llh_ift = result_ift.values.llh
Δχ²_old = -2 .* (llh_old .- maximum(llh_old))
Δχ²_ift = -2 .* (llh_ift .- maximum(llh_ift))

dm31_old  = result_old.values.Δm²₃₁
dm31_ift  = result_ift.values.Δm²₃₁
escale_old = result_old.values.orca_energy_scale
escale_ift = result_ift.values.orca_energy_scale
norm_old  = result_old.values.orca_norm_all
norm_ift  = result_ift.values.orca_norm_all

fig = Figure(size=(900, 1100))

ax1 = Axis(fig[1, 1],
    xlabel = "θ₂₃",
    ylabel = "Δχ²",
    title  = "ORCA 1D θ₂₃ profile: sequential profile vs ift_profile (42 nuisances)")

lines!(ax1, θ_grid, Δχ²_old, label="profile (sequential)", color=:steelblue, linewidth=2)
scatter!(ax1, θ_grid, Δχ²_old, color=:steelblue, markersize=6)
lines!(ax1, θ_grid, Δχ²_ift, label="ift_profile (polish)", color=:crimson,
       linewidth=2, linestyle=:dash)
scatter!(ax1, θ_grid, Δχ²_ift, color=:crimson, markersize=6, marker=:diamond)
hlines!(ax1, [1.0, 4.0], color=:gray, linestyle=:dot, linewidth=1)
axislegend(ax1, position=:lt)

ax2 = Axis(fig[2, 1],
    xlabel = "θ₂₃",
    ylabel = "Δm²₃₁  [eV²]",
    title  = "Nuisance: Δm²₃₁")

lines!(ax2, θ_grid, dm31_old, color=:steelblue, linewidth=2, label="profile")
scatter!(ax2, θ_grid, dm31_old, color=:steelblue, markersize=6)
lines!(ax2, θ_grid, dm31_ift, color=:crimson, linewidth=2, linestyle=:dash, label="ift_profile")
scatter!(ax2, θ_grid, dm31_ift, color=:crimson, markersize=6, marker=:diamond)
axislegend(ax2, position=:lt)

ax3 = Axis(fig[3, 1],
    xlabel = "θ₂₃",
    ylabel = "orca_energy_scale",
    title  = "Nuisance: orca_energy_scale")

lines!(ax3, θ_grid, escale_old, color=:steelblue, linewidth=2, label="profile")
scatter!(ax3, θ_grid, escale_old, color=:steelblue, markersize=6)
lines!(ax3, θ_grid, escale_ift, color=:crimson, linewidth=2, linestyle=:dash, label="ift_profile")
scatter!(ax3, θ_grid, escale_ift, color=:crimson, markersize=6, marker=:diamond)
axislegend(ax3, position=:lt)

ax4 = Axis(fig[4, 1],
    xlabel = "θ₂₃",
    ylabel = "orca_norm_all",
    title  = "Nuisance: orca_norm_all")

lines!(ax4, θ_grid, norm_old, color=:steelblue, linewidth=2, label="profile")
scatter!(ax4, θ_grid, norm_old, color=:steelblue, markersize=6)
lines!(ax4, θ_grid, norm_ift, color=:crimson, linewidth=2, linestyle=:dash, label="ift_profile")
scatter!(ax4, θ_grid, norm_ift, color=:crimson, markersize=6, marker=:diamond)
axislegend(ax4, position=:lt)

Label(fig[5, 1],
    "profile: $(round(t_old, digits=1))s   |   ift_profile: $(round(t_ift, digits=1))s",
    tellwidth=false, fontsize=13)

save("compare_profile_ift.png", fig)
@info "Saved compare_profile_ift.png"
display(fig)
