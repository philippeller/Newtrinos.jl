using Newtrinos
using CairoMakie
using LinearAlgebra
using StatsBase: midpoints

# --- Setup ---
earth_layers = Newtrinos.earth_layers.configure()
layers = earth_layers.compute_layers()

osc_basic = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI()))
osc_spray = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI(), propagation=Newtrinos.osc.Spray()))

params = Newtrinos.get_params((; osc=osc_basic, earth_layers))

# Grid: log10(E) from -1 to 2, cz from -1 to 0 (upgoing)
loge_grid = LinRange(-1, 2, 301)
cz_grid = LinRange(-1.0, 0.0, 201)

E = 10.0 .^ midpoints(loge_grid)
cz = midpoints(cz_grid)
paths = earth_layers.compute_paths(cz, layers)
dldcz = Newtrinos.earth_layers.compute_dldcz(cz, layers)

dlogE = Float64(step(loge_grid))
dCZ = Float64(step(cz_grid))

# --- Compute oscillation probabilities ---

# 1) No smoothing (Basic)
println("Computing Basic (no smoothing)...")
p_basic = osc_basic.osc_prob(E, paths, layers, params)
p_basic_anti = osc_basic.osc_prob(E, paths, layers, params; anti=true)

# 2) Gaussian blur (Basic + kernel smoothing, old Super-K method)
println("Computing Gaussian blur...")
K_E_mat = Newtrinos.super_k.make_gaussian_kernel_matrix(length(E), 0.10 / dlogE)
K_CZ_mat = Newtrinos.super_k.make_gaussian_kernel_matrix(length(cz), 0.10 / dCZ)
p_gauss = Newtrinos.super_k.smooth_osc_prob(p_basic, K_E_mat, K_CZ_mat)
p_gauss_anti = Newtrinos.super_k.smooth_osc_prob(p_basic_anti, K_E_mat, K_CZ_mat)

# 3) Spray E-only
println("Computing Spray (E only)...")
Delta_E = 0.10 .* E
p_spray_E = osc_spray.osc_prob(E, paths, layers, params; Delta_E)
p_spray_E_anti = osc_spray.osc_prob(E, paths, layers, params; anti=true, Delta_E)

# 4) Spray E+CZ
println("Computing Spray (E+CZ)...")
p_spray_ECZ = osc_spray.osc_prob(E, paths, layers, params; Delta_E, Delta_CZ=0.10, dldcz)
p_spray_ECZ_anti = osc_spray.osc_prob(E, paths, layers, params; anti=true, Delta_E, Delta_CZ=0.10, dldcz)

# --- Plot ---
logE_mid = midpoints(loge_grid)
cz_mid = midpoints(cz_grid)

function make_oscillogram_figure(probs_list, titles, channel_name, channel_idx; clims=(0, 1))
    n = length(probs_list)
    fig = Figure(size=(400*n + 80, 420))
    for (i, (p, title)) in enumerate(zip(probs_list, titles))
        ax = Axis(fig[1, i];
            xlabel = "log₁₀(E / GeV)",
            ylabel = i == 1 ? "cos θ_z" : "",
            title = title,
            titlesize = 14,
        )
        hm = heatmap!(ax, logE_mid, cz_mid, p[:, :, channel_idx...]';
            colorrange = clims,
            colormap = :RdBu,
        )
        if i == n
            Colorbar(fig[1, n+1], hm; label = channel_name)
        end
    end
    fig
end

println("Plotting...")

titles = ["No smoothing", "Gaussian blur", "Spray (E only)", "Spray (E+CZ)"]

# νμ → νμ survival
fig_surv = make_oscillogram_figure(
    [p_basic, p_gauss, p_spray_E, p_spray_ECZ],
    titles, "P(νμ → νμ)", (2, 2)
)
save("src/analysis/oscillogram_numu_survival.png", fig_surv; px_per_unit=2)
println("Saved oscillogram_numu_survival.png")

# νμ → νe appearance
fig_app = make_oscillogram_figure(
    [p_basic, p_gauss, p_spray_E, p_spray_ECZ],
    titles, "P(νμ → νe)", (2, 1);
    clims=(0, 0.5)
)
save("src/analysis/oscillogram_numu_appearance.png", fig_app; px_per_unit=2)
println("Saved oscillogram_numu_appearance.png")

# ν̄μ → ν̄μ survival (antineutrino)
fig_surv_anti = make_oscillogram_figure(
    [p_basic_anti, p_gauss_anti, p_spray_E_anti, p_spray_ECZ_anti],
    titles, "P(ν̄μ → ν̄μ)", (2, 2)
)
save("src/analysis/oscillogram_numubar_survival.png", fig_surv_anti; px_per_unit=2)
println("Saved oscillogram_numubar_survival.png")

# Difference: Spray(E+CZ) vs Spray(E only) — shows the CZ averaging effect
fig_diff = Figure(size=(1000, 420))
for (i, (channel_idx, channel_name, cl)) in enumerate([
    ((2, 2), "ΔP(νμ → νμ)", (-0.15, 0.15)),
    ((2, 1), "ΔP(νμ → νe)", (-0.1, 0.1)),
])
    ax = Axis(fig_diff[1, 2*(i-1)+1];
        xlabel = "log₁₀(E / GeV)",
        ylabel = i == 1 ? "cos θ_z" : "",
        title = "Spray(E+CZ) − Spray(E): $(channel_name)",
        titlesize = 14,
    )
    diff = p_spray_ECZ[:, :, channel_idx...] .- p_spray_E[:, :, channel_idx...]
    hm = heatmap!(ax, logE_mid, cz_mid, diff';
        colorrange = cl,
        colormap = :RdBu,
    )
    Colorbar(fig_diff[1, 2*i], hm; label = channel_name)
end
save("src/analysis/oscillogram_spray_ecz_vs_e.png", fig_diff; px_per_unit=2)
println("Saved oscillogram_spray_ecz_vs_e.png")

println("Done!")
