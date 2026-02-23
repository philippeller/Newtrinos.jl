
using Base.Threads
println("Threads available: ", nthreads())

# --- Parse command-line arguments: MODEL ORDERING [--dry-run] ---
# Usage: julia junotao.jl [NNM|NND] [NO|IO] [--dry-run]
dry_run = "--dry-run" in ARGS
positional = filter(a -> !startswith(a, "-"), ARGS)

model_name = length(positional) >= 1 ? uppercase(positional[1]) : "NNM"
ordering_name = length(positional) >= 2 ? uppercase(positional[2]) : "NO"

model_name in ("NNM", "NND") || error("Unknown model: $model_name. Use NNM or NND.")
ordering_name in ("NO", "IO") || error("Unknown ordering: $ordering_name. Use NO or IO.")

ordering_sym = Symbol(ordering_name)
println("Running: model=$model_name, ordering=$ordering_name", dry_run ? " [DRY RUN]" : "")

using LinearAlgebra
using Distributions
using LaTeXStrings
using Printf
using FileIO
import JLD2
using DataFrames
using Accessors
using DensityInterface: logdensityof

using Newtrinos
using CairoMakie

OUTDIR = joinpath(@__DIR__, "results", "junotao")
mkpath(OUTDIR)

# --- Build oscillation config for the chosen model ---
three_flavour = Newtrinos.osc.ThreeFlavour(ordering=ordering_sym)

flavour = if model_name == "NNM"
    Newtrinos.osc.NNM(three_flavour=three_flavour)
elseif model_name == "NND"
    Newtrinos.osc.NND(three_flavour=three_flavour)
end

osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=flavour,
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
)

osc = Newtrinos.osc.configure(osc_cfg)

atm_flux = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec = Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec);

experiments = (
   juno = Newtrinos.juno.configure(physics),
   tao = Newtrinos.tao.configure(physics),
);

p = Newtrinos.get_params(experiments)

all_priors = Newtrinos.get_priors(experiments)

# Override r prior: use LogUniform instead of Uniform
@reset all_priors.r = LogUniform(1e-8, 1.0)

# --- SM configuration for toy data generation ---
osc_cfg_SM = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.ThreeFlavour(ordering=ordering_sym),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
)

osc_SM = Newtrinos.osc.configure(osc_cfg_SM)

atm_flux_SM = Newtrinos.atm_flux.configure()
earth_layers_SM = Newtrinos.earth_layers.configure()
xsec_SM = Newtrinos.xsec.configure()

physics_SM = (; osc=osc_SM, atm_flux=atm_flux_SM, earth_layers=earth_layers_SM, xsec=xsec_SM);

experiments_SM = (
   juno = Newtrinos.juno.configure(physics_SM),
   tao = Newtrinos.tao.configure(physics_SM),
);

p_SM = Newtrinos.get_params(experiments_SM)

# Complete parameters with TAO systematics
p_complete = merge(p, (
        tao_detection_epsilon = 1.0,
        tao_res_a = 0.015,
        tao_res_b = 0.0,
        tao_res_c = 0.0,
        tao_accidental_norm = 1.0,
        tao_fast_neutron_norm = 1.0,
        tao_lihe_norm = 1.0,
))

p_complete_SM = merge(p_SM, (
        tao_detection_epsilon = 1.0,
        tao_res_a = 0.015,
        tao_res_b = 0.0,
        tao_res_c = 0.0,
        tao_accidental_norm = 1.0,
        tao_fast_neutron_norm = 1.0,
        tao_lihe_norm = 1.0,
))

# Generate toy data from SM
println("Generating toy data (SM)...")
toy_data_j = Newtrinos.generate_toy_data(experiments_SM.juno, p_complete_SM)
toy_data_j = Float64.(toy_data_j)

toy_data_t = Newtrinos.generate_toy_data(experiments_SM.tao, p_complete_SM)
toy_data_t = Float64.(toy_data_t)

toy_data = (juno = toy_data_j, tao = toy_data_t)
println("  JUNO toy data: $(length(toy_data_j)) bins, sum=$(sum(toy_data_j))")
println("  TAO  toy data: $(length(toy_data_t)) bins, sum=$(sum(toy_data_t))")

# --- Scan ---
m0_values = [1e-2]

tag = "$(model_name)_$(ordering_name)"

for i in 1:length(m0_values)

    m0 = m0_values[i]
    p_complete_new = merge(p_complete, (m₀ = m0,))

    vars_to_scan = (r=31, N=31)
    n_points = prod(values(vars_to_scan))

    likelihood = Newtrinos.generate_likelihood(experiments, toy_data);

    # --- Dry run: validate setup without running full scan ---
    if dry_run
        println("\n=== DRY RUN: $tag, m0=$m0 ===")
        println("Scan grid: $(vars_to_scan) -> $n_points points")
        println("Priors:")
        for (k, v) in pairs(all_priors)
            println("  $k: $v")
        end
        println("Parameters (non-prior):")
        scan_keys = keys(vars_to_scan)
        prior_keys = keys(all_priors)
        for (k, v) in pairs(p_complete_new)
            k in prior_keys && continue
            println("  $k = $v")
        end

        # Single-point likelihood evaluation at default params
        println("\nSingle-point likelihood evaluation at default params...")
        t = @elapsed llh_val = logdensityof(likelihood, p_complete_new)
        println("  loglikelihood = $llh_val")
        println("  time = $(round(t, digits=3))s")

        # Test at a corner point (small r, small N)
        p_corner = merge(p_complete_new, (r=1e-8, N=2))
        println("\nSingle-point at r=1e-8, N=2...")
        t = @elapsed llh_corner = logdensityof(likelihood, p_corner)
        println("  loglikelihood = $llh_corner")
        println("  time = $(round(t, digits=3))s")

        # Test at opposite corner (r=1, large N)
        p_corner2 = merge(p_complete_new, (r=1.0, N=80))
        println("\nSingle-point at r=1, N=80...")
        t = @elapsed llh_corner2 = logdensityof(likelihood, p_corner2)
        println("  loglikelihood = $llh_corner2")
        println("  time = $(round(t, digits=3))s")

        # Check output paths are writable
        test_jld2 = joinpath(OUTDIR, "junotao_rN_$(tag)_m0=$m0.jld2")
        test_png  = joinpath(OUTDIR, "junotao_rN_$(tag)_m0=$m0.png")
        println("\nOutput paths:")
        println("  JLD2: $test_jld2")
        println("  PNG:  $test_png")
        println("  OUTDIR exists: $(isdir(OUTDIR))")
        println("  OUTDIR writable: $(try; touch(joinpath(OUTDIR, ".write_test")); rm(joinpath(OUTDIR, ".write_test")); true; catch; false; end)")

        println("\n=== DRY RUN PASSED ===")
        continue
    end

    # --- Full scan ---
    result = Newtrinos.scan(likelihood, all_priors, vars_to_scan, p_complete_new)

    JLD2.@save joinpath(OUTDIR, "junotao_rN_$(tag)_m0=$m0.jld2") result

    img = CairoMakie.plot(result; title="JUNO+TAO $tag - LogLikelihood r vs N, m0=$m0", log=0, mass=0)

    save(joinpath(OUTDIR, "junotao_rN_$(tag)_m0=$m0.png"), img)

end
