# CLI Analysis
Usually, the main analysis entry point is `src/analysis/analysis.jl`. You run this file from the command line and specify the the experiments, analysis tasks, number of threads, and number of distributed workers. You must also specify a base name for the output files. If you want you can also add the plot feature and generate a plot of results right away.

## Usage

You use the analysis script in the command line via:

```bash
julia --project src/analysis/analysis.jl \
    --experiments deepcore dayabay \
    --name myrun \
    --task profile \
    --workers 4 \
    --threads 2 \
    --plot
```

### Arguments

Some arguments are optional, so here is a short overview of the different options:

| Argument | Required | Default | Description |
|:---------|:---------|:--------|:------------|
| `--experiments` | Yes | — | Space-separated list of experiments |
| `--name` | Yes | — | Base name for output files (`.jld2`, `.png`) |
| `--task` | Yes | — | One of: `Scan`, `Profile`, `NestedSampling`, `ImportanceSampling` |
| `--workers` | No | 1 | Number of distributed workers |
| `--threads` | No | 1 | Threads per worker (when `--workers > 1`) |
| `--plot` | No | false | Generate a PNG plot of results |

### Physics modifications

The `analysis.jl` file use the default physics configuration. If you want to use a different physics model you must uncomment the physics configuration block in the file and specify it as described in [here](./physics_model.md).

```julia
# To use defaults:
experiments = configure_experiments(args["experiments"])

# To override physics (e.g. for IO, sterile models, custom flux, etc.), uncomment and modify:
# osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(
#     flavour=Newtrinos.osc.ThreeFlavour(ordering=:IO),
#     interaction=Newtrinos.osc.SI(),
# ))
# atm_flux = Newtrinos.atm_flux.configure()
# earth_layers = Newtrinos.earth_layers.configure()
# xsec = Newtrinos.xsec.configure()
# physics = (; osc, atm_flux, earth_layers, xsec)
# experiments = configure_experiments(args["experiments"], physics)
```

### Tasks

#### Scan and Profile
The scan grid and conditional parameters for a conditional likelihood or a likelihood profile scan must be specified in the `analysis.jl` file. You do this at the end of the `PHYSICS CONFIG` block by putting in the parameters that you need.

```julia
# Variables to condition on (=fix)
conditional_vars = Dict(:θ₁₂=>p.θ₁₂, :δCP=>-1.89, :Δm²₂₁=>p.Δm²₂₁)

# For profile / scan task only: choose scan grid
vars_to_scan = OrderedDict()
vars_to_scan[:θ₂₃] = 11
vars_to_scan[:Δm²₃₁] = 11
```

##### Scan

Evaluates the likelihood on a grid of parameter values. Fast but does not account for nuisance parameter variations.

```bash
julia -t 4 --project src/analysis/analysis.jl \
    --experiments dayabay --name scan_test --task scan
```
This example uses 4 julia threads for executing the scan task.

##### Profile

Profile likelihood scan: at each grid point, optimizes over all nuisance parameters using LBFGS. Results are cached to disk for resumability.

```bash
julia --project src/analysis/analysis.jl \
    --experiments deepcore dayabay --name profile_test --task profile --workers 4
```

This example uses 4 distributed workers for executing the profile task.

#### NestedSampling

Bayesian inference via nested sampling (requires UltraNest). Nested Sampling is a Bayesian inference algorithm that estimates the model evidence (marginal likelihood) by iteratively replacing the least likely point in a set of "live points" with a new sample of higher likelihood, effectively shrinking from the full prior toward the peak of the posterior.

#### ImportanceSampling

Adaptive importance sampling via the mole-whacking algorithm which was covered in the [analysis tutorial](./analyse_data.md).

#### Parallelism
To accelerate the execution of or task we can use the parallelism options provided by the `Newtrinos.jl` package. This includes julias multithreading where we run threads in parallel on several CPU cores but inside the same memory space. We can also use distributed workers to separate julia and parallelize processes, each with their own memory space. For specifying the number of threads and workers via `--threads` and `--workers` you need to make sure that the specifications are compatible with your system.

- **`--workers 1`** (default): Grid points parallelized via Julia threads. Control the thread count at launch with `julia -t N`.
- **`--workers N`**: Spawns N distributed processes via `addprocs`. Grid points distributed via `pmap`. Use `--threads M` to set M threads per worker.

## Output

- `<name>.jld2` — Results file containing a `NewtrinosResult` struct
- `<name>.png` — Contour plot (if `--plot` is set)

## Benchmarks

```bash
julia -t 4 --project benchmark/bench_likelihood.jl --experiments deepcore
julia --project benchmark/bench_osc.jl
```
