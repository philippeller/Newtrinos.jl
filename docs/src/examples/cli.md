# CLI Reference

The main analysis entry point is `src/analysis/analysis.jl`.

## Usage

```bash
julia --project src/analysis/analysis.jl \
    --experiments deepcore dayabay \
    --name myrun \
    --task profile \
    --workers 4 \
    --threads 2 \
    --plot
```

## Arguments

| Argument | Required | Default | Description |
|:---------|:---------|:--------|:------------|
| `--experiments` | Yes | — | Space-separated list of experiments |
| `--name` | Yes | — | Base name for output files (`.jld2`, `.png`) |
| `--task` | Yes | — | One of: `Scan`, `Profile`, `NestedSampling`, `ImportanceSampling` |
| `--workers` | No | 1 | Number of distributed workers |
| `--threads` | No | 1 | Threads per worker (when `--workers > 1`) |
| `--plot` | No | false | Generate a PNG plot of results |

## Tasks

### Scan

Evaluates the likelihood on a grid of parameter values. Fast but does not account for nuisance parameter variations.

```bash
julia -t 4 --project src/analysis/analysis.jl \
    --experiments dayabay --name scan_test --task scan
```

### Profile

Profile likelihood scan: at each grid point, optimizes over all nuisance parameters using LBFGS. Results are cached to disk for resumability.

```bash
julia --project src/analysis/analysis.jl \
    --experiments deepcore dayabay --name profile_test --task profile --workers 4
```

### NestedSampling

Bayesian inference via nested sampling (requires UltraNest).

### ImportanceSampling

Adaptive importance sampling via the mole-whacking algorithm.

## Parallelism

- **`--workers 1`** (default): Grid points parallelized via Julia threads. Control thread count at launch with `julia -t N`.
- **`--workers N`**: Spawns N distributed processes via `addprocs`. Grid points distributed via `pmap`. Use `--threads M` to set threads per worker.

## Output

- `<name>.jld2` — Results file containing a `NewtrinosResult` struct
- `<name>.png` — Contour plot (if `--plot` is set)

## Benchmarks

```bash
julia -t 4 --project benchmark/bench_likelihood.jl --experiments deepcore
julia --project benchmark/bench_osc.jl
```
