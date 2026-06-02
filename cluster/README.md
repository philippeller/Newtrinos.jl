# Cluster Execution for JUNO-TAO NNM Profile

This directory contains scripts to run the profile likelihood computation on a Slurm cluster with **one point per job** (no threading).

## Structure

- `single_point.jl`  - Worker script: computes ONE scan point
- `submit.sh`       - Slurm submission script (job array)
- `combine.jl`      - Combines all results into final output
- `results/`        - Directory for per-point results (created automatically)
- `logs/`           - Directory for Slurm logs (created automatically)

## Quick Start

### 1. Submit all jobs
```bash
sbatch cluster/submit.sh
```

This submits 961 jobs (31 × 31 grid). Each job:
- Uses 1 CPU core
- Requests 2GB memory
- Runs for up to 1 hour
- Saves output to `cluster/results/point_<N>.jld2`

### 2. Monitor progress
```bash
squeue -u $USER
ls cluster/results/ | wc -l  # Count completed files
```

### 3. Combine results (after all jobs finish)
```bash
julia cluster/combine.jl
```

Output: `juno_NND_profile_combined.jld2`

## Customization

### Change grid resolution
Edit `vars_to_scan` in `single_point.jl`:
```julia
vars_to_scan = (r=31, N=31)  # Change these values
```
Then update `TOTAL_POINTS` in `submit.sh` and `combine.jl` to match `r * N`.

### Memory/Time limits
Edit `submit.sh`:
```bash
#SBATCH --mem=2G        # Increase if jobs fail with OOM
#SBATCH --time=1:00:00  # Increase if jobs timeout
```

### Julia environment
The scripts use `--project=@.` to activate the current project. If your cluster has Julia modules:
```bash
# Uncomment and adjust in submit.sh:
module load julia/1.9.0
```

## Notes

- Each job **independently recreates** the physics configuration and toy data. This ensures reproducibility but may increase startup time.
- The `cache/` directory is shared across jobs (same as threaded version).
- Results are saved as individual `.jld2` files for fault tolerance. If a job fails, only that point needs to be recomputed.
