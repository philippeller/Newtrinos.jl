#!/bin/bash
# Slurm submission script for JUNO-TAO profile likelihood
# Total points: 31 x 31 = 961
# Each job computes ONE point

#SBATCH --job-name=nnd_profile
#SBATCH --array=1-961%25
#SBATCH --output=cluster/logs/slurm_%A_%a.out
#SBATCH --error=cluster/logs/slurm_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1:00:00

# Load required modules (adjust for your cluster)
# module load julia/1.9.0

# Run the worker script
echo "Starting job $SLURM_ARRAY_TASK_ID at $(date)"
julia --project=@. cluster/single_point.jl $SLURM_ARRAY_TASK_ID 961
echo "Finished job $SLURM_ARRAY_TASK_ID at $(date)"
