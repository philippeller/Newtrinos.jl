#!/bin/bash
# Slurm submission script for JUNO-TAO profile likelihood
# Total points: 31 x 31 = 961
# Each job computes ONE point

#SBATCH --job-name=nnd_profile
#SBATCH --array=101-125
#SBATCH --output=cluster/logs/slurm_%A_%a.out
#SBATCH --error=cluster/logs/slurm_%A_%a.err
#SBATCH --get-user-env
#SBATCH --export=NONE


#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny

#SBATCH --cpus-per-task=17
#SBATCH --mem=2G
#SBATCH --time=1:00:00

module load slurm_setup
module load julia/1.11.7
export JULIA_PROJECT=/dss/dsshome1/08/go67jac2/julia/my_env
MYPROG=/dss/dsshome1/08/go67jac2/cluster/single_point.jl
julia $MYPROG $SLURM_ARRAY_TASK_ID 961
