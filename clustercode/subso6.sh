#!/bin/bash

#SBATCH --cpus-per-task=10                  # use multi-threading with 4 cpu threads (= 2 physical cores + hyperthreading)
#SBATCH --mem=128G                           # request this amount of memory for each task
#SBATCH --job-name=SO6DMRG_lx$1_K$2_pbc$5_D$4_Job&7
#SBATCH --partition=fat6348
#SBATCH --ntasks-per-node=2
#SBATCH --output=out%x_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # number of CPUs per node, total for all the tasks below.

lx=$1
J=$2
K=$3
D=$4
pbc=$5
sweeps=$6
job=$7

echo "Execute job on host $HOSTNAME at $(date)"

python so6dmrg.py -lx=${lx} -J=${J} -K=${K} -D=${D} -pbc=${pbc} -sweeps=${sweeps} -job=${job}

echo "Finished job on host $HOSTNAME at $(date)"