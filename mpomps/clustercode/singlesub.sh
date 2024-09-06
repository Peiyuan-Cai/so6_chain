#!/bin/bash

#SBATCH --cpus-per-task=64                  # use multi-threading with 4 cpu threads (= 2 physical cores + hyperthreading)
#SBATCH --mem=200G                           # request this amount of memory for each task
#SBATCH --job-name=SO6MPOStest
#SBATCH --partition=xhacnormala
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # number of CPUs per node, total for all the tasks below.

echo "Execute job on host $HOSTNAME at $(date)"

python ~/so6/so6_chain/mpomps/so6/mpos2bc.py -lx 6 -pbc 2

echo "Finished job on host $HOSTNAME at $(date)"