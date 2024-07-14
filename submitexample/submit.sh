#!/bin/bash

# hardware requirements
#SBATCH --time=47:00:00                    # enter a maximum runtime for the job. (format: DD-HH:MM:SS, or just HH:MM:SS)
#SBATCH --cpus-per-task=16                  # use multi-threading with 4 cpu threads (= 2 physical cores + hyperthreading)
#SBATCH --mem=128G                           # request this amount of memory for each task
 
#SBATCH --partition=cpu                    # optional, cpu is default. needed for gpu/classes. See `sinfo` for options
#SBATCH --qos=normal                        # Submit debug job for quick test. See `sacctmgr show qos` for options
 
# some further useful options, uncomment as needed/desired
#SBATCH --job-name Nhole_$4_t$1_h1_$2_h2_$3_D_$5
#SBATCH --output %x.%j.out                 # this is where the (text) output goes. %x=Job name, %j=Jobd id, %N=node.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # number of CPUs per node, total for all the tasks below.

TENPYPATH="/space/cmt/tenpy_cluster/tenpy_v0.10.0"
export PYTHONPATH="${TENPYPATH}:$PYTHONPATH"

Ly=$8
Lx=$1
P=$2
t=1.0
U=$3
bcy=$4
chi=$5
job=$6
Sweeps=3
init=$7
chiF=$9
chiS=$10
ansatz=CSL

echo "Execute job on host $HOSTNAME at $(date)"
which python

ls ${init}
python runKitaev.py -Lx ${Lx}  -Ly ${Ly} -chi ${chi} -t 1.0 -U ${U} -job ${job} -P ${P}  -Sweeps ${Sweeps} -init ${init} -bcy ${bcy} -chiF ${chiF} -chiS ${chiS}

echo "Finished job at $(date)"

