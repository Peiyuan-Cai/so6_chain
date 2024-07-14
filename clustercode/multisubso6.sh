#!/bin/bash

lx=36
J=1.
K=1.
D=512
pbc=1
sweeps=10

for pbc in 1 0
do
for K in 0.0 0.2 0.33333 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 4.0 8.0 16.0 32.0 64.0 128.0 256.0
do
echo K = $K, pbc = ${pbc}
sbatch -J SO6DMRG_lx${lx}_K${K}_pbc${pbc}_D${D}_Sweeps${sweeps} subso6.sh ${lx} ${J} ${K} ${D} ${pbc} ${sweeps}
done
done