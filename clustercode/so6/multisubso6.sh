#!/bin/bash

lx=36
J=1.
K=1.
D=2048
pbc=1
sweeps=10
job=dmrg

for D in 1000 2000 3000
do
for pbc in 1 0
do
for K in 0.0 0.2 0.3 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 1.9 2.0 2.1 2.2 3.8 3.9 4.0 4.1 4.2 7.8 7.9 8.0 8.1 8.2 10.0 12.0 16.0
do
echo Job = ${job} J = $J K = $K, pbc = ${pbc}, D = ${D}
jname=SO6DMRG_Job${job}_lx${lx}_J${J}_K${K}_pbc${pbc}_D${D}_Sweeps${sweeps}
sbatch -J ${jname} subso6.sh ${lx} ${J} ${K} ${D} ${pbc} ${sweeps} ${job}
done
done
done