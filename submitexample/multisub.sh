#!/bin/bash

Ly=4
Lx=8
D=2000
P=10.0
U=0.0
job=spinon
bcy=-1
chiF=12
chiS=128
init=data/KitaevH_half_filling_Ly_4_Lx_8_txyz_1.0_1.0_1.0/SMPOMPS_Kitaev_Psi_P_10.0_WY_${bcy}_chiS_256_chiF_8

for P in 7.0 7.2 7.4 7.6 7.8 8.0 8.2 8.4 8.6 8.8 9.0 9.2 9.4 9.6 9.8 10.0 10.2 10.4 10.6 10.8 11.0 11.2 11.4 11.6 11.8 12.0 12.2 12.4 12.6 12.8 13.0 13.2 13.4 13.6 13.8 14.0 14.2 14.4 14.6 14.8 15.0 15.2 15.4 15.6 15.8 16.0
#for bcy in 1 -1
do
for chiF in 20 28 36
do

#init=data/KitaevH_half_filling_Ly_4_Lx_8_txyz_1.0_1.0_1.0/DMRG_Psi_SMPOMPS_Y_1_P_10.0_chi_128_chiF_12_t_1_U_30.0_D_2000
#init=data/KitaevH_half_filling_Ly_4_Lx_8_txyz_1.0_1.0_1.0/DMRG_Psi_SMPOMPS_Kitaev_Psi_P_10.0_WY_-1_chiS_256_chiF_8_t_1_U_12.0_D_2000
init=i
echo P = $P , chiF = ${chiF}
sbatch -J ${job}Y${Ly}Lx${Lx}Y${bcy}U${U}P${P}M${D}_${chiF}_${chiS} submit.sh ${Lx} ${P} ${U} ${bcy} ${D} ${job} ${init} ${Ly} ${chiF} ${chiS}
done
done
