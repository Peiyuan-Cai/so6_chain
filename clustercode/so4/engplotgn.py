"""
Energy plot version 2

Puiyuen 241029
    1. 241106 different D
"""
import numpy as np
import numpy.linalg as LA
from copy import deepcopy
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import SpinSite, FermionSite, Site
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO, MPOEnvironment
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge, ChargeInfo
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig
import pickle
import matplotlib.pyplot as plt
from so4bbqham import *
import scipy.linalg as spLA

lxlist = [8,12,16,20,24,28,32,36,40]
Klist = [0.245, 0.255]
engdifflist_2000 = []
engdifflist_3000 = []

import os
homepath = os.getcwd()
datapath = homepath + '/data/'

for K in Klist:
    engdifflisttemp_K = []
    for lx in lxlist:
        path = homepath + '/data/' + "SO4DMRG_lx{}_J1.0_K{}_pbc1/".format(lx,K)
        fname1 = path + "psidmrg_jobmposdmrg_lx{}_J1.0_K{}_pbc1_D2000_sweeps10".format(lx,K)
        fname2 = path + "psidmrg_jobmposdmrg2_lx{}_J1.0_K{}_pbc1_D2000_sweeps10_no_orth".format(lx,K)
        with open(fname1, 'rb') as f:
            psi1 = pickle.load(f)
        with open(fname2, 'rb') as f:
            psi2 = pickle.load(f)

        model_paras = dict(cons_N=None, cons_S='U1', Lx = lx, bc='periodic', J=1.0, K=K, D=2000, sweeps=10, verbose=2)
        SO4BBQ = BBQJKSO4(model_paras)
        BBQMPO = SO4BBQ.calc_H_MPO()

        hmat = np.zeros((2,2))
        ovlp_mat = np.zeros((2,2))

        ovlp_mat[0,0] = psi1.overlap(psi1); ovlp_mat[0,1] = psi1.overlap(psi2); 
        ovlp_mat[1,0] = psi2.overlap(psi1); ovlp_mat[1,1] = psi2.overlap(psi2); 

        hmat[0,0] = BBQMPO.expectation_value(psi1)
        hmat[1,1] = BBQMPO.expectation_value(psi2)
        env12 = MPOEnvironment(psi1, BBQMPO, psi2)
        env21 = MPOEnvironment(psi2, BBQMPO, psi1)
        hmat[0,1] = env12.full_contraction(0)
        hmat[1,0] = env21.full_contraction(0)

        val, vec = spLA.eigh(hmat, ovlp_mat)

        absengdiff = abs(val[0]-val[1])
        engdifflisttemp_K.append(absengdiff)
    engdifflist_2000.append(engdifflisttemp_K)

for K in Klist:
    engdifflisttemp_K = []
    for lx in lxlist:
        path = homepath + '/data/' + "SO4DMRG_lx{}_J1.0_K{}_pbc1/".format(lx,K)
        fname1 = path + "psidmrg_jobmposdmrg_lx{}_J1.0_K{}_pbc1_D3000_sweeps10".format(lx,K)
        fname2 = path + "psidmrg_jobmposdmrg2_lx{}_J1.0_K{}_pbc1_D3000_sweeps10_no_orth".format(lx,K)
        with open(fname1, 'rb') as f:
            psi1 = pickle.load(f)
        with open(fname2, 'rb') as f:
            psi2 = pickle.load(f)

        model_paras = dict(cons_N=None, cons_S='U1', Lx = lx, bc='periodic', J=1.0, K=K, D=3000, sweeps=10, verbose=2)
        SO4BBQ = BBQJKSO4(model_paras)
        BBQMPO = SO4BBQ.calc_H_MPO()

        hmat = np.zeros((2,2))
        ovlp_mat = np.zeros((2,2))

        ovlp_mat[0,0] = psi1.overlap(psi1); ovlp_mat[0,1] = psi1.overlap(psi2); 
        ovlp_mat[1,0] = psi2.overlap(psi1); ovlp_mat[1,1] = psi2.overlap(psi2); 

        hmat[0,0] = BBQMPO.expectation_value(psi1)
        hmat[1,1] = BBQMPO.expectation_value(psi2)
        env12 = MPOEnvironment(psi1, BBQMPO, psi2)
        env21 = MPOEnvironment(psi2, BBQMPO, psi1)
        hmat[0,1] = env12.full_contraction(0)
        hmat[1,0] = env21.full_contraction(0)

        val, vec = spLA.eigh(hmat, ovlp_mat)

        absengdiff = abs(val[0]-val[1])
        engdifflisttemp_K.append(absengdiff)
    engdifflist_3000.append(engdifflisttemp_K)

fig, ax = plt.subplots(figsize=(15, 10))
for ki in Klist:
    ax.plot(lxlist, [engdifflist_2000[Klist.index(ki)][i] for i in range(len(lxlist))], '-o', label='K={}, D=2000'.format(ki))
for ki in Klist:
    ax.plot(lxlist, [engdifflist_3000[Klist.index(ki)][i] for i in range(len(lxlist))], '-x', label='K={}, D=3000'.format(ki))
ax.set_title('|E1-E2|(log scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn.png')