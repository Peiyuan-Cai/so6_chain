"""
Energy plot version 2

Puiyuen 241206
    1. 241206 cancel error bar, new K list

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

lxlist = [16,20,24,28,32]
#Klist = [0.245, 0.246, 0.247, 0.248, 0.249, 0.251, 0.252, 0.253, 0.254, 0.255]
Klist_1 = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
Klist_1 = np.arange(0.1, 0.156, 0.004)
Klist_1 = np.round(Klist_1, 3)
Klist_1 = Klist_1.tolist()
engdifflist_1 = []
engdifflist_2 = []
D = 2000

import os
homepath = os.getcwd()
datapath = homepath + '/data/'

E_error_1_dmrg1 = []
trunc_error_1_dmrg1 = []
e_error_1_dmrg1 = []
E_error_1_dmrg2 = []
trunc_error_1_dmrg2 = []
e_error_1_dmrg2 = []

#load errordata
for K in Klist_1:
    E_error_dmrg1_temp = []
    trunc_error_dmrg1_temp = []
    e_error_dmrg1_temp = []
    E_error_dmrg2_temp = []
    trunc_error_dmrg2_temp = []
    e_error_dmrg2_temp = []
    for lx in lxlist:
        errorpath = datapath + "SO4DMRG_lx{}_J1.0_K{}_pbc1/".format(lx,K)
        errorfname_dmrg1 = errorpath + "errordata_jobmposdmrg_lx{}_J1.0_K{}_pbc1_D{}_sweeps10".format(lx, K, D)
        errorfname_dmrg2 = errorpath + "errordata_jobmposdmrg2_lx{}_J1.0_K{}_pbc1_D{}_sweeps10".format(lx, K, D)
        errordata_dmrg1 = pickle.load(open(errorfname_dmrg1, 'rb'))
        errordata_dmrg2 = pickle.load(open(errorfname_dmrg2, 'rb'))
        E_error_dmrg1_temp.append(errordata_dmrg1[0][-1])
        trunc_error_dmrg1_temp.append(errordata_dmrg1[1][-1])
        e_error_dmrg1_temp.append(errordata_dmrg1[2][-1])
        E_error_dmrg2_temp.append(errordata_dmrg2[0][-1])
        trunc_error_dmrg2_temp.append(errordata_dmrg2[1][-1])
        e_error_dmrg2_temp.append(errordata_dmrg2[2][-1])
    E_error_1_dmrg1.append(E_error_dmrg1_temp)
    trunc_error_1_dmrg1.append(trunc_error_dmrg1_temp)
    e_error_1_dmrg1.append(e_error_dmrg1_temp)
    E_error_1_dmrg2.append(E_error_dmrg2_temp)
    trunc_error_1_dmrg2.append(trunc_error_dmrg2_temp)
    e_error_1_dmrg2.append(e_error_dmrg2_temp)

trunc_error_1_dmrg1 = np.array(trunc_error_1_dmrg1)
trunc_error_1_dmrg2 = np.array(trunc_error_1_dmrg2)
e_error_1_dmrg1 = np.array(e_error_1_dmrg1)
e_error_1_dmrg2 = np.array(e_error_1_dmrg2)
E_error_1_dmrg1 = np.array(E_error_1_dmrg1)
E_error_1_dmrg2 = np.array(E_error_1_dmrg2)

trunc_error_1 = trunc_error_1_dmrg1 + trunc_error_1_dmrg2
e_error_1 = e_error_1_dmrg1 + e_error_1_dmrg2
E_error_1 = abs(E_error_1_dmrg1) + abs(E_error_1_dmrg2)

trunc_error_1 = trunc_error_1.tolist()
e_error_1 = e_error_1.tolist()
E_error_1 = E_error_1.tolist()

#error list saving
error_list_fname = homepath + '/errorlist_K{}to{}_D{}'.format(Klist_1[0],Klist_1[-1],D)
with open(error_list_fname, 'wb') as f:
    pickle.dump([E_error_1, e_error_1, trunc_error_1], f)
print("error list saved")

# trunc_error_2_dmrg1 = np.array(trunc_error_2_dmrg1)
# trunc_error_2_dmrg2 = np.array(trunc_error_2_dmrg2)
# e_error_2_dmrg1 = np.array(e_error_2_dmrg1)
# e_error_2_dmrg2 = np.array(e_error_2_dmrg2)
# E_error_2_dmrg1 = np.array(E_error_2_dmrg1)
# E_error_2_dmrg2 = np.array(E_error_2_dmrg2)

# trunc_error_2 = trunc_error_2_dmrg1 + trunc_error_2_dmrg2
# e_error_2 = e_error_2_dmrg1 + e_error_2_dmrg2
# E_error_2 = abs(E_error_2_dmrg1) + abs(E_error_2_dmrg2)

# trunc_error_2 = trunc_error_2.tolist()
# e_error_2 = e_error_2.tolist()
# E_error_2 = E_error_2.tolist()

for K in Klist_1:
    engdifflisttemp_K = []
    for lx in lxlist:
        print("calculating lx",lx,"K",K)
        path = homepath + '/data2/' + "SO4DMRG_lx{}_J1.0_K{}_pbc1/".format(lx,K)
        fname1 = path + "psidmrg_jobmposdmrg_lx{}_J1.0_K{}_pbc1_D{}_sweeps10".format(lx,K,D)
        fname2 = path + "psidmrg_jobmposdmrg2_lx{}_J1.0_K{}_pbc1_D{}_sweeps10_no_orth".format(lx,K,D)
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
    engdifflist_2.append(engdifflisttemp_K)

#energy saving
eng_list_2_fname = homepath + '/englist2_K{}to{}_D{}'.format(Klist_1[0],Klist_1[-1],D)
with open(eng_list_2_fname, 'wb') as f:
    pickle.dump(engdifflist_2, f)

#dimer side
fig, ax = plt.subplots(figsize=(15, 10))
for ki in Klist_1:
    ax.plot(lxlist, [engdifflist_2[Klist_1.index(ki)][i] for i in range(len(lxlist))], '-x', label='K={}, D={}'.format(ki,D))
ax.set_title('|E1-E2|(log-linear scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn{}_K{}to{}_log_linear_1_new.pdf'.format(D,Klist_1[0],Klist_1[-1]))

# fig, ax = plt.subplots(figsize=(15, 10))
# for ki in Klist_1:
#     ax.plot(lxlist, [engdifflist_2[Klist_1.index(ki)][i] for i in range(len(lxlist))], '-x', label='K={}, D={}'.format(ki,D))
# ax.set_title('|E1-E2|(log-log scale)')
# ax.set_xlabel('lx')
# ax.set_ylabel('|E1-E2|')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend()
# plt.tight_layout()
# plt.savefig('edplotgn{}_log_log_2.pdf'.format(D))

#dimer side with errorbar
fig, ax = plt.subplots(figsize=(15, 10))
for ki in Klist_1:
    ax.errorbar(lxlist, [engdifflist_2[Klist_1.index(ki)][i] for i in range(len(lxlist))], yerr=E_error_1[Klist_1.index(ki)], capsize=3, fmt='-x', label='K={}, D={}'.format(ki,D))
ax.set_title('|E1-E2|(log-linear scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn{}_K{}to{}_log_linear_2_error_new.pdf'.format(D,Klist_1[0],Klist_1[-1]))

# fig, ax = plt.subplots(figsize=(15, 10))
# for ki in Klist_1:
#     ax.errorbar(lxlist, [engdifflist_2[Klist_1.index(ki)][i] for i in range(len(lxlist))], yerr=E_error_2[Klist_1.index(ki)], capsize=3, fmt='-x', label='K={}, D={}'.format(ki,D))
# ax.set_title('|E1-E2|(log-log scale)')
# ax.set_xlabel('lx')
# ax.set_ylabel('|E1-E2|')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend()
# plt.tight_layout()
# plt.savefig('edplotgn{}_log_log_2_error.pdf'.format(D))
