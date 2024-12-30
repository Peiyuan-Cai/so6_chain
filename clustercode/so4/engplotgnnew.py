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

lxlist = [20,24,28,32,36,40]
#Klist = [0.245, 0.246, 0.247, 0.248, 0.249, 0.251, 0.252, 0.253, 0.254, 0.255]
Klist_1 = [0.24, 0.241, 0.242, 0.243, 0.244]
engdifflist_1 = []
engdifflist_2 = []
D = 2000

import os
homepath = os.getcwd()
datapath = homepath + '/data2/'

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
        E_error_1_dmrg1.append(errordata_dmrg1[0][-1])
        trunc_error_1_dmrg1.append(errordata_dmrg1[1][-1])
        e_error_1_dmrg1.append(errordata_dmrg1[2][-1])
        E_error_1_dmrg2.append(errordata_dmrg2[0][-1])
        trunc_error_1_dmrg2.append(errordata_dmrg2[1][-1])
        e_error_1_dmrg2.append(errordata_dmrg2[2][-1])
    E_error_dmrg1_temp.append(E_error_dmrg1_temp)
    trunc_error_dmrg1_temp.append(trunc_error_dmrg1_temp)
    e_error_dmrg1_temp.append(e_error_dmrg1_temp)
    E_error_dmrg2_temp.append(E_error_dmrg2_temp)
    trunc_error_dmrg2_temp.append(trunc_error_dmrg2_temp)
    e_error_dmrg2_temp.append(e_error_dmrg2_temp)

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


# trunc_error_2_dmrg1 = [[4.3075e-14, 2.1952e-16, 1.7352e-16, 1.6045e-16, 1.3132e-16, 8.6299e-17],
#                        [2.1926e-15, 1.7865e-15, 1.6069e-15, 1.5572e-15, 1.4897e-15, 1.1401e-15],
#                        [7.1281e-15, 7.4986e-15, 7.7145e-15, 6.9549e-15, 6.4449e-15, 3.6542e-15],
#                        [2.2214e-14, 2.2016e-14, 2.4448e-14, 1.9002e-14, 1.5245e-14, 8.3607e-15],
#                        [4.3075e-14, 6.2662e-14, 4.6404e-14, 4.1312e-14, 4.2372e-14, 1.8082e-14]]
# e_error_2_dmrg1 = [[3.7303e-13, 2.8422e-14, 3.5527e-14, 3.9080e-14, 4.2633e-14, 6.7502e-14],
#                    [5.8620e-14, 7.4607e-14, 7.8160e-14, 7.1054e-14, 9.2371e-14, 9.2371e-14],
#                    [6.3949e-14, 1.1724e-13, 1.1369e-13, 1.2079e-13, 1.1724e-13, 1.0658e-13],
#                    [1.9007e-13, 2.0606e-13, 1.9540e-13, 2.5935e-13, 1.8829e-13, 1.5987e-13],
#                    [3.7303e-13, 4.6896e-13, 3.5527e-13, 3.7303e-13, 3.9790e-13, 1.9895e-13]]
# E_error_2_dmrg1 = [[2.3093e-14, -3.5527e-15, -5.3291e-14, -4.2633e-14, -4.4054e-13, -6.2172e-13],
#                    [1.9540e-14, -3.5527e-15, -1.3145e-13, -4.0501e-13, -1.5987e-12, -5.0164e-12],
#                    [3.9080e-14, -2.6290e-13, -7.4607e-14, -8.3844e-13, -1.4669e-11, -4.2885e-11],
#                    [1.7764e-14, -7.1054e-14, -5.0449e-13, -1.3145e-13, -4.2881e-12, -1.2619e-11],
#                    [2.8422e-14, -6.3594e-13, -1.8119e-13, -4.7606e-13, -6.3828e-11, -3.0234e-12]]

# trunc_error_2_dmrg2 = [[2.94e-16, 2.1613e-16, 1.6915e-16, 1.2375e-16, 1.0687e-16, 6.3633e-17],
#                        [2.37e-15, 2.1076e-15, 1.8121e-15, 1.3350e-15, 1.1615e-15, 7.8021e-16],
#                        [9.1275e-15, 7.3324e-15, 8.0013e-15, 6.1844e-15, 5.2525e-15, 4.4171e-15],
#                        [2.5002e-14, 2.4164e-14, 2.2557e-14, 1.9223e-14, 1.7035e-14, 8.0177e-15],
#                        [5.61e-14, 8.7595e-14, 5.4303e-14, 4.3669e-14, 3.6515e-14, 1.8088e-14]]
# e_error_2_dmrg2 = [[4.80e-14, 1.7764e-14, 2.8422e-14, 2.1316e-14, 5.6843e-14, 7.4607e-14],
#                    [5.51e-14, 5.6843e-14, 8.8818e-14, 1.0303e-13, 6.3949e-14, 8.8818e-14],
#                    [8.3489e-14, 1.2790e-13, 8.5265e-14, 1.1724e-13, 1.4921e-13, 7.8160e-14],
#                    [1.9362e-13, 1.7764e-13, 1.7764e-13, 1.7764e-13, 2.1672e-13, 1.4211e-13],
#                    [4.32e-13, 5.4712e-13, 4.4409e-13, 4.1211e-13, 4.1922e-13, 1.9185e-13]]
# E_error_2_dmrg2 = [[-1.5987e-14, 2.4869e-14, -1.2790e-13, -7.8160e-14, -1.5206e-12, -1.1369e-11],
#                    [7.1054e-15, 7.1054e-15, -2.3448e-13, -4.5475e-13, -2.4514e-12, -6.4375e-12],
#                    [-2.8422e-14, -2.1316e-14, -1.2434e-13, -2.0322e-12, -1.0168e-11, -5.3291e-13],
#                    [5.3291e-14, -1.3856e-13, -1.2790e-13, -2.9772e-12, -1.3767e-11, -9.5888e-12],
#                    [-2.1316e-14, -3.5527e-15, -2.9132e-13, -5.6097e-11, -5.2466e-11, -1.1532e-11]]

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

        model_paras = dict(cons_N=None, cons_S='U1', Lx = lx, bc='periodic', J=1.0, K=K, D=D, sweeps=10, verbose=2)
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
eng_list_2_fname = homepath + '/englist2_D{}'.format(D)
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
plt.savefig('edplotgn{}_log_linear_1_new.pdf'.format(D))

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
# fig, ax = plt.subplots(figsize=(15, 10))
# for ki in Klist_1:
#     ax.errorbar(lxlist, [engdifflist_2[Klist_1.index(ki)][i] for i in range(len(lxlist))], yerr=E_error_2[Klist_1.index(ki)], capsize=3, fmt='-x', label='K={}, D={}'.format(ki,D))
# ax.set_title('|E1-E2|(log-linear scale)')
# ax.set_xlabel('lx')
# ax.set_ylabel('|E1-E2|')
# ax.set_yscale('log')
# ax.legend()
# plt.tight_layout()
# plt.savefig('edplotgn{}_log_linear_2_error.pdf'.format(D))

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