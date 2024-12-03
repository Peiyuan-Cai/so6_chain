"""
Energy plot version 2

Puiyuen 241029
    1. 241106 different D
    2. 241129 separate two directions
    3. 241129 add errorbar
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
Klist = [0.245, 0.246, 0.247, 0.248, 0.249, 0.251, 0.252, 0.253, 0.254, 0.255]
Klist_1 = [0.245, 0.246, 0.247, 0.248, 0.249] #the dimer side
Klist_2 = [0.251, 0.252, 0.253, 0.254, 0.255] #the critical side
engdifflist_2000_1 = [] #dimer side
engdifflist_2000_2 = [] #critical side

trunc_error_1_dmrg1 = [[2.3652e-14, 3.1591e-14, 2.9448e-14, 2.5563e-14, 2.9443e-14, 2.9337e-14],
                       [1.1787e-14, 1.4496e-14, 1.5045e-14, 1.1747e-14, 1.3313e-14, 1.3076e-14],
                       [4.7961e-15, 6.3199e-15, 6.2886e-15, 5.0823e-15, 4.8514e-15, 4.8131e-15],
                       [1.4861e-15, 1.4113e-15, 1.5552e-15, 1.3223e-15, 1.1685e-15, 1.1662e-15],
                       [2.7567e-16, 2.0343e-16, 1.8800e-16, 1.3215e-16, 1.1638e-16, 1.0830e-16]]
e_error_1_dmrg1 = [[2.1494e-13, 2.9843e-13, 3.1974e-13, 2.8422e-13, 2.6290e-13, 3.1619e-13],
                   [1.6698e-13, 1.7053e-13, 1.6342e-13, 1.3145e-13, 1.7764e-13, 1.5277e-13],
                   [4.4409e-14, 7.4607e-14, 8.1712e-14, 7.8160e-14, 7.1054e-14, 8.1712e-14],
                   [4.9738e-14, 6.7502e-14, 1.3145e-13, 7.1054e-14, 9.9476e-14, 1.2434e-13],
                   [3.0198e-14, 3.9080e-14, 4.2633e-14, 5.6843e-14, 6.0396e-14, 3.5527e-14]]
E_error_1_dmrg1 = [[4.2633e-14, -3.1974e-13, -6.8923e-13, -2.8422e-14, -3.2685e-12, -3.5698e-11],
                   [2.1316e-14, 2.8422e-14, -7.6739e-13, -1.9185e-13, -5.7625e-12, -1.6140e-11],
                   [-1.0658e-14, -2.8422e-14, -1.3145e-12, -1.3145e-13, -1.2307e-11, -1.2967e-11],
                   [-1.4211e-14, -6.3949e-14, 3.1974e-14, -2.1672e-13, -1.8368e-12, -8.2814e-12],
                   [-8.8818e-15, -1.7764e-14, 7.1054e-15, -1.7408e-13, -1.6342e-13, -8.7397e-13]]

trunc_error_2_dmrg1 = [[4.3075e-14, 2.1952e-16, 1.7352e-16, 1.6045e-16, 1.3132e-16, 8.6299e-17],
                       [2.1926e-15, 1.7865e-15, 1.6069e-15, 1.5572e-15, 1.4897e-15, 1.1401e-15],
                       [7.1281e-15, 7.4986e-15, 7.7145e-15, 6.9549e-15, 6.4449e-15, 3.6542e-15],
                       [2.2214e-14, 2.2016e-14, 2.4448e-14, 1.9002e-14, 1.5245e-14, 8.3607e-15],
                       [4.3075e-14, 6.2662e-14, 4.6404e-14, 4.1312e-14, 4.2372e-14, 1.8082e-14]]
e_error_2_dmrg1 = [[3.7303e-13, 2.8422e-14, 3.5527e-14, 3.9080e-14, 4.2633e-14, 6.7502e-14],
                   [5.8620e-14, 7.4607e-14, 7.8160e-14, 7.1054e-14, 9.2371e-14, 9.2371e-14],
                   [6.3949e-14, 1.1724e-13, 1.1369e-13, 1.2079e-13, 1.1724e-13, 1.0658e-13],
                   [1.9007e-13, 2.0606e-13, 1.9540e-13, 2.5935e-13, 1.8829e-13, 1.5987e-13],
                   [3.7303e-13, 4.6896e-13, 3.5527e-13, 3.7303e-13, 3.9790e-13, 1.9895e-13]]
E_error_2_dmrg1 = [[2.3093e-14, -3.5527e-15, -5.3291e-14, -4.2633e-14, -4.4054e-13, -6.2172e-13],
                   [1.9540e-14, -3.5527e-15, -1.3145e-13, -4.0501e-13, -1.5987e-12, -5.0164e-12],
                   [3.9080e-14, -2.6290e-13, -7.4607e-14, -8.3844e-13, -1.4669e-11, -4.2885e-11],
                   [1.7764e-14, -7.1054e-14, -5.0449e-13, -1.3145e-13, -4.2881e-12, -1.2619e-11],
                   [2.8422e-14, -6.3594e-13, -1.8119e-13, -4.7606e-13, -6.3828e-11, -3.0234e-12]]

trunc_error_1_dmrg2 = [[2.36e-14, 3.0170e-14, 3.2053e-14, 3.3224e-14, 2.6930e-14, 2.4879e-14],
                       [1.18e-14, 1.4013e-14, 1.4468e-14, 1.3793e-14, 1.2192e-14, 1.1303e-14],
                       [5.75e-15, 5.3470e-15, 5.3631e-15, 5.0120e-15, 4.5145e-15, 4.2773e-15],
                       [1.72e-15, 1.3777e-15, 1.3432e-15, 1.1298e-15, 1.0276e-15, 9.1235e-16],
                       [2.38e-16, 2.2110e-16, 1.7445e-16, 1.1174e-16, 1.0567e-16, 8.1869e-17]]
e_error_1_dmrg2 = [[2.17e-13, 2.8777e-13, 3.4106e-13, 2.8066e-13, 2.5935e-13, 3.0198e-13],
                   [1.46e-13, 1.4211e-13, 1.7764e-13, 1.3145e-13, 1.7408e-13, 1.7408e-13],
                   [6.22e-14, 7.1054e-14, 9.2371e-14, 4.6185e-14, 5.3291e-14, 7.1054e-14],
                   [3.38e-14, 8.5265e-14, 9.2371e-14, 1.0658e-13, 7.4607e-14, 1.3145e-13],
                   [5.86e-14, 7.1054e-14, 4.9738e-14, 5.6843e-14, 6.3949e-14, 7.8160e-14]]
E_error_1_dmrg2 = [[-1.7337e-12, -8.5265e-14, -1.1013e-13, -8.1002e-13, -7.0742e-11, -6.3423e-11],
                   [-1.0658e-14, -1.2434e-13, -8.1712e-14, -5.2225e-13, -5.0676e-11, -3.8362e-11],
                   [4.2633e-14, 1.4211e-14, -4.9738e-14, -3.5882e-13, -3.3118e-11, -1.5454e-12],
                   [3.5527e-15, 3.9080e-14, -1.4211e-14, 3.1974e-14, -8.5265e-12, -1.9007e-12],
                   [1.0658e-14, 1.0658e-14, 6.3949e-14, 2.8422e-14, -3.3396e-13, -3.1264e-12]]

trunc_error_2_dmrg2 = [[2.94e-16, 2.1613e-16, 1.6915e-16, 1.2375e-16, 1.0687e-16, 6.3633e-17],
                       [2.37e-15, 2.1076e-15, 1.8121e-15, 1.3350e-15, 1.1615e-15, 7.8021e-16],
                       [9.1275e-15, 7.3324e-15, 8.0013e-15, 6.1844e-15, 5.2525e-15, 4.4171e-15],
                       [2.5002e-14, 2.4164e-14, 2.2557e-14, 1.9223e-14, 1.7035e-14, 8.0177e-15],
                       [5.61e-14, 8.7595e-14, 5.4303e-14, 4.3669e-14, 3.6515e-14, 1.8088e-14]]
e_error_2_dmrg2 = [[4.80e-14, 1.7764e-14, 2.8422e-14, 2.1316e-14, 5.6843e-14, 7.4607e-14],
                   [5.51e-14, 5.6843e-14, 8.8818e-14, 1.0303e-13, 6.3949e-14, 8.8818e-14],
                   [8.3489e-14, 1.2790e-13, 8.5265e-14, 1.1724e-13, 1.4921e-13, 7.8160e-14],
                   [1.9362e-13, 1.7764e-13, 1.7764e-13, 1.7764e-13, 2.1672e-13, 1.4211e-13],
                   [4.32e-13, 5.4712e-13, 4.4409e-13, 4.1211e-13, 4.1922e-13, 1.9185e-13]]
E_error_2_dmrg2 = [[-1.5987e-14, 2.4869e-14, -1.2790e-13, -7.8160e-14, -1.5206e-12, -1.1369e-11],
                   [7.1054e-15, 7.1054e-15, -2.3448e-13, -4.5475e-13, -2.4514e-12, -6.4375e-12],
                   [-2.8422e-14, -2.1316e-14, -1.2434e-13, -2.0322e-12, -1.0168e-11, -5.3291e-13],
                   [5.3291e-14, -1.3856e-13, -1.2790e-13, -2.9772e-12, -1.3767e-11, -9.5888e-12],
                   [-2.1316e-14, -3.5527e-15, -2.9132e-13, -5.6097e-11, -5.2466e-11, -1.1532e-11]]

trunc_error_1_dmrg1 = np.array(trunc_error_1_dmrg1)
trunc_error_1_dmrg2 = np.array(trunc_error_1_dmrg2)
trunc_error_2_dmrg1 = np.array(trunc_error_2_dmrg1)
trunc_error_2_dmrg2 = np.array(trunc_error_2_dmrg2)
e_error_1_dmrg1 = np.array(e_error_1_dmrg1)
e_error_1_dmrg2 = np.array(e_error_1_dmrg2)
e_error_2_dmrg1 = np.array(e_error_2_dmrg1)
e_error_2_dmrg2 = np.array(e_error_2_dmrg2)
E_error_1_dmrg1 = np.array(E_error_1_dmrg1)
E_error_1_dmrg2 = np.array(E_error_1_dmrg2)
E_error_2_dmrg1 = np.array(E_error_2_dmrg1)
E_error_2_dmrg2 = np.array(E_error_2_dmrg2)

trunc_error_1 = trunc_error_1_dmrg1 + trunc_error_1_dmrg2
trunc_error_2 = trunc_error_2_dmrg1 + trunc_error_2_dmrg2
e_error_1 = e_error_1_dmrg1 + e_error_1_dmrg2
e_error_2 = e_error_2_dmrg1 + e_error_2_dmrg2
E_error_1 = abs(E_error_1_dmrg1) + abs(E_error_1_dmrg2)
E_error_2 = abs(E_error_2_dmrg1) + abs(E_error_2_dmrg2)

trunc_error_1 = trunc_error_1.tolist()
trunc_error_2 = trunc_error_2.tolist()
e_error_1 = e_error_1.tolist()
e_error_2 = e_error_2.tolist()
E_error_1 = E_error_1.tolist()
E_error_2 = E_error_2.tolist()

import os
homepath = os.getcwd()

#energy reading
eng_list_1_fname = homepath + '/englist1'
with open(eng_list_1_fname, 'rb') as f:
    engdifflist_2000_1 = pickle.load(f)
eng_list_2_fname = homepath + '/englist2'
with open(eng_list_2_fname, 'rb') as f:
    engdifflist_2000_2 = pickle.load(f)

#dimer side
# fig, ax = plt.subplots(figsize=(15, 10))
# for ki in Klist_1:
#     ax.plot(lxlist, [engdifflist_2000_1[Klist_1.index(ki)][i] for i in range(len(lxlist))], 'o-', label='K={}, D=2000'.format(ki))
# ax.set_title('|E1-E2|(log-linear scale)')
# ax.set_xlabel('lx')
# ax.set_ylabel('|E1-E2|')
# ax.set_yscale('log')
# ax.legend()
# plt.tight_layout()
# plt.savefig('edplotgn2000_log_linear_1.pdf')

# fig, ax = plt.subplots(figsize=(15, 10))
# for ki in Klist_1:
#     ax.plot(lxlist, [engdifflist_2000_1[Klist_1.index(ki)][i] for i in range(len(lxlist))], 'o-', label='K={}, D=2000'.format(ki))
# ax.set_title('|E1-E2|(log-log scale)')
# ax.set_xlabel('lx')
# ax.set_ylabel('|E1-E2|')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend()
# plt.tight_layout()
# plt.savefig('edplotgn2000_log_log_1.pdf')

# #critical side
# fig, ax = plt.subplots(figsize=(15, 10))
# for ki in Klist_2:
#     ax.plot(lxlist, [engdifflist_2000_2[Klist_2.index(ki)][i] for i in range(len(lxlist))], 'x-', label='K={}, D=2000'.format(ki))
# ax.set_title('|E1-E2|(log-linear scale)')
# ax.set_xlabel('lx')
# ax.set_ylabel('|E1-E2|')
# ax.set_yscale('log')
# ax.legend()
# plt.tight_layout()
# plt.savefig('edplotgn2000_log_linear_2.pdf')

# fig, ax = plt.subplots(figsize=(15, 10))
# for ki in Klist_2:
#     ax.plot(lxlist, [engdifflist_2000_2[Klist_2.index(ki)][i] for i in range(len(lxlist))], 'x-', label='K={}, D=2000'.format(ki))
# ax.set_title('|E1-E2|(log-log scale)')
# ax.set_xlabel('lx')
# ax.set_ylabel('|E1-E2|')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend()
# plt.tight_layout()
# plt.savefig('edplotgn2000_log_log_2.pdf')

#dimer side with errorbar
fig, ax = plt.subplots(figsize=(15, 10))
for ki in Klist_1:
    ax.errorbar(lxlist, [engdifflist_2000_1[Klist_1.index(ki)][i] for i in range(len(lxlist))], yerr=E_error_1[Klist_1.index(ki)], fmt='o-', capsize=3, label='K={}, D=2000'.format(ki))
ax.set_title('|E1-E2|(log-linear scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn2000_log_linear_1_error.pdf')

fig, ax = plt.subplots(figsize=(15, 10))
for ki in Klist_1:
    ax.errorbar(lxlist, [engdifflist_2000_1[Klist_1.index(ki)][i] for i in range(len(lxlist))], yerr=E_error_1[Klist_1.index(ki)], fmt='o-', capsize=3, label='K={}, D=2000'.format(ki))
ax.set_title('|E1-E2|(log-log scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn2000_log_log_1_error.pdf')

#critical side with errorbar
fig, ax = plt.subplots(figsize=(15, 10))
for ki in Klist_2:
    ax.errorbar(lxlist, [engdifflist_2000_2[Klist_2.index(ki)][i] for i in range(len(lxlist))], yerr=E_error_2[Klist_2.index(ki)], fmt='x-', capsize=3, label='K={}, D=2000'.format(ki))
ax.set_title('|E1-E2|(log-linear scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn2000_log_linear_2_error.pdf')

fig, ax = plt.subplots(figsize=(15, 10))
for ki in Klist_2:
    ax.errorbar(lxlist, [engdifflist_2000_2[Klist_2.index(ki)][i] for i in range(len(lxlist))], yerr=E_error_2[Klist_2.index(ki)], fmt='x-', capsize=3, label='K={}, D=2000'.format(ki))
ax.set_title('|E1-E2|(log-log scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn2000_log_log_2_error.pdf')