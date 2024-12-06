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
Klist_2 = [0.251, 0.252, 0.253, 0.254, 0.255] #the critical side
#Klist_2 = [0.251, 0.253, 0.255] #the critical side
engdifflist_2 = [] #critical side
D = 4000

E_error_2_dmrg1 = [[-1.3812e-09, -7.1054e-15, -4.6185e-14, 3.5527e-14, 3.5527e-15, 4.9738e-14],
                   [1.7764e-15, 1.0658e-14, 7.1054e-15, -4.6185e-14, -1.3500e-13, 3.1974e-14],
                   [1.9540e-14, 1.0658e-14, -2.4869e-14, -3.5527e-14, -4.9738e-14, 1.1724e-13],
                   [2.1316e-14, -9.2371e-14, -1.0658e-14, -1.1013e-13, 2.8422e-14, -1.3500e-13],
                   [1.2434e-14, -2.1316e-14, 1.4211e-14, 1.0658e-14, -7.1054e-15, 3.9080e-14]]
trunc_error_2_dmrg1 = [[6.1304e-16, 4.7606e-18, 3.6555e-18, 3.5564e-18, 4.3113e-18, 3.8211e-18],
                       [4.0087e-16, 5.0156e-17, 4.9425e-17, 4.0894e-17, 4.8099e-17, 3.7933e-17],
                       [1.6266e-16, 2.1516e-16, 2.2161e-16, 2.1825e-16, 1.7582e-16, 1.5643e-16],
                       [4.6545e-16, 5.8807e-16, 6.6204e-16, 6.6663e-16, 5.0948e-16, 5.2737e-16],
                       [1.2229e-15, 1.3132e-15, 1.2946e-15, 1.5370e-15, 1.0508e-15, 1.2012e-15]]
e_error_2_dmrg1 = [[3.3751e-14, 1.4211e-14, 1.0658e-14, 6.3949e-14, 2.8422e-14, 5.6843e-14],
                   [2.6645e-14, 1.0658e-14, 6.0396e-14, 4.9738e-14, 3.5527e-14, 6.7502e-14],
                   [1.7764e-14, 3.9080e-14, 3.1974e-14, 7.4607e-14, 7.4607e-14, 6.7502e-14],
                   [3.1974e-14, 2.4869e-14, 1.0658e-14, 2.8422e-14, 3.5527e-14, 3.1974e-14],
                   [2.3093e-14, 3.9080e-14, 6.3949e-14, 7.8160e-14, 3.9080e-14, 1.2434e-13]]

E_error_2_dmrg2 = [[-1.7764e-15, 3.1974e-14, -7.1054e-15, 1.4211e-14, 2.8422e-14, 7.1054e-14],
                   [-7.1054e-15, -4.6185e-14, -1.2079e-13, 4.2633e-14, 4.2633e-14, 6.7502e-14],
                   [3.1974e-14, -5.6843e-14, 1.7764e-14, 1.4211e-14, -1.4211e-13, -6.3949e-14],
                   [-5.3291e-15, -2.4869e-14, -1.7764e-14, -5.6843e-14, 2.1316e-14, -2.1316e-13],
                   [1.7764e-14, 4.9738e-14, -4.9738e-14, 2.1316e-14, 3.5527e-15, -3.5527e-15]]
trunc_error_2_dmrg2 = [[4.2744e-18, 4.6244e-18, 3.6091e-18, 3.7144e-18, 3.5784e-18, 3.3492e-18],
                       [3.9229e-17, 4.8012e-17, 4.7858e-17, 3.9374e-17, 3.4688e-17, 3.6468e-17],
                       [1.5417e-16, 2.0280e-16, 1.8408e-16, 1.8221e-16, 1.5775e-16, 1.6008e-16],
                       [4.4736e-16, 5.7498e-16, 5.9879e-16, 6.2731e-16, 5.0547e-16, 5.0704e-16],
                       [1.0588e-15, 1.2274e-15, 1.4294e-15, 1.5348e-15, 1.2594e-15, 1.1717e-15]]
e_error_2_dmrg2 = [[3.5527e-14, 2.8422e-14, 1.7764e-14, 4.2633e-14, 1.7764e-14, 4.9738e-14],
                   [1.7764e-14, 2.8422e-14, 2.4869e-14, 6.3949e-14, 2.4869e-14, 4.9738e-14],
                   [4.4409e-14, 7.8160e-14, 4.2633e-14, 5.6843e-14, 5.6843e-14, 4.6185e-14],
                   [2.8422e-14, 5.3291e-14, 1.4211e-14, 1.4211e-14, 3.1974e-14, 3.9080e-14],
                   [3.1974e-14, 5.3291e-14, 3.5527e-14, 6.7502e-14, 6.0396e-14, 6.3949e-14]]

trunc_error_2_dmrg1 = np.array(trunc_error_2_dmrg1)
trunc_error_2_dmrg2 = np.array(trunc_error_2_dmrg2)
e_error_2_dmrg1 = np.array(e_error_2_dmrg1)
e_error_2_dmrg2 = np.array(e_error_2_dmrg2)
E_error_2_dmrg1 = np.array(E_error_2_dmrg1)
E_error_2_dmrg2 = np.array(E_error_2_dmrg2)

trunc_error_2 = trunc_error_2_dmrg1 + trunc_error_2_dmrg2
e_error_2 = e_error_2_dmrg1 + e_error_2_dmrg2
E_error_2 = abs(E_error_2_dmrg1) + abs(E_error_2_dmrg2)

trunc_error_2 = trunc_error_2.tolist()
e_error_2 = e_error_2.tolist()
E_error_2 = E_error_2.tolist()

import os
homepath = os.getcwd()

#energy reading
eng_list_2_fname = homepath + '/englist2_D{}'.format(D)
with open(eng_list_2_fname, 'rb') as f:
    engdifflist_2 = pickle.load(f)

# #critical side
fig, ax = plt.subplots(figsize=(12, 8))
for ki in Klist_2:
    ax.plot(lxlist, [engdifflist_2[Klist_2.index(ki)][i] for i in range(len(lxlist))], 'x-', label='K={}, D={}'.format(ki,D))
ax.set_title('$|E_1-E_2|$(log-linear scale)')
ax.set_xlabel('$N$')
ax.set_ylabel('$|E_1-E_2|$')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn{}_log_linear_2.pdf'.format(D))

fig, ax = plt.subplots(figsize=(12, 8))
for ki in Klist_2:
    ax.plot(lxlist, [engdifflist_2[Klist_2.index(ki)][i] for i in range(len(lxlist))], 'x-', label='K={}, D={}'.format(ki,D))
ax.set_title('$|E_1-E_2|$(log-log scale)')
ax.set_xlabel('$N$')
ax.set_ylabel('$|E_1-E_2|$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn{}_log_log_2.pdf'.format(D))

#critical side with errorbar
fig, ax = plt.subplots(figsize=(12, 8))
for ki in Klist_2:
    ax.errorbar(lxlist, [engdifflist_2[Klist_2.index(ki)][i] for i in range(len(lxlist))], yerr=e_error_2[Klist_2.index(ki)], fmt='x-', capsize=3, label='K={}, D={}'.format(ki,D))
ax.set_title('$|E_1-E_2|$(log-linear scale)')
ax.set_xlabel('$N$')
ax.set_ylabel('$|E_1-E_2|$')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn{}_log_linear_2_error.pdf'.format(D))

fig, ax = plt.subplots(figsize=(12, 8))
for ki in Klist_2:
    ax.errorbar(lxlist, [engdifflist_2[Klist_2.index(ki)][i] for i in range(len(lxlist))], yerr=e_error_2[Klist_2.index(ki)], fmt='x-', capsize=3, label='K={}, D={}'.format(ki,D))
ax.set_title('$|E_1-E_2|$(log-log scale)')
ax.set_xlabel('$N$')
ax.set_ylabel('$|E_1-E_2|$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn{}_log_log_2_error.pdf'.format(D))