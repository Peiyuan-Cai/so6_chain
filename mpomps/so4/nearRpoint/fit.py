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
Klist_1 = np.arange(0.1, 0.144, 0.004)
Klist_1 = np.round(Klist_1, 3)
Klist_1 = Klist_1.tolist()
engdifflist_1 = []
engdifflist_2 = []
D = 2000

import os
homepath = os.getcwd()

E_error_1_dmrg1 = []
trunc_error_1_dmrg1 = []
e_error_1_dmrg1 = []
E_error_1_dmrg2 = []
trunc_error_1_dmrg2 = []
e_error_1_dmrg2 = []

#read error list
error_list_fname = homepath + '/errorlist_K{}to{}_D{}'.format(Klist_1[0],Klist_1[-1],D)
with open(error_list_fname, 'rb') as f:
    E_error_1, e_error_1, trunc_error_1 = pickle.load(f)

#energy loading
eng_list_2_fname = homepath + '/englist2_K{}to{}_D{}'.format(Klist_1[0],Klist_1[-1],D)
with open(eng_list_2_fname, 'rb') as f:
    engdifflist_2 = pickle.load(f)

#plot
# fig, ax = plt.subplots(figsize=(15, 10))
# for ki in Klist_1:
#     ax.plot(lxlist, [engdifflist_2[Klist_1.index(ki)][i] for i in range(len(lxlist))], '-x', label='K={}, D={}'.format(ki,D))
# ax.set_title('|E1-E2|(log-linear scale)')
# ax.set_xlabel('lx')
# ax.set_ylabel('|E1-E2|')
# ax.set_yscale('log')
# ax.legend()
# plt.tight_layout()
# plt.savefig('edplotgn{}_K{}to{}_log_linear_1_new.pdf'.format(D,Klist_1[0],Klist_1[-1]))

#plot errorbar
fig, ax = plt.subplots(figsize=(15, 10))
for ki in Klist_1:
    ax.errorbar(np.array(lxlist), [engdifflist_2[Klist_1.index(ki)][i] for i in range(len(lxlist))], yerr=E_error_1[Klist_1.index(ki)], capsize=3, fmt='-x', label='K={}, D={}'.format(ki,D))
ax.set_title('|E1-E2|(log-linear scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplotgn{}_K{}to{}_log_linear_2_error_new.pdf'.format(D,Klist_1[0],Klist_1[-1]))