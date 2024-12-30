"""
Spin-Peierls correlation function calculation for SO(4)

Puiyuen 241212
    1. 
"""
import numpy as np
import numpy.linalg as LA
from copy import deepcopy
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import SpinSite, FermionSite, Site
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge, ChargeInfo
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig
import pickle
import matplotlib.pyplot as plt
from so4bbqham import *

"""
main functions
"""
import os
homepath = os.getcwd()
datapath = homepath + '/data2/'
lxlist = [20,24,28,32,36,40]
Klist = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
D = 4000
ii = 0 #the i in spin-peierls correlation function <P_i P_{i+L/2}>
print("----------Start Spin-Peierls Correlation Calculation----------")

SP_value_list = []

for lx in lxlist:
    jj = ii + int(lx/2) - 1
    SP_value_list_temp = [] #spin-peierls correlation value for each lx
    for K in Klist:
        path = homepath + '/data2/' + "SO4DMRG_lx{}_J1.0_K{}_pbc1/".format(lx,K)
        fname1 = path + "psidmrg_jobmposdmrg_lx{}_J1.0_K{}_pbc1_D{}_sweeps10".format(lx,K,D) #load only 1 state
        # fname2 = path + "psidmrg_jobmposdmrg2_lx{}_J1.0_K{}_pbc1_D{}_sweeps10_no_orth".format(lx,K,D)
        with open(fname1, 'rb') as f:
            psi1 = pickle.load(f)
        # with open(fname2, 'rb') as f:
        #     psi2 = pickle.load(f)

        SP_value_temp = psi1.expectation_value_term([('L0',ii), ('L0',ii+1), ('L0',jj), ('L0',jj+1)]) -  psi1.expectation_value_term([('L0',ii), ('L0',ii+1), ('L0',jj+1), ('L0',jj+2)]) - psi1.expectation_value_term([('L0',ii+1), ('L0',ii+2), ('L0',jj), ('L0',jj+1)]) + psi1.expectation_value_term([('L0',ii+1), ('L0',ii+2), ('L0',jj+1), ('L0',jj+2)])
        SP_value_list_temp.append(SP_value_temp)
    SP_value_list.append(SP_value_list_temp)

fig, ax = plt.subplots(figsize=(15,10))
for lx in lxlist:
    ax.plot(Klist, SP_value_list[lxlist.index(lx)], label='Lx={}'.format(lx))
ax.legend()
plt.tight_layout()
plt.savefig('Spin_Peierls_corr_around_K0.png')