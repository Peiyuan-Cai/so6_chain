"""
SO4 chain DMRG code compact version for HPC use. Use U(1)xU(1) symmetry by default. 

Gutzwiller guided DMRG code for the SO4 chain.

Puiyuen 241029
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

lxlist = [24,28,32,36,40,44,48,52,56,60,64,68,72]
Klist = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
engdifflist = []

import os
homepath = os.getcwd()
datapath = homepath + '/data/'

for K in Klist:
    engdifflisttemp_K = []
    for lx in lxlist:
        path = homepath + '/data/' + "SO4DMRG_lx{}_J1.0_K{}_pbc1/".format(lx,K)
        fname1 = path + "psidmrg_jobmposdmrg_lx{}_J1.0_K{}_pbc1_D2000_sweeps10".format(lx,K)
        fname2 = path + "psidmrg_jobmposdmrg2_lx{}_J1.0_K{}_pbc1_D2000_sweeps10".format(lx,K)
        with open(fname1, 'rb') as f:
            psi1 = pickle.load(f)
        with open(fname2, 'rb') as f:
            psi2 = pickle.load(f)
        
        model_paras = dict(cons_N=None, cons_S='U1', Lx = lx, bc='periodic', J=1.0, K=K, D=2000, sweeps=10, verbose=2)
        SO4BBQ = BBQJKSO4(model_paras)
        BBQMPO = SO4BBQ.calc_H_MPO()
        
        eng1 = BBQMPO.expectation_value(psi1)
        eng2 = BBQMPO.expectation_value(psi2)
        
        absengdiff = np.abs(eng2 - eng1)
        engdifflisttemp_K.append(absengdiff)
    engdifflist.append(engdifflisttemp_K)

fname = homepath + 'engdifflist'
with open(fname, 'wb') as f:
    pickle.dump(engdifflist, f)
    
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(lxlist, [engdifflist[0][i] for i in range(len(lxlist))], '-o', label='K=0.05')
ax.plot(lxlist, [engdifflist[1][i] for i in range(len(lxlist))], '-o', label='K=0.1')
ax.plot(lxlist, [engdifflist[2][i] for i in range(len(lxlist))], '-o', label='K=0.15')
ax.plot(lxlist, [engdifflist[3][i] for i in range(len(lxlist))], '-o', label='K=0.2')
ax.plot(lxlist, [engdifflist[4][i] for i in range(len(lxlist))], '-o', label='K=0.25')
ax.plot(lxlist, [engdifflist[5][i] for i in range(len(lxlist))], '-o', label='K=0.3')
ax.plot(lxlist, [engdifflist[6][i] for i in range(len(lxlist))], '-o', label='K=0.35')
ax.plot(lxlist, [engdifflist[7][i] for i in range(len(lxlist))], '-o', label='K=0.4')
ax.plot(lxlist, [engdifflist[8][i] for i in range(len(lxlist))], '-o', label='K=0.45')
ax.set_title('|E1-E2|(log scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplot.png')