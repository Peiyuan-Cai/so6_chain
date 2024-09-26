import numpy as np
import pickle
from so6bbqham import *
import matplotlib.pyplot as plt

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

import os
homepath  = os.getcwd()
if os.path.isdir(homepath+'/data/') == False:
    os.mkdir(homepath+'/data/')

J = 1.
pbc = 1
sweeps = 10
D = 2000

Edlist = []
lxlist = [24, 30, 36, 42, 48, 54]
Klist = [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24]

for lx in lxlist:
    for K in Klist:
        Edlist_temp = []
        
        model_paras = dict(cons_N=None, cons_S=None, Lx = lx, bc='periodic', J=J, K=K, D=D, sweeps=sweeps, verbose=2)
        so6bbq = BBQJK(model_paras)

        path = homepath + '/data/' + "SO6DMRG_lx{}_J{}_K{}_pbc{}/".format(lx, J, K, pbc)

        #DMRG state loading
        fname = path+'psidmrg_jobmposdmrg_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, D, sweeps)
        with open(fname, 'rb') as f:
            psi1 = pickle.load(f)
        fname = path+'psidmrg_jobmposdmrg2_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, D, sweeps)
        with open(fname, 'rb') as f:
            psi2 = pickle.load(f)

        bbqmpo = so6bbq.calc_H_MPO()
        eng1 = bbqmpo.expectation_value(psi1)
        eng2 = bbqmpo.expectation_value(psi2)
        Ed = np.abs(eng1-eng2)
        Edlist_temp.append(Ed)
    Edlist.append(Edlist_temp)

#plot
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(lxlist, [Klist[i][0] for i in range(len(lxlist))], '-o', label='K=0.13')
ax.plot(lxlist, [Klist[i][1] for i in range(len(lxlist))], '-o', label='K=0.14')
ax.plot(lxlist, [Klist[i][2] for i in range(len(lxlist))], '-o', label='K=0.15')
ax.plot(lxlist, [Klist[i][3] for i in range(len(lxlist))], '-o', label='K=0.16')
ax.plot(lxlist, [Klist[i][4] for i in range(len(lxlist))], '-o', label='K=0.17')
ax.plot(lxlist, [Klist[i][5] for i in range(len(lxlist))], '-o', label='K=0.18')
ax.plot(lxlist, [Klist[i][6] for i in range(len(lxlist))], '-o', label='K=0.19')
ax.plot(lxlist, [Klist[i][7] for i in range(len(lxlist))], '-o', label='K=0.20')
ax.plot(lxlist, [Klist[i][8] for i in range(len(lxlist))], '-o', label='K=0.21')
ax.plot(lxlist, [Klist[i][9] for i in range(len(lxlist))], '-o', label='K=0.22')
ax.plot(lxlist, [Klist[i][10] for i in range(len(lxlist))], '-o', label='K=0.23')
ax.plot(lxlist, [Klist[i][11] for i in range(len(lxlist))], '-o', label='K=0.24')

ax.set_title('|E1-E2|(log scale)')
ax.set_xlabel('lx')
ax.set_ylabel('|E1-E2|')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('edplot.png')