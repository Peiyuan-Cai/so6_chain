"""
single DMRG for SO(4) Reshetikhin point

Puiyuen 240112
    1. 240112 initial version
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=8)
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=1.)
    parser.add_argument("-lamb", type=float, default=2.)
    parser.add_argument("-Dmpos", type=int, default=2000)
    parser.add_argument("-Ddmrg", type=int, default=4000)
    parser.add_argument("-sweeps", type=int, default=6)
    parser.add_argument("-pbc", type=int, default=2)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=0.)
    parser.add_argument("-verbose", type=int, default=1)
    args = parser.parse_args()
    
    import logging
    logging.basicConfig(level=args.verbose)
    for _ in ['parso.python.diff', 'parso.cache', 'parso.python.diff', 
              'parso.cache', 'matplotlib.font_manager', 'tenpy.tools.cache', 
              'tenpy.algorithms.mps_common', 'tenpy.linalg.lanczos', 'tenpy.tools.params']:
        logging.getLogger(_).disabled = True
        
    np.random.seed(0)
    
    chi = args.chi
    delta = args.delta
    lamb = args.lamb
    lx = args.lx
    pbc = args.pbc
    Dmpos = args.Dmpos
    Ddmrg = args.Ddmrg
    J = args.J
    K = args.K
    sweeps = args.sweeps
    verbose = args.verbose
    conn = None
    cons = 'U1'

    import os
    homepath = os.getcwd()
    datapath = homepath + '/data/'

    path = homepath + '/data/' + 'so4psimpos_lx{}_delta{}_lambda{}/'.format(lx, delta, lamb)
    fname = path+'psidmrg_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, Ddmrg, sweeps)

    model_paras = dict(cons_N=None, cons_S='U1', Lx = lx, bc='periodic', J=J, K=K, D=Ddmrg, sweeps=sweeps, verbose=2)
    model_paras['init'] = homepath+'/data/so4psimpos_lx{}_delta1.0_lambda{}'.format(lx, lamb)+'/so4psimpos_lx{}_delta1.0_lambda{}_Dmpos{}_APBC'.format(lx, lamb, Dmpos)

    so4bbq = BBQJKSO4(model_paras)

    psi_dmrg, E = so4bbq.run_dmrg()
    print("DMRG results")
    print("DMRG psi", psi_dmrg)
    
    #DMRG state saving
    with open(fname, 'wb') as f:
        pickle.dump(psi_dmrg, f)
    print("DMRG state saved at", fname)