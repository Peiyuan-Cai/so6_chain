"""
central charge fitting for SO(4) Reshetikhin point

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

    #read dmrg psi
    with open(fname, 'rb') as f:
        psi_dmrg = pickle.load(f)

    #plot central charge fitting
    x = np.arange(1, lx)
    xaxis = np.log(np.sin(np.pi*x/lx))/4
    yaxis = psi_dmrg.entanglement_entropy(2)
    plt.plot(xaxis, yaxis, 'o', label='central charge scattering')

    #fit the central charge
    from scipy.optimize import curve_fit
    def func(x, k, b):
        return k*x + b
    popt, pcov = curve_fit(func, xaxis, yaxis)
    plt.plot(xaxis, func(xaxis, *popt), label='fit: y=%5.3f*x + %5.3f' % tuple(popt))
    plt.title('The central charge scattering')
    plt.xlabel('$\log[\sin(\pi x/L)]/4$')
    plt.ylabel('$S^{(2)}(x)$')
    plt.ylim(0,2.5)
    plt.legend()
    plt.savefig('ccfit_lx{}_K{}.pdf'.format(lx, K))