"""
The MPO-MPS method for SO(6) BBQ model

Puiyuen 2024.08.27-

"I will make it better"
"""
import matplotlib.pyplot as plt
from tenpy.tools.params import asConfig
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import Site, SpinSite, SpinHalfFermionSite, FermionSite
from tenpy.models.lattice import Chain, Square
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from bdgpack import *
from tenpy.algorithms import dmrg
import logging
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True

class KitaevSingleChain():
    def __init__(self, chi, delta, lamb, L, bc='open'):
        """
        The Single Kitaev chain class. 
        
        The Hamiltonian is in the BdG form, the matrix is written under the Bogoliubov *QUASIHOLE* representation. 

        Args:
            chi (float): variational parameter $\chi$
            delta (float): variational parameter $\delta$
            lamb (float): variational parameter $\lambda$
            L (int): Chain length
            bc (str, optional): Boundary condition. Defaults to 'open'.
            
        Raises:
            Check bc must be open or periodic: check your boundary condition input
            
        Note:
            240827: We should now have three types of boundary conditions namely 0(open), 1(periodic) and -1(anti-periodic)
        """
        self.L = L
        self.chi = chi
        self.delta = delta
        self.lamb = lamb
        self.bc = bc
        self.model = "Single Kitaev chain parameters_L{}_chi{}_delta{}_lambda{}_bc{}".format(L, round(chi, 3), round(delta, 3), round(lamb, 3), bc)
        self.dtype = np.float64
        if bc == 'open':
            self.pbc = 0
        elif bc == 'periodic':
            self.pbc = 1
        else:
            raise "Check bc must be open or periodic"
        
    def calc_hamiltonian(self):
        """
        BdG Hamiltonian calculator. 

        As a !type function. Calculate the $t$ and $d$ matrices, the real space Hamiltonian, the $V$ and $U$ matrices and the $M$=[[V,U^*],[U,V^*]] matrix. 
        """
        L = self.L
        t = -self.chi
        d = self.delta
        mu = self.lamb
        self.tmat = np.zeros((L,L), dtype=self.dtype)
        self.dmat = np.zeros((L,L), dtype=self.dtype)
        for i in range(L):
            self.tmat[i, i] = mu/2 
        for i in range(L-1):
            self.tmat[i, (i+1)%L] = t 
            self.dmat[i, (i+1)%L] = d 
        self.parity = 1
        if self.pbc:
            parity = - 1 + 2 * ( L % 2 )
            self.parity = parity
            self.tmat[L-1, 0] = t*parity 
            self.dmat[L-1, 0] = d*parity 
        self.tmat += self.tmat.T.conj() 
        self.dmat -= self.dmat.T
        
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])
        self.eig_eng, self.eig_vec = bdgeig(self.ham)
        print("the eig energies", self.eig_eng)
        self.V, self.U = m2vu(self.eig_vec)
        self.M = vu2m(self.V, self.U)