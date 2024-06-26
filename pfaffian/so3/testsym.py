"""
Finding the single chain symmetry
"""

import numpy as np
from gaussianbdg import *
from tenpy.tools.params import asConfig
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import SpinSite, FermionSite, Site
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge, ChargeInfo
from copy import deepcopy
from tenpy.algorithms import dmrg
from tenpy.models.aklt import AKLTChain
import matplotlib.pyplot as plt
import numpy.linalg as LA

class singlechain(bdg):
    """
    The Bogoliubov-de Gennes form of a SINGLE Kitaev chain.
    """
    def __init__(self, chi, d, lamb, Nx, D, pbc):
        self.model = "Kitaev single chain_L{}_chi{}_d{}_lambda{}_D{}".format(Nx, round(chi,6), round(d,6), round(lamb,6), D)
        super().__init__(Nx=Nx, Ny=1, model=self.model, D=D)
        self.t, self.d = round(-chi, 6), round(d, 6)
        self.mu = lamb
        self.dtype = np.float64
        self.pbc = pbc
        self.Nlatt = Nx #the REAL site number
        
    def hamiltonian(self):
        N = self.Nlatt
        self.tmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        self.dmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        t, d = self.t, self.d
        mu = self.mu
        print("t=",self.t, "d=",self.d, "mu=",self.mu)
        for i in range(N):
            self.tmat[i, i] = mu/2 #why devided by 2 -> we will add the diagonal terms twice below
        for i in range(N-1):
            self.tmat[i, (i+1)%N] = t 
            self.dmat[i, (i+1)%N] = d 
        self.parity = 1
        if self.pbc:
            parity = - 1 + 2 * ( N % 2 )
            self.parity = parity
            self.tmat[N-1, 0] = t*parity 
            self.dmat[N-1, 0] = d*parity 
        self.tmat += self.tmat.T.conj() #here we doubled the diagonal elements
        self.dmat -= self.dmat.T
        
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])
        self.eig_eng, self.eig_vec = bdgeig_zeromode(self.ham, ref=0.0)
        print("the eig energies", self.eig_eng)
        self.exact_EgsXY = -self.eig_eng[:self.Nlatt].sum()/2 + np.trace(self.tmat)/2 - self.Nlatt*mu/2
        print("the exact energy by em and trt", -self.eig_eng[:self.Nlatt].sum()/2 + np.trace(self.tmat)/2 - self.Nlatt*mu/2)
        self.u, self.v = m2uv(self.eig_vec)
        self.m = uv2m(self.u, self.v)
        print("check the hamiltonian diagonalization: HM=ME, M=[[u,v*];[v,u*]]")
        check_orthonormal(self.ham, self.m)
        self.covariance()

class spinhalf(bdg):
    """
    The Spin half BdG hamiltonian, It's a type of full hamiltonian, with 2 copies seperated by U(1) symmetry: S^z up and down. 
    """
    def __init__(self, chi, d, lamb, Nx, D, pbc):
        self.model = "Kitaev single chain_L{}_chi{}_d{}_lambda{}_D{}".format(Nx, round(chi,6), round(d,6), round(lamb,6), D)
        super().__init__(Nx=Nx, Ny=1, model=self.model, D=D)
        self.t, self.d = round(-chi, 6), round(d, 6)
        self.mu = lamb
        self.dtype = np.float64
        self.pbc = pbc
        self.Nlatt = Nx #the REAL site number
        
    def hamiltonian(self):
        N = self.Nlatt
        self.tmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        self.dmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        t, d = self.t, self.d
        mu = self.mu
        print("t=",self.t, "d=",self.d, "mu=",self.mu)
        for i in range(N):
            self.tmat[i, i] = mu/2 #why devided by 2 -> we will add the diagonal terms twice below
        for i in range(N-1):
            self.tmat[i, (i+1)%N] = t 
            self.dmat[i, (i+1)%N] = d 
        self.parity = 1
        if self.pbc:
            parity = - 1 + 2 * ( N % 2 )
            self.parity = parity
            self.tmat[N-1, 0] = t*parity 
            self.dmat[N-1, 0] = d*parity 
        self.tmat += self.tmat.T.conj() #here we doubled the diagonal elements
        self.dmat -= self.dmat.T
        
        self.tmat = np.kron(self.tmat, np.diag([1,1]))
        self.dmat = np.kron(self.dmat, np.array([[0,1],[1,0]]))
        
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])
        self.eig_eng, self.eig_vec = bdgeig_zeromode(self.ham, ref=0.0)
        print("the eig energies", self.eig_eng)
        self.exact_EgsXY = -self.eig_eng[:self.Nlatt].sum()/2 + np.trace(self.tmat)/2 - self.Nlatt*mu/2
        print("the exact energy by em and trt", -self.eig_eng[:self.Nlatt].sum()/2 + np.trace(self.tmat)/2 - self.Nlatt*mu/2)
        self.u, self.v = m2uv(self.eig_vec)
        self.m = uv2m(self.u, self.v)
        print("check the hamiltonian diagonalization: HM=ME, M=[[u,v*];[v,u*]]")
        check_orthonormal(self.ham, self.m)
        self.covariance()
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=6)
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=0.98)
    parser.add_argument("-lamb", type=float, default=1.78)
    parser.add_argument("-D", type=int, default=16)
    parser.add_argument("-pbc", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(0)

    chi, delta, lamb = round(args.chi, 6), round(args.delta, 6), round(args.lamb, 6)
    lx, D, pbc = args.lx, args.D, args.pbc

    single_kitaev_chain = singlechain(chi, delta, lamb, lx, D, pbc)
    single_kitaev_chain.hamiltonian()
    
    covmat = single_kitaev_chain.cov_mat
    print("the correlation matrix is", covmat)

    fig, axs = plt.subplots(1, 2)
    axs[0].matshow(single_kitaev_chain.ham)
    axs[0].set_title('ham')
    axs[1].matshow(single_kitaev_chain.cov_mat)
    axs[1].set_title('covmat')
    plt.show()
    
    spinhalfchain = spinhalf(chi, delta, lamb, lx, D, pbc)
    spinhalfchain.hamiltonian()
    
    fig, axs = plt.subplots(1, 2)
    axs[0].matshow(spinhalfchain.ham)
    axs[0].set_title('ham')
    axs[1].matshow(spinhalfchain.cov_mat)
    axs[1].set_title('covmat')
    plt.show()
    
    