"""
The Pfaffian method on Kitaev chain model, work!

Peiyuan@202403

Comments:
    1. (2024.03.20) We haven't consider the anti-periodic condition
    2. (2024.04.25) Take the code into its minimum
    3. (2024.05.23) The theoretic exact energy is correct now, and t=(jx+jy)/4 now
"""
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import sparse
from gaussianbdg import *
from tenpy.tools.params import asConfig
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge, ChargeInfo
from copy import deepcopy

import matplotlib.pyplot as plt

class XYChain(CouplingModel):
    def __init__(self, model_params):
        print(model_params)
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        Lx = model_params.get('Lx', 4)
        S = model_params.get('S', 0.5)
        bc = model_params.get('bc', ['periodic'])
        bc_MPS = model_params.get('bc_MPS', 'finite')
        conserve = model_params.get('conserve', 'parity')
        self.verbose = model_params.get('verbose', 2)
        
        site = SpinSite(S=0.5, conserve=conserve)
        self.sites = [site]*Lx
        self.lat = Chain(m.Nlatt, site, bc=bc)
        CouplingModel.__init__(self, self.lat, explicit_plus_hc=False)
        self.init_terms(model_params)
        H_MPO = self.calc_H_MPO()
        if model_params.get('sort_mpo_legs', False):
            H_MPO.sort_legcharges()
        MPOModel.__init__(self, self.lat, H_MPO)
        model_params.warn_unused()

    def init_terms(self, model_params):
        self._Jx = model_params.get('jx', 1)
        self._Jy = model_params.get('jy', 1)
        self._hz = model_params.get('hz', 0)
        if abs(self._hz) > 1e-12:
           self.add_onsite(self._hz, 0, "Sz")
        self.add_coupling(self._Jx, 0, "Sx", 0, "Sx", np.array([1]))
        self.add_coupling(self._Jy, 0, "Sy", 0, "Sy", np.array([1]))

class kitaevchain(bdg):
    """
    Create a bdg class of Kitaev chain
    """
    def __init__(self, jx, jy, mu, Nx, D):
        self.jx, self.jy = jx, jy
        self.model = "Kitaevchain_L{}_jx{}_jy{}_hz{}_D{}".format(Nx, round(jx,6), round(jy,6), round(mu,6), D)
        super().__init__(Nx=Nx, Ny=1, model=self.model, D=D)
        self.t, self.d = round(( jx + jy )/4, 6), round(( jx - jy )/4, 6) #20240321 why devided by 8 here -> Sx = 1/2 sigma_x and doubled when doing JW trans
        self.mu = mu
        self.path = "./gaussiandata/"+self.model
        self.dtype = np.float64
        self.pbc = 1
        
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
        
        '''
        fig, axs = plt.subplots(1, 2)
        axs[0].matshow(self.tmat)
        axs[0].set_title('tmat')
        axs[1].matshow(self.dmat)
        axs[1].set_title('dmat')
        plt.show()
        '''

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
    import os
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=4)
    parser.add_argument("-jx", type=float, default=1.)
    parser.add_argument("-jy", type=float, default=0.)
    parser.add_argument("-hz", type=float, default=0.)
    parser.add_argument("-D", type=int, default=10)
    parser.add_argument("-pbc", type=int, default=1)
    parser.add_argument("-isite", type=int, default=1)
    parser.add_argument("-fsite", type=int, default=4)
    parser.add_argument("-jobs", type=str, default='mps')
    parser.add_argument("-fname", type=str, default='')
    parser.add_argument("-ifpl", type=int, default=0)
    args = parser.parse_args()
    if args.fsite > args.lx:
        args.fsite = args.lx

    np.random.seed(0)
    #sp.random.seed(0)

    jx, jy, mu = round(args.jx,6), round(args.jy,6), np.round(args.hz,6) #2024.03.21 12:16 (latest) but we will have error when jx=jy if we don't divided this by 2, so just keep this /2
    D = args.D
    lx = args.lx
    pbc = args.pbc

    m = kitaevchain(jx, jy, mu, lx, D)
    m.pbc = args.pbc

    if m.pbc == 1:
        bc = "periodic"
    model_params = dict(Lx=m.Nlatt, jx=jx, jy=jy, hz=mu, bc=bc)
    latt = XYChain(model_params)
    sites = latt.sites
    
    m.hamiltonian()
    mps = []
    for i in range(1,lx+1): #i=1,2,...,16
        tensortemp = m.A_nba(i)
        mps.append(tensortemp)
        
    bflat = deepcopy(mps)
    for i in range(lx):
        if i == 0:
            bflat[i] = np.reshape(bflat[i], (1,2,2))
            bflat[i] = np.transpose(bflat[i], (1,0,2))
        elif i == lx-1:
            bflat[i] = np.reshape(bflat[i], (2,2,1))
    #now bflat is a list of B-tensor in the leg label ['p','vL','vR']
        
    for i in range(1,lx+1):
        print("shape of mps", np.shape(mps[i-1]))
        
    #BUILD FROM BFLAT
    ppsi = MPS.from_Bflat(sites, bflat)
    #canonical form of ppsi
    ppsi.canonical_form()

    #sandwich energy
    xympo = latt.H_MPO
    
    #DMRG from random initial state
    print("DMRG from random state")
    psi = MPS.from_product_state(sites, ['up'] * lx, "finite")
    from tenpy.algorithms import dmrg
    dmrg_params = dict(mixer=True, max_E_err=1.e-14, max_chi = 20)
    eng = dmrg.TwoSiteDMRGEngine(psi, latt, dmrg_params)
    E, psi = eng.run()
    print("psi after dmrg is", psi)
    print("psi after dmrg has shpae")
    for i in range(lx):
        print("psi", i, 'has shape', np.shape(psi._B[i]))
    print("DMRG eng = ", E)
    
    #check the sandwich of psi
    print(" ")
    print("Expectation value of XY model DMRG result", xympo.expectation_value(psi))
    
    #check the sandwich of ppsi
    print(" ")
    sandwich_ppsi = xympo.expectation_value(ppsi)
    print("Expectation value of PFAFFIAN result", sandwich_ppsi)
    
    #the overlap of DMRG result and PFAFFIAN result
    print(" ")
    print("overlap of DMRG and PFAFFIAN results", psi.overlap(ppsi))
    
    #ppsi as DMRG initial state
    peng = dmrg.TwoSiteDMRGEngine(ppsi, latt, dmrg_params)
    E, ppsi = peng.run()
    print("ppsi DMRG eng = ", E)
    
    #exact energy analytically
    exact_energy =  m.exact_EgsXY
    print("exct eng of XY model (with shift)= Egs - L*hz/2",  exact_energy)
    print("the energy deviation", np.abs(exact_energy - sandwich_ppsi)/exact_energy)