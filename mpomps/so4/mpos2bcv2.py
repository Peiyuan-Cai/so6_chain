"""
The MPO-MPS method for SO(4) BBQ model with good quantum number U(1)xU(1) given by Cartan subalgebra. 

Puiyuen 2024.10.24
    2024.10.24: code collection
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
import itertools
import matplotlib.pyplot as plt
import numpy as np
from bdgpack import *
from so4bbqham import *
from tenpy.algorithms import dmrg
import logging
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True

class SpinDoubleChain():
    def __init__(self, chi, delta, lamb, L, pbc):
        """
        The Double Spin chain class, defined under the standard basis. 
        
        The Hamiltonian is in the BdG form, the matrix is written under the Bogoliubov *QUASIHOLE* representation. 
        """
        self.L = L
        self.chi = chi
        self.delta = delta
        self.lamb = lamb
        self.pbc = pbc
        self.model = "Single Kitaev chain parameters_L{}_chi{}_delta{}_lambda{}_pbc{}".format(L, round(chi, 3), round(delta, 3), round(lamb, 3), pbc)
        self.dtype = np.float64
        if pbc not in [1, 0, -1]:
            raise "Check pbc must be 0:open or 1:periodic or -1:anti-periodic"
        
    def Wannier_Z2(self, g1, g2, N=1):
        norbital, n = g1.shape        
        position = np.power( list(range(1,n+1)), N)
        position = np.diag(position) 
        position12 = g1.conj() @ position @ g1.T + g2.conj() @ position @ g2.T
        position12 = (position12 + position12.T.conj())/2.
        D, U = np.linalg.eigh(position12)
        index = np.argsort(D)
        print('The MLWOs locate at', D)
        U = U[:,index]
        g3 = U.T @ g1
        g4 = U.T @ g2
        index1 = np.zeros(norbital, dtype=int)
        if norbital%2 == 0:
            index1[0:(norbital):2] = np.ceil( range(0, norbital//2) ).astype(int)
            index1[1:(norbital):2] = np.ceil( range(norbital-1, norbital//2-1, -1) ).astype(int)
        else:
            index1[0:(norbital):2] = np.ceil( range(0, norbital//2+1) ).astype(int)
            index1[1:(norbital):2] = np.ceil( range(norbital-1, norbital//2, -1) ).astype(int)
        g3 = g3[index1,:]
        g4 = g4[index1,:]
        return g3.T, g4.T
            
    def calc_hamiltonian(self):
        L = self.L
        t = -self.chi
        d = self.delta
        mu = self.lamb
        self.tmat = np.zeros((L,L), dtype=float)
        self.dmat = np.zeros((L,L), dtype=float)
        for i in range(L):
            self.tmat[i, i] = mu/2 
        for i in range(L-1):
            self.tmat[i, (i+1)%L] = t 
            self.dmat[i, (i+1)%L] = d 

        if self.pbc == 1:
            self.tmat[L-1, 0] = t
            self.dmat[L-1, 0] = d
            print("Periodic terms added in SINGLE Kitaev chain. ")
        elif self.pbc == -1:
            self.tmat[L-1, 0] = -t
            self.dmat[L-1, 0] = -d
            print("Anti-periodic terms added in SINGLE Kitaev chain. ")
        else:
            print("No periodic term added. ")
        
        self.tmat += self.tmat.T.conj() 
        self.dmat -= self.dmat.T

        self.bigtmat = np.block([[self.tmat, np.zeros((L,L))],[np.zeros((L,L)), self.tmat]])
        self.bigdmat = np.block([[np.zeros((L,L)), -self.dmat],[self.dmat, np.zeros((L,L))]])
        
        self.ham = np.block([[self.bigtmat, self.bigdmat],[-self.bigdmat.conj(), -self.bigtmat.conj()]])
        
        self.eig_eng, self.eig_vec = bdgeig(self.ham)
        print("the eig energies", np.real(self.eig_eng))
        
        self.H14 = np.block([[self.tmat, -self.dmat], [-self.dmat.conj().T, -self.tmat.conj()]])
        self.H23 = np.block([[self.tmat, self.dmat], [self.dmat.conj().T, -self.tmat.conj()]])
        eig_eng14, eig_vec14 = bdgeig(self.H14)
        self.smallM14 = eig_vec14
        eig_eng23, eig_vec23 = bdgeig(self.H23)
        self.smallM23 = eig_vec23
        
        self.V14, self.U14 = m2vu(self.smallM14)
        self.V23, self.U23 = m2vu(self.smallM23)
        
        zeromat = np.zeros((L,L))
        
        self.bigM = np.block([[self.V14, zeromat, zeromat, self.U14.conj()],
                       [zeromat, self.V23, self.U23.conj(), zeromat],
                       [zeromat, self.U23, self.V23.conj(), zeromat],
                       [self.U14, zeromat, zeromat, self.V14.conj()]])
        
        self.V, self.U = m2vu(self.bigM)
        
class partonsite(Site):
    def __init__(self, cons_N=None, cons_S=None):
        """
        The 4 in 1 parton site for MPO-MPS method, meaning that we are doing 2 pairs of 2-coupled chain, filling 1 parton each site. 
    
        Local physical leg dimension = 16 = 2**4. Parton flavors are w,x,y,z for the hatted 1 to 4, but it is in the SO(4) standard basis not the real parton flavor. We are going to use the U(1)xU(1) symmetry directly. 
        
        Args:
            cons_N (str, optional): good quantum number: the parton number. Defaults to None. Optional to be 'N', 'Z2'
            cons_S (str, optional): good quantum number: the parton flavor. Defaults to None. Optional to be 'U1'

        Notes:
            There are two different types of cons_N, 'N' for the parton filling number, and 'Z2' for a fake quantum number which is useful after projection. 
        """
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S

        #itertools counting part
        flavors = ['w','x','y','z']
        combinations = []
        for i in range(1, len(flavors) + 1):
            for combo in itertools.combinations(flavors, i):
                combinations.append(''.join(combo))

        flavorqn = [[0,0]] #the first state is empty, and the qn set of 'S' and 'T' is [0,0]
        for i in range(len(combinations)):
            flavorqn.append(self.flavor_qn(combinations[i]))

        leglist0 = [[0,0]]; leglist1 = []; leglist2 = []
        for i in range(len(flavorqn)):
            leglist1.append([1, flavorqn[i]])
            leglist2.append(flavorqn[i])
        for i in range(len(flavorqn)-1):
            leglist0.append([len(combinations[i]), flavorqn[i+1]])

        if cons_N == 'N' and cons_S == 'U1': #changed flavor into U1
            chinfo = npc.ChargeInfo([1, 1], ['N', 'U1'])
            leg = npc.LegCharge.from_qflat(chinfo, leglist0)
        elif cons_N == 'Z2' and cons_S == 'U1':
            chinfo = npc.ChargeInfo([1, 1], ['Z2', 'U1'])
            leg = npc.LegCharge.from_qflat(chinfo, leglist1)
        elif cons_N == None and cons_S == 'U1':
            chinfo = npc.ChargeInfo([1, 1], ['S','T'])
            leg = npc.LegCharge.from_qflat(chinfo, leglist2)
        elif cons_N == None and cons_S == None:
            print("No symmetry used in site 'sixparton'. ")
            leg = npc.LegCharge.from_trivial(16)
        else:
            raise ValueError("Check your conserve quantities. ")

        names = ['empty']+combinations #now names are the str form of 64 basis

        #operators
        id16 = np.eye(len(names))

        JW = np.eye(len(names))
        for i in range(1,len(names)):
            if len(names[i]) %2 == 1:
                JW[i,i] = -1
        
        Fw = self.fmatrix('w',names); 
        Fx = self.fmatrix('x',names); Fy = self.fmatrix('y',names); Fz = self.fmatrix('z',names); 
        
        awdag = self.adaggermatrix('w',names); 
        axdag = self.adaggermatrix('x',names); aydag = self.adaggermatrix('y',names); azdag = self.adaggermatrix('z',names); 
        
        cwdag = awdag
        cxdag = axdag @ Fw
        cydag = aydag @ Fx @ Fw
        czdag = azdag @ Fy @ Fx @ Fw
        
        #print('Should be true', np.allclose(Fz@Fy@Fx@Fw@Fv@Fu,JW))
        
        cw = cwdag.T; cx = cxdag.T; cy = cydag.T; cz = czdag.T; 
        
        ops = dict(id16=id16, JW=JW, 
                   cwdag=cwdag, cxdag=cxdag, cydag=cydag, czdag=czdag, 
                   cw=cw, cx=cx, cy=cy, cz=cz)
        
        Site.__init__(self, leg, names, **ops)
        
    def flavor_qn(self, combination):
        qn_map = {'w': [-1,0], 'x': [0,-1], 'y': [0,1], 'z': [1,0]}
        totalqn = [0,0]
        for char in combination:
            for i in range(len(totalqn)):
                totalqn[i] += qn_map[char][i]
        return totalqn

    #a function to help you calculate the matrix form of c_u^\dagger under the 64-dim basis
    def adaggermatrix(self, flavor, basis):
        """
        Calculate matrix form of a_[flavor]^\dagger under [basis]

        Args:
            flavor (str): the flavor of parton, chosen from u,v,w,x,y,z
            basis (list of str): exactly the var [name] defined in Site.__init__, it's 64-dim, ['empty','u','v',...]

        Returns:
            amatrixform (ndarray): the matrix form of a_flavor^\dagger
            
        Notes:
            It's ugly, but it works well. 
        """
        basislength = len(basis) #for SO(4), it's 16
        adaggermatrixform = np.zeros((basislength,basislength))
        for l in range(basislength):
            if basis[l] == 'empty':
                setL = set()
            else:
                setL = set(basis[l])
            for r in range(basislength):
                if basis[r] == 'empty':
                    setR = set()
                else:
                    setR = set(basis[r])
                
                if (flavor in setL) and (flavor not in setR):
                    diff = setL - setR
                    listdiff = list(diff)
                    
                    if len(setL)-len(setR)==1 and len(listdiff) == 1 and listdiff[0] == flavor:
                        adaggermatrixform[l,r] = 1
        return adaggermatrixform

    def fmatrix(self, flavor, basis):
        """
        Calculate the Jordan-Wigner F matrix of a given flavor. Convention is in consist with the one on ITensor doc

        Args:
            flavor (str): the flavor of parton, chosen from u,v,w,x,y,z
            basis (list of str): exactly the var [name] defined in Site.__init__, it's 64-dim, ['empty','u','v',...]
            
        Returns:
            fmat (ndarray): the in-site Jordan-Wigner matrix of the given flavor
        """
        flist = [1]
        for i in range(1,len(basis)):
            if flavor in basis[i]:
                flist.append(-1)
            else:
                flist.append(1)
        fmat = np.diag(flist)
        return fmat