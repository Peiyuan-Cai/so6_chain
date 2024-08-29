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
import itertools
import matplotlib.pyplot as plt
import numpy as np
from bdgpack import *
from tenpy.algorithms import dmrg
import logging
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True

class KitaevSingleChain():
    def __init__(self, chi, delta, lamb, L, pbc=0):
        """
        The Single Kitaev chain class. 
        
        The Hamiltonian is in the BdG form, the matrix is written under the Bogoliubov *QUASIHOLE* representation. 

        Args:
            chi (float): variational parameter $\chi$
            delta (float): variational parameter $\delta$
            lamb (float): variational parameter $\lambda$
            L (int): Chain length
            pbc (int, optional): Boundary condition. Defaults to be 0. 0 for OBC, 1 for PBC, 2 for APBC. 
            
        Raises:
            Check pbc must be 0:open or 1:periodic or -1:anti-periodic: check your boundary condition input
            
        Notes:
            240827: We should now have three types of boundary conditions namely 0(open), 1(periodic) and -1(anti-periodic), and calcel the 'bc' stuff
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
        '''
        20240828: any necessarity to consider parity? I don't think so.  
        self.parity = 1
        if self.pbc == 1:
            parity = - 1 + 2 * ( L % 2 )
            self.parity = parity
            self.tmat[L-1, 0] = t*parity 
            self.dmat[L-1, 0] = d*parity 
        '''
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
        
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])
        self.eig_eng, self.eig_vec = bdgeig(self.ham)
        print("the eig energies", self.eig_eng)
        self.V, self.U = m2vu(self.eig_vec)
        self.M = vu2m(self.V, self.U)

'''
---------------------------6 parton site designed function begin---------------------------
'''
#a funtion to calculate the values in the sixparton flavor quantum number
def flavor_qn(combination):
    qn_map = {'u': -5/2, 'v': -3/2, 'w': -1/2, 'x': 1/2, 'y': 3/2, 'z': 5/2}
    totalqn = 0
    for char in combination:
        totalqn += qn_map[char]
    return totalqn

#a function to help you calculate the matrix form of c_u^\dagger under the 64-dim basis
def adaggermatrix(flavor, basis):
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
    basislength = len(basis) #for SO(6) case, it's 64
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
                    print('l=',setL,'r=',setR)
    return adaggermatrixform

def fmatrix(flavor, basis):
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

'''
---------------------------6 parton site designed function end-----------------------------
'''

class sixparton(Site):
    def __init__(self, cons_N=None, cons_S=None):
        """
        The 6 in 1 parton site for MPO-MPS method, meaning that we are doing 6 decoupled Kitaev chain, filling 1 parton each site. 
    
        Local physical leg dimension = 64 = 2**6. Parton flavors are u,v,w,x,y,z for the hatted 1 to 6, the SO(6) basis. 
        
        Args:
            cons_N (str, optional): good quantum number: the parton number. Defaults to None. Optional to be 'N', 'Z2'
            cons_S (str, optional): good quantum number: the parton flavor. Defaults to None. Optional to be 'flavor'

        Notes:
            There are two different types of cons_N, 'N' for the parton filling number, and 'Z2' for a fake quantum number which is useful after projection. 
        """
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S

        #itertools counting part
        flavors = ['u','v','w','x','y','z']
        combinations = []
        for i in range(1, len(flavors) + 1):
            for combo in itertools.combinations(flavors, i):
                combinations.append(''.join(combo))

        flavorqn = [0] #the first state is empty, and the qn is 0
        for i in range(len(combinations)):
            flavorqn.append(flavor_qn(combinations[i]))

        leglist0 = [[0,0]]; leglist1 = []; leglist2 = []
        for i in range(len(flavorqn)):
            leglist1.append([1, flavorqn[i]])
            leglist2.append([flavorqn[i]])
        for i in range(len(flavorqn)-1):
            leglist0.append([len(combinations[i]), flavorqn[i+1]])

        if cons_N == 'N' and cons_S == 'flavor':
            chinfo = npc.ChargeInfo([1, 1], ['N', 'flavor'])
            leg = npc.LegCharge.from_qflat(chinfo, leglist0)
        elif cons_N == 'Z2' and cons_S == 'flavor':
            chinfo = npc.ChargeInfo([1, 1], ['Z2', 'flavor'])
            leg = npc.LegCharge.from_qflat(chinfo, leglist1)
        elif cons_N == None and cons_S == 'flavor':
            chinfo = npc.ChargeInfo([1], ['flavor'])
            leg = npc.LegCharge.from_qflat(chinfo, leglist2)
        else:
            print("No symmetry used in site 'sixparton'. ")
            leg = npc.LegCharge.from_trivial(64)

        names = ['empty']+combinations #now names are the str form of 64 basis

        #operators
        id64 = np.diag([1]*64)

        JW = np.eye(64)
        for i in range(1,64):
            if len(names[i]) %2 == 1:
                JW[i,i] = -1
        
        Fu = fmatrix('u',names); Fv = fmatrix('v',names); Fw = fmatrix('w',names); 
        Fx = fmatrix('x',names); Fy = fmatrix('y',names); Fz = fmatrix('z',names); 
        
        audag = adaggermatrix('u',names); avdag = adaggermatrix('v',names); awdag = adaggermatrix('w',names); 
        axdag = adaggermatrix('x',names); aydag = adaggermatrix('y',names); azdag = adaggermatrix('z',names); 
        
        cudag = audag
        cvdag = avdag @ Fu
        cwdag = awdag @ Fv @ Fu
        cxdag = axdag @ Fw @ Fv @ Fu
        cydag = aydag @ Fx @ Fw @ Fv @ Fu
        czdag = azdag @ Fy @ Fx @ Fw @ Fv @ Fu
        
        cu = cudag.T; cv = cvdag.T; cw = cwdag.T; cx = cxdag.T; cy = cydag.T; cz = czdag.T; 
        
        ops = dict(id64=id64, JW=JW, 
                   cudag=cudag, cvdag=cvdag, cwdag=cwdag, cxdag=cxdag, cydag=cydag, czdag=czdag, 
                   cu=cu, cv=cv, cw=cw, cx=cx, cy=cy, cz=cz)
        
        Site.__init__(self, leg, names, **ops)