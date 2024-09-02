"""
The MPO-MPS method for SO(3) BBQ model

Puiyuen 2024.06.03-2024.06.11

2024.08.27: this code does work. 
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
import logging
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
import matplotlib.pyplot as plt
import numpy as np
from bdgpack import *
from tenpy.algorithms import dmrg
from tenpy.models import AKLTChain


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
        """
        self.L = L
        self.chi = chi
        self.delta = delta
        self.lamb = lamb
        self.bc = bc
        self.model = "Single Kitaev chain_L{}_chi{}_delta{}_lambda{}_bc{}".format(L, round(chi, 3), round(delta, 3), round(lamb, 3), bc)
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
        
        #matshow module
        #fig, axs = plt.subplots(1, 2)
        #axs[0].matshow(self.tmat)
        #axs[0].set_title('tmat')
        #axs[1].matshow(self.dmat)
        #axs[1].set_title('dmat')
        #plt.show()
        
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])
        self.eig_eng, self.eig_vec = bdgeig(self.ham)
        print("the eig energies", self.eig_eng)
        self.V, self.U = m2vu(self.eig_vec)
        self.M = vu2m(self.V, self.U)
        
        #matshow module
        #fig, axs = plt.subplots(1, 2)
        #axs[0].matshow(np.real(self.V))
        #axs[0].set_title('realV')
        #axs[1].matshow(np.real(self.U))
        #axs[1].set_title('realU')
        #plt.show()

class threeparton(Site):
    def __init__(self, cons_N=None, cons_S=None):
        """
        The 3 in 1 parton site for MPO-MPS method, different from the one for DMRG
    
        local physical leg dimension = 8 = 2**3
        empty, single occupancy of(x,y,z), double occupancy, full
        
        this site is a combination of 3 parton sites, and for the MPOS method, there is no need to define operators here
        Args:
            cons_N (str, optional): good quantum number: the parton number. Defaults to None.
            cons_S (str, optional): good quantum number: the parton flavor. Defaults to None.
        """
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        if cons_N == 'N' and cons_S == 'xyz':
            chinfo = npc.ChargeInfo([1, 1], ['N', 'xyz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[0, 0], [1, 1], [1, -1], [1, 0], [2, 0], [2, 1], [2, -1], [3, 0]])
        elif cons_N == 'Z2' and cons_S == 'xyz':
            chinfo = npc.ChargeInfo([1, 1], ['N', 'xyz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 0], [1, 1], [1, -1], [1, 0], [1, 0], [1, 1], [1, -1], [1, 0]])
        else:
            print("No symmetry used in site 'threeparton'. ")
            leg = npc.LegCharge.from_trivial(8)
        
        names = ['empty', 'x', 'y', 'z', 'xy', 'zx', 'yz', 'xyz']
        
        '''
        ###
        archieved 20240829, they are wrong. See the developer log for more details. 
        Besisdes JW, we have to bring Fx and Fy, these two are the JW matrices inside the i-th site, JW is the JW matrix of the i-th site. I think this is enough to wake you up. 
        ###
        
        JW = np.diag([1,-1,-1,-1,1,1,1,-1]) #This is the JW string of the whole site-i (20240829)
        id8 = np.diag([1,1,1,1,1,1,1,1])
        
        cxdag = np.zeros((8,8))
        cxdag[1,0] = 1; cxdag[4,2] = 1; cxdag[5,3] = 1; cxdag[7,6] = 1; #these are actually a matrices? without any sign brought by JW (20240827)
        cx = cxdag.T
        cydag = np.zeros((8,8))
        cydag[2,0] = 1; cydag[4,1] = 1; cydag[6,3] = 1; cydag[7,5] = 1; 
        cy = cydag.T
        czdag = np.zeros((8,8))
        czdag[3,0] = 1; czdag[5,1] = 1; czdag[6,2] = 1; czdag[7,4] = 1; 
        cz = czdag.T

        cxdagF = cxdag @ JW
        Fcx = JW @ cx
        cydagF = cydag @ JW
        Fcy = JW @ cy
        czdagF = czdag @ JW
        Fcz = JW @ cz

        ops = dict(JW=JW, id=id8, 
                   cxdag = cxdag, cx = cx, cydag = cydag, cy = cy, czdag = czdag, cz = cz, 
                   cxdagF = cxdagF, Fcx = Fcx, cydagF = cydagF, Fcy = Fcy, czdagF = czdagF, Fcz = Fcz)
        
        Site.__init__(self, leg, names, **ops)
        '''
        JW = np.diag([1,-1,-1,-1,1,1,1,-1])
        Fx = np.diag([1,-1,1,1,-1,-1,1,-1])
        Fy = np.diag([1,1,-1,1,-1,1,-1,-1])
        id8 = np.diag([1,1,1,1,1,1,1,1])
        
        axdag = np.zeros((8,8))
        axdag[1,0] = 1; axdag[4,2] = 1; axdag[5,3] = 1; axdag[7,6] = 1; 
        ax = axdag.T
        aydag = np.zeros((8,8))
        aydag[2,0] = 1; aydag[4,1] = 1; aydag[6,3] = 1; aydag[7,5] = 1; 
        ay = aydag.T
        azdag = np.zeros((8,8))
        azdag[3,0] = 1; azdag[5,1] = 1; azdag[6,2] = 1; azdag[7,4] = 1; 
        az = azdag.T

        cxdag = axdag
        cx = ax
        cydag = aydag @ Fx
        cy = Fx @ ay
        czdag = azdag @ Fy @ Fx
        cz = Fx @ Fy @ az

        ops = dict(JW=JW, Fx=Fx, Fy=Fy, id8=id8, 
                   cxdag = cxdag, cx = cx, cydag = cydag, cy = cy, czdag = czdag, cz = cz) #with the JW string embedded in c operators, there is no need to add a operators into ops dict. 
        
        Site.__init__(self, leg, names, **ops)
        
class MPOMPS():
    def __init__(self, v, u, **kwargs):
        self.cons_N = kwargs.get("cons_N", "Z2")
        self.cons_S = kwargs.get("cons_S", "xyz")
        self.trunc_params = kwargs.get("trunc_params", dict(chi_max=20) )
        
        assert v.ndim == 2
        self._V = v
        self._U = u
        assert self._U.shape == self._V.shape
        self.projection_type = kwargs.get("projection_type", "Gutz")
        #self.L = self.Llat = u.shape[1]//3 #for 3 in 1 site, only 1 Kitaev chain should be calculated
        self.L = self.Llat = u.shape[0] #the length of real sites
        
        self.site = threeparton(self.cons_N, self.cons_S)
        self.init_mps()
        
    def init_mps(self, init=None):
        L = self.L
        if init == None:
            init = [0] * L #all empty
        site = self.site
        self.init_psi = MPS.from_product_state([site]*L, init)
        self.psi = self.init_psi.copy()
        self.n_omode = 0
        return self.psi
    
    def get_mpo_Z2U1(self, v, u, xyz):
        chinfo = self.site.leg.chinfo
        pleg = self.site.leg #physical leg
        
        if xyz == 1: #v cxdag + u cy
            qn = [0, 1]
        elif xyz == 0: #v czdag + u cz
            qn = [0, 0]
        elif xyz == -1: #v cydag + u cx
            qn = [0,-1]
        else:
            raise "quantum number of d mode i.e. xyz should be 1,0,-1"
        
        firstleg = npc.LegCharge.from_qflat(chinfo, [[0, 0]], 1)
        lastleg = npc.LegCharge.from_qflat(chinfo, [qn], -1)
        bulkleg = npc.LegCharge.from_qflat(chinfo, [qn, [0, 0]], 1)
        #legs arrange in order 'wL', 'wR', 'p', 'p*'
        legs_first = [firstleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_bulk = [bulkleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_last = [bulkleg, lastleg, pleg, pleg.conj()]
        
        mpo = []
        L = self.L
        
        t0 = npc.zeros(legs_first, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        i = 0
        if xyz == 1:
            t0[0, 0, 1, 0] = v[0]; t0[0, 0, 4, 2] = v[0]; t0[0, 0, 5, 3] = v[0]; t0[0, 0, 7, 6] = v[0]; #v cxdag
            t0[0, 0, 0, 2] = u[0]; t0[0, 0, 1, 4] = -u[0]; t0[0, 0, 3, 6] = u[0]; t0[0, 0, 5, 7] = -u[0]; #u F cy
        elif xyz == 0:
            t0[0, 0, 3, 0] = v[0]; t0[0, 0, 5, 1] = -v[0]; t0[0, 0, 6, 2] = -v[0]; t0[0, 0, 7, 4] = v[0]; #v czdag F
            t0[0, 0, 0, 3] = u[0]; t0[0, 0, 1, 5] = -u[0]; t0[0, 0, 2, 6] = -u[0]; t0[0, 0, 4, 7] = u[0]; #u F cz
        elif xyz == -1:
            t0[0, 0, 2, 0] = v[0]; t0[0, 0, 4, 1] = -v[0]; t0[0, 0, 6, 3] = v[0]; t0[0, 0, 7, 5] = -v[0]; #v cydag F
            t0[0, 0, 0, 1] = u[0]; t0[0, 0, 2, 4] = u[0]; t0[0, 0, 3, 5] = u[0]; t0[0, 0, 6, 7] = u[0]; #u cx
            
        t0[0, 1, 0, 0] = 1; t0[0, 1, 1, 1] = -1; t0[0, 1, 2, 2] = -1; t0[0, 1, 3, 3] = -1; 
        t0[0, 1, 4, 4] = 1; t0[0, 1, 5, 5] = 1; t0[0, 1, 6, 6] = 1; t0[0, 1, 7, 7] = -1; 
        mpo.append(t0)
        
        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
            ti[0,0,0,0] = 1; ti[0,0,1,1] = 1; ti[0,0,2,2] = 1; ti[0,0,3,3] = 1; 
            ti[0,0,4,4] = 1; ti[0,0,5,5] = 1; ti[0,0,6,6] = 1; ti[0,0,7,7] = 1; 
            if xyz == 1:
                ti[1, 0, 1, 0] = v[i]; ti[1, 0, 4, 2] = v[i]; ti[1, 0, 5, 3] = v[i]; ti[1, 0, 7, 6] = v[i]; 
                ti[1, 0, 0, 2] = u[i]; ti[1, 0, 1, 4] = -u[i]; ti[1, 0, 3, 6] = u[i]; ti[1, 0, 5, 7] = -u[i]; 
            elif xyz == 0:
                ti[1, 0, 3, 0] = v[i]; ti[1, 0, 5, 1] = -v[i]; ti[1, 0, 6, 2] = -v[i]; ti[1, 0, 7, 4] = v[i]; 
                ti[1, 0, 0, 3] = u[i]; ti[1, 0, 1, 5] = -u[i]; ti[1, 0, 2, 6] = -u[i]; ti[1, 0, 4, 7] = u[i]; 
            elif xyz == -1:
                ti[1, 0, 2, 0] = v[i]; ti[1, 0, 4, 1] = -v[i]; ti[1, 0, 6, 3] = v[i]; ti[1, 0, 7, 5] = -v[i]; 
                ti[1, 0, 0, 1] = u[i]; ti[1, 0, 2, 4] = u[i]; ti[1, 0, 3, 5] = u[i]; ti[1, 0, 6, 7] = u[i]; 
            ti[1, 1, 0, 0] = 1; ti[1, 1, 1, 1] = -1; ti[1, 1, 2, 2] = -1; ti[1, 1, 3, 3] = -1; 
            ti[1, 1, 4, 4] = 1; ti[1, 1, 5, 5] = 1; ti[1, 1, 6, 6] = 1; ti[1, 1, 7, 7] = -1; 
            mpo.append(ti)
                
        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        tL[0,0,0,0] = 1; tL[0,0,1,1] = 1; tL[0,0,2,2] = 1; tL[0,0,3,3] = 1; 
        tL[0,0,4,4] = 1; tL[0,0,5,5] = 1; tL[0,0,6,6] = 1; tL[0,0,7,7] = 1; 
        if xyz == 1:
            tL[1, 0, 1, 0] = v[i]; tL[1, 0, 4, 2] = v[i]; tL[1, 0, 5, 3] = v[i]; tL[1, 0, 7, 6] = v[i]; 
            tL[1, 0, 0, 2] = u[i]; tL[1, 0, 1, 4] = -u[i]; tL[1, 0, 3, 6] = u[i]; tL[1, 0, 5, 7] = -u[i]; 
        elif xyz == 0:
            tL[1, 0, 3, 0] = v[i]; tL[1, 0, 5, 1] = -v[i]; tL[1, 0, 6, 2] = -v[i]; tL[1, 0, 7, 4] = v[i]; 
            tL[1, 0, 0, 3] = u[i]; tL[1, 0, 1, 5] = -u[i]; tL[1, 0, 2, 6] = -u[i]; tL[1, 0, 4, 7] = u[i]; 
        elif xyz == -1:
            tL[1, 0, 2, 0] = v[i]; tL[1, 0, 4, 1] = -v[i]; tL[1, 0, 6, 3] = v[i]; tL[1, 0, 7, 5] = -v[i]; 
            tL[1, 0, 0, 1] = u[i]; tL[1, 0, 2, 4] = u[i]; tL[1, 0, 3, 5] = u[i]; tL[1, 0, 6, 7] = u[i]; 
        mpo.append(tL)
        
        return mpo

    def get_mpo_trivial(self, v, u, xyz):
        chinfo = self.site.leg.chinfo
        pleg = self.site.leg #physical leg

        firstleg = npc.LegCharge.from_trivial(1)
        lastleg = npc.LegCharge.from_trivial(1)
        bulkleg = npc.LegCharge.from_trivial(2)
        #legs arrange in order 'wL', 'wR', 'p', 'p*'
        legs_first = [firstleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_bulk = [bulkleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_last = [bulkleg, lastleg, pleg, pleg.conj()]
        
        mpo = []
        L = self.L
        
        '''
        ###
        #archieved 20240829, we are now trying to writen them in operators defined in the threeparton site
        ###
        
        t0 = npc.zeros(legs_first, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        i = 0
        if xyz == 1:
            t0[0, 0, 1, 0] = v[0]; t0[0, 0, 4, 2] = v[0]; t0[0, 0, 5, 3] = v[0]; t0[0, 0, 7, 6] = v[0]; #v cxdag
            t0[0, 0, 0, 1] = u[0]; t0[0, 0, 2, 4] = u[0]; t0[0, 0, 3, 5] = u[0]; t0[0, 0, 6, 7] = u[0]; #u cx
        elif xyz == 0:
            t0[0, 0, 3, 0] = v[0]; t0[0, 0, 5, 1] = -v[0]; t0[0, 0, 6, 2] = -v[0]; t0[0, 0, 7, 4] = v[0]; #v czdag F
            t0[0, 0, 0, 3] = u[0]; t0[0, 0, 1, 5] = -u[0]; t0[0, 0, 2, 6] = -u[0]; t0[0, 0, 4, 7] = u[0]; #u F cz
        elif xyz == -1:
            t0[0, 0, 2, 0] = v[0]; t0[0, 0, 4, 1] = -v[0]; t0[0, 0, 6, 3] = v[0]; t0[0, 0, 7, 5] = -v[0]; #v cydag F
            t0[0, 0, 0, 2] = u[0]; t0[0, 0, 1, 4] = -u[0]; t0[0, 0, 3, 6] = u[0]; t0[0, 0, 5, 7] = -u[0]; #u F cy
            
        t0[0, 1, 0, 0] = 1; t0[0, 1, 1, 1] = -1; t0[0, 1, 2, 2] = -1; t0[0, 1, 3, 3] = -1; 
        t0[0, 1, 4, 4] = 1; t0[0, 1, 5, 5] = 1; t0[0, 1, 6, 6] = 1; t0[0, 1, 7, 7] = -1; 
        mpo.append(t0)
        
        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
            ti[0,0,0,0] = 1; ti[0,0,1,1] = 1; ti[0,0,2,2] = 1; ti[0,0,3,3] = 1; 
            ti[0,0,4,4] = 1; ti[0,0,5,5] = 1; ti[0,0,6,6] = 1; ti[0,0,7,7] = 1; 
            if xyz == 1:
                ti[1, 0, 1, 0] = v[i]; ti[1, 0, 4, 2] = v[i]; ti[1, 0, 5, 3] = v[i]; ti[1, 0, 7, 6] = v[i]; 
                ti[1, 0, 0, 1] = u[i]; ti[1, 0, 2, 4] = u[i]; ti[1, 0, 3, 5] = u[i]; ti[1, 0, 6, 7] = u[i]; 
            elif xyz == 0:
                ti[1, 0, 3, 0] = v[i]; ti[1, 0, 5, 1] = -v[i]; ti[1, 0, 6, 2] = -v[i]; ti[1, 0, 7, 4] = v[i]; 
                ti[1, 0, 0, 3] = u[i]; ti[1, 0, 1, 5] = -u[i]; ti[1, 0, 2, 6] = -u[i]; ti[1, 0, 4, 7] = u[i]; 
            elif xyz == -1:
                ti[1, 0, 2, 0] = v[i]; ti[1, 0, 4, 1] = -v[i]; ti[1, 0, 6, 3] = v[i]; ti[1, 0, 7, 5] = -v[i]; 
                ti[1, 0, 0, 2] = u[i]; ti[1, 0, 1, 4] = -u[i]; ti[1, 0, 3, 6] = u[i]; ti[1, 0, 5, 7] = -u[i]; 
            ti[1, 1, 0, 0] = 1; ti[1, 1, 1, 1] = -1; ti[1, 1, 2, 2] = -1; ti[1, 1, 3, 3] = -1; 
            ti[1, 1, 4, 4] = 1; ti[1, 1, 5, 5] = 1; ti[1, 1, 6, 6] = 1; ti[1, 1, 7, 7] = -1; 
            mpo.append(ti)
                
        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        tL[0,0,0,0] = 1; tL[0,0,1,1] = 1; tL[0,0,2,2] = 1; tL[0,0,3,3] = 1; 
        tL[0,0,4,4] = 1; tL[0,0,5,5] = 1; tL[0,0,6,6] = 1; tL[0,0,7,7] = 1; 
        if xyz == 1:
            tL[1, 0, 1, 0] = v[i]; tL[1, 0, 4, 2] = v[i]; tL[1, 0, 5, 3] = v[i]; tL[1, 0, 7, 6] = v[i]; 
            tL[1, 0, 0, 1] = u[i]; tL[1, 0, 2, 4] = u[i]; tL[1, 0, 3, 5] = u[i]; tL[1, 0, 6, 7] = u[i]; 
        elif xyz == 0:
            tL[1, 0, 3, 0] = v[i]; tL[1, 0, 5, 1] = -v[i]; tL[1, 0, 6, 2] = -v[i]; tL[1, 0, 7, 4] = v[i]; 
            tL[1, 0, 0, 3] = u[i]; tL[1, 0, 1, 5] = -u[i]; tL[1, 0, 2, 6] = -u[i]; tL[1, 0, 4, 7] = u[i]; 
        elif xyz == -1:
            tL[1, 0, 2, 0] = v[i]; tL[1, 0, 4, 1] = -v[i]; tL[1, 0, 6, 3] = v[i]; tL[1, 0, 7, 5] = -v[i];    
            tL[1, 0, 0, 2] = u[i]; tL[1, 0, 1, 4] = -u[i]; tL[1, 0, 3, 6] = u[i]; tL[1, 0, 5, 7] = -u[i]; 
        mpo.append(tL)
        '''
        
        '''
        #archieved 20240830
        t0 = npc.zeros(legs_first, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        i = 0
        if xyz == 1:
            t0[0, 0, :, :] = v[0]*self.site.get_op('cxdag') + u[0]*self.site.get_op('cx')
        elif xyz == 0:
            t0[0, 0, :, :] = v[0]*self.site.get_op('czdag') + u[0]*self.site.get_op('cz')
        elif xyz == -1:
            t0[0, 0, :, :] = v[0]*self.site.get_op('cydag') + u[0]*self.site.get_op('cy')
            
        t0[0, 1, :, :] = self.site.get_op('JW')
        mpo.append(t0)
        
        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
            ti[0,0,:,:] = self.site.get_op('id8')
            if xyz == 1:
                ti[1, 0, :, :] = v[i]*self.site.get_op('cxdag') + u[i]*self.site.get_op('cx')
            elif xyz == 0:
                ti[1, 0, :, :] = v[i]*self.site.get_op('czdag') + u[i]*self.site.get_op('cz')
            elif xyz == -1:
                ti[1, 0, :, :] = v[i]*self.site.get_op('cydag') + u[i]*self.site.get_op('cy')
            ti[1, 1, :, :] = self.site.get_op('JW')
            mpo.append(ti)
                
        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        tL[0,0,:,:] = self.site.get_op('id8')
        if xyz == 1:
            tL[1, 0, :, :] = v[i]*self.site.get_op('cxdag') + u[i]*self.site.get_op('cx')
        elif xyz == 0:
            tL[1, 0, :, :] = v[i]*self.site.get_op('czdag') + u[i]*self.site.get_op('cz')
        elif xyz == -1:
            tL[1, 0, :, :] = v[i]*self.site.get_op('cydag') + u[i]*self.site.get_op('cy')
        mpo.append(tL)
        
        #And it works well
        '''
        
        op_dict = {1: ('cxdag', 'cx'), 0: ('czdag', 'cz'), -1: ('cydag', 'cy')}
        
        t0 = npc.zeros(legs_first, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        i = 0
        if xyz in op_dict:
            cr, an = op_dict[xyz]
            t0[0, 0, :, :] = v[i]*self.site.get_op(cr) + u[i]*self.site.get_op(an)
        t0[0, 1, :, :] = self.site.get_op('JW')
        mpo.append(t0)
        
        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
            ti[0,0,:,:] = self.site.get_op('id8')
            if xyz in op_dict:
                cr, an = op_dict[xyz]
                ti[1, 0, :, :] = v[i]*self.site.get_op(cr) + u[i]*self.site.get_op(an)
            ti[1, 1, :, :] = self.site.get_op('JW')
            mpo.append(ti)
                
        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        tL[0,0,:,:] = self.site.get_op('id8')
        if xyz in op_dict:
            cr, an = op_dict[xyz]
            tL[1, 0, :, :] = v[i]*self.site.get_op(cr) + u[i]*self.site.get_op(an)
        mpo.append(tL)
        
        return mpo
    
    def mpomps_step_1time(self, m, xyz):
        vm = self._V[:,m]
        um = self._U[:,m]
        mps = self.psi
        if self.cons_N==None and self.cons_S==None:
            mpo = self.get_mpo_trivial(vm, um, xyz)
        elif self.cons_N=='Z2' and self.cons_S=='xyz':
            mpo = self.get_mpo_Z2U1(vm, um, xyz)
        else:
            raise "Symmetry set of N and S is not allowed. "
        halflength = self.L//2
        for i in range(self.L):
            B = npc.tensordot(mps.get_B(i,'B'), mpo[i], axes=('p','p*'))
            B = B.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
            B.ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
            B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
            B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
            mps._B[i] = B
        err = mps.compress_svd(self.trunc_params)
        return err, mps
        
    def run(self, init=None):
        self.fidelity = 1
        if self.n_omode > 0:
            print("initialize the mpo-mps calculation mps")
            self.init_mps(init=init)
            self.n_omode = 0
        nmode = self._U.shape[0]
        print("MPO-MPS application start")
        
        xyzlist = [-1, 0, -1]
        for m in range(nmode):
            for xyz in xyzlist:
                err, self.psi = self.mpomps_step_1time(m, xyz)
                self.fidelity *= 1-err.eps
                self.chi_max = np.max(self.psi.chi)
                print( "applied the {}-th {} mode, the fidelity is {}, the largest bond dimension is {}. ".format( self.n_omode, xyz, self.fidelity, self.chi_max) )
            self.n_omode += 1

class Spin1(Site):
    """
    Customized Spin-1 site, local operators are generators of SU(3)
    """
    def __init__(self, cons_N=None, cons_S=None):
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        if cons_N is None and cons_S == 'xyz':
            chinfo = npc.ChargeInfo([1, 1], ['Z2', 'xyz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 1], [1, -1], [1, 0]])
        elif cons_N == 'Z2' and cons_S == 'xyz':
            chinfo = npc.ChargeInfo([1, 1], ['Z2', 'xyz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 1], [1, -1], [1, 0]])
        elif cons_N == 'Z2' and cons_S == None:
            chinfo = npc.ChargeInfo([2], ['Z2'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1],[1],[1]])
        else:
            print("No symmetry used in site 'Spin1'. ")
            leg = npc.LegCharge.from_trivial(3)

        JW = np.diag([-1, -1, -1])

        Syz = np.zeros((3,3), float); Syz[1, 2] = 1;
        Szy = Syz.T
        Sxz = np.zeros((3,3), float); Sxz[0, 2] = 1;
        Szx = Sxz.T
        Sxy = np.zeros((3,3), float); Sxy[0, 1] = 1;
        Syx = Sxy.T

        Q1 = np.diag([1,-1,0])
        Q2 = np.diag([1,1,-2])/np.sqrt(3)

        Sxx = np.diag([1., 0., 0.])
        Syy = np.diag([0., 1., 0.])
        Szz = np.diag([0., 0., 1.])
        
        #Q11 = np.diag([1.,1.,0.])
        #Q12 = np.diag([1,-1,0]) / np.sqrt(3)
        #Q21 = Q12
        #Q22 = np.diag([1,1,4])/3

        Q11 = Q1 @ Q1
        Q12 = Q1 @ Q2
        Q21 = Q12
        Q22 = Q2 @ Q2

        ops = dict(JW=JW,
                   Sxx=Sxx, Syy=Syy, Szz=Szz,
                   Syz=Syz, Szy=Szy, 
                   Sxz=Sxz, Szx=Szx,
                   Sxy=Sxy, Syx=Syx,
                   Q1=Q1, Q2=Q2,
                   Q11=Q11, Q12=Q12, Q21=Q21, Q22=Q22)
        names = ['x', 'y', 'z']
        Site.__init__(self, leg, names, **ops)

    def __repr__(self):
        return "site for spin-1 in trs basis with conserve = {}".format(["N", self.cons_S])

class BBQJK(CouplingModel):
    def __init__(self, model_params):
        print(model_params)
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        self.Lx = model_params.get('Lx', 12)
        self.S = model_params.get('S', 1)
        self.bc = model_params.get('bc', 'periodic')
        self.J = model_params.get('J', 1)
        self.K = model_params.get('K', 1/3)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        conserve = model_params.get('conserve', 'parity')
        self.verbose = model_params.get('verbose', 2)
        
        site = Spin1(cons_N=None, cons_S=None)
        self.sites = [site]*self.Lx
        self.lat = Chain(self.Lx, site, bc=bc)
        CouplingModel.__init__(self, self.lat, explicit_plus_hc=False)
        self.init_terms(model_params)
        H_MPO = self.calc_H_MPO()
        if model_params.get('sort_mpo_legs', False):
            H_MPO.sort_legcharges()
        MPOModel.__init__(self, self.lat, H_MPO)
        model_params.warn_unused()
    
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        K = model_params.get('K', 1/3)
        for l in range(self.Lx):
            if l < self.Lx - 1:
                i0, i1 = l, (l+1)%self.Lx
            elif l == self.Lx-1 and bc == 'periodic':
                i0, i1 = 0, self.Lx-1
                print("periodic terms added")
            else:
                break
            self.add_coupling_term(J,  i0, i1, "Sxy", "Syx")
            self.add_coupling_term(J,  i0, i1, "Syx", "Sxy")
            self.add_coupling_term(J,  i0, i1, "Syz", "Szy")
            self.add_coupling_term(J,  i0, i1, "Szy", "Syz")
            self.add_coupling_term(J,  i0, i1, "Szx", "Sxz")
            self.add_coupling_term(J,  i0, i1, "Sxz", "Szx")

            self.add_coupling_term(K-J,  i0, i1, "Sxy", "Sxy")
            self.add_coupling_term(K-J,  i0, i1, "Sxz", "Sxz")
            self.add_coupling_term(K-J,  i0, i1, "Syx", "Syx")
            self.add_coupling_term(K-J,  i0, i1, "Syz", "Syz")
            self.add_coupling_term(K-J,  i0, i1, "Szx", "Szx")
            self.add_coupling_term(K-J,  i0, i1, "Szy", "Szy")

            self.add_coupling_term(K/2,  i0, i1, "Q1", "Q1")
            self.add_coupling_term(K/2,  i0, i1, "Q2", "Q2")
            #self.add_coupling_term(K*4/3, i0, i1, "Id", "Id") #this is a constant term
    
    def run_dmrg(self, **kwargs):
        mixer      = kwargs.get('mixer', True)
        chi_max    = kwargs.get('chi_max', 100)
        max_E_err  = kwargs.get('max_E_err', 1e-12)
        max_sweeps = kwargs.get('max_sweeps', 6)
        min_sweeps = kwargs.get('min_sweeps', min(3, max_sweeps) )
        dmrg_params = dict(mixer=mixer, 
                           trunc_params=dict(chi_max=chi_max),
                           max_E_err=max_E_err, 
                           max_sweeps=max_sweeps,
                           min_sweeps=min_sweeps)

        init = kwargs.get('init', None)
        if init is None:
            N = self.lat.N_sites
            init = [0]*(N//3)+[1]*(N//3)+[2]*(N-N//3-N//3)
            np.random.shuffle(init)
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)
            psiinit.norm = 1
            psiinit.canonical_form()
        elif isinstance(init, str):
            with open (init, 'rb') as f:
                psiinit = pickle.load(f)
            dmrg_params['mixer'] = False
        elif isinstance(init, list):
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)            
        elif isinstance(init, MPS):
            psiinit = init
        else:
            print("wrong init")
            
        eng = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = eng.run()
        print("Eng = ", E)
        self.psidmrg = psidmrg
        return psidmrg, E
            
def GutzwillerProjection2Spin1(psi):
    '''
    The 8-dim(threeparton) to 3-dim(Spin1) projection. 

    Inputs:
        1. psi: tenpy.MPS, the unprojected MPS obtained from MPO-MPS method. 

    Outputs:
        1. gp_psi_spin: tenpy.MPS, the 3-dim GUTZWILLER PROJECTED MPS under Spin1 site
    '''
    threeparton_site = psi.sites[0]
    cons_N, cons_S = threeparton_site.conserve
    threeparton_leg = threeparton_site.leg

    spin1_site = Spin1(cons_N=cons_N, cons_S=None)
    spin1_leg = spin1_site.leg

    #the projection shouldn't change the qns
    if cons_N == 'Z2' and cons_S == 'xyz':
        qtotal = [0, 0]
    else:
        qtotal = None

    projector = npc.zeros([spin1_leg, threeparton_leg.conj()], qtotal=qtotal, labels=['p','p*'], dtype=psi.dtype)
    projector[0,1] = 1 #x parton -> 1
    projector[1,2] = 1 #y parton -> -1
    projector[2,3] = 1 #z parton -> 0
    L = psi.L
    gp_psi_spin1 = MPS.from_product_state([spin1_site]*L, [0]*L)
    for i in range(L):
        t1 = npc.tensordot(psi._B[i], projector, axes=(['p'],['p*']))
        gp_psi_spin1.set_B(i, t1, form=None)
    gp_psi_spin1.canonical_form()
    return gp_psi_spin1

def Wannier_Z2(g1, g2, N=1):
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=6)
    parser.add_argument("-theta", type=float, default=np.arctan(1/3))
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=1.)
    parser.add_argument("-lamb", type=float, default=0.)
    parser.add_argument("-D", type=int, default=10)
    parser.add_argument("-pbc", type=int, default=0)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=1/3)
    args = parser.parse_args()
    
    np.random.seed(0)
    
    theta = args.theta
    chi = args.chi
    delta = args.delta
    lamb = args.lamb
    lx = args.lx
    pbc = args.pbc
    D = args.D
    J = args.J
    K = args.K
    
    if pbc == 1:
        bc = 'periodic'
    elif pbc == 0:
        bc = 'open'
    else:
        raise "pbc must be 1(periodic) or 0(open)"
    
    if pbc == 1:
        Econst = (4/3) * K * lx
    elif pbc == 0:
        Econst = (4/3) * K * (lx-1)
    
    model_params = dict(Lx=lx, theta=theta, bc=bc, J=J, K=K)
    
    print("----------Build single Kitaev chain Hamiltonian----------")
    singlechain = KitaevSingleChain(chi, delta, lamb, lx, bc)
    singlechain.calc_hamiltonian()
    vmat = singlechain.V
    umat = singlechain.U

    print("----------Build MLWO----------")
    wv, wu = Wannier_Z2(vmat.T, umat.T)

    print("----------Z2U1 MPO-MPS method: dm----------")
    params_mpompsz2u1 = dict(cons_N=None, cons_S=None, trunc_params=dict(chi_max=D))
    mpos = MPOMPS(vmat, umat, **params_mpompsz2u1)
    mpos.run()

    print("----------Gutzwiller projection to Spin1 site----------")
    psi1 = mpos.psi
    gppsi = GutzwillerProjection2Spin1(psi1)

    print("----------Z2U1 MPO-MPS method: MLWO----------")
    params_mpompsz2u1 = dict(cons_N=None, cons_S=None, trunc_params=dict(chi_max=D))
    mpos = MPOMPS(wv, wu, **params_mpompsz2u1)
    mpos.run()

    print("----------Gutzwiller projection to Spin1 site----------")
    psimlwo = mpos.psi
    gppsimlwo = GutzwillerProjection2Spin1(psimlwo)

    print("----------SU(3) Spin1 model DMRG---------")
    su3dmrgmodel = BBQJK(model_params)
    #sites2 = su3dmrgmodel.sites
    sites2 = [Spin1(cons_N=None, cons_S=None)] * lx
    psi2, E2 = su3dmrgmodel.run_dmrg()
    print("SU3 site DMRG results")
    print("psi2 after DMRG is", psi2)
    print("E2 is", E2 + Econst)

    print("----------Build-in AKLT----------")
    akltparams = dict(L = lx, J=1, bc = bc, conserve = 'parity')
    aklt = AKLTChain(akltparams)
    psiaklt = aklt.psi_AKLT()
    print("expected open boundary AKLT energy from theory (J=1, K=1/3)", -2/3 *(lx-1)*1)
    print(" ")

    print("----------sandwiches----------")
    bbqmpo = su3dmrgmodel.calc_H_MPO()
    print(" ")
    print("the sandwich of projected psi and SO(3) MPO is", bbqmpo.expectation_value(gppsi)+Econst)
    print(" ")
    print("the sandwich of projected psimlwo and SO(3) MPO is", bbqmpo.expectation_value(gppsimlwo)+Econst)
    print(" ")
    print("the overlap of gppsi and gppsimlwo", gppsi.overlap(gppsimlwo))
    print(" ")
    print("the overlap of psidmrg and gppsi", psi2.overlap(gppsi))
    print(" ")
    print("the overlap of psidmrg and gppsimlwo", psi2.overlap(gppsimlwo))