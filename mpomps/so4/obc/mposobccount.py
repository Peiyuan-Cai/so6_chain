"""
The MPO-MPS method for SO(4) BBQ model with good quantum number U(1)xU(1) given by Cartan subalgebra. 

For OBC counting use

Puiyuen 2024.10.25
    2024.10.25: OBC counting first commit
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
        self.model = "double Kitaev chain parameters_L{}_chi{}_delta{}_lambda{}_pbc{}".format(L, round(chi, 3), round(delta, 3), round(lamb, 3), pbc)
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
            print("Periodic terms added in double Kitaev chain. ")
        elif self.pbc == -1:
            self.tmat[L-1, 0] = -t
            self.dmat[L-1, 0] = -d
            print("Anti-periodic terms added in double Kitaev chain. ")
        else:
            print("No periodic term added. ")
        
        self.tmat += self.tmat.T.conj() 
        self.dmat -= self.dmat.T
        zeromat = np.zeros((L,L))

        self.bigtmat = np.block([[self.tmat, zeromat],[zeromat, self.tmat]])
        self.bigdmat = np.block([[zeromat, -self.dmat],[self.dmat, zeromat]])
        
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
        
        self.bigM = np.block([[self.V14, zeromat, zeromat, self.U14.conj()],
                       [zeromat, self.V23, self.U23.conj(), zeromat],
                       [zeromat, self.U23, self.V23.conj(), zeromat],
                       [self.U14, zeromat, zeromat, self.V14.conj()]])
        
        self.V, self.U = m2vu(self.bigM)
        
        self.wV14, self.wU14 = self.Wannier_Z2(self.V14.T, self.U14.T)
        self.wV23, self.wU23 = self.Wannier_Z2(self.V23.T, self.U23.T)
        
        self.wV = np.block([[self.wV14, zeromat],[zeromat, self.wV23]])
        self.wU = np.block([[zeromat, self.wU23],[self.wU14, zeromat]])
        
        self.wM = np.block([[self.wV, self.wU.conj()],
                            [self.wU, self.wV.conj()]])
        
        print('eigen energies of the BdG Hamiltonian', np.diag(self.wM.conj().T @ self.ham @ self.wM))
    
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
    
class MPOMPS():
    """
    input: wannier orbital v and u
    """
    def __init__(self, v, u, **kwargs):
        self.cons_N = kwargs.get("cons_N", None)
        self.cons_S = kwargs.get("cons_S", None)
        self.trunc_params = kwargs.get("trunc_params", dict(chi_max=64) )
        self.pbc = kwargs.get("pbc", -1)
        self.init = kwargs.get("init", None)
        
        #the V and U are the whole matrix, not the V11, V22, U12, U21
        assert v.ndim == 2
        self._V = v
        self._U = u
        assert self._U.shape == self._V.shape
        self.projection_type = kwargs.get("projection_type", "Gutz")

        self.L = self.Llat = u.shape[0]
        if self.cons_S == 'U1':
            self.L = self.Llat = u.shape[0] // 2 #the length of real sites: the half of the length of the 2-coupled chain
        
        self.v11, self.v22 = v_to_v1v2(self._V)
        self.u12, self.u21 = u_to_u1u2(self._U)
        
        self.site = partonsite(self.cons_N, self.cons_S)
        self.init_mps(self.init)
        
    def init_mps(self, init=None):
        L = self.L
        if init is None:
            if self.pbc == -1 or self.pbc == 0:
                if self.cons_S == None:
                    init = [0] * L #all empty
                elif self.cons_S == 'U1':
                    init = [0] * L #particle-hole symmetry used
            if self.pbc == 1:
                if self.cons_S == None:
                    init = [15] + [0]*(L-1) #a_{1,u}^\dagger a_{1,v}^\dagger ... a_{1,z}^\dagger \ket{0}_a
                elif self.cons_S == 'U1':
                    init = [10] * L
        site = self.site
        self.init_psi = MPS.from_product_state([site]*L, init)
        self.psi = self.init_psi.copy()
        self.n_omode = 0
        print("the initial state is", init)
        return self.psi
    
    def get_mpo_U1(self, v11, v22, u12, u21, qn):
        chinfo = self.site.leg.chinfo
        pleg = self.site.leg

        if qn == 3:
            fqn = [-1, 0] #cwdag + cz
        elif qn == 1:
            fqn = [0, -1] #cxdag + cy
        elif qn == -1:
            fqn = [0, 1] #cydag + cx
        elif qn == -3:
            fqn = [1, 0] #czdag + cw

        firstleg = npc.LegCharge.from_qflat(chinfo, [[0,0]], 1)
        lastleg = npc.LegCharge.from_qflat(chinfo, [fqn], -1)
        bulkleg = npc.LegCharge.from_qflat(chinfo, [fqn, [0,0]], 1)
        #legs arrange in order 'wL', 'wR', 'p', 'p*'
        legs_first = [firstleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_bulk = [bulkleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_last = [bulkleg, lastleg, pleg, pleg.conj()]
        
        mpo = []
        L = self.L
        
        op_dict = {'w': ('cwdag', 'cw'), 'x': ('cxdag', 'cx'), 'y': ('cydag', 'cy'), 'z': ('czdag', 'cz')}
        
        qn_dict = {3: ('w', 'z'), 1: ('x', 'y'), -1: ('y', 'x'), -3: ('z', 'w')}
        
        t0 = npc.zeros(legs_first, labels=['wL', 'wR', 'p', 'p*'], dtype=float)
        i = 0
        if qn in qn_dict:
            cr_op, an_op = qn_dict[qn]
            cr = op_dict[cr_op][0]
            an = op_dict[an_op][1]
            if qn == 3 or qn == -3:
                t0[0, 0, :, :] = v11[i]*self.site.get_op(cr) + u21[i]*self.site.get_op(an)
            elif qn == 1 or qn == -1:
                t0[0, 0, :, :] = v22[i]*self.site.get_op(cr) + u12[i]*self.site.get_op(an)
        t0[0, 1, :, :] = self.site.get_op('JW')
        mpo.append(t0)
        
        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=float)
            ti[0,0,:,:] = self.site.get_op('id16')
            if qn in qn_dict:
                cr_op, an_op = qn_dict[qn]
                cr = op_dict[cr_op][0]
                an = op_dict[an_op][1]
                if qn == 3 or qn == -3:
                    ti[1, 0, :, :] = v11[i]*self.site.get_op(cr) + u21[i]*self.site.get_op(an)
                elif qn == 1 or qn == -1:
                    ti[1, 0, :, :] = v22[i]*self.site.get_op(cr) + u12[i]*self.site.get_op(an)
            ti[1, 1, :, :] = self.site.get_op('JW')
            mpo.append(ti)
            
        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=float)
        tL[0,0,:,:] = self.site.get_op('id16')
        if qn in qn_dict:
            cr_op, an_op = qn_dict[qn]
            cr = op_dict[cr_op][0]
            an = op_dict[an_op][1]
            if qn == 3 or qn == -3:
                tL[1, 0, :, :] = v11[i]*self.site.get_op(cr) + u21[i]*self.site.get_op(an)
            elif qn == 1 or qn == -1:
                tL[1, 0, :, :] = v22[i]*self.site.get_op(cr) + u12[i]*self.site.get_op(an)
        mpo.append(tL)
        
        return mpo
    
    def get_mpo_U1_an(self, v11, v22, u12, u21, qn):
        #the annihilator version of the get_mpo_U1
        chinfo = self.site.leg.chinfo
        pleg = self.site.leg

        if qn == 3:
            fqn = [1, 0] #cw + czdag
        elif qn == 1:
            fqn = [0, 1] #cx + cydag
        elif qn == -1:
            fqn = [0, -1] #cy + cxdag
        elif qn == -3:
            fqn = [-1, 0] #cz + cwdag

        firstleg = npc.LegCharge.from_qflat(chinfo, [[0,0]], 1)
        lastleg = npc.LegCharge.from_qflat(chinfo, [fqn], -1)
        bulkleg = npc.LegCharge.from_qflat(chinfo, [fqn, [0,0]], 1)
        #legs arrange in order 'wL', 'wR', 'p', 'p*'
        legs_first = [firstleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_bulk = [bulkleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_last = [bulkleg, lastleg, pleg, pleg.conj()]
        
        mpo = []
        L = self.L
        
        op_dict = {'w': ('cwdag', 'cw'), 'x': ('cxdag', 'cx'), 'y': ('cydag', 'cy'), 'z': ('czdag', 'cz')}
        
        qn_dict = {3: ('w', 'z'), 1: ('x', 'y'), -1: ('y', 'x'), -3: ('z', 'w')}
        
        t0 = npc.zeros(legs_first, labels=['wL', 'wR', 'p', 'p*'], dtype=float)
        i = 0
        if qn in qn_dict:
            cr_op, an_op = qn_dict[qn]
            cr = op_dict[cr_op][1]
            an = op_dict[an_op][0]
            if qn == 3 or qn == -3:
                t0[0, 0, :, :] = v11[i].conj()*self.site.get_op(cr) + u21[i].conj()*self.site.get_op(an)
            elif qn == 1 or qn == -1:
                t0[0, 0, :, :] = v22[i].conj()*self.site.get_op(cr) + u12[i].conj()*self.site.get_op(an)
        t0[0, 1, :, :] = self.site.get_op('JW')
        mpo.append(t0)
        
        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=float)
            ti[0,0,:,:] = self.site.get_op('id16')
            if qn in qn_dict:
                cr_op, an_op = qn_dict[qn]
                cr = op_dict[cr_op][1]
                an = op_dict[an_op][0]
                if qn == 3 or qn == -3:
                    ti[1, 0, :, :] = v11[i].conj()*self.site.get_op(cr) + u21[i].conj()*self.site.get_op(an)
                elif qn == 1 or qn == -1:
                    ti[1, 0, :, :] = v22[i].conj()*self.site.get_op(cr) + u12[i].conj()*self.site.get_op(an)
            ti[1, 1, :, :] = self.site.get_op('JW')
            mpo.append(ti)
            
        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=float)
        tL[0,0,:,:] = self.site.get_op('id16')
        if qn in qn_dict:
            cr_op, an_op = qn_dict[qn]
            cr = op_dict[cr_op][1]
            an = op_dict[an_op][0]
            if qn == 3 or qn == -3:
                tL[1, 0, :, :] = v11[i].conj()*self.site.get_op(cr) + u21[i].conj()*self.site.get_op(an)
            elif qn == 1 or qn == -1:
                tL[1, 0, :, :] = v22[i].conj()*self.site.get_op(cr) + u12[i].conj()*self.site.get_op(an)
        mpo.append(tL)
        
        return mpo
    
    def mpomps_step_1time(self, m, flavor):
        v11, v22 = self.v11[:,m], self.v22[:,m]
        u12, u21 = self.u12[:,m], self.u21[:,m]
        mps = self.psi
        if self.cons_N==None and self.cons_S=='U1':
            mpo = self.get_mpo_U1(v11, v22, u12, u21, flavor)
        else:
            raise "Symmetry set of N and S is not allowed. "
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
        nmode = self._U.shape[0] // 2 #the number of the modes = the real site length
        print("MPO-MPS application start")
        
        if self.cons_N == None and self.cons_S == 'U1':
            qnlist = [3, 1, -1, -3] #use particle-hole symmetry
            for m in range(nmode):
                for qn in qnlist:
                    err, self.psi = self.mpomps_step_1time(m, qn)
                    self.fidelity *= 1-err.eps
                    self.chi_max = np.max(self.psi.chi)
                    print( "applied the {}-th {} mode, the fidelity is {}, the largest bond dimension is {}. ".format( self.n_omode, qn, self.fidelity, self.chi_max) )
                self.n_omode += 1
                
def GutzwillerProjectionParton2Spin(partonpsi):
    partonsite = partonpsi.sites[0]
    cons_N, cons_S = partonsite.conserve
    partonleg = partonsite.leg
    
    so4gen, cmn, dmn = get_so4_opr_list_new()
    spinsite = SO4Site(so4gen, cons_N, cons_S)
    spinleg = spinsite.leg
    
    middleleg = npc.LegCharge.from_trivial(4)
    
    if cons_N == None and cons_S == 'U1':
        qtotal = [0,0]
        middleleg = spinleg
    else:
        qtotal = None
        
    projector = npc.zeros([middleleg, partonleg.conj()], qtotal=qtotal, labels=['p','p*'], dtype=partonpsi.dtype)
    projector[0,1] = 1 #0th spin index <=> w parton
    projector[1,2] = 1
    projector[2,3] = 1
    projector[3,4] = 1
    
    L = partonpsi.L
    spinpsi = MPS.from_product_state([spinsite]*L, [0]*L)
    for i in range(L):
        t1 = npc.tensordot(partonpsi._B[i], projector, axes=(['p'],['p*']))
        spinpsi.set_B(i, t1, form=None)
    spinpsi.canonical_form()
    
    return spinpsi

def apply_mpo(mpo, mps, i0=0):
    """
    Inputs:
        1. mpo, list of npc.Array
        2. mps, tenpy.MPS object
        3. i0, int, the mpo starts from i0-th site
    Output:
        1. mps, tenpy.MPS object
    
    It's a ! type function, changing the input mps at the same time
    """
    L = len(mpo)
    for i in range( i0, i0+L ):
        B = npc.tensordot(mps.get_B(i, 'B'), mpo[i-i0], axes=('p', 'p*'))
        B = B.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
        B.ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
        B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
        B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
        mps._B[i] = B#.itranspose(('vL', 'p', 'vR'))
    return mps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=8)
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=1.)
    parser.add_argument("-lamb", type=float, default=0.)
    parser.add_argument("-Dmpos", type=int, default=64)
    parser.add_argument("-Ddmrg", type=int, default=216)
    parser.add_argument("-sweeps", type=int, default=6)
    parser.add_argument("-pbc", type=int, default=0)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=0.25)
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

    print(" ")
    print("OBC case MPOMPS")
    print("----------Build double spin chain Hamiltonian----------")
    doublechain = SpinDoubleChain(chi, delta, lamb, lx, pbc)
    doublechain.calc_hamiltonian()
    vmat = doublechain.wV
    umat = doublechain.wU

    print("----------MPO-MPS method: MLWO----------")
    params_mpomps = dict(cons_N=conn, cons_S=cons, trunc_params=dict(chi_max=Dmpos), pbc=pbc, init=None)
    mpos = MPOMPS(vmat, umat, **params_mpomps)
    mpos.run()
    
    print("----------Gutzwiller projection to SO(4) site----------")
    psimlwo_obc = mpos.psi
    gppsimlwo_obc = GutzwillerProjectionParton2Spin(psimlwo_obc)
    print("Gutzwiller projected MLWO MPO-MPS result is", gppsimlwo_obc)
    
    print(" ")
    print("----------SO(4) Spin1 model DMRG---------")
    params_dmrg = dict(cons_N=conn, cons_S=cons, Lx = lx, pbc=pbc, J=J, K=K, D=Ddmrg, sweeps=sweeps, verbose=verbose)
    so4dmrgmodel = BBQJKSO4(params_dmrg)
    psidmrg, Edmrg = so4dmrgmodel.run_dmrg()
    print("SO(4) DMRG results")
    print("One of the many degenerated psi after DMRG is", psidmrg)
    
    print(" ")
    print("----------sandwiches----------")
    bbqmpo = so4dmrgmodel.calc_H_MPO()
    #print("the overlap of psidmrg and psidmrg2", psidmrg.overlap(psidmrg2))
    print(" ")
    print("the sandwich of projected psimlwo_obc and SO(4) MPO is", bbqmpo.expectation_value(gppsimlwo_obc))
    print("the overlap of psidmrg and gppsimlwo_obc", psidmrg.overlap(gppsimlwo_obc))
    
    print(" ")
    print(" ")
    print("----------OBC counting----------")
    def mpomps_obc(init):
        try:
            params_mpomps = dict(cons_N=conn, cons_S=cons, trunc_params=dict(chi_max=Dmpos), pbc=pbc, init=init)
            mpos = MPOMPS(vmat, umat, **params_mpomps)
            mpos.run()

            psi = mpos.psi
            gppsi = GutzwillerProjectionParton2Spin(psi)
            
            print("----------OBC sandwiches----------")
            bbqmpo = so4dmrgmodel.calc_H_MPO()
            print(" ")
            print("init is", init)
            
            expectation_value = bbqmpo.expectation_value(gppsi)
            print("the sandwich of projected psi and SO(4) MPO is", expectation_value)
            
            overlap_value = gppsimlwo_obc.overlap(gppsi)
            print("the overlap of gppsimlwo_obc and gppsimlwo_obc", overlap_value)

            if abs(expectation_value-Edmrg) < 1e-6:
                print("Degenerated init:", init)
                if abs(overlap_value) < 1e-6:
                    print("Orthogonal init:", init)
            return psi, gppsi
            
        except Exception as e:
            print(f"init={init} error: {e}")
            return None
        
    initgs1 = [0,0,0,0]
    gs1, gpgs1 = mpomps_obc(initgs1)
    initgs2 = [0,10,0,0]
    gs2, gpgs2 = mpomps_obc(initgs2)
    initgs3 = [0,10,2,2]
    gs3, gpgs3 = mpomps_obc(initgs3)
    initgs4 = [0,10,11,11]
    gs4, gpgs4 = mpomps_obc(initgs4)
    initgs5 = [0,10,5,5]
    gs5, gpgs5 = mpomps_obc(initgs5)
    initgs6 = [2,6,10,4]
    gs6, gpgs6 = mpomps_obc(initgs6)
    initgs7 = [2,6,10,14]
    gs7, gpgs7 = mpomps_obc(initgs7)
    
    print(" ")
    print("OBC MPS genereation")
    gscopy = deepcopy(psimlwo_obc)
    
    vmat11, vmat22 = mpos.v11[:,0], mpos.v22[:,0]
    umat12, umat21 = mpos.u12[:,0], mpos.u21[:,0]
    boundary_mpo = mpos.get_mpo_U1_an(vmat11, vmat22, umat12, umat21, 3)
    gscopy = apply_mpo(boundary_mpo, gscopy, i0=0)
    gscopy.canonical_form()
    
    vmat11, vmat22 = mpos.v11[:,0], mpos.v22[:,0]
    umat12, umat21 = mpos.u12[:,0], mpos.u21[:,0]
    boundary_mpo = mpos.get_mpo_U1_an(vmat11, vmat22, umat12, umat21, 1)
    gscopy = apply_mpo(boundary_mpo, gscopy, i0=0)
    gscopy.canonical_form()

    vmat11, vmat22 = mpos.v11[:,0], mpos.v22[:,0]
    umat12, umat21 = mpos.u12[:,0], mpos.u21[:,0]
    boundary_mpo = mpos.get_mpo_U1_an(vmat11, vmat22, umat12, umat21, -1)
    gscopy = apply_mpo(boundary_mpo, gscopy, i0=0)
    gscopy.canonical_form()
    
    vmat11, vmat22 = mpos.v11[:,0], mpos.v22[:,0]
    umat12, umat21 = mpos.u12[:,0], mpos.u21[:,0]
    boundary_mpo = mpos.get_mpo_U1_an(vmat11, vmat22, umat12, umat21, -3)
    gscopy = apply_mpo(boundary_mpo, gscopy, i0=0)
    gscopy.canonical_form()

    gscopy1 = deepcopy(gscopy)
    gscopy2 = deepcopy(gscopy)
    vmat11, vmat22 = mpos.v11[:,0], mpos.v22[:,0]
    umat12, umat21 = mpos.u12[:,0], mpos.u21[:,0]
    boundary_mpo = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, -3)
    gscopy1 = apply_mpo(boundary_mpo, gscopy, i0=0)
    gscopy1.canonical_form()
    vmat11, vmat22 = mpos.v11[:,3], mpos.v22[:,3]
    umat12, umat21 = mpos.u12[:,3], mpos.u21[:,3]
    boundary_mpo = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, -3)
    gscopy2 = apply_mpo(boundary_mpo, gscopy, i0=0)
    gscopy2.canonical_form()
    gscopy = gscopy1.add(gscopy2,1j/np.sqrt(2),1/np.sqrt(2))
    gscopy.canonical_form()
    
    gpgscopy = GutzwillerProjectionParton2Spin(gscopy)
    
    print('bbqmpo.expectation_value after-gs application', bbqmpo.expectation_value(gpgscopy))
    print('gpgs1.overlap(gpgscopy)', gpgs1.overlap(gpgscopy))
    print('gpgs2.overlap(gpgscopy)', gpgs2.overlap(gpgscopy))
    print('gpgs3.overlap(gpgscopy)', gpgs3.overlap(gpgscopy))
    print('gpgs4.overlap(gpgscopy)', gpgs4.overlap(gpgscopy))
    print('gpgs5.overlap(gpgscopy)', gpgs5.overlap(gpgscopy))
    print('gpgs6.overlap(gpgscopy)', gpgs6.overlap(gpgscopy))
    print('gpgs7.overlap(gpgscopy)', gpgs7.overlap(gpgscopy))