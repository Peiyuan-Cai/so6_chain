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
        
        self.eig_eng, self.eig_vec = bdgeig_zeromode(self.ham)
        print("the eig energies", np.real(self.eig_eng))
        
        self.H14 = np.block([[self.tmat, -self.dmat], [-self.dmat.conj().T, -self.tmat.conj()]])
        self.H23 = np.block([[self.tmat, self.dmat], [self.dmat.conj().T, -self.tmat.conj()]])
        eig_eng14, eig_vec14 = bdgeig(self.H14)
        print('eigen energies of H_{+3}', eig_eng14)
        self.smallM14 = eig_vec14
        eig_eng23, eig_vec23 = bdgeig(self.H23)
        print('eigen energies of H_{-3}', eig_eng23)
        self.smallM23 = eig_vec23
        
        self.V11, self.U21 = m2vu(self.smallM14)
        self.V22, self.U12 = m2vu(self.smallM23)
        
        self.bigM = np.block([[self.V11, zeromat, zeromat, self.U21.conj()],
                       [zeromat, self.V22, self.U12.conj(), zeromat],
                       [zeromat, self.U12, self.V22.conj(), zeromat],
                       [self.U21, zeromat, zeromat, self.V11.conj()]])
        
        self.V, self.U = m2vu(self.bigM)
        
        self.wV11, self.wU21 = self.Wannier_Z2(self.V11.T, self.U21.T)
        self.wV22, self.wU12 = self.Wannier_Z2(self.V22.T, self.U12.T)
        
        self.wV = np.block([[self.wV11, zeromat],[zeromat, self.wV22]])
        self.wU = np.block([[zeromat, self.wU12],[self.wU21, zeromat]])
        
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
        
        #here w,x,y,z are not the decoupled parton flavors, they are uu ud du dd flavors
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
                    if m != 1: 
                        err, self.psi = self.mpomps_step_1time(m, qn)
                        self.fidelity *= 1-err.eps
                        self.chi_max = np.max(self.psi.chi)
                        print( "applied the {}-th {} mode, the fidelity is {}, the largest bond dimension is {}. ".format( self.n_omode, qn, self.fidelity, self.chi_max) )
                    
                self.n_omode += 1

    def init_mps_from_init(self, init):
        L = self.L
        site = self.site
        self.init_psi = MPS.from_product_state([site]*L, init)
        self.psi = self.init_psi.copy()
        self.n_omode = 0
        print("the initial state is", init)
        return self.psi
        
    def run_zeromode_skipper(self, zm_index, skip_spin, init=None):
        '''
        The ground state generator with zero mode skipper, the skip_spin is the spin index to skip
        '''
        self.fidelity = 1
        if self.n_omode > 0:
            print("initialize the mpo-mps calculation mps")
            self.init_mps(init=init)
            self.n_omode = 0
        nmode = self._U.shape[0] // 2
        print("MPO-MPS application with zero mode skipping start")

        if self.cons_N == None and self.cons_S == 'U1':
            qnlist = [3, 1, -1, -3] #not using the particle-hole symmetry
            for m in range(nmode):
                for qn in qnlist:
                    if m != zm_index: 
                        err, self.psi = self.mpomps_step_1time(m, qn)
                        self.fidelity *= 1-err.eps
                        self.chi_max = np.max(self.psi.chi)
                        print( "applied the {}-th {} mode, the fidelity is {}, the largest bond dimension is {}. ".format( self.n_omode, qn, self.fidelity, self.chi_max) )
                    
                    elif qn not in skip_spin: 
                        err, self.psi = self.mpomps_step_1time(m, qn)
                        self.fidelity *= 1-err.eps
                        self.chi_max = np.max(self.psi.chi)
                        print( "applied the {}-th {} mode, the fidelity is {}, the largest bond dimension is {}. ".format( self.n_omode, qn, self.fidelity, self.chi_max) )
                    
                self.n_omode += 1
                
    def run_ph_skipper(self, zm_index, skip_spin, init, qnlist):
        self.fidelity = 1
        if self.n_omode > 0:
            print("initialize the mpo-mps calculation mps")
            self.init_mps(init=init)
            self.n_omode = 0
        nmode = self._U.shape[0] // 2
        print("MPO-MPS application with zero mode skipping with PH symmetry used start")
        
        if self.cons_N == None and self.cons_S == 'U1':
            for m in range(nmode):
                for qn in qnlist:
                    if m != zm_index: 
                        err, self.psi = self.mpomps_step_1time(m, qn)
                        self.fidelity *= 1-err.eps
                        self.chi_max = np.max(self.psi.chi)
                        print( "applied the {}-th {} mode, the fidelity is {}, the largest bond dimension is {}. ".format( self.n_omode, qn, self.fidelity, self.chi_max) )
                    
                    elif qn not in skip_spin: 
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
    parser.add_argument("-sweeps", type=int, default=10)
    parser.add_argument("-pbc", type=int, default=0)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=0.25)
    parser.add_argument("-verbose", type=int, default=1)
    args = parser.parse_args()
    
    '''
    import logging
    logging.basicConfig(level=args.verbose)
    for _ in ['parso.python.diff', 'parso.cache', 'parso.python.diff', 
              'parso.cache', 'matplotlib.font_manager', 'tenpy.tools.cache', 
              'tenpy.algorithms.mps_common', 'tenpy.linalg.lanczos', 'tenpy.tools.params']:
        logging.getLogger(_).disabled = True
    '''
        
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

    print(" ")
    print("OBC MPS genereation")
    gscopy = deepcopy(psimlwo_obc)
    gscopy.canonical_form()

    def get_zeromode(name, flavor): 
        #mpos is defined as a global variable
        if name == 'c':
            if flavor == 'w':
                print('getting majorana fermion c_w')
                vmat11, vmat22 = mpos.v11[:,0], mpos.v22[:,0]
                umat12, umat21 = mpos.u12[:,0], mpos.u21[:,0]
                vmat11[:] = 0; vmat11[0] = 1; vmat22[:] = 0; vmat22[0] = 1
                umat21[:] = 0; umat21[0] = 1; umat12[:] = 0; umat12[0] = 1
                a1 = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, 3)
                vmat11[:] = 0; vmat11[-1] = 1; vmat22[:] = 0; vmat22[-1] = 1
                umat21[:] = 0; umat21[-1] = -1; umat12[:] = 0; umat12[-1] = -1
                aN = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, 3)
                c1 = 1j/np.sqrt(2); cN = 1/np.sqrt(2)
                return a1, aN, c1, cN
            elif flavor == 'x':
                print('getting majorana fermion c_x')
                vmat11, vmat22 = mpos.v11[:,0], mpos.v22[:,0]
                umat12, umat21 = mpos.u12[:,0], mpos.u21[:,0]
                vmat11[:] = 0; vmat11[0] = 1; vmat22[:] = 0; vmat22[0] = 1
                umat21[:] = 0; umat21[0] = -1; umat12[:] = 0; umat12[0] = -1
                a1 = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, 1)
                vmat11[:] = 0; vmat11[-1] = 1; vmat22[:] = 0; vmat22[-1] = 1
                umat21[:] = 0; umat21[-1] = 1; umat12[:] = 0; umat12[-1] = 1
                aN = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, 1)
                c1 = 1/np.sqrt(2); cN = -1j/np.sqrt(2)
                return a1, aN, c1, cN
        elif name == 'd':
            if flavor == 'w':
                print('getting majorana fermion d_w')
                vmat11, vmat22 = mpos.v11[:,0], mpos.v22[:,0]
                umat12, umat21 = mpos.u12[:,0], mpos.u21[:,0]
                vmat11[:] = 0; vmat11[0] = 1; vmat22[:] = 0; vmat22[0] = 1
                umat21[:] = 0; umat21[0] = 1; umat12[:] = 0; umat12[0] = 1
                a1 = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, -3)
                vmat11[:] = 0; vmat11[-1] = 1; vmat22[:] = 0; vmat22[-1] = 1
                umat21[:] = 0; umat21[-1] = -1; umat12[:] = 0; umat12[-1] = -1
                aN = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, -3)
                c1 = -1j/np.sqrt(2); cN = -1/np.sqrt(2)
                return a1, aN, c1, cN
            elif flavor == 'x':
                print('getting majorana fermion d_x')
                vmat11, vmat22 = mpos.v11[:,0], mpos.v22[:,0]
                umat12, umat21 = mpos.u12[:,0], mpos.u21[:,0]
                vmat11[:] = 0; vmat11[0] = 1; vmat22[:] = 0; vmat22[0] = 1
                umat21[:] = 0; umat21[0] = -1; umat12[:] = 0; umat12[0] = -1
                a1 = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, -1)
                vmat11[:] = 0; vmat11[-1] = 1; vmat22[:] = 0; vmat22[-1] = 1
                umat21[:] = 0; umat21[-1] = 1; umat12[:] = 0; umat12[-1] = 1
                aN = mpos.get_mpo_U1(vmat11, vmat22, umat12, umat21, -1)
                c1 = -1/np.sqrt(2); cN = 1j/np.sqrt(2)
                return a1, aN, c1, cN

    def apply_zero_mode_on_gs(op1, opN, co1, coN, gs):
        #not gonna change the skipped_gs
        gscp1 = deepcopy(gs)
        gscp1.canonical_form()
        gscp2 = deepcopy(gs)
        gscp2.canonical_form()

        gscp1 = apply_mpo(op1, gscp1, i0=0)
        gscp1.canonical_form()
        print("applied a1 on gs")
        gscp2 = apply_mpo(opN, gscp2, i0=0)
        gscp2.canonical_form()
        print("applied aN on gs")

        gscp = gscp1.add(gscp2,co1,coN)
        gscp.canonical_form()
        return gscp




    #-------------------------------Majorana application--------------------------------
    #the no-skipped ground state
    mpos.run_zeromode_skipper(1, [3, 1, -1, -3]) #generate a ground state with spin -3
    no_skipped_gs = mpos.psi

    #the majorana application

    #compare with the all-skipped ground state, gscopy is the gsmlwo_obc, the zero-mode-all-skipped ground state
    gscopy.canonical_form()
    no_skipped_gs.canonical_form()
    print("the overlap of all-skipped GS and full GS is", gscopy.overlap(no_skipped_gs))

    try:
        gpfullgs = GutzwillerProjectionParton2Spin(no_skipped_gs)
        print('bbqmpo.expectation_value after application', bbqmpo.expectation_value(gpfullgs))
        print('gppsimlwo.overlap(gpfullgs)', gppsimlwo_obc.overlap(gpfullgs))
        print('survived MFGS')
    except Exception as e:
        print('error:', e)

    def obc_gs_generator(majorana_opr_list, init, qnlist):
        #majoprlist = [['d','w'],['c','w']] written in the reversed order
        print(' ')
        print('OBC counting of applying MZM', majorana_opr_list)
        
        zm_skip_list = []
        for majopr in majorana_opr_list:
            if majopr == ['c','w']:
                zm_skip_list.append(3)
            elif majopr == ['d','w']:
                zm_skip_list.append(-3)
            elif majopr == ['c','x']:
                zm_skip_list.append(1)
            elif majopr == ['d','x']:
                zm_skip_list.append(-1)
        mpos.run_ph_skipper(1, zm_skip_list, init, qnlist)
        appgs = mpos.psi

        try:
            #the maj application
            for majopr in majorana_opr_list:
                operator1, operatorN, coefficient1, coefficientN = get_zeromode(majopr[0], majopr[1])
                appgs = apply_zero_mode_on_gs(operator1, operatorN, coefficient1, coefficientN, appgs)
                appgs.compress_svd(trunc_par=dict(chi_max=64))
            
            #compare with the all-skipped ground state
            gscopy.canonical_form()
            appgs.canonical_form()
            print("the overlap (BEFORE GP) of all-skipped GS and applied GS is", gscopy.overlap(appgs))

            #Gutzwiller projection
            gpappgs = GutzwillerProjectionParton2Spin(appgs)
            print('bbqmpo.expectation_value after application', bbqmpo.expectation_value(gpappgs))
            print('(AFTER GP) gppsimlwo.overlap(gpappgs)', gppsimlwo_obc.overlap(gpappgs))
            print('survived MFGS')
            if np.abs(bbqmpo.expectation_value(gppsimlwo_obc)-bbqmpo.expectation_value(gpappgs)) < 1e-8:
                print('DEGENERATED MFGS with Majorana ZM', majorana_opr_list)
        
            return appgs, gpappgs
        except Exception as e:
            print('error:', e)
            return appgs, None

    majoprlstlst = [[['c','w']], 
                    [['c','x']], 
                    [['c','w'], ['c','x']], 
                    [],
                    [['c','w']], 
                    [['d','x']], 
                    [['c','w'], ['d','x']], 
                    [],
                    [['d','w']], 
                    [['c','x']], 
                    [['d','w'], ['c','x']], 
                    [],
                    [['d','w']], 
                    [['d','x']], 
                    [['d','w'], ['d','x']], 
                    []]
    
    initlst = [[10]*lx,
               [10]*lx,
               [10]*lx,
               [10]*lx,
               [9]*lx,
               [9]*lx,
               [9]*lx,
               [9]*lx,
               [6]*lx,
               [6]*lx,
               [6]*lx,
               [6]*lx,
               [5]*lx,
               [5]*lx,
               [5]*lx,
               [5]*lx]
    
    qnlstlst = [[3,1],
                [3,1],
                [3,1],
                [3,1],
                [3,-1],
                [3,-1],
                [3,-1],
                [3,-1],
                [-3,1],
                [-3,1],
                [-3,1],
                [-3,1],
                [-3,-1],
                [-3,-1],
                [-3,-1],
                [-3,-1]]
    
    survive_appgs_list = []
    appgs_list = []

    for majoprlst in majoprlstlst:
        surgs, gpsurgs = obc_gs_generator(majoprlst, initlst[majoprlstlst.index(majoprlst)], qnlstlst[majoprlstlst.index(majoprlst)])
        appgs_list.append(surgs)
        if gpsurgs is not None:
            survive_appgs_list.append(gpsurgs)

    gp_ovlp_mat = np.zeros((len(survive_appgs_list), len(survive_appgs_list)))
    for i in range(len(survive_appgs_list)):
        for j in range(len(survive_appgs_list)):
            gp_ovlp_mat[i,j] = np.real(survive_appgs_list[i].overlap(survive_appgs_list[j]))