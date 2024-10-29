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
                init = [0] * L #all empty
            if self.pbc == 1:
                init = [15] + [0]*(L-1) #a_{1,u}^\dagger a_{1,v}^\dagger ... a_{1,z}^\dagger \ket{0}_a
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

    # 定义mpomps_obc函数
    def mpomps_obc(init):
        try:
            # 设置并运行mpos
            params_mpomps = dict(cons_N=conn, cons_S=cons, trunc_params=dict(chi_max=Dmpos), pbc=pbc, init=init)
            mpos = MPOMPS(vmat, umat, **params_mpomps)
            mpos.run()

            # 投影操作和计算能量期望
            psi = mpos.psi
            gppsi = GutzwillerProjectionParton2Spin(psi)
            
            print("----------OBC sandwiches----------")
            bbqmpo = so4dmrgmodel.calc_H_MPO()
            print(" ")
            print("init is", init)
            
            expectation_value = bbqmpo.expectation_value(gppsi)
            print("the sandwich of projected psi and SO(4) MPO is", expectation_value)
            
            # 计算重叠
            overlap_value = psidmrg.overlap(gppsi)
            print("the overlap of psidmrg and gppsimlwo_obc", overlap_value)
            
            # 检查期望值是否等于目标值
            if abs(expectation_value-Edmrg) < 1e-6:
                print("找到目标 init:", init)
                return gppsi
            
        except Exception as e:
            # 捕获异常，打印错误信息，并继续循环
            print(f"运行init={init}时出错: {e}")
            return None
        
    init_list = [[0, 0, 0, 0], [0, 0, 0, 7], [0, 0, 0, 8], [0, 0, 0, 10], [0, 0, 0, 15], [0, 0, 3, 2], [0, 0, 3, 4], [0, 0, 3, 12], [0, 0, 3, 14], [0, 0, 4, 1], [0, 0, 4, 3], [0, 0, 4, 11], [0, 0, 4, 13], [0, 0, 7, 0], [0, 0, 7, 7], [0, 0, 7, 8], [0, 0, 7, 10], [0, 0, 7, 15], [0, 0, 8, 0], [0, 0, 8, 7], [0, 0, 8, 8], [0, 0, 8, 10], [0, 0, 8, 15], [0, 0, 10, 0], [0, 0, 10, 5], [0, 0, 10, 7], [0, 0, 10, 8], [0, 0, 10, 15], [0, 0, 13, 2], [0, 0, 13, 4], [0, 0, 13, 12], [0, 0, 13, 14], [0, 0, 14, 1], [0, 0, 14, 3], [0, 0, 14, 11], [0, 0, 14, 13], [0, 0, 15, 0], [0, 0, 15, 7], [0, 0, 15, 8], [0, 0, 15, 10], [0, 0, 15, 15], [0, 1, 4, 10], [0, 1, 10, 4], [0, 1, 10, 14], [0, 1, 14, 10], [0, 2, 3, 10], [0, 2, 10, 3], [0, 2, 10, 13], [0, 2, 13, 10], [0, 3, 0, 2], [0, 3, 0, 4], [0, 3, 0, 12], [0, 3, 0, 14], [0, 3, 2, 0], [0, 3, 2, 7], [0, 3, 2, 8], [0, 3, 2, 9], [0, 3, 2, 10], [0, 3, 2, 15], [0, 3, 4, 0], [0, 3, 4, 5], [0, 3, 4, 7], [0, 3, 4, 8], [0, 3, 4, 15], [0, 3, 7, 2], [0, 3, 7, 4], [0, 3, 7, 12], [0, 3, 7, 14], [0, 3, 8, 2], [0, 3, 8, 4], [0, 3, 8, 12], [0, 3, 8, 14], [0, 3, 9, 1], [0, 3, 9, 2], [0, 3, 9, 3], [0, 3, 9, 11], [0, 3, 9, 12], [0, 3, 9, 13], [0, 3, 12, 0], [0, 3, 12, 7], [0, 3, 12, 8], [0, 3, 12, 9], [0, 3, 12, 10], [0, 3, 12, 15], [0, 3, 14, 0], [0, 3, 14, 5], [0, 3, 14, 7], [0, 3, 14, 8], [0, 3, 14, 15], [0, 3, 15, 2], [0, 3, 15, 4], [0, 3, 15, 12], [0, 3, 15, 14], [0, 4, 0, 1], [0, 4, 0, 3], [0, 4, 0, 11], [0, 4, 0, 13], [0, 4, 1, 0], [0, 4, 1, 6], [0, 4, 1, 7], [0, 4, 1, 8], [0, 4, 1, 10], [0, 4, 1, 15], [0, 4, 3, 0], [0, 4, 3, 5], [0, 4, 3, 7], [0, 4, 3, 8], [0, 4, 3, 15], [0, 4, 6, 1], [0, 4, 6, 2], [0, 4, 6, 4], [0, 4, 6, 11], [0, 4, 6, 12], [0, 4, 6, 14], [0, 4, 7, 1], [0, 4, 7, 3], [0, 4, 7, 11], [0, 4, 7, 13], [0, 4, 8, 1], [0, 4, 8, 3], [0, 4, 8, 11], [0, 4, 8, 13], [0, 4, 11, 0], [0, 4, 11, 6], [0, 4, 11, 7], [0, 4, 11, 8], [0, 4, 11, 10], [0, 4, 11, 15], [0, 4, 13, 0], [0, 4, 13, 5], [0, 4, 13, 7], [0, 4, 13, 8], [0, 4, 13, 15], [0, 4, 15, 1], [0, 4, 15, 3], [0, 4, 15, 11], [0, 4, 15, 13], [0, 6, 4, 4], [0, 6, 4, 14], [0, 6, 9, 9], [0, 6, 9, 10], [0, 6, 14, 4], [0, 6, 14, 14], [0, 7, 0, 0], [0, 7, 0, 7], [0, 7, 0, 8], [0, 7, 0, 10], [0, 7, 0, 15], [0, 7, 3, 2], [0, 7, 3, 4], [0, 7, 3, 12], [0, 7, 3, 14], [0, 7, 4, 1], [0, 7, 4, 3], [0, 7, 4, 11], [0, 7, 4, 13], [0, 7, 7, 0], [0, 7, 7, 7], [0, 7, 7, 8], [0, 7, 7, 10], [0, 7, 7, 15], [0, 7, 8, 0], [0, 7, 8, 7], [0, 7, 8, 8], [0, 7, 8, 10], [0, 7, 8, 15], [0, 7, 10, 0], [0, 7, 10, 5], [0, 7, 10, 7], [0, 7, 10, 8], [0, 7, 10, 15], [0, 7, 13, 2], [0, 7, 13, 4], [0, 7, 13, 12], [0, 7, 13, 14], [0, 7, 14, 1], [0, 7, 14, 3], [0, 7, 14, 11], [0, 7, 14, 13], [0, 7, 15, 0], [0, 7, 15, 7], [0, 7, 15, 8], [0, 7, 15, 10], [0, 7, 15, 15], [0, 8, 0, 0], [0, 8, 0, 7], [0, 8, 0, 8], [0, 8, 0, 10], [0, 8, 0, 15], [0, 8, 3, 2], [0, 8, 3, 4], [0, 8, 3, 12], [0, 8, 3, 14], [0, 8, 4, 1], [0, 8, 4, 3], [0, 8, 4, 11], [0, 8, 4, 13], [0, 8, 7, 0], [0, 8, 7, 7], [0, 8, 7, 8], [0, 8, 7, 10], [0, 8, 7, 15], [0, 8, 8, 0], [0, 8, 8, 7], [0, 8, 8, 8], [0, 8, 8, 10], [0, 8, 8, 15], [0, 8, 10, 0], [0, 8, 10, 5], [0, 8, 10, 7], [0, 8, 10, 8], [0, 8, 10, 15], [0, 8, 13, 2], [0, 8, 13, 4], [0, 8, 13, 12], [0, 8, 13, 14], [0, 8, 14, 1], [0, 8, 14, 3], [0, 8, 14, 11], [0, 8, 14, 13], [0, 8, 15, 0], [0, 8, 15, 7], [0, 8, 15, 8], [0, 8, 15, 10], [0, 8, 15, 15], [0, 9, 3, 3], [0, 9, 3, 13], [0, 9, 6, 6], [0, 9, 6, 10], [0, 9, 13, 3], [0, 9, 13, 13], [0, 10, 0, 0], [0, 10, 0, 5], [0, 10, 0, 7], [0, 10, 0, 8], [0, 10, 0, 15], [0, 10, 1, 1], [0, 10, 1, 2], [0, 10, 1, 4], [0, 10, 1, 11], [0, 10, 1, 12], [0, 10, 1, 14], [0, 10, 2, 1], [0, 10, 2, 2], [0, 10, 2, 3], [0, 10, 2, 11], [0, 10, 2, 12], [0, 10, 2, 13], [0, 10, 5, 0], [0, 10, 5, 5], [0, 10, 5, 6], [0, 10, 5, 7], [0, 10, 5, 8], [0, 10, 5, 9], [0, 10, 5, 10], [0, 10, 5, 15], [0, 10, 7, 0], [0, 10, 7, 5], [0, 10, 7, 7], [0, 10, 7, 8], [0, 10, 7, 15], [0, 10, 8, 0], [0, 10, 8, 5], [0, 10, 8, 7], [0, 10, 8, 8], [0, 10, 8, 15], [0, 10, 11, 1], [0, 10, 11, 2], [0, 10, 11, 4], [0, 10, 11, 11], [0, 10, 11, 12], [0, 10, 11, 14], [0, 10, 12, 1], [0, 10, 12, 2], [0, 10, 12, 3], [0, 10, 12, 11], [0, 10, 12, 12], [0, 10, 12, 13], [0, 10, 15, 0], [0, 10, 15, 5], [0, 10, 15, 7], [0, 10, 15, 8], [0, 10, 15, 15], [0, 11, 4, 10], [0, 11, 10, 4], [0, 11, 10, 14], [0, 11, 14, 10], [0, 12, 3, 10], [0, 12, 10, 3], [0, 12, 10, 13], [0, 12, 13, 10], [0, 13, 0, 2], [0, 13, 0, 4], [0, 13, 0, 12], [0, 13, 0, 14], [0, 13, 2, 0], [0, 13, 2, 7], [0, 13, 2, 8], [0, 13, 2, 9], [0, 13, 2, 10], [0, 13, 2, 15], [0, 13, 4, 0], [0, 13, 4, 5], [0, 13, 4, 7], [0, 13, 4, 8], [0, 13, 4, 15], [0, 13, 7, 2], [0, 13, 7, 4], [0, 13, 7, 12], [0, 13, 7, 14], [0, 13, 8, 2], [0, 13, 8, 4], [0, 13, 8, 12], [0, 13, 8, 14], [0, 13, 9, 1], [0, 13, 9, 2], [0, 13, 9, 3], [0, 13, 9, 11], [0, 13, 9, 12], [0, 13, 9, 13], [0, 13, 12, 0], [0, 13, 12, 7], [0, 13, 12, 8], [0, 13, 12, 9], [0, 13, 12, 10], [0, 13, 12, 15], [0, 13, 14, 0], [0, 13, 14, 5], [0, 13, 14, 7], [0, 13, 14, 8], [0, 13, 14, 15], [0, 13, 15, 2], [0, 13, 15, 4], [0, 13, 15, 12], [0, 13, 15, 14], [0, 14, 0, 1], [0, 14, 0, 3], [0, 14, 0, 11], [0, 14, 0, 13], [0, 14, 1, 0], [0, 14, 1, 6], [0, 14, 1, 7], [0, 14, 1, 8], [0, 14, 1, 10], [0, 14, 1, 15], [0, 14, 3, 0], [0, 14, 3, 5], [0, 14, 3, 7], [0, 14, 3, 8], [0, 14, 3, 15], [0, 14, 6, 1], [0, 14, 6, 2], [0, 14, 6, 4], [0, 14, 6, 11], [0, 14, 6, 12], [0, 14, 6, 14], [0, 14, 7, 1], [0, 14, 7, 3], [0, 14, 7, 11], [0, 14, 7, 13], [0, 14, 8, 1], [0, 14, 8, 3], [0, 14, 8, 11], [0, 14, 8, 13], [0, 14, 11, 0], [0, 14, 11, 6], [0, 14, 11, 7], [0, 14, 11, 8], [0, 14, 11, 10], [0, 14, 11, 15], [0, 14, 13, 0], [0, 14, 13, 5], [0, 14, 13, 7], [0, 14, 13, 8], [0, 14, 13, 15], [0, 14, 15, 1], [0, 14, 15, 3], [0, 14, 15, 11], [0, 14, 15, 13], [0, 15, 0, 0], [0, 15, 0, 7], [0, 15, 0, 8], [0, 15, 0, 10], [0, 15, 0, 15], [0, 15, 3, 2], [0, 15, 3, 4], [0, 15, 3, 12], [0, 15, 3, 14], [0, 15, 4, 1], [0, 15, 4, 3], [0, 15, 4, 11], [0, 15, 4, 13], [0, 15, 7, 0], [0, 15, 7, 7], [0, 15, 7, 8], [0, 15, 7, 10], [0, 15, 7, 15], [0, 15, 8, 0], [0, 15, 8, 7], [0, 15, 8, 8], [0, 15, 8, 10], [0, 15, 8, 15], [0, 15, 10, 0], [0, 15, 10, 5], [0, 15, 10, 7], [0, 15, 10, 8], [0, 15, 10, 15], [0, 15, 13, 2], [0, 15, 13, 4], [0, 15, 13, 12], [0, 15, 13, 14], [0, 15, 14, 1], [0, 15, 14, 3], [0, 15, 14, 11], [0, 15, 14, 13], [0, 15, 15, 0], [0, 15, 15, 7], [0, 15, 15, 8], [0, 15, 15, 10], [0, 15, 15, 15], [1, 0, 0, 4], [1, 0, 0, 14], [1, 0, 3, 9], [1, 0, 4, 0], [1, 0, 4, 7], [1, 0, 4, 8], [1, 0, 4, 10], [1, 0, 4, 15], [1, 0, 7, 4], [1, 0, 7, 14], [1, 0, 8, 4], [1, 0, 8, 14], [1, 0, 10, 2], [1, 0, 10, 4], [1, 0, 10, 12], [1, 0, 10, 14], [1, 0, 13, 9], [1, 0, 14, 0], [1, 0, 14, 7], [1, 0, 14, 8], [1, 0, 14, 10], [1, 0, 14, 15], [1, 0, 15, 4], [1, 0, 15, 14], [1, 2, 10, 10], [1, 3, 0, 9], [1, 3, 2, 4], [1, 3, 2, 14], [1, 3, 4, 2], [1, 3, 4, 4], [1, 3, 4, 12], [1, 3, 4, 14], [1, 3, 7, 9], [1, 3, 8, 9], [1, 3, 9, 0], [1, 3, 9, 7], [1, 3, 9, 8], [1, 3, 9, 9], [1, 3, 9, 10], [1, 3, 9, 15], [1, 3, 12, 4], [1, 3, 12, 14], [1, 3, 14, 2], [1, 3, 14, 4], [1, 3, 14, 12], [1, 3, 14, 14], [1, 3, 15, 9], [1, 4, 0, 0], [1, 4, 0, 7], [1, 4, 0, 8], [1, 4, 0, 10], [1, 4, 0, 15], [1, 4, 1, 4], [1, 4, 1, 14], [1, 4, 3, 2], [1, 4, 3, 4], [1, 4, 3, 12], [1, 4, 3, 14], [1, 4, 4, 1], [1, 4, 4, 3], [1, 4, 4, 11], [1, 4, 4, 13], [1, 4, 6, 9], [1, 4, 7, 0], [1, 4, 7, 7], [1, 4, 7, 8], [1, 4, 7, 10], [1, 4, 7, 15], [1, 4, 8, 0], [1, 4, 8, 7], [1, 4, 8, 8], [1, 4, 8, 10], [1, 4, 8, 15], [1, 4, 10, 0], [1, 4, 10, 5], [1, 4, 10, 7], [1, 4, 10, 8], [1, 4, 10, 15], [1, 4, 11, 4], [1, 4, 11, 14], [1, 4, 13, 2], [1, 4, 13, 4], [1, 4, 13, 12], [1, 4, 13, 14], [1, 4, 14, 1], [1, 4, 14, 3], [1, 4, 14, 11], [1, 4, 14, 13], [1, 4, 15, 0], [1, 4, 15, 7], [1, 4, 15, 8], [1, 4, 15, 10], [1, 4, 15, 15], [1, 7, 0, 4], [1, 7, 0, 14], [1, 7, 3, 9], [1, 7, 4, 0], [1, 7, 4, 7], [1, 7, 4, 8], [1, 7, 4, 10], [1, 7, 4, 15], [1, 7, 7, 4], [1, 7, 7, 14], [1, 7, 8, 4], [1, 7, 8, 14], [1, 7, 10, 2], [1, 7, 10, 4], [1, 7, 10, 12], [1, 7, 10, 14], [1, 7, 13, 9], [1, 7, 14, 0], [1, 7, 14, 7], [1, 7, 14, 8], [1, 7, 14, 10], [1, 7, 14, 15], [1, 7, 15, 4], [1, 7, 15, 14], [1, 8, 0, 4], [1, 8, 0, 14], [1, 8, 3, 9], [1, 8, 4, 0], [1, 8, 4, 7], [1, 8, 4, 8], [1, 8, 4, 10], [1, 8, 4, 15], [1, 8, 7, 4], [1, 8, 7, 14], [1, 8, 8, 4], [1, 8, 8, 14], [1, 8, 10, 2], [1, 8, 10, 4], [1, 8, 10, 12], [1, 8, 10, 14], [1, 8, 13, 9], [1, 8, 14, 0], [1, 8, 14, 7], [1, 8, 14, 8], [1, 8, 14, 10], [1, 8, 14, 15], [1, 8, 15, 4], [1, 8, 15, 14], [1, 9, 3, 10], [1, 9, 10, 3], [1, 9, 10, 13], [1, 9, 13, 10], [1, 10, 0, 2], [1, 10, 0, 4], [1, 10, 0, 12], [1, 10, 0, 14], [1, 10, 1, 9], [1, 10, 2, 0], [1, 10, 2, 7], [1, 10, 2, 8], [1, 10, 2, 9], [1, 10, 2, 10], [1, 10, 2, 15], [1, 10, 4, 0], [1, 10, 4, 5], [1, 10, 4, 7], [1, 10, 4, 8], [1, 10, 4, 15], [1, 10, 5, 4], [1, 10, 5, 14], [1, 10, 7, 2], [1, 10, 7, 4], [1, 10, 7, 12], [1, 10, 7, 14], [1, 10, 8, 2], [1, 10, 8, 4], [1, 10, 8, 12], [1, 10, 8, 14], [1, 10, 9, 1], [1, 10, 9, 2], [1, 10, 9, 3], [1, 10, 9, 11], [1, 10, 9, 12], [1, 10, 9, 13], [1, 10, 11, 9], [1, 10, 12, 0], [1, 10, 12, 7], [1, 10, 12, 8], [1, 10, 12, 9], [1, 10, 12, 10], [1, 10, 12, 15], [1, 10, 14, 0], [1, 10, 14, 5], [1, 10, 14, 7], [1, 10, 14, 8], [1, 10, 14, 15], [1, 10, 15, 2], [1, 10, 15, 4], [1, 10, 15, 12], [1, 10, 15, 14], [1, 12, 10, 10], [1, 13, 0, 9], [1, 13, 2, 4], [1, 13, 2, 14], [1, 13, 4, 2], [1, 13, 4, 4], [1, 13, 4, 12], [1, 13, 4, 14], [1, 13, 7, 9], [1, 13, 8, 9], [1, 13, 9, 0], [1, 13, 9, 7], [1, 13, 9, 8], [1, 13, 9, 9], [1, 13, 9, 10], [1, 13, 9, 15], [1, 13, 12, 4], [1, 13, 12, 14], [1, 13, 14, 2], [1, 13, 14, 4], [1, 13, 14, 12], [1, 13, 14, 14], [1, 13, 15, 9], [1, 14, 0, 0], [1, 14, 0, 7], [1, 14, 0, 8], [1, 14, 0, 10], [1, 14, 0, 15], [1, 14, 1, 4], [1, 14, 1, 14], [1, 14, 3, 2], [1, 14, 3, 4], [1, 14, 3, 12], [1, 14, 3, 14], [1, 14, 4, 1], [1, 14, 4, 3], [1, 14, 4, 11], [1, 14, 4, 13], [1, 14, 6, 9], [1, 14, 7, 0], [1, 14, 7, 7], [1, 14, 7, 8], [1, 14, 7, 10], [1, 14, 7, 15], [1, 14, 8, 0], [1, 14, 8, 7], [1, 14, 8, 8], [1, 14, 8, 10], [1, 14, 8, 15], [1, 14, 10, 0], [1, 14, 10, 5], [1, 14, 10, 7], [1, 14, 10, 8], [1, 14, 10, 15], [1, 14, 11, 4], [1, 14, 11, 14], [1, 14, 13, 2], [1, 14, 13, 4], [1, 14, 13, 12], [1, 14, 13, 14], [1, 14, 14, 1], [1, 14, 14, 3], [1, 14, 14, 11], [1, 14, 14, 13], [1, 14, 15, 0], [1, 14, 15, 7], [1, 14, 15, 8], [1, 14, 15, 10], [1, 14, 15, 15], [1, 15, 0, 4], [1, 15, 0, 14], [1, 15, 3, 9], [1, 15, 4, 0], [1, 15, 4, 7], [1, 15, 4, 8], [1, 15, 4, 10], [1, 15, 4, 15], [1, 15, 7, 4], [1, 15, 7, 14], [1, 15, 8, 4], [1, 15, 8, 14], [1, 15, 10, 2], [1, 15, 10, 4], [1, 15, 10, 12], [1, 15, 10, 14], [1, 15, 13, 9], [1, 15, 14, 0], [1, 15, 14, 7], [1, 15, 14, 8], [1, 15, 14, 10], [1, 15, 14, 15], [1, 15, 15, 4], [1, 15, 15, 14], [2, 0, 0, 3], [2, 0, 0, 13], [2, 0, 3, 0], [2, 0, 3, 7], [2, 0, 3, 8], [2, 0, 3, 10], [2, 0, 3, 15], [2, 0, 4, 6], [2, 0, 7, 3], [2, 0, 7, 13], [2, 0, 8, 3], [2, 0, 8, 13], [2, 0, 10, 1], [2, 0, 10, 3], [2, 0, 10, 11], [2, 0, 10, 13], [2, 0, 13, 0], [2, 0, 13, 7], [2, 0, 13, 8], [2, 0, 13, 10], [2, 0, 13, 15], [2, 0, 14, 6], [2, 0, 15, 3], [2, 0, 15, 13], [2, 1, 10, 10], [2, 3, 0, 0], [2, 3, 0, 7], [2, 3, 0, 8], [2, 3, 0, 10], [2, 3, 0, 15], [2, 3, 2, 3], [2, 3, 2, 13], [2, 3, 3, 2], [2, 3, 3, 4], [2, 3, 3, 12], [2, 3, 3, 14], [2, 3, 4, 1], [2, 3, 4, 3], [2, 3, 4, 11], [2, 3, 4, 13], [2, 3, 7, 0], [2, 3, 7, 7], [2, 3, 7, 8], [2, 3, 7, 10], [2, 3, 7, 15], [2, 3, 8, 0], [2, 3, 8, 7], [2, 3, 8, 8], [2, 3, 8, 10], [2, 3, 8, 15], [2, 3, 9, 6], [2, 3, 10, 0], [2, 3, 10, 5], [2, 3, 10, 7], [2, 3, 10, 8], [2, 3, 10, 15], [2, 3, 12, 3], [2, 3, 12, 13], [2, 3, 13, 2], [2, 3, 13, 4], [2, 3, 13, 12], [2, 3, 13, 14], [2, 3, 14, 1], [2, 3, 14, 3], [2, 3, 14, 11], [2, 3, 14, 13], [2, 3, 15, 0], [2, 3, 15, 7], [2, 3, 15, 8], [2, 3, 15, 10], [2, 3, 15, 15], [2, 4, 0, 6], [2, 4, 1, 3], [2, 4, 1, 13], [2, 4, 3, 1], [2, 4, 3, 3], [2, 4, 3, 11], [2, 4, 3, 13], [2, 4, 6, 0], [2, 4, 6, 6], [2, 4, 6, 7], [2, 4, 6, 8], [2, 4, 6, 10], [2, 4, 6, 15], [2, 4, 7, 6], [2, 4, 8, 6], [2, 4, 11, 3], [2, 4, 11, 13], [2, 4, 13, 1], [2, 4, 13, 3], [2, 4, 13, 11], [2, 4, 13, 13], [2, 4, 15, 6], [2, 6, 4, 10], [2, 6, 10, 4], [2, 6, 10, 14], [2, 6, 14, 10], [2, 7, 0, 3], [2, 7, 0, 13], [2, 7, 3, 0], [2, 7, 3, 7], [2, 7, 3, 8], [2, 7, 3, 10], [2, 7, 3, 15], [2, 7, 4, 6], [2, 7, 7, 3], [2, 7, 7, 13], [2, 7, 8, 3], [2, 7, 8, 13], [2, 7, 10, 1], [2, 7, 10, 3], [2, 7, 10, 11], [2, 7, 10, 13], [2, 7, 13, 0], [2, 7, 13, 7], [2, 7, 13, 8], [2, 7, 13, 10], [2, 7, 13, 15], [2, 7, 14, 6], [2, 7, 15, 3], [2, 7, 15, 13], [2, 8, 0, 3], [2, 8, 0, 13], [2, 8, 3, 0], [2, 8, 3, 7], [2, 8, 3, 8], [2, 8, 3, 10], [2, 8, 3, 15], [2, 8, 4, 6], [2, 8, 7, 3], [2, 8, 7, 13], [2, 8, 8, 3], [2, 8, 8, 13], [2, 8, 10, 1], [2, 8, 10, 3], [2, 8, 10, 11], [2, 8, 10, 13], [2, 8, 13, 0], [2, 8, 13, 7], [2, 8, 13, 8], [2, 8, 13, 10], [2, 8, 13, 15], [2, 8, 14, 6], [2, 8, 15, 3], [2, 8, 15, 13], [2, 10, 0, 1], [2, 10, 0, 3], [2, 10, 0, 11], [2, 10, 0, 13], [2, 10, 1, 0], [2, 10, 1, 6], [2, 10, 1, 7], [2, 10, 1, 8], [2, 10, 1, 10], [2, 10, 1, 15], [2, 10, 2, 6], [2, 10, 3, 0], [2, 10, 3, 5], [2, 10, 3, 7], [2, 10, 3, 8], [2, 10, 3, 15], [2, 10, 5, 3], [2, 10, 5, 13], [2, 10, 6, 1], [2, 10, 6, 2], [2, 10, 6, 4], [2, 10, 6, 11], [2, 10, 6, 12], [2, 10, 6, 14], [2, 10, 7, 1], [2, 10, 7, 3], [2, 10, 7, 11], [2, 10, 7, 13], [2, 10, 8, 1], [2, 10, 8, 3], [2, 10, 8, 11], [2, 10, 8, 13], [2, 10, 11, 0], [2, 10, 11, 6], [2, 10, 11, 7], [2, 10, 11, 8], [2, 10, 11, 10], [2, 10, 11, 15], [2, 10, 12, 6], [2, 10, 13, 0], [2, 10, 13, 5], [2, 10, 13, 7], [2, 10, 13, 8], [2, 10, 13, 15], [2, 10, 15, 1], [2, 10, 15, 3], [2, 10, 15, 11], [2, 10, 15, 13], [2, 11, 10, 10], [2, 13, 0, 0], [2, 13, 0, 7], [2, 13, 0, 8], [2, 13, 0, 10], [2, 13, 0, 15], [2, 13, 2, 3], [2, 13, 2, 13], [2, 13, 3, 2], [2, 13, 3, 4], [2, 13, 3, 12], [2, 13, 3, 14], [2, 13, 4, 1], [2, 13, 4, 3], [2, 13, 4, 11], [2, 13, 4, 13], [2, 13, 7, 0], [2, 13, 7, 7], [2, 13, 7, 8], [2, 13, 7, 10], [2, 13, 7, 15], [2, 13, 8, 0], [2, 13, 8, 7], [2, 13, 8, 8], [2, 13, 8, 10], [2, 13, 8, 15], [2, 13, 9, 6], [2, 13, 10, 0], [2, 13, 10, 5], [2, 13, 10, 7], [2, 13, 10, 8], [2, 13, 10, 15], [2, 13, 12, 3], [2, 13, 12, 13], [2, 13, 13, 2], [2, 13, 13, 4], [2, 13, 13, 12], [2, 13, 13, 14], [2, 13, 14, 1], [2, 13, 14, 3], [2, 13, 14, 11], [2, 13, 14, 13], [2, 13, 15, 0], [2, 13, 15, 7], [2, 13, 15, 8], [2, 13, 15, 10], [2, 13, 15, 15], [2, 14, 0, 6], [2, 14, 1, 3], [2, 14, 1, 13], [2, 14, 3, 1], [2, 14, 3, 3], [2, 14, 3, 11], [2, 14, 3, 13], [2, 14, 6, 0], [2, 14, 6, 6], [2, 14, 6, 7], [2, 14, 6, 8], [2, 14, 6, 10], [2, 14, 6, 15], [2, 14, 7, 6], [2, 14, 8, 6], [2, 14, 11, 3], [2, 14, 11, 13], [2, 14, 13, 1], [2, 14, 13, 3], [2, 14, 13, 11], [2, 14, 13, 13], [2, 14, 15, 6], [2, 15, 0, 3], [2, 15, 0, 13], [2, 15, 3, 0], [2, 15, 3, 7], [2, 15, 3, 8], [2, 15, 3, 10], [2, 15, 3, 15], [2, 15, 4, 6], [2, 15, 7, 3], [2, 15, 7, 13], [2, 15, 8, 3], [2, 15, 8, 13], [2, 15, 10, 1], [2, 15, 10, 3], [2, 15, 10, 11], [2, 15, 10, 13], [2, 15, 13, 0], [2, 15, 13, 7], [2, 15, 13, 8], [2, 15, 13, 10], [2, 15, 13, 15], [2, 15, 14, 6], [2, 15, 15, 3], [2, 15, 15, 13], [3, 0, 0, 2], [3, 0, 0, 4], [3, 0, 0, 12], [3, 0, 0, 14], [3, 0, 2, 0], [3, 0, 2, 7], [3, 0, 2, 8], [3, 0, 2, 9], [3, 0, 2, 10], [3, 0, 2, 15], [3, 0, 4, 0], [3, 0, 4, 5], [3, 0, 4, 7], [3, 0, 4, 8], [3, 0, 4, 15], [3, 0, 7, 2], [3, 0, 7, 4], [3, 0, 7, 12], [3, 0, 7, 14], [3, 0, 8, 2], [3, 0, 8, 4], [3, 0, 8, 12], [3, 0, 8, 14], [3, 0, 9, 1], [3, 0, 9, 2], [3, 0, 9, 3], [3, 0, 9, 11], [3, 0, 9, 12], [3, 0, 9, 13], [3, 0, 12, 0], [3, 0, 12, 7], [3, 0, 12, 8], [3, 0, 12, 9], [3, 0, 12, 10], [3, 0, 12, 15], [3, 0, 14, 0], [3, 0, 14, 5], [3, 0, 14, 7], [3, 0, 14, 8], [3, 0, 14, 15], [3, 0, 15, 2], [3, 0, 15, 4], [3, 0, 15, 12], [3, 0, 15, 14], [3, 1, 4, 4], [3, 1, 4, 14], [3, 1, 9, 9], [3, 1, 9, 10], [3, 1, 14, 4], [3, 1, 14, 14], [3, 2, 0, 0], [3, 2, 0, 7], [3, 2, 0, 8], [3, 2, 0, 10], [3, 2, 0, 15], [3, 2, 3, 2], [3, 2, 3, 4], [3, 2, 3, 12], [3, 2, 3, 14], [3, 2, 4, 1], [3, 2, 4, 3], [3, 2, 4, 11], [3, 2, 4, 13], [3, 2, 7, 0], [3, 2, 7, 7], [3, 2, 7, 8], [3, 2, 7, 10], [3, 2, 7, 15], [3, 2, 8, 0], [3, 2, 8, 7], [3, 2, 8, 8], [3, 2, 8, 10], [3, 2, 8, 15], [3, 2, 10, 0], [3, 2, 10, 5], [3, 2, 10, 7], [3, 2, 10, 8], [3, 2, 10, 15], [3, 2, 13, 2], [3, 2, 13, 4], [3, 2, 13, 12], [3, 2, 13, 14], [3, 2, 14, 1], [3, 2, 14, 3], [3, 2, 14, 11], [3, 2, 14, 13], [3, 2, 15, 0], [3, 2, 15, 7], [3, 2, 15, 8], [3, 2, 15, 10], [3, 2, 15, 15], [3, 3, 2, 2], [3, 3, 2, 12], [3, 3, 9, 5], [3, 3, 12, 2], [3, 3, 12, 12], [3, 4, 0, 0], [3, 4, 0, 5], [3, 4, 0, 7], [3, 4, 0, 8], [3, 4, 0, 15], [3, 4, 1, 1], [3, 4, 1, 2], [3, 4, 1, 4], [3, 4, 1, 11], [3, 4, 1, 12], [3, 4, 1, 14], [3, 4, 2, 1], [3, 4, 2, 2], [3, 4, 2, 3], [3, 4, 2, 11], [3, 4, 2, 12], [3, 4, 2, 13], [3, 4, 5, 0], [3, 4, 5, 5], [3, 4, 5, 6], [3, 4, 5, 7], [3, 4, 5, 8], [3, 4, 5, 9], [3, 4, 5, 10], [3, 4, 5, 15], [3, 4, 7, 0], [3, 4, 7, 5], [3, 4, 7, 7], [3, 4, 7, 8], [3, 4, 7, 15], [3, 4, 8, 0], [3, 4, 8, 5], [3, 4, 8, 7], [3, 4, 8, 8], [3, 4, 8, 15], [3, 4, 11, 1], [3, 4, 11, 2], [3, 4, 11, 4], [3, 4, 11, 11], [3, 4, 11, 12], [3, 4, 11, 14], [3, 4, 12, 1], [3, 4, 12, 2], [3, 4, 12, 3], [3, 4, 12, 11], [3, 4, 12, 12], [3, 4, 12, 13], [3, 4, 15, 0], [3, 4, 15, 5], [3, 4, 15, 7], [3, 4, 15, 8], [3, 4, 15, 15], [3, 5, 4, 10], [3, 5, 10, 4], [3, 5, 10, 14], [3, 5, 14, 10], [3, 7, 0, 2], [3, 7, 0, 4], [3, 7, 0, 12], [3, 7, 0, 14], [3, 7, 2, 0], [3, 7, 2, 7], [3, 7, 2, 8], [3, 7, 2, 9], [3, 7, 2, 10], [3, 7, 2, 15], [3, 7, 4, 0], [3, 7, 4, 5], [3, 7, 4, 7], [3, 7, 4, 8], [3, 7, 4, 15], [3, 7, 7, 2], [3, 7, 7, 4], [3, 7, 7, 12], [3, 7, 7, 14], [3, 7, 8, 2], [3, 7, 8, 4], [3, 7, 8, 12], [3, 7, 8, 14], [3, 7, 9, 1], [3, 7, 9, 2], [3, 7, 9, 3], [3, 7, 9, 11], [3, 7, 9, 12], [3, 7, 9, 13], [3, 7, 12, 0], [3, 7, 12, 7], [3, 7, 12, 8], [3, 7, 12, 9], [3, 7, 12, 10], [3, 7, 12, 15], [3, 7, 14, 0], [3, 7, 14, 5], [3, 7, 14, 7], [3, 7, 14, 8], [3, 7, 14, 15], [3, 7, 15, 2], [3, 7, 15, 4], [3, 7, 15, 12], [3, 7, 15, 14], [3, 8, 0, 2], [3, 8, 0, 4], [3, 8, 0, 12], [3, 8, 0, 14], [3, 8, 2, 0], [3, 8, 2, 7], [3, 8, 2, 8], [3, 8, 2, 9], [3, 8, 2, 10], [3, 8, 2, 15], [3, 8, 4, 0], [3, 8, 4, 5], [3, 8, 4, 7], [3, 8, 4, 8], [3, 8, 4, 15], [3, 8, 7, 2], [3, 8, 7, 4], [3, 8, 7, 12], [3, 8, 7, 14], [3, 8, 8, 2], [3, 8, 8, 4], [3, 8, 8, 12], [3, 8, 8, 14], [3, 8, 9, 1], [3, 8, 9, 2], [3, 8, 9, 3], [3, 8, 9, 11], [3, 8, 9, 12], [3, 8, 9, 13], [3, 8, 12, 0], [3, 8, 12, 7], [3, 8, 12, 8], [3, 8, 12, 9], [3, 8, 12, 10], [3, 8, 12, 15], [3, 8, 14, 0], [3, 8, 14, 5], [3, 8, 14, 7], [3, 8, 14, 8], [3, 8, 14, 15], [3, 8, 15, 2], [3, 8, 15, 4], [3, 8, 15, 12], [3, 8, 15, 14], [3, 9, 0, 1], [3, 9, 0, 3], [3, 9, 0, 11], [3, 9, 0, 13], [3, 9, 1, 0], [3, 9, 1, 6], [3, 9, 1, 7], [3, 9, 1, 8], [3, 9, 1, 10], [3, 9, 1, 15], [3, 9, 3, 0], [3, 9, 3, 5], [3, 9, 3, 7], [3, 9, 3, 8], [3, 9, 3, 15], [3, 9, 6, 1], [3, 9, 6, 2], [3, 9, 6, 4], [3, 9, 6, 11], [3, 9, 6, 12], [3, 9, 6, 14], [3, 9, 7, 1], [3, 9, 7, 3], [3, 9, 7, 11], [3, 9, 7, 13], [3, 9, 8, 1], [3, 9, 8, 3], [3, 9, 8, 11], [3, 9, 8, 13], [3, 9, 11, 0], [3, 9, 11, 6], [3, 9, 11, 7], [3, 9, 11, 8], [3, 9, 11, 10], [3, 9, 11, 15], [3, 9, 13, 0], [3, 9, 13, 5], [3, 9, 13, 7], [3, 9, 13, 8], [3, 9, 13, 15], [3, 9, 15, 1], [3, 9, 15, 3], [3, 9, 15, 11], [3, 9, 15, 13], [3, 10, 2, 5], [3, 10, 5, 2], [3, 10, 5, 12], [3, 10, 12, 5], [3, 11, 4, 4], [3, 11, 4, 14], [3, 11, 9, 9], [3, 11, 9, 10], [3, 11, 14, 4], [3, 11, 14, 14], [3, 12, 0, 0], [3, 12, 0, 7], [3, 12, 0, 8], [3, 12, 0, 10], [3, 12, 0, 15], [3, 12, 3, 2], [3, 12, 3, 4], [3, 12, 3, 12], [3, 12, 3, 14], [3, 12, 4, 1], [3, 12, 4, 3], [3, 12, 4, 11], [3, 12, 4, 13], [3, 12, 7, 0], [3, 12, 7, 7], [3, 12, 7, 8], [3, 12, 7, 10], [3, 12, 7, 15], [3, 12, 8, 0], [3, 12, 8, 7], [3, 12, 8, 8], [3, 12, 8, 10], [3, 12, 8, 15], [3, 12, 10, 0], [3, 12, 10, 5], [3, 12, 10, 7], [3, 12, 10, 8], [3, 12, 10, 15], [3, 12, 13, 2], [3, 12, 13, 4], [3, 12, 13, 12], [3, 12, 13, 14], [3, 12, 14, 1], [3, 12, 14, 3], [3, 12, 14, 11], [3, 12, 14, 13], [3, 12, 15, 0], [3, 12, 15, 7], [3, 12, 15, 8], [3, 12, 15, 10], [3, 12, 15, 15], [3, 13, 2, 2], [3, 13, 2, 12], [3, 13, 9, 5], [3, 13, 12, 2], [3, 13, 12, 12], [3, 14, 0, 0], [3, 14, 0, 5], [3, 14, 0, 7], [3, 14, 0, 8], [3, 14, 0, 15], [3, 14, 1, 1], [3, 14, 1, 2], [3, 14, 1, 4], [3, 14, 1, 11], [3, 14, 1, 12], [3, 14, 1, 14], [3, 14, 2, 1], [3, 14, 2, 2], [3, 14, 2, 3], [3, 14, 2, 11], [3, 14, 2, 12], [3, 14, 2, 13], [3, 14, 5, 0], [3, 14, 5, 5], [3, 14, 5, 6], [3, 14, 5, 7], [3, 14, 5, 8], [3, 14, 5, 9], [3, 14, 5, 10], [3, 14, 5, 15], [3, 14, 7, 0], [3, 14, 7, 5], [3, 14, 7, 7], [3, 14, 7, 8], [3, 14, 7, 15], [3, 14, 8, 0], [3, 14, 8, 5], [3, 14, 8, 7], [3, 14, 8, 8], [3, 14, 8, 15], [3, 14, 11, 1], [3, 14, 11, 2], [3, 14, 11, 4], [3, 14, 11, 11], [3, 14, 11, 12], [3, 14, 11, 14], [3, 14, 12, 1], [3, 14, 12, 2], [3, 14, 12, 3], [3, 14, 12, 11], [3, 14, 12, 12], [3, 14, 12, 13], [3, 14, 15, 0], [3, 14, 15, 5], [3, 14, 15, 7], [3, 14, 15, 8], [3, 14, 15, 15], [3, 15, 0, 2], [3, 15, 0, 4], [3, 15, 0, 12], [3, 15, 0, 14], [3, 15, 2, 0], [3, 15, 2, 7], [3, 15, 2, 8], [3, 15, 2, 9], [3, 15, 2, 10], [3, 15, 2, 15], [3, 15, 4, 0], [3, 15, 4, 5], [3, 15, 4, 7], [3, 15, 4, 8], [3, 15, 4, 15], [3, 15, 7, 2], [3, 15, 7, 4], [3, 15, 7, 12], [3, 15, 7, 14], [3, 15, 8, 2], [3, 15, 8, 4], [3, 15, 8, 12], [3, 15, 8, 14], [3, 15, 9, 1], [3, 15, 9, 2], [3, 15, 9, 3], [3, 15, 9, 11], [3, 15, 9, 12], [3, 15, 9, 13], [3, 15, 12, 0], [3, 15, 12, 7], [3, 15, 12, 8], [3, 15, 12, 9], [3, 15, 12, 10], [3, 15, 12, 15], [3, 15, 14, 0], [3, 15, 14, 5], [3, 15, 14, 7], [3, 15, 14, 8], [3, 15, 14, 15], [3, 15, 15, 2], [3, 15, 15, 4], [3, 15, 15, 12], [3, 15, 15, 14], [4, 0, 0, 1], [4, 0, 0, 3], [4, 0, 0, 11], [4, 0, 0, 13], [4, 0, 1, 0], [4, 0, 1, 6], [4, 0, 1, 7], [4, 0, 1, 8], [4, 0, 1, 10], [4, 0, 1, 15], [4, 0, 3, 0], [4, 0, 3, 5], [4, 0, 3, 7], [4, 0, 3, 8], [4, 0, 3, 15], [4, 0, 6, 1], [4, 0, 6, 2], [4, 0, 6, 4], [4, 0, 6, 11], [4, 0, 6, 12], [4, 0, 6, 14], [4, 0, 7, 1], [4, 0, 7, 3], [4, 0, 7, 11], [4, 0, 7, 13], [4, 0, 8, 1], [4, 0, 8, 3], [4, 0, 8, 11], [4, 0, 8, 13], [4, 0, 11, 0], [4, 0, 11, 6], [4, 0, 11, 7], [4, 0, 11, 8], [4, 0, 11, 10], [4, 0, 11, 15], [4, 0, 13, 0], [4, 0, 13, 5], [4, 0, 13, 7], [4, 0, 13, 8], [4, 0, 13, 15], [4, 0, 15, 1], [4, 0, 15, 3], [4, 0, 15, 11], [4, 0, 15, 13], [4, 1, 0, 0], [4, 1, 0, 7], [4, 1, 0, 8], [4, 1, 0, 10], [4, 1, 0, 15], [4, 1, 3, 2], [4, 1, 3, 4], [4, 1, 3, 12], [4, 1, 3, 14], [4, 1, 4, 1], [4, 1, 4, 3], [4, 1, 4, 11], [4, 1, 4, 13], [4, 1, 7, 0], [4, 1, 7, 7], [4, 1, 7, 8], [4, 1, 7, 10], [4, 1, 7, 15], [4, 1, 8, 0], [4, 1, 8, 7], [4, 1, 8, 8], [4, 1, 8, 10], [4, 1, 8, 15], [4, 1, 10, 0], [4, 1, 10, 5], [4, 1, 10, 7], [4, 1, 10, 8], [4, 1, 10, 15], [4, 1, 13, 2], [4, 1, 13, 4], [4, 1, 13, 12], [4, 1, 13, 14], [4, 1, 14, 1], [4, 1, 14, 3], [4, 1, 14, 11], [4, 1, 14, 13], [4, 1, 15, 0], [4, 1, 15, 7], [4, 1, 15, 8], [4, 1, 15, 10], [4, 1, 15, 15], [4, 2, 3, 3], [4, 2, 3, 13], [4, 2, 6, 6], [4, 2, 6, 10], [4, 2, 13, 3], [4, 2, 13, 13], [4, 3, 0, 0], [4, 3, 0, 5], [4, 3, 0, 7], [4, 3, 0, 8], [4, 3, 0, 15], [4, 3, 1, 1], [4, 3, 1, 2], [4, 3, 1, 4], [4, 3, 1, 11], [4, 3, 1, 12], [4, 3, 1, 14], [4, 3, 2, 1], [4, 3, 2, 2], [4, 3, 2, 3], [4, 3, 2, 11], [4, 3, 2, 12], [4, 3, 2, 13], [4, 3, 5, 0], [4, 3, 5, 5], [4, 3, 5, 6], [4, 3, 5, 7], [4, 3, 5, 8], [4, 3, 5, 9], [4, 3, 5, 10], [4, 3, 5, 15], [4, 3, 7, 0], [4, 3, 7, 5], [4, 3, 7, 7], [4, 3, 7, 8], [4, 3, 7, 15], [4, 3, 8, 0], [4, 3, 8, 5], [4, 3, 8, 7], [4, 3, 8, 8], [4, 3, 8, 15], [4, 3, 11, 1], [4, 3, 11, 2], [4, 3, 11, 4], [4, 3, 11, 11], [4, 3, 11, 12], [4, 3, 11, 14], [4, 3, 12, 1], [4, 3, 12, 2], [4, 3, 12, 3], [4, 3, 12, 11], [4, 3, 12, 12], [4, 3, 12, 13], [4, 3, 15, 0], [4, 3, 15, 5], [4, 3, 15, 7], [4, 3, 15, 8], [4, 3, 15, 15], [4, 4, 1, 1], [4, 4, 1, 11], [4, 4, 6, 5], [4, 4, 11, 1], [4, 4, 11, 11], [4, 5, 3, 10], [4, 5, 10, 3], [4, 5, 10, 13], [4, 5, 13, 10], [4, 6, 0, 2], [4, 6, 0, 4], [4, 6, 0, 12], [4, 6, 0, 14], [4, 6, 2, 0], [4, 6, 2, 7], [4, 6, 2, 8], [4, 6, 2, 9], [4, 6, 2, 10], [4, 6, 2, 15], [4, 6, 4, 0], [4, 6, 4, 5], [4, 6, 4, 7], [4, 6, 4, 8], [4, 6, 4, 15], [4, 6, 7, 2], [4, 6, 7, 4], [4, 6, 7, 12], [4, 6, 7, 14], [4, 6, 8, 2], [4, 6, 8, 4], [4, 6, 8, 12], [4, 6, 8, 14], [4, 6, 9, 1], [4, 6, 9, 2], [4, 6, 9, 3], [4, 6, 9, 11], [4, 6, 9, 12], [4, 6, 9, 13], [4, 6, 12, 0], [4, 6, 12, 7], [4, 6, 12, 8], [4, 6, 12, 9], [4, 6, 12, 10], [4, 6, 12, 15], [4, 6, 14, 0], [4, 6, 14, 5], [4, 6, 14, 7], [4, 6, 14, 8], [4, 6, 14, 15], [4, 6, 15, 2], [4, 6, 15, 4], [4, 6, 15, 12], [4, 6, 15, 14], [4, 7, 0, 1], [4, 7, 0, 3], [4, 7, 0, 11], [4, 7, 0, 13], [4, 7, 1, 0], [4, 7, 1, 6], [4, 7, 1, 7], [4, 7, 1, 8], [4, 7, 1, 10], [4, 7, 1, 15], [4, 7, 3, 0], [4, 7, 3, 5], [4, 7, 3, 7], [4, 7, 3, 8], [4, 7, 3, 15], [4, 7, 6, 1], [4, 7, 6, 2], [4, 7, 6, 4], [4, 7, 6, 11], [4, 7, 6, 12], [4, 7, 6, 14], [4, 7, 7, 1], [4, 7, 7, 3], [4, 7, 7, 11], [4, 7, 7, 13], [4, 7, 8, 1], [4, 7, 8, 3], [4, 7, 8, 11], [4, 7, 8, 13], [4, 7, 11, 0], [4, 7, 11, 6], [4, 7, 11, 7], [4, 7, 11, 8], [4, 7, 11, 10], [4, 7, 11, 15], [4, 7, 13, 0], [4, 7, 13, 5], [4, 7, 13, 7], [4, 7, 13, 8], [4, 7, 13, 15], [4, 7, 15, 1], [4, 7, 15, 3], [4, 7, 15, 11], [4, 7, 15, 13], [4, 8, 0, 1], [4, 8, 0, 3], [4, 8, 0, 11], [4, 8, 0, 13], [4, 8, 1, 0], [4, 8, 1, 6], [4, 8, 1, 7], [4, 8, 1, 8], [4, 8, 1, 10], [4, 8, 1, 15], [4, 8, 3, 0], [4, 8, 3, 5], [4, 8, 3, 7], [4, 8, 3, 8], [4, 8, 3, 15], [4, 8, 6, 1], [4, 8, 6, 2], [4, 8, 6, 4], [4, 8, 6, 11], [4, 8, 6, 12], [4, 8, 6, 14], [4, 8, 7, 1], [4, 8, 7, 3], [4, 8, 7, 11], [4, 8, 7, 13], [4, 8, 8, 1], [4, 8, 8, 3], [4, 8, 8, 11], [4, 8, 8, 13], [4, 8, 11, 0], [4, 8, 11, 6], [4, 8, 11, 7], [4, 8, 11, 8], [4, 8, 11, 10], [4, 8, 11, 15], [4, 8, 13, 0], [4, 8, 13, 5], [4, 8, 13, 7], [4, 8, 13, 8], [4, 8, 13, 15], [4, 8, 15, 1], [4, 8, 15, 3], [4, 8, 15, 11], [4, 8, 15, 13], [4, 10, 1, 5], [4, 10, 5, 1], [4, 10, 5, 11], [4, 10, 11, 5], [4, 11, 0, 0], [4, 11, 0, 7], [4, 11, 0, 8], [4, 11, 0, 10], [4, 11, 0, 15], [4, 11, 3, 2], [4, 11, 3, 4], [4, 11, 3, 12], [4, 11, 3, 14], [4, 11, 4, 1], [4, 11, 4, 3], [4, 11, 4, 11], [4, 11, 4, 13], [4, 11, 7, 0], [4, 11, 7, 7], [4, 11, 7, 8], [4, 11, 7, 10], [4, 11, 7, 15], [4, 11, 8, 0], [4, 11, 8, 7], [4, 11, 8, 8], [4, 11, 8, 10], [4, 11, 8, 15], [4, 11, 10, 0], [4, 11, 10, 5], [4, 11, 10, 7], [4, 11, 10, 8], [4, 11, 10, 15], [4, 11, 13, 2], [4, 11, 13, 4], [4, 11, 13, 12], [4, 11, 13, 14], [4, 11, 14, 1], [4, 11, 14, 3], [4, 11, 14, 11], [4, 11, 14, 13], [4, 11, 15, 0], [4, 11, 15, 7], [4, 11, 15, 8], [4, 11, 15, 10], [4, 11, 15, 15], [4, 12, 3, 3], [4, 12, 3, 13], [4, 12, 6, 6], [4, 12, 6, 10], [4, 12, 13, 3], [4, 12, 13, 13], [4, 13, 0, 0], [4, 13, 0, 5], [4, 13, 0, 7], [4, 13, 0, 8], [4, 13, 0, 15], [4, 13, 1, 1], [4, 13, 1, 2], [4, 13, 1, 4], [4, 13, 1, 11], [4, 13, 1, 12], [4, 13, 1, 14], [4, 13, 2, 1], [4, 13, 2, 2], [4, 13, 2, 3], [4, 13, 2, 11], [4, 13, 2, 12], [4, 13, 2, 13], [4, 13, 5, 0], [4, 13, 5, 5], [4, 13, 5, 6], [4, 13, 5, 7], [4, 13, 5, 8], [4, 13, 5, 9], [4, 13, 5, 10], [4, 13, 5, 15], [4, 13, 7, 0], [4, 13, 7, 5], [4, 13, 7, 7], [4, 13, 7, 8], [4, 13, 7, 15], [4, 13, 8, 0], [4, 13, 8, 5], [4, 13, 8, 7], [4, 13, 8, 8], [4, 13, 8, 15], [4, 13, 11, 1], [4, 13, 11, 2], [4, 13, 11, 4], [4, 13, 11, 11], [4, 13, 11, 12], [4, 13, 11, 14], [4, 13, 12, 1], [4, 13, 12, 2], [4, 13, 12, 3], [4, 13, 12, 11], [4, 13, 12, 12], [4, 13, 12, 13], [4, 13, 15, 0], [4, 13, 15, 5], [4, 13, 15, 7], [4, 13, 15, 8], [4, 13, 15, 15], [4, 14, 1, 1], [4, 14, 1, 11], [4, 14, 6, 5], [4, 14, 11, 1], [4, 14, 11, 11], [4, 15, 0, 1], [4, 15, 0, 3], [4, 15, 0, 11], [4, 15, 0, 13], [4, 15, 1, 0], [4, 15, 1, 6], [4, 15, 1, 7], [4, 15, 1, 8], [4, 15, 1, 10], [4, 15, 1, 15], [4, 15, 3, 0], [4, 15, 3, 5], [4, 15, 3, 7], [4, 15, 3, 8], [4, 15, 3, 15], [4, 15, 6, 1], [4, 15, 6, 2], [4, 15, 6, 4], [4, 15, 6, 11], [4, 15, 6, 12], [4, 15, 6, 14], [4, 15, 7, 1], [4, 15, 7, 3], [4, 15, 7, 11], [4, 15, 7, 13], [4, 15, 8, 1], [4, 15, 8, 3], [4, 15, 8, 11], [4, 15, 8, 13], [4, 15, 11, 0], [4, 15, 11, 6], [4, 15, 11, 7], [4, 15, 11, 8], [4, 15, 11, 10], [4, 15, 11, 15], [4, 15, 13, 0], [4, 15, 13, 5], [4, 15, 13, 7], [4, 15, 13, 8], [4, 15, 13, 15], [4, 15, 15, 1], [4, 15, 15, 3], [4, 15, 15, 11], [4, 15, 15, 13], [5, 0, 0, 10], [5, 0, 3, 4], [5, 0, 3, 14], [5, 0, 4, 3], [5, 0, 4, 13], [5, 0, 7, 10], [5, 0, 8, 10], [5, 0, 10, 0], [5, 0, 10, 7], [5, 0, 10, 8], [5, 0, 10, 10], [5, 0, 10, 15], [5, 0, 13, 4], [5, 0, 13, 14], [5, 0, 14, 3], [5, 0, 14, 13], [5, 0, 15, 10], [5, 3, 0, 4], [5, 3, 0, 14], [5, 3, 2, 10], [5, 3, 3, 9], [5, 3, 4, 0], [5, 3, 4, 7], [5, 3, 4, 8], [5, 3, 4, 10], [5, 3, 4, 15], [5, 3, 7, 4], [5, 3, 7, 14], [5, 3, 8, 4], [5, 3, 8, 14], [5, 3, 9, 3], [5, 3, 9, 13], [5, 3, 10, 2], [5, 3, 10, 4], [5, 3, 10, 12], [5, 3, 10, 14], [5, 3, 12, 10], [5, 3, 13, 9], [5, 3, 14, 0], [5, 3, 14, 7], [5, 3, 14, 8], [5, 3, 14, 10], [5, 3, 14, 15], [5, 3, 15, 4], [5, 3, 15, 14], [5, 4, 0, 3], [5, 4, 0, 13], [5, 4, 1, 10], [5, 4, 3, 0], [5, 4, 3, 7], [5, 4, 3, 8], [5, 4, 3, 10], [5, 4, 3, 15], [5, 4, 4, 6], [5, 4, 6, 4], [5, 4, 6, 14], [5, 4, 7, 3], [5, 4, 7, 13], [5, 4, 8, 3], [5, 4, 8, 13], [5, 4, 10, 1], [5, 4, 10, 3], [5, 4, 10, 11], [5, 4, 10, 13], [5, 4, 11, 10], [5, 4, 13, 0], [5, 4, 13, 7], [5, 4, 13, 8], [5, 4, 13, 10], [5, 4, 13, 15], [5, 4, 14, 6], [5, 4, 15, 3], [5, 4, 15, 13], [5, 7, 0, 10], [5, 7, 3, 4], [5, 7, 3, 14], [5, 7, 4, 3], [5, 7, 4, 13], [5, 7, 7, 10], [5, 7, 8, 10], [5, 7, 10, 0], [5, 7, 10, 7], [5, 7, 10, 8], [5, 7, 10, 10], [5, 7, 10, 15], [5, 7, 13, 4], [5, 7, 13, 14], [5, 7, 14, 3], [5, 7, 14, 13], [5, 7, 15, 10], [5, 8, 0, 10], [5, 8, 3, 4], [5, 8, 3, 14], [5, 8, 4, 3], [5, 8, 4, 13], [5, 8, 7, 10], [5, 8, 8, 10], [5, 8, 10, 0], [5, 8, 10, 7], [5, 8, 10, 8], [5, 8, 10, 10], [5, 8, 10, 15], [5, 8, 13, 4], [5, 8, 13, 14], [5, 8, 14, 3], [5, 8, 14, 13], [5, 8, 15, 10], [5, 10, 0, 0], [5, 10, 0, 7], [5, 10, 0, 8], [5, 10, 0, 10], [5, 10, 0, 15], [5, 10, 1, 4], [5, 10, 1, 14], [5, 10, 2, 3], [5, 10, 2, 13], [5, 10, 3, 2], [5, 10, 3, 4], [5, 10, 3, 12], [5, 10, 3, 14], [5, 10, 4, 1], [5, 10, 4, 3], [5, 10, 4, 11], [5, 10, 4, 13], [5, 10, 5, 10], [5, 10, 6, 9], [5, 10, 7, 0], [5, 10, 7, 7], [5, 10, 7, 8], [5, 10, 7, 10], [5, 10, 7, 15], [5, 10, 8, 0], [5, 10, 8, 7], [5, 10, 8, 8], [5, 10, 8, 10], [5, 10, 8, 15], [5, 10, 9, 6], [5, 10, 10, 0], [5, 10, 10, 5], [5, 10, 10, 7], [5, 10, 10, 8], [5, 10, 10, 15], [5, 10, 11, 4], [5, 10, 11, 14], [5, 10, 12, 3], [5, 10, 12, 13], [5, 10, 13, 2], [5, 10, 13, 4], [5, 10, 13, 12], [5, 10, 13, 14], [5, 10, 14, 1], [5, 10, 14, 3], [5, 10, 14, 11], [5, 10, 14, 13], [5, 10, 15, 0], [5, 10, 15, 7], [5, 10, 15, 8], [5, 10, 15, 10], [5, 10, 15, 15], [5, 13, 0, 4], [5, 13, 0, 14], [5, 13, 2, 10], [5, 13, 3, 9], [5, 13, 4, 0], [5, 13, 4, 7], [5, 13, 4, 8], [5, 13, 4, 10], [5, 13, 4, 15], [5, 13, 7, 4], [5, 13, 7, 14], [5, 13, 8, 4], [5, 13, 8, 14], [5, 13, 9, 3], [5, 13, 9, 13], [5, 13, 10, 2], [5, 13, 10, 4], [5, 13, 10, 12], [5, 13, 10, 14], [5, 13, 12, 10], [5, 13, 13, 9], [5, 13, 14, 0], [5, 13, 14, 7], [5, 13, 14, 8], [5, 13, 14, 10], [5, 13, 14, 15], [5, 13, 15, 4], [5, 13, 15, 14], [5, 14, 0, 3], [5, 14, 0, 13], [5, 14, 1, 10], [5, 14, 3, 0], [5, 14, 3, 7], [5, 14, 3, 8], [5, 14, 3, 10], [5, 14, 3, 15], [5, 14, 4, 6], [5, 14, 6, 4], [5, 14, 6, 14], [5, 14, 7, 3], [5, 14, 7, 13], [5, 14, 8, 3], [5, 14, 8, 13], [5, 14, 10, 1], [5, 14, 10, 3], [5, 14, 10, 11], [5, 14, 10, 13], [5, 14, 11, 10], [5, 14, 13, 0], [5, 14, 13, 7], [5, 14, 13, 8], [5, 14, 13, 10], [5, 14, 13, 15], [5, 14, 14, 6], [5, 14, 15, 3], [5, 14, 15, 13], [5, 15, 0, 10], [5, 15, 3, 4], [5, 15, 3, 14], [5, 15, 4, 3], [5, 15, 4, 13], [5, 15, 7, 10], [5, 15, 8, 10], [5, 15, 10, 0], [5, 15, 10, 7], [5, 15, 10, 8], [5, 15, 10, 10], [5, 15, 10, 15], [5, 15, 13, 4], [5, 15, 13, 14], [5, 15, 14, 3], [5, 15, 14, 13], [5, 15, 15, 10], [6, 0, 0, 9], [6, 0, 2, 4], [6, 0, 2, 14], [6, 0, 4, 2], [6, 0, 4, 4], [6, 0, 4, 12], [6, 0, 4, 14], [6, 0, 7, 9], [6, 0, 8, 9], [6, 0, 9, 0], [6, 0, 9, 7], [6, 0, 9, 8], [6, 0, 9, 9], [6, 0, 9, 10], [6, 0, 9, 15], [6, 0, 12, 4], [6, 0, 12, 14], [6, 0, 14, 2], [6, 0, 14, 4], [6, 0, 14, 12], [6, 0, 14, 14], [6, 0, 15, 9], [6, 2, 0, 4], [6, 2, 0, 14], [6, 2, 3, 9], [6, 2, 4, 0], [6, 2, 4, 7], [6, 2, 4, 8], [6, 2, 4, 10], [6, 2, 4, 15], [6, 2, 7, 4], [6, 2, 7, 14], [6, 2, 8, 4], [6, 2, 8, 14], [6, 2, 10, 2], [6, 2, 10, 4], [6, 2, 10, 12], [6, 2, 10, 14], [6, 2, 13, 9], [6, 2, 14, 0], [6, 2, 14, 7], [6, 2, 14, 8], [6, 2, 14, 10], [6, 2, 14, 15], [6, 2, 15, 4], [6, 2, 15, 14], [6, 3, 2, 9], [6, 3, 9, 2], [6, 3, 9, 12], [6, 3, 12, 9], [6, 4, 0, 2], [6, 4, 0, 4], [6, 4, 0, 12], [6, 4, 0, 14], [6, 4, 1, 9], [6, 4, 2, 0], [6, 4, 2, 7], [6, 4, 2, 8], [6, 4, 2, 9], [6, 4, 2, 10], [6, 4, 2, 15], [6, 4, 4, 0], [6, 4, 4, 5], [6, 4, 4, 7], [6, 4, 4, 8], [6, 4, 4, 15], [6, 4, 5, 4], [6, 4, 5, 14], [6, 4, 7, 2], [6, 4, 7, 4], [6, 4, 7, 12], [6, 4, 7, 14], [6, 4, 8, 2], [6, 4, 8, 4], [6, 4, 8, 12], [6, 4, 8, 14], [6, 4, 9, 1], [6, 4, 9, 2], [6, 4, 9, 3], [6, 4, 9, 11], [6, 4, 9, 12], [6, 4, 9, 13], [6, 4, 11, 9], [6, 4, 12, 0], [6, 4, 12, 7], [6, 4, 12, 8], [6, 4, 12, 9], [6, 4, 12, 10], [6, 4, 12, 15], [6, 4, 14, 0], [6, 4, 14, 5], [6, 4, 14, 7], [6, 4, 14, 8], [6, 4, 14, 15], [6, 4, 15, 2], [6, 4, 15, 4], [6, 4, 15, 12], [6, 4, 15, 14], [6, 7, 0, 9], [6, 7, 2, 4], [6, 7, 2, 14], [6, 7, 4, 2], [6, 7, 4, 4], [6, 7, 4, 12], [6, 7, 4, 14], [6, 7, 7, 9], [6, 7, 8, 9], [6, 7, 9, 0], [6, 7, 9, 7], [6, 7, 9, 8], [6, 7, 9, 9], [6, 7, 9, 10], [6, 7, 9, 15], [6, 7, 12, 4], [6, 7, 12, 14], [6, 7, 14, 2], [6, 7, 14, 4], [6, 7, 14, 12], [6, 7, 14, 14], [6, 7, 15, 9], [6, 8, 0, 9], [6, 8, 2, 4], [6, 8, 2, 14], [6, 8, 4, 2], [6, 8, 4, 4], [6, 8, 4, 12], [6, 8, 4, 14], [6, 8, 7, 9], [6, 8, 8, 9], [6, 8, 9, 0], [6, 8, 9, 7], [6, 8, 9, 8], [6, 8, 9, 9], [6, 8, 9, 10], [6, 8, 9, 15], [6, 8, 12, 4], [6, 8, 12, 14], [6, 8, 14, 2], [6, 8, 14, 4], [6, 8, 14, 12], [6, 8, 14, 14], [6, 8, 15, 9], [6, 9, 0, 0], [6, 9, 0, 7], [6, 9, 0, 8], [6, 9, 0, 10], [6, 9, 0, 15], [6, 9, 1, 4], [6, 9, 1, 14], [6, 9, 3, 2], [6, 9, 3, 4], [6, 9, 3, 12], [6, 9, 3, 14], [6, 9, 4, 1], [6, 9, 4, 3], [6, 9, 4, 11], [6, 9, 4, 13], [6, 9, 6, 9], [6, 9, 7, 0], [6, 9, 7, 7], [6, 9, 7, 8], [6, 9, 7, 10], [6, 9, 7, 15], [6, 9, 8, 0], [6, 9, 8, 7], [6, 9, 8, 8], [6, 9, 8, 10], [6, 9, 8, 15], [6, 9, 10, 0], [6, 9, 10, 5], [6, 9, 10, 7], [6, 9, 10, 8], [6, 9, 10, 15], [6, 9, 11, 4], [6, 9, 11, 14], [6, 9, 13, 2], [6, 9, 13, 4], [6, 9, 13, 12], [6, 9, 13, 14], [6, 9, 14, 1], [6, 9, 14, 3], [6, 9, 14, 11], [6, 9, 14, 13], [6, 9, 15, 0], [6, 9, 15, 7], [6, 9, 15, 8], [6, 9, 15, 10], [6, 9, 15, 15], [6, 10, 2, 2], [6, 10, 2, 12], [6, 10, 5, 9], [6, 10, 9, 5], [6, 10, 12, 2], [6, 10, 12, 12], [6, 12, 0, 4], [6, 12, 0, 14], [6, 12, 3, 9], [6, 12, 4, 0], [6, 12, 4, 7], [6, 12, 4, 8], [6, 12, 4, 10], [6, 12, 4, 15], [6, 12, 7, 4], [6, 12, 7, 14], [6, 12, 8, 4], [6, 12, 8, 14], [6, 12, 10, 2], [6, 12, 10, 4], [6, 12, 10, 12], [6, 12, 10, 14], [6, 12, 13, 9], [6, 12, 14, 0], [6, 12, 14, 7], [6, 12, 14, 8], [6, 12, 14, 10], [6, 12, 14, 15], [6, 12, 15, 4], [6, 12, 15, 14], [6, 13, 2, 9], [6, 13, 9, 2], [6, 13, 9, 12], [6, 13, 12, 9], [6, 14, 0, 2], [6, 14, 0, 4], [6, 14, 0, 12], [6, 14, 0, 14], [6, 14, 1, 9], [6, 14, 2, 0], [6, 14, 2, 7], [6, 14, 2, 8], [6, 14, 2, 9], [6, 14, 2, 10], [6, 14, 2, 15], [6, 14, 4, 0], [6, 14, 4, 5], [6, 14, 4, 7], [6, 14, 4, 8], [6, 14, 4, 15], [6, 14, 5, 4], [6, 14, 5, 14], [6, 14, 7, 2], [6, 14, 7, 4], [6, 14, 7, 12], [6, 14, 7, 14], [6, 14, 8, 2], [6, 14, 8, 4], [6, 14, 8, 12], [6, 14, 8, 14], [6, 14, 9, 1], [6, 14, 9, 2], [6, 14, 9, 3], [6, 14, 9, 11], [6, 14, 9, 12], [6, 14, 9, 13], [6, 14, 11, 9], [6, 14, 12, 0], [6, 14, 12, 7], [6, 14, 12, 8], [6, 14, 12, 9], [6, 14, 12, 10], [6, 14, 12, 15], [6, 14, 14, 0], [6, 14, 14, 5], [6, 14, 14, 7], [6, 14, 14, 8], [6, 14, 14, 15], [6, 14, 15, 2], [6, 14, 15, 4], [6, 14, 15, 12], [6, 14, 15, 14], [6, 15, 0, 9], [6, 15, 2, 4], [6, 15, 2, 14], [6, 15, 4, 2], [6, 15, 4, 4], [6, 15, 4, 12], [6, 15, 4, 14], [6, 15, 7, 9], [6, 15, 8, 9], [6, 15, 9, 0], [6, 15, 9, 7], [6, 15, 9, 8], [6, 15, 9, 9], [6, 15, 9, 10], [6, 15, 9, 15], [6, 15, 12, 4], [6, 15, 12, 14], [6, 15, 14, 2], [6, 15, 14, 4], [6, 15, 14, 12], [6, 15, 14, 14], [6, 15, 15, 9], [7, 0, 0, 0], [7, 0, 0, 7], [7, 0, 0, 8], [7, 0, 0, 10], [7, 0, 0, 15], [7, 0, 3, 2], [7, 0, 3, 4], [7, 0, 3, 12], [7, 0, 3, 14], [7, 0, 4, 1], [7, 0, 4, 3], [7, 0, 4, 11], [7, 0, 4, 13], [7, 0, 7, 0], [7, 0, 7, 7], [7, 0, 7, 8], [7, 0, 7, 10], [7, 0, 7, 15], [7, 0, 8, 0], [7, 0, 8, 7], [7, 0, 8, 8], [7, 0, 8, 10], [7, 0, 8, 15], [7, 0, 10, 0], [7, 0, 10, 5], [7, 0, 10, 7], [7, 0, 10, 8], [7, 0, 10, 15], [7, 0, 13, 2], [7, 0, 13, 4], [7, 0, 13, 12], [7, 0, 13, 14], [7, 0, 14, 1], [7, 0, 14, 3], [7, 0, 14, 11], [7, 0, 14, 13], [7, 0, 15, 0], [7, 0, 15, 7], [7, 0, 15, 8], [7, 0, 15, 10], [7, 0, 15, 15], [7, 1, 4, 10], [7, 1, 10, 4], [7, 1, 10, 14], [7, 1, 14, 10], [7, 2, 3, 10], [7, 2, 10, 3], [7, 2, 10, 13], [7, 2, 13, 10], [7, 3, 0, 2], [7, 3, 0, 4], [7, 3, 0, 12], [7, 3, 0, 14], [7, 3, 2, 0], [7, 3, 2, 7], [7, 3, 2, 8], [7, 3, 2, 9], [7, 3, 2, 10], [7, 3, 2, 15], [7, 3, 4, 0], [7, 3, 4, 5], [7, 3, 4, 7], [7, 3, 4, 8], [7, 3, 4, 15], [7, 3, 7, 2], [7, 3, 7, 4], [7, 3, 7, 12], [7, 3, 7, 14], [7, 3, 8, 2], [7, 3, 8, 4], [7, 3, 8, 12], [7, 3, 8, 14], [7, 3, 9, 1], [7, 3, 9, 2], [7, 3, 9, 3], [7, 3, 9, 11], [7, 3, 9, 12], [7, 3, 9, 13], [7, 3, 12, 0], [7, 3, 12, 7], [7, 3, 12, 8], [7, 3, 12, 9], [7, 3, 12, 10], [7, 3, 12, 15], [7, 3, 14, 0], [7, 3, 14, 5], [7, 3, 14, 7], [7, 3, 14, 8], [7, 3, 14, 15], [7, 3, 15, 2], [7, 3, 15, 4], [7, 3, 15, 12], [7, 3, 15, 14], [7, 4, 0, 1], [7, 4, 0, 3], [7, 4, 0, 11], [7, 4, 0, 13], [7, 4, 1, 0], [7, 4, 1, 6], [7, 4, 1, 7], [7, 4, 1, 8], [7, 4, 1, 10], [7, 4, 1, 15], [7, 4, 3, 0], [7, 4, 3, 5], [7, 4, 3, 7], [7, 4, 3, 8], [7, 4, 3, 15], [7, 4, 6, 1], [7, 4, 6, 2], [7, 4, 6, 4], [7, 4, 6, 11], [7, 4, 6, 12], [7, 4, 6, 14], [7, 4, 7, 1], [7, 4, 7, 3], [7, 4, 7, 11], [7, 4, 7, 13], [7, 4, 8, 1], [7, 4, 8, 3], [7, 4, 8, 11], [7, 4, 8, 13], [7, 4, 11, 0], [7, 4, 11, 6], [7, 4, 11, 7], [7, 4, 11, 8], [7, 4, 11, 10], [7, 4, 11, 15], [7, 4, 13, 0], [7, 4, 13, 5], [7, 4, 13, 7], [7, 4, 13, 8], [7, 4, 13, 15], [7, 4, 15, 1], [7, 4, 15, 3], [7, 4, 15, 11], [7, 4, 15, 13], [7, 6, 4, 4], [7, 6, 4, 14], [7, 6, 9, 9], [7, 6, 9, 10], [7, 6, 14, 4], [7, 6, 14, 14], [7, 7, 0, 0], [7, 7, 0, 7], [7, 7, 0, 8], [7, 7, 0, 10], [7, 7, 0, 15], [7, 7, 3, 2], [7, 7, 3, 4], [7, 7, 3, 12], [7, 7, 3, 14], [7, 7, 4, 1], [7, 7, 4, 3], [7, 7, 4, 11], [7, 7, 4, 13], [7, 7, 7, 0], [7, 7, 7, 7], [7, 7, 7, 8], [7, 7, 7, 10], [7, 7, 7, 15], [7, 7, 8, 0], [7, 7, 8, 7], [7, 7, 8, 8], [7, 7, 8, 10], [7, 7, 8, 15], [7, 7, 10, 0], [7, 7, 10, 5], [7, 7, 10, 7], [7, 7, 10, 8], [7, 7, 10, 15], [7, 7, 13, 2], [7, 7, 13, 4], [7, 7, 13, 12], [7, 7, 13, 14], [7, 7, 14, 1], [7, 7, 14, 3], [7, 7, 14, 11], [7, 7, 14, 13], [7, 7, 15, 0], [7, 7, 15, 7], [7, 7, 15, 8], [7, 7, 15, 10], [7, 7, 15, 15], [7, 8, 0, 0], [7, 8, 0, 7], [7, 8, 0, 8], [7, 8, 0, 10], [7, 8, 0, 15], [7, 8, 3, 2], [7, 8, 3, 4], [7, 8, 3, 12], [7, 8, 3, 14], [7, 8, 4, 1], [7, 8, 4, 3], [7, 8, 4, 11], [7, 8, 4, 13], [7, 8, 7, 0], [7, 8, 7, 7], [7, 8, 7, 8], [7, 8, 7, 10], [7, 8, 7, 15], [7, 8, 8, 0], [7, 8, 8, 7], [7, 8, 8, 8], [7, 8, 8, 10], [7, 8, 8, 15], [7, 8, 10, 0], [7, 8, 10, 5], [7, 8, 10, 7], [7, 8, 10, 8], [7, 8, 10, 15], [7, 8, 13, 2], [7, 8, 13, 4], [7, 8, 13, 12], [7, 8, 13, 14], [7, 8, 14, 1], [7, 8, 14, 3], [7, 8, 14, 11], [7, 8, 14, 13], [7, 8, 15, 0], [7, 8, 15, 7], [7, 8, 15, 8], [7, 8, 15, 10], [7, 8, 15, 15], [7, 9, 3, 3], [7, 9, 3, 13], [7, 9, 6, 6], [7, 9, 6, 10], [7, 9, 13, 3], [7, 9, 13, 13], [7, 10, 0, 0], [7, 10, 0, 5], [7, 10, 0, 7], [7, 10, 0, 8], [7, 10, 0, 15], [7, 10, 1, 1], [7, 10, 1, 2], [7, 10, 1, 4], [7, 10, 1, 11], [7, 10, 1, 12], [7, 10, 1, 14], [7, 10, 2, 1], [7, 10, 2, 2], [7, 10, 2, 3], [7, 10, 2, 11], [7, 10, 2, 12], [7, 10, 2, 13], [7, 10, 5, 0], [7, 10, 5, 5], [7, 10, 5, 6], [7, 10, 5, 7], [7, 10, 5, 8], [7, 10, 5, 9], [7, 10, 5, 10], [7, 10, 5, 15], [7, 10, 7, 0], [7, 10, 7, 5], [7, 10, 7, 7], [7, 10, 7, 8], [7, 10, 7, 15], [7, 10, 8, 0], [7, 10, 8, 5], [7, 10, 8, 7], [7, 10, 8, 8], [7, 10, 8, 15], [7, 10, 11, 1], [7, 10, 11, 2], [7, 10, 11, 4], [7, 10, 11, 11], [7, 10, 11, 12], [7, 10, 11, 14], [7, 10, 12, 1], [7, 10, 12, 2], [7, 10, 12, 3], [7, 10, 12, 11], [7, 10, 12, 12], [7, 10, 12, 13], [7, 10, 15, 0], [7, 10, 15, 5], [7, 10, 15, 7], [7, 10, 15, 8], [7, 10, 15, 15], [7, 11, 4, 10], [7, 11, 10, 4], [7, 11, 10, 14], [7, 11, 14, 10], [7, 12, 3, 10], [7, 12, 10, 3], [7, 12, 10, 13], [7, 12, 13, 10], [7, 13, 0, 2], [7, 13, 0, 4], [7, 13, 0, 12], [7, 13, 0, 14], [7, 13, 2, 0], [7, 13, 2, 7], [7, 13, 2, 8], [7, 13, 2, 9], [7, 13, 2, 10], [7, 13, 2, 15], [7, 13, 4, 0], [7, 13, 4, 5], [7, 13, 4, 7], [7, 13, 4, 8], [7, 13, 4, 15], [7, 13, 7, 2], [7, 13, 7, 4], [7, 13, 7, 12], [7, 13, 7, 14], [7, 13, 8, 2], [7, 13, 8, 4], [7, 13, 8, 12], [7, 13, 8, 14], [7, 13, 9, 1], [7, 13, 9, 2], [7, 13, 9, 3], [7, 13, 9, 11], [7, 13, 9, 12], [7, 13, 9, 13], [7, 13, 12, 0], [7, 13, 12, 7], [7, 13, 12, 8], [7, 13, 12, 9], [7, 13, 12, 10], [7, 13, 12, 15], [7, 13, 14, 0], [7, 13, 14, 5], [7, 13, 14, 7], [7, 13, 14, 8], [7, 13, 14, 15], [7, 13, 15, 2], [7, 13, 15, 4], [7, 13, 15, 12], [7, 13, 15, 14], [7, 14, 0, 1], [7, 14, 0, 3], [7, 14, 0, 11], [7, 14, 0, 13], [7, 14, 1, 0], [7, 14, 1, 6], [7, 14, 1, 7], [7, 14, 1, 8], [7, 14, 1, 10], [7, 14, 1, 15], [7, 14, 3, 0], [7, 14, 3, 5], [7, 14, 3, 7], [7, 14, 3, 8], [7, 14, 3, 15], [7, 14, 6, 1], [7, 14, 6, 2], [7, 14, 6, 4], [7, 14, 6, 11], [7, 14, 6, 12], [7, 14, 6, 14], [7, 14, 7, 1], [7, 14, 7, 3], [7, 14, 7, 11], [7, 14, 7, 13], [7, 14, 8, 1], [7, 14, 8, 3], [7, 14, 8, 11], [7, 14, 8, 13], [7, 14, 11, 0], [7, 14, 11, 6], [7, 14, 11, 7], [7, 14, 11, 8], [7, 14, 11, 10], [7, 14, 11, 15], [7, 14, 13, 0], [7, 14, 13, 5], [7, 14, 13, 7], [7, 14, 13, 8], [7, 14, 13, 15], [7, 14, 15, 1], [7, 14, 15, 3], [7, 14, 15, 11], [7, 14, 15, 13], [7, 15, 0, 0], [7, 15, 0, 7], [7, 15, 0, 8], [7, 15, 0, 10], [7, 15, 0, 15], [7, 15, 3, 2], [7, 15, 3, 4], [7, 15, 3, 12], [7, 15, 3, 14], [7, 15, 4, 1], [7, 15, 4, 3], [7, 15, 4, 11], [7, 15, 4, 13], [7, 15, 7, 0], [7, 15, 7, 7], [7, 15, 7, 8], [7, 15, 7, 10], [7, 15, 7, 15], [7, 15, 8, 0], [7, 15, 8, 7], [7, 15, 8, 8], [7, 15, 8, 10], [7, 15, 8, 15], [7, 15, 10, 0], [7, 15, 10, 5], [7, 15, 10, 7], [7, 15, 10, 8], [7, 15, 10, 15], [7, 15, 13, 2], [7, 15, 13, 4], [7, 15, 13, 12], [7, 15, 13, 14], [7, 15, 14, 1], [7, 15, 14, 3], [7, 15, 14, 11], [7, 15, 14, 13], [7, 15, 15, 0], [7, 15, 15, 7], [7, 15, 15, 8], [7, 15, 15, 10], [7, 15, 15, 15], [8, 0, 0, 0], [8, 0, 0, 7], [8, 0, 0, 8], [8, 0, 0, 10], [8, 0, 0, 15], [8, 0, 3, 2], [8, 0, 3, 4], [8, 0, 3, 12], [8, 0, 3, 14], [8, 0, 4, 1], [8, 0, 4, 3], [8, 0, 4, 11], [8, 0, 4, 13], [8, 0, 7, 0], [8, 0, 7, 7], [8, 0, 7, 8], [8, 0, 7, 10], [8, 0, 7, 15], [8, 0, 8, 0], [8, 0, 8, 7], [8, 0, 8, 8], [8, 0, 8, 10], [8, 0, 8, 15], [8, 0, 10, 0], [8, 0, 10, 5], [8, 0, 10, 7], [8, 0, 10, 8], [8, 0, 10, 15], [8, 0, 13, 2], [8, 0, 13, 4], [8, 0, 13, 12], [8, 0, 13, 14], [8, 0, 14, 1], [8, 0, 14, 3], [8, 0, 14, 11], [8, 0, 14, 13], [8, 0, 15, 0], [8, 0, 15, 7], [8, 0, 15, 8], [8, 0, 15, 10], [8, 0, 15, 15], [8, 1, 4, 10], [8, 1, 10, 4], [8, 1, 10, 14], [8, 1, 14, 10], [8, 2, 3, 10], [8, 2, 10, 3], [8, 2, 10, 13], [8, 2, 13, 10], [8, 3, 0, 2], [8, 3, 0, 4], [8, 3, 0, 12], [8, 3, 0, 14], [8, 3, 2, 0], [8, 3, 2, 7], [8, 3, 2, 8], [8, 3, 2, 9], [8, 3, 2, 10], [8, 3, 2, 15], [8, 3, 4, 0], [8, 3, 4, 5], [8, 3, 4, 7], [8, 3, 4, 8], [8, 3, 4, 15], [8, 3, 7, 2], [8, 3, 7, 4], [8, 3, 7, 12], [8, 3, 7, 14], [8, 3, 8, 2], [8, 3, 8, 4], [8, 3, 8, 12], [8, 3, 8, 14], [8, 3, 9, 1], [8, 3, 9, 2], [8, 3, 9, 3], [8, 3, 9, 11], [8, 3, 9, 12], [8, 3, 9, 13], [8, 3, 12, 0], [8, 3, 12, 7], [8, 3, 12, 8], [8, 3, 12, 9], [8, 3, 12, 10], [8, 3, 12, 15], [8, 3, 14, 0], [8, 3, 14, 5], [8, 3, 14, 7], [8, 3, 14, 8], [8, 3, 14, 15], [8, 3, 15, 2], [8, 3, 15, 4], [8, 3, 15, 12], [8, 3, 15, 14], [8, 4, 0, 1], [8, 4, 0, 3], [8, 4, 0, 11], [8, 4, 0, 13], [8, 4, 1, 0], [8, 4, 1, 6], [8, 4, 1, 7], [8, 4, 1, 8], [8, 4, 1, 10], [8, 4, 1, 15], [8, 4, 3, 0], [8, 4, 3, 5], [8, 4, 3, 7], [8, 4, 3, 8], [8, 4, 3, 15], [8, 4, 6, 1], [8, 4, 6, 2], [8, 4, 6, 4], [8, 4, 6, 11], [8, 4, 6, 12], [8, 4, 6, 14], [8, 4, 7, 1], [8, 4, 7, 3], [8, 4, 7, 11], [8, 4, 7, 13], [8, 4, 8, 1], [8, 4, 8, 3], [8, 4, 8, 11], [8, 4, 8, 13], [8, 4, 11, 0], [8, 4, 11, 6], [8, 4, 11, 7], [8, 4, 11, 8], [8, 4, 11, 10], [8, 4, 11, 15], [8, 4, 13, 0], [8, 4, 13, 5], [8, 4, 13, 7], [8, 4, 13, 8], [8, 4, 13, 15], [8, 4, 15, 1], [8, 4, 15, 3], [8, 4, 15, 11], [8, 4, 15, 13], [8, 6, 4, 4], [8, 6, 4, 14], [8, 6, 9, 9], [8, 6, 9, 10], [8, 6, 14, 4], [8, 6, 14, 14], [8, 7, 0, 0], [8, 7, 0, 7], [8, 7, 0, 8], [8, 7, 0, 10], [8, 7, 0, 15], [8, 7, 3, 2], [8, 7, 3, 4], [8, 7, 3, 12], [8, 7, 3, 14], [8, 7, 4, 1], [8, 7, 4, 3], [8, 7, 4, 11], [8, 7, 4, 13], [8, 7, 7, 0], [8, 7, 7, 7], [8, 7, 7, 8], [8, 7, 7, 10], [8, 7, 7, 15], [8, 7, 8, 0], [8, 7, 8, 7], [8, 7, 8, 8], [8, 7, 8, 10], [8, 7, 8, 15], [8, 7, 10, 0], [8, 7, 10, 5], [8, 7, 10, 7], [8, 7, 10, 8], [8, 7, 10, 15], [8, 7, 13, 2], [8, 7, 13, 4], [8, 7, 13, 12], [8, 7, 13, 14], [8, 7, 14, 1], [8, 7, 14, 3], [8, 7, 14, 11], [8, 7, 14, 13], [8, 7, 15, 0], [8, 7, 15, 7], [8, 7, 15, 8], [8, 7, 15, 10], [8, 7, 15, 15], [8, 8, 0, 0], [8, 8, 0, 7], [8, 8, 0, 8], [8, 8, 0, 10], [8, 8, 0, 15], [8, 8, 3, 2], [8, 8, 3, 4], [8, 8, 3, 12], [8, 8, 3, 14], [8, 8, 4, 1], [8, 8, 4, 3], [8, 8, 4, 11], [8, 8, 4, 13], [8, 8, 7, 0], [8, 8, 7, 7], [8, 8, 7, 8], [8, 8, 7, 10], [8, 8, 7, 15], [8, 8, 8, 0], [8, 8, 8, 7], [8, 8, 8, 8], [8, 8, 8, 10], [8, 8, 8, 15], [8, 8, 10, 0], [8, 8, 10, 5], [8, 8, 10, 7], [8, 8, 10, 8], [8, 8, 10, 15], [8, 8, 13, 2], [8, 8, 13, 4], [8, 8, 13, 12], [8, 8, 13, 14], [8, 8, 14, 1], [8, 8, 14, 3], [8, 8, 14, 11], [8, 8, 14, 13], [8, 8, 15, 0], [8, 8, 15, 7], [8, 8, 15, 8], [8, 8, 15, 10], [8, 8, 15, 15], [8, 9, 3, 3], [8, 9, 3, 13], [8, 9, 6, 6], [8, 9, 6, 10], [8, 9, 13, 3], [8, 9, 13, 13], [8, 10, 0, 0], [8, 10, 0, 5], [8, 10, 0, 7], [8, 10, 0, 8], [8, 10, 0, 15], [8, 10, 1, 1], [8, 10, 1, 2], [8, 10, 1, 4], [8, 10, 1, 11], [8, 10, 1, 12], [8, 10, 1, 14], [8, 10, 2, 1], [8, 10, 2, 2], [8, 10, 2, 3], [8, 10, 2, 11], [8, 10, 2, 12], [8, 10, 2, 13], [8, 10, 5, 0], [8, 10, 5, 5], [8, 10, 5, 6], [8, 10, 5, 7], [8, 10, 5, 8], [8, 10, 5, 9], [8, 10, 5, 10], [8, 10, 5, 15], [8, 10, 7, 0], [8, 10, 7, 5], [8, 10, 7, 7], [8, 10, 7, 8], [8, 10, 7, 15], [8, 10, 8, 0], [8, 10, 8, 5], [8, 10, 8, 7], [8, 10, 8, 8], [8, 10, 8, 15], [8, 10, 11, 1], [8, 10, 11, 2], [8, 10, 11, 4], [8, 10, 11, 11], [8, 10, 11, 12], [8, 10, 11, 14], [8, 10, 12, 1], [8, 10, 12, 2], [8, 10, 12, 3], [8, 10, 12, 11], [8, 10, 12, 12], [8, 10, 12, 13], [8, 10, 15, 0], [8, 10, 15, 5], [8, 10, 15, 7], [8, 10, 15, 8], [8, 10, 15, 15], [8, 11, 4, 10], [8, 11, 10, 4], [8, 11, 10, 14], [8, 11, 14, 10], [8, 12, 3, 10], [8, 12, 10, 3], [8, 12, 10, 13], [8, 12, 13, 10], [8, 13, 0, 2], [8, 13, 0, 4], [8, 13, 0, 12], [8, 13, 0, 14], [8, 13, 2, 0], [8, 13, 2, 7], [8, 13, 2, 8], [8, 13, 2, 9], [8, 13, 2, 10], [8, 13, 2, 15], [8, 13, 4, 0], [8, 13, 4, 5], [8, 13, 4, 7], [8, 13, 4, 8], [8, 13, 4, 15], [8, 13, 7, 2], [8, 13, 7, 4], [8, 13, 7, 12], [8, 13, 7, 14], [8, 13, 8, 2], [8, 13, 8, 4], [8, 13, 8, 12], [8, 13, 8, 14], [8, 13, 9, 1], [8, 13, 9, 2], [8, 13, 9, 3], [8, 13, 9, 11], [8, 13, 9, 12], [8, 13, 9, 13], [8, 13, 12, 0], [8, 13, 12, 7], [8, 13, 12, 8], [8, 13, 12, 9], [8, 13, 12, 10], [8, 13, 12, 15], [8, 13, 14, 0], [8, 13, 14, 5], [8, 13, 14, 7], [8, 13, 14, 8], [8, 13, 14, 15], [8, 13, 15, 2], [8, 13, 15, 4], [8, 13, 15, 12], [8, 13, 15, 14], [8, 14, 0, 1], [8, 14, 0, 3], [8, 14, 0, 11], [8, 14, 0, 13], [8, 14, 1, 0], [8, 14, 1, 6], [8, 14, 1, 7], [8, 14, 1, 8], [8, 14, 1, 10], [8, 14, 1, 15], [8, 14, 3, 0], [8, 14, 3, 5], [8, 14, 3, 7], [8, 14, 3, 8], [8, 14, 3, 15], [8, 14, 6, 1], [8, 14, 6, 2], [8, 14, 6, 4], [8, 14, 6, 11], [8, 14, 6, 12], [8, 14, 6, 14], [8, 14, 7, 1], [8, 14, 7, 3], [8, 14, 7, 11], [8, 14, 7, 13], [8, 14, 8, 1], [8, 14, 8, 3], [8, 14, 8, 11], [8, 14, 8, 13], [8, 14, 11, 0], [8, 14, 11, 6], [8, 14, 11, 7], [8, 14, 11, 8], [8, 14, 11, 10], [8, 14, 11, 15], [8, 14, 13, 0], [8, 14, 13, 5], [8, 14, 13, 7], [8, 14, 13, 8], [8, 14, 13, 15], [8, 14, 15, 1], [8, 14, 15, 3], [8, 14, 15, 11], [8, 14, 15, 13], [8, 15, 0, 0], [8, 15, 0, 7], [8, 15, 0, 8], [8, 15, 0, 10], [8, 15, 0, 15], [8, 15, 3, 2], [8, 15, 3, 4], [8, 15, 3, 12], [8, 15, 3, 14], [8, 15, 4, 1], [8, 15, 4, 3], [8, 15, 4, 11], [8, 15, 4, 13], [8, 15, 7, 0], [8, 15, 7, 7], [8, 15, 7, 8], [8, 15, 7, 10], [8, 15, 7, 15], [8, 15, 8, 0], [8, 15, 8, 7], [8, 15, 8, 8], [8, 15, 8, 10], [8, 15, 8, 15], [8, 15, 10, 0], [8, 15, 10, 5], [8, 15, 10, 7], [8, 15, 10, 8], [8, 15, 10, 15], [8, 15, 13, 2], [8, 15, 13, 4], [8, 15, 13, 12], [8, 15, 13, 14], [8, 15, 14, 1], [8, 15, 14, 3], [8, 15, 14, 11], [8, 15, 14, 13], [8, 15, 15, 0], [8, 15, 15, 7], [8, 15, 15, 8], [8, 15, 15, 10], [8, 15, 15, 15], [9, 0, 0, 6], [9, 0, 1, 3], [9, 0, 1, 13], [9, 0, 3, 1], [9, 0, 3, 3], [9, 0, 3, 11], [9, 0, 3, 13], [9, 0, 6, 0], [9, 0, 6, 6], [9, 0, 6, 7], [9, 0, 6, 8], [9, 0, 6, 10], [9, 0, 6, 15], [9, 0, 7, 6], [9, 0, 8, 6], [9, 0, 11, 3], [9, 0, 11, 13], [9, 0, 13, 1], [9, 0, 13, 3], [9, 0, 13, 11], [9, 0, 13, 13], [9, 0, 15, 6], [9, 1, 0, 3], [9, 1, 0, 13], [9, 1, 3, 0], [9, 1, 3, 7], [9, 1, 3, 8], [9, 1, 3, 10], [9, 1, 3, 15], [9, 1, 4, 6], [9, 1, 7, 3], [9, 1, 7, 13], [9, 1, 8, 3], [9, 1, 8, 13], [9, 1, 10, 1], [9, 1, 10, 3], [9, 1, 10, 11], [9, 1, 10, 13], [9, 1, 13, 0], [9, 1, 13, 7], [9, 1, 13, 8], [9, 1, 13, 10], [9, 1, 13, 15], [9, 1, 14, 6], [9, 1, 15, 3], [9, 1, 15, 13], [9, 3, 0, 1], [9, 3, 0, 3], [9, 3, 0, 11], [9, 3, 0, 13], [9, 3, 1, 0], [9, 3, 1, 6], [9, 3, 1, 7], [9, 3, 1, 8], [9, 3, 1, 10], [9, 3, 1, 15], [9, 3, 2, 6], [9, 3, 3, 0], [9, 3, 3, 5], [9, 3, 3, 7], [9, 3, 3, 8], [9, 3, 3, 15], [9, 3, 5, 3], [9, 3, 5, 13], [9, 3, 6, 1], [9, 3, 6, 2], [9, 3, 6, 4], [9, 3, 6, 11], [9, 3, 6, 12], [9, 3, 6, 14], [9, 3, 7, 1], [9, 3, 7, 3], [9, 3, 7, 11], [9, 3, 7, 13], [9, 3, 8, 1], [9, 3, 8, 3], [9, 3, 8, 11], [9, 3, 8, 13], [9, 3, 11, 0], [9, 3, 11, 6], [9, 3, 11, 7], [9, 3, 11, 8], [9, 3, 11, 10], [9, 3, 11, 15], [9, 3, 12, 6], [9, 3, 13, 0], [9, 3, 13, 5], [9, 3, 13, 7], [9, 3, 13, 8], [9, 3, 13, 15], [9, 3, 15, 1], [9, 3, 15, 3], [9, 3, 15, 11], [9, 3, 15, 13], [9, 4, 1, 6], [9, 4, 6, 1], [9, 4, 6, 11], [9, 4, 11, 6], [9, 6, 0, 0], [9, 6, 0, 7], [9, 6, 0, 8], [9, 6, 0, 10], [9, 6, 0, 15], [9, 6, 2, 3], [9, 6, 2, 13], [9, 6, 3, 2], [9, 6, 3, 4], [9, 6, 3, 12], [9, 6, 3, 14], [9, 6, 4, 1], [9, 6, 4, 3], [9, 6, 4, 11], [9, 6, 4, 13], [9, 6, 7, 0], [9, 6, 7, 7], [9, 6, 7, 8], [9, 6, 7, 10], [9, 6, 7, 15], [9, 6, 8, 0], [9, 6, 8, 7], [9, 6, 8, 8], [9, 6, 8, 10], [9, 6, 8, 15], [9, 6, 9, 6], [9, 6, 10, 0], [9, 6, 10, 5], [9, 6, 10, 7], [9, 6, 10, 8], [9, 6, 10, 15], [9, 6, 12, 3], [9, 6, 12, 13], [9, 6, 13, 2], [9, 6, 13, 4], [9, 6, 13, 12], [9, 6, 13, 14], [9, 6, 14, 1], [9, 6, 14, 3], [9, 6, 14, 11], [9, 6, 14, 13], [9, 6, 15, 0], [9, 6, 15, 7], [9, 6, 15, 8], [9, 6, 15, 10], [9, 6, 15, 15], [9, 7, 0, 6], [9, 7, 1, 3], [9, 7, 1, 13], [9, 7, 3, 1], [9, 7, 3, 3], [9, 7, 3, 11], [9, 7, 3, 13], [9, 7, 6, 0], [9, 7, 6, 6], [9, 7, 6, 7], [9, 7, 6, 8], [9, 7, 6, 10], [9, 7, 6, 15], [9, 7, 7, 6], [9, 7, 8, 6], [9, 7, 11, 3], [9, 7, 11, 13], [9, 7, 13, 1], [9, 7, 13, 3], [9, 7, 13, 11], [9, 7, 13, 13], [9, 7, 15, 6], [9, 8, 0, 6], [9, 8, 1, 3], [9, 8, 1, 13], [9, 8, 3, 1], [9, 8, 3, 3], [9, 8, 3, 11], [9, 8, 3, 13], [9, 8, 6, 0], [9, 8, 6, 6], [9, 8, 6, 7], [9, 8, 6, 8], [9, 8, 6, 10], [9, 8, 6, 15], [9, 8, 7, 6], [9, 8, 8, 6], [9, 8, 11, 3], [9, 8, 11, 13], [9, 8, 13, 1], [9, 8, 13, 3], [9, 8, 13, 11], [9, 8, 13, 13], [9, 8, 15, 6], [9, 10, 1, 1], [9, 10, 1, 11], [9, 10, 5, 6], [9, 10, 6, 5], [9, 10, 11, 1], [9, 10, 11, 11], [9, 11, 0, 3], [9, 11, 0, 13], [9, 11, 3, 0], [9, 11, 3, 7], [9, 11, 3, 8], [9, 11, 3, 10], [9, 11, 3, 15], [9, 11, 4, 6], [9, 11, 7, 3], [9, 11, 7, 13], [9, 11, 8, 3], [9, 11, 8, 13], [9, 11, 10, 1], [9, 11, 10, 3], [9, 11, 10, 11], [9, 11, 10, 13], [9, 11, 13, 0], [9, 11, 13, 7], [9, 11, 13, 8], [9, 11, 13, 10], [9, 11, 13, 15], [9, 11, 14, 6], [9, 11, 15, 3], [9, 11, 15, 13], [9, 13, 0, 1], [9, 13, 0, 3], [9, 13, 0, 11], [9, 13, 0, 13], [9, 13, 1, 0], [9, 13, 1, 6], [9, 13, 1, 7], [9, 13, 1, 8], [9, 13, 1, 10], [9, 13, 1, 15], [9, 13, 2, 6], [9, 13, 3, 0], [9, 13, 3, 5], [9, 13, 3, 7], [9, 13, 3, 8], [9, 13, 3, 15], [9, 13, 5, 3], [9, 13, 5, 13], [9, 13, 6, 1], [9, 13, 6, 2], [9, 13, 6, 4], [9, 13, 6, 11], [9, 13, 6, 12], [9, 13, 6, 14], [9, 13, 7, 1], [9, 13, 7, 3], [9, 13, 7, 11], [9, 13, 7, 13], [9, 13, 8, 1], [9, 13, 8, 3], [9, 13, 8, 11], [9, 13, 8, 13], [9, 13, 11, 0], [9, 13, 11, 6], [9, 13, 11, 7], [9, 13, 11, 8], [9, 13, 11, 10], [9, 13, 11, 15], [9, 13, 12, 6], [9, 13, 13, 0], [9, 13, 13, 5], [9, 13, 13, 7], [9, 13, 13, 8], [9, 13, 13, 15], [9, 13, 15, 1], [9, 13, 15, 3], [9, 13, 15, 11], [9, 13, 15, 13], [9, 14, 1, 6], [9, 14, 6, 1], [9, 14, 6, 11], [9, 14, 11, 6], [9, 15, 0, 6], [9, 15, 1, 3], [9, 15, 1, 13], [9, 15, 3, 1], [9, 15, 3, 3], [9, 15, 3, 11], [9, 15, 3, 13], [9, 15, 6, 0], [9, 15, 6, 6], [9, 15, 6, 7], [9, 15, 6, 8], [9, 15, 6, 10], [9, 15, 6, 15], [9, 15, 7, 6], [9, 15, 8, 6], [9, 15, 11, 3], [9, 15, 11, 13], [9, 15, 13, 1], [9, 15, 13, 3], [9, 15, 13, 11], [9, 15, 13, 13], [9, 15, 15, 6], [10, 0, 0, 0], [10, 0, 0, 5], [10, 0, 0, 7], [10, 0, 0, 8], [10, 0, 0, 15], [10, 0, 1, 1], [10, 0, 1, 2], [10, 0, 1, 4], [10, 0, 1, 11], [10, 0, 1, 12], [10, 0, 1, 14], [10, 0, 2, 1], [10, 0, 2, 2], [10, 0, 2, 3], [10, 0, 2, 11], [10, 0, 2, 12], [10, 0, 2, 13], [10, 0, 5, 0], [10, 0, 5, 5], [10, 0, 5, 6], [10, 0, 5, 7], [10, 0, 5, 8], [10, 0, 5, 9], [10, 0, 5, 10], [10, 0, 5, 15], [10, 0, 7, 0], [10, 0, 7, 5], [10, 0, 7, 7], [10, 0, 7, 8], [10, 0, 7, 15], [10, 0, 8, 0], [10, 0, 8, 5], [10, 0, 8, 7], [10, 0, 8, 8], [10, 0, 8, 15], [10, 0, 11, 1], [10, 0, 11, 2], [10, 0, 11, 4], [10, 0, 11, 11], [10, 0, 11, 12], [10, 0, 11, 14], [10, 0, 12, 1], [10, 0, 12, 2], [10, 0, 12, 3], [10, 0, 12, 11], [10, 0, 12, 12], [10, 0, 12, 13], [10, 0, 15, 0], [10, 0, 15, 5], [10, 0, 15, 7], [10, 0, 15, 8], [10, 0, 15, 15], [10, 1, 0, 2], [10, 1, 0, 4], [10, 1, 0, 12], [10, 1, 0, 14], [10, 1, 2, 0], [10, 1, 2, 7], [10, 1, 2, 8], [10, 1, 2, 9], [10, 1, 2, 10], [10, 1, 2, 15], [10, 1, 4, 0], [10, 1, 4, 5], [10, 1, 4, 7], [10, 1, 4, 8], [10, 1, 4, 15], [10, 1, 7, 2], [10, 1, 7, 4], [10, 1, 7, 12], [10, 1, 7, 14], [10, 1, 8, 2], [10, 1, 8, 4], [10, 1, 8, 12], [10, 1, 8, 14], [10, 1, 9, 1], [10, 1, 9, 2], [10, 1, 9, 3], [10, 1, 9, 11], [10, 1, 9, 12], [10, 1, 9, 13], [10, 1, 12, 0], [10, 1, 12, 7], [10, 1, 12, 8], [10, 1, 12, 9], [10, 1, 12, 10], [10, 1, 12, 15], [10, 1, 14, 0], [10, 1, 14, 5], [10, 1, 14, 7], [10, 1, 14, 8], [10, 1, 14, 15], [10, 1, 15, 2], [10, 1, 15, 4], [10, 1, 15, 12], [10, 1, 15, 14], [10, 2, 0, 1], [10, 2, 0, 3], [10, 2, 0, 11], [10, 2, 0, 13], [10, 2, 1, 0], [10, 2, 1, 6], [10, 2, 1, 7], [10, 2, 1, 8], [10, 2, 1, 10], [10, 2, 1, 15], [10, 2, 3, 0], [10, 2, 3, 5], [10, 2, 3, 7], [10, 2, 3, 8], [10, 2, 3, 15], [10, 2, 6, 1], [10, 2, 6, 2], [10, 2, 6, 4], [10, 2, 6, 11], [10, 2, 6, 12], [10, 2, 6, 14], [10, 2, 7, 1], [10, 2, 7, 3], [10, 2, 7, 11], [10, 2, 7, 13], [10, 2, 8, 1], [10, 2, 8, 3], [10, 2, 8, 11], [10, 2, 8, 13], [10, 2, 11, 0], [10, 2, 11, 6], [10, 2, 11, 7], [10, 2, 11, 8], [10, 2, 11, 10], [10, 2, 11, 15], [10, 2, 13, 0], [10, 2, 13, 5], [10, 2, 13, 7], [10, 2, 13, 8], [10, 2, 13, 15], [10, 2, 15, 1], [10, 2, 15, 3], [10, 2, 15, 11], [10, 2, 15, 13], [10, 3, 2, 5], [10, 3, 5, 2], [10, 3, 5, 12], [10, 3, 12, 5], [10, 4, 1, 5], [10, 4, 5, 1], [10, 4, 5, 11], [10, 4, 11, 5], [10, 5, 0, 0], [10, 5, 0, 7], [10, 5, 0, 8], [10, 5, 0, 10], [10, 5, 0, 15], [10, 5, 3, 2], [10, 5, 3, 4], [10, 5, 3, 12], [10, 5, 3, 14], [10, 5, 4, 1], [10, 5, 4, 3], [10, 5, 4, 11], [10, 5, 4, 13], [10, 5, 7, 0], [10, 5, 7, 7], [10, 5, 7, 8], [10, 5, 7, 10], [10, 5, 7, 15], [10, 5, 8, 0], [10, 5, 8, 7], [10, 5, 8, 8], [10, 5, 8, 10], [10, 5, 8, 15], [10, 5, 10, 0], [10, 5, 10, 5], [10, 5, 10, 7], [10, 5, 10, 8], [10, 5, 10, 15], [10, 5, 13, 2], [10, 5, 13, 4], [10, 5, 13, 12], [10, 5, 13, 14], [10, 5, 14, 1], [10, 5, 14, 3], [10, 5, 14, 11], [10, 5, 14, 13], [10, 5, 15, 0], [10, 5, 15, 7], [10, 5, 15, 8], [10, 5, 15, 10], [10, 5, 15, 15], [10, 6, 2, 2], [10, 6, 2, 12], [10, 6, 9, 5], [10, 6, 12, 2], [10, 6, 12, 12], [10, 7, 0, 0], [10, 7, 0, 5], [10, 7, 0, 7], [10, 7, 0, 8], [10, 7, 0, 15], [10, 7, 1, 1], [10, 7, 1, 2], [10, 7, 1, 4], [10, 7, 1, 11], [10, 7, 1, 12], [10, 7, 1, 14], [10, 7, 2, 1], [10, 7, 2, 2], [10, 7, 2, 3], [10, 7, 2, 11], [10, 7, 2, 12], [10, 7, 2, 13], [10, 7, 5, 0], [10, 7, 5, 5], [10, 7, 5, 6], [10, 7, 5, 7], [10, 7, 5, 8], [10, 7, 5, 9], [10, 7, 5, 10], [10, 7, 5, 15], [10, 7, 7, 0], [10, 7, 7, 5], [10, 7, 7, 7], [10, 7, 7, 8], [10, 7, 7, 15], [10, 7, 8, 0], [10, 7, 8, 5], [10, 7, 8, 7], [10, 7, 8, 8], [10, 7, 8, 15], [10, 7, 11, 1], [10, 7, 11, 2], [10, 7, 11, 4], [10, 7, 11, 11], [10, 7, 11, 12], [10, 7, 11, 14], [10, 7, 12, 1], [10, 7, 12, 2], [10, 7, 12, 3], [10, 7, 12, 11], [10, 7, 12, 12], [10, 7, 12, 13], [10, 7, 15, 0], [10, 7, 15, 5], [10, 7, 15, 7], [10, 7, 15, 8], [10, 7, 15, 15], [10, 8, 0, 0], [10, 8, 0, 5], [10, 8, 0, 7], [10, 8, 0, 8], [10, 8, 0, 15], [10, 8, 1, 1], [10, 8, 1, 2], [10, 8, 1, 4], [10, 8, 1, 11], [10, 8, 1, 12], [10, 8, 1, 14], [10, 8, 2, 1], [10, 8, 2, 2], [10, 8, 2, 3], [10, 8, 2, 11], [10, 8, 2, 12], [10, 8, 2, 13], [10, 8, 5, 0], [10, 8, 5, 5], [10, 8, 5, 6], [10, 8, 5, 7], [10, 8, 5, 8], [10, 8, 5, 9], [10, 8, 5, 10], [10, 8, 5, 15], [10, 8, 7, 0], [10, 8, 7, 5], [10, 8, 7, 7], [10, 8, 7, 8], [10, 8, 7, 15], [10, 8, 8, 0], [10, 8, 8, 5], [10, 8, 8, 7], [10, 8, 8, 8], [10, 8, 8, 15], [10, 8, 11, 1], [10, 8, 11, 2], [10, 8, 11, 4], [10, 8, 11, 11], [10, 8, 11, 12], [10, 8, 11, 14], [10, 8, 12, 1], [10, 8, 12, 2], [10, 8, 12, 3], [10, 8, 12, 11], [10, 8, 12, 12], [10, 8, 12, 13], [10, 8, 15, 0], [10, 8, 15, 5], [10, 8, 15, 7], [10, 8, 15, 8], [10, 8, 15, 15], [10, 9, 1, 1], [10, 9, 1, 11], [10, 9, 6, 5], [10, 9, 11, 1], [10, 9, 11, 11], [10, 10, 5, 5], [10, 11, 0, 2], [10, 11, 0, 4], [10, 11, 0, 12], [10, 11, 0, 14], [10, 11, 2, 0], [10, 11, 2, 7], [10, 11, 2, 8], [10, 11, 2, 9], [10, 11, 2, 10], [10, 11, 2, 15], [10, 11, 4, 0], [10, 11, 4, 5], [10, 11, 4, 7], [10, 11, 4, 8], [10, 11, 4, 15], [10, 11, 7, 2], [10, 11, 7, 4], [10, 11, 7, 12], [10, 11, 7, 14], [10, 11, 8, 2], [10, 11, 8, 4], [10, 11, 8, 12], [10, 11, 8, 14], [10, 11, 9, 1], [10, 11, 9, 2], [10, 11, 9, 3], [10, 11, 9, 11], [10, 11, 9, 12], [10, 11, 9, 13], [10, 11, 12, 0], [10, 11, 12, 7], [10, 11, 12, 8], [10, 11, 12, 9], [10, 11, 12, 10], [10, 11, 12, 15], [10, 11, 14, 0], [10, 11, 14, 5], [10, 11, 14, 7], [10, 11, 14, 8], [10, 11, 14, 15], [10, 11, 15, 2], [10, 11, 15, 4], [10, 11, 15, 12], [10, 11, 15, 14], [10, 12, 0, 1], [10, 12, 0, 3], [10, 12, 0, 11], [10, 12, 0, 13], [10, 12, 1, 0], [10, 12, 1, 6], [10, 12, 1, 7], [10, 12, 1, 8], [10, 12, 1, 10], [10, 12, 1, 15], [10, 12, 3, 0], [10, 12, 3, 5], [10, 12, 3, 7], [10, 12, 3, 8], [10, 12, 3, 15], [10, 12, 6, 1], [10, 12, 6, 2], [10, 12, 6, 4], [10, 12, 6, 11], [10, 12, 6, 12], [10, 12, 6, 14], [10, 12, 7, 1], [10, 12, 7, 3], [10, 12, 7, 11], [10, 12, 7, 13], [10, 12, 8, 1], [10, 12, 8, 3], [10, 12, 8, 11], [10, 12, 8, 13], [10, 12, 11, 0], [10, 12, 11, 6], [10, 12, 11, 7], [10, 12, 11, 8], [10, 12, 11, 10], [10, 12, 11, 15], [10, 12, 13, 0], [10, 12, 13, 5], [10, 12, 13, 7], [10, 12, 13, 8], [10, 12, 13, 15], [10, 12, 15, 1], [10, 12, 15, 3], [10, 12, 15, 11], [10, 12, 15, 13], [10, 13, 2, 5], [10, 13, 5, 2], [10, 13, 5, 12], [10, 13, 12, 5], [10, 14, 1, 5], [10, 14, 5, 1], [10, 14, 5, 11], [10, 14, 11, 5], [10, 15, 0, 0], [10, 15, 0, 5], [10, 15, 0, 7], [10, 15, 0, 8], [10, 15, 0, 15], [10, 15, 1, 1], [10, 15, 1, 2], [10, 15, 1, 4], [10, 15, 1, 11], [10, 15, 1, 12], [10, 15, 1, 14], [10, 15, 2, 1], [10, 15, 2, 2], [10, 15, 2, 3], [10, 15, 2, 11], [10, 15, 2, 12], [10, 15, 2, 13], [10, 15, 5, 0], [10, 15, 5, 5], [10, 15, 5, 6], [10, 15, 5, 7], [10, 15, 5, 8], [10, 15, 5, 9], [10, 15, 5, 10], [10, 15, 5, 15], [10, 15, 7, 0], [10, 15, 7, 5], [10, 15, 7, 7], [10, 15, 7, 8], [10, 15, 7, 15], [10, 15, 8, 0], [10, 15, 8, 5], [10, 15, 8, 7], [10, 15, 8, 8], [10, 15, 8, 15], [10, 15, 11, 1], [10, 15, 11, 2], [10, 15, 11, 4], [10, 15, 11, 11], [10, 15, 11, 12], [10, 15, 11, 14], [10, 15, 12, 1], [10, 15, 12, 2], [10, 15, 12, 3], [10, 15, 12, 11], [10, 15, 12, 12], [10, 15, 12, 13], [10, 15, 15, 0], [10, 15, 15, 5], [10, 15, 15, 7], [10, 15, 15, 8], [10, 15, 15, 15], [11, 0, 0, 4], [11, 0, 0, 14], [11, 0, 3, 9], [11, 0, 4, 0], [11, 0, 4, 7], [11, 0, 4, 8], [11, 0, 4, 10], [11, 0, 4, 15], [11, 0, 7, 4], [11, 0, 7, 14], [11, 0, 8, 4], [11, 0, 8, 14], [11, 0, 10, 2], [11, 0, 10, 4], [11, 0, 10, 12], [11, 0, 10, 14], [11, 0, 13, 9], [11, 0, 14, 0], [11, 0, 14, 7], [11, 0, 14, 8], [11, 0, 14, 10], [11, 0, 14, 15], [11, 0, 15, 4], [11, 0, 15, 14], [11, 2, 10, 10], [11, 3, 0, 9], [11, 3, 2, 4], [11, 3, 2, 14], [11, 3, 4, 2], [11, 3, 4, 4], [11, 3, 4, 12], [11, 3, 4, 14], [11, 3, 7, 9], [11, 3, 8, 9], [11, 3, 9, 0], [11, 3, 9, 7], [11, 3, 9, 8], [11, 3, 9, 9], [11, 3, 9, 10], [11, 3, 9, 15], [11, 3, 12, 4], [11, 3, 12, 14], [11, 3, 14, 2], [11, 3, 14, 4], [11, 3, 14, 12], [11, 3, 14, 14], [11, 3, 15, 9], [11, 4, 0, 0], [11, 4, 0, 7], [11, 4, 0, 8], [11, 4, 0, 10], [11, 4, 0, 15], [11, 4, 1, 4], [11, 4, 1, 14], [11, 4, 3, 2], [11, 4, 3, 4], [11, 4, 3, 12], [11, 4, 3, 14], [11, 4, 4, 1], [11, 4, 4, 3], [11, 4, 4, 11], [11, 4, 4, 13], [11, 4, 6, 9], [11, 4, 7, 0], [11, 4, 7, 7], [11, 4, 7, 8], [11, 4, 7, 10], [11, 4, 7, 15], [11, 4, 8, 0], [11, 4, 8, 7], [11, 4, 8, 8], [11, 4, 8, 10], [11, 4, 8, 15], [11, 4, 10, 0], [11, 4, 10, 5], [11, 4, 10, 7], [11, 4, 10, 8], [11, 4, 10, 15], [11, 4, 11, 4], [11, 4, 11, 14], [11, 4, 13, 2], [11, 4, 13, 4], [11, 4, 13, 12], [11, 4, 13, 14], [11, 4, 14, 1], [11, 4, 14, 3], [11, 4, 14, 11], [11, 4, 14, 13], [11, 4, 15, 0], [11, 4, 15, 7], [11, 4, 15, 8], [11, 4, 15, 10], [11, 4, 15, 15], [11, 7, 0, 4], [11, 7, 0, 14], [11, 7, 3, 9], [11, 7, 4, 0], [11, 7, 4, 7], [11, 7, 4, 8], [11, 7, 4, 10], [11, 7, 4, 15], [11, 7, 7, 4], [11, 7, 7, 14], [11, 7, 8, 4], [11, 7, 8, 14], [11, 7, 10, 2], [11, 7, 10, 4], [11, 7, 10, 12], [11, 7, 10, 14], [11, 7, 13, 9], [11, 7, 14, 0], [11, 7, 14, 7], [11, 7, 14, 8], [11, 7, 14, 10], [11, 7, 14, 15], [11, 7, 15, 4], [11, 7, 15, 14], [11, 8, 0, 4], [11, 8, 0, 14], [11, 8, 3, 9], [11, 8, 4, 0], [11, 8, 4, 7], [11, 8, 4, 8], [11, 8, 4, 10], [11, 8, 4, 15], [11, 8, 7, 4], [11, 8, 7, 14], [11, 8, 8, 4], [11, 8, 8, 14], [11, 8, 10, 2], [11, 8, 10, 4], [11, 8, 10, 12], [11, 8, 10, 14], [11, 8, 13, 9], [11, 8, 14, 0], [11, 8, 14, 7], [11, 8, 14, 8], [11, 8, 14, 10], [11, 8, 14, 15], [11, 8, 15, 4], [11, 8, 15, 14], [11, 9, 3, 10], [11, 9, 10, 3], [11, 9, 10, 13], [11, 9, 13, 10], [11, 10, 0, 2], [11, 10, 0, 4], [11, 10, 0, 12], [11, 10, 0, 14], [11, 10, 1, 9], [11, 10, 2, 0], [11, 10, 2, 7], [11, 10, 2, 8], [11, 10, 2, 9], [11, 10, 2, 10], [11, 10, 2, 15], [11, 10, 4, 0], [11, 10, 4, 5], [11, 10, 4, 7], [11, 10, 4, 8], [11, 10, 4, 15], [11, 10, 5, 4], [11, 10, 5, 14], [11, 10, 7, 2], [11, 10, 7, 4], [11, 10, 7, 12], [11, 10, 7, 14], [11, 10, 8, 2], [11, 10, 8, 4], [11, 10, 8, 12], [11, 10, 8, 14], [11, 10, 9, 1], [11, 10, 9, 2], [11, 10, 9, 3], [11, 10, 9, 11], [11, 10, 9, 12], [11, 10, 9, 13], [11, 10, 11, 9], [11, 10, 12, 0], [11, 10, 12, 7], [11, 10, 12, 8], [11, 10, 12, 9], [11, 10, 12, 10], [11, 10, 12, 15], [11, 10, 14, 0], [11, 10, 14, 5], [11, 10, 14, 7], [11, 10, 14, 8], [11, 10, 14, 15], [11, 10, 15, 2], [11, 10, 15, 4], [11, 10, 15, 12], [11, 10, 15, 14], [11, 12, 10, 10], [11, 13, 0, 9], [11, 13, 2, 4], [11, 13, 2, 14], [11, 13, 4, 2], [11, 13, 4, 4], [11, 13, 4, 12], [11, 13, 4, 14], [11, 13, 7, 9], [11, 13, 8, 9], [11, 13, 9, 0], [11, 13, 9, 7], [11, 13, 9, 8], [11, 13, 9, 9], [11, 13, 9, 10], [11, 13, 9, 15], [11, 13, 12, 4], [11, 13, 12, 14], [11, 13, 14, 2], [11, 13, 14, 4], [11, 13, 14, 12], [11, 13, 14, 14], [11, 13, 15, 9], [11, 14, 0, 0], [11, 14, 0, 7], [11, 14, 0, 8], [11, 14, 0, 10], [11, 14, 0, 15], [11, 14, 1, 4], [11, 14, 1, 14], [11, 14, 3, 2], [11, 14, 3, 4], [11, 14, 3, 12], [11, 14, 3, 14], [11, 14, 4, 1], [11, 14, 4, 3], [11, 14, 4, 11], [11, 14, 4, 13], [11, 14, 6, 9], [11, 14, 7, 0], [11, 14, 7, 7], [11, 14, 7, 8], [11, 14, 7, 10], [11, 14, 7, 15], [11, 14, 8, 0], [11, 14, 8, 7], [11, 14, 8, 8], [11, 14, 8, 10], [11, 14, 8, 15], [11, 14, 10, 0], [11, 14, 10, 5], [11, 14, 10, 7], [11, 14, 10, 8], [11, 14, 10, 15], [11, 14, 11, 4], [11, 14, 11, 14], [11, 14, 13, 2], [11, 14, 13, 4], [11, 14, 13, 12], [11, 14, 13, 14], [11, 14, 14, 1], [11, 14, 14, 3], [11, 14, 14, 11], [11, 14, 14, 13], [11, 14, 15, 0], [11, 14, 15, 7], [11, 14, 15, 8], [11, 14, 15, 10], [11, 14, 15, 15], [11, 15, 0, 4], [11, 15, 0, 14], [11, 15, 3, 9], [11, 15, 4, 0], [11, 15, 4, 7], [11, 15, 4, 8], [11, 15, 4, 10], [11, 15, 4, 15], [11, 15, 7, 4], [11, 15, 7, 14], [11, 15, 8, 4], [11, 15, 8, 14], [11, 15, 10, 2], [11, 15, 10, 4], [11, 15, 10, 12], [11, 15, 10, 14], [11, 15, 13, 9], [11, 15, 14, 0], [11, 15, 14, 7], [11, 15, 14, 8], [11, 15, 14, 10], [11, 15, 14, 15], [11, 15, 15, 4], [11, 15, 15, 14], [12, 0, 0, 3], [12, 0, 0, 13], [12, 0, 3, 0], [12, 0, 3, 7], [12, 0, 3, 8], [12, 0, 3, 10], [12, 0, 3, 15], [12, 0, 4, 6], [12, 0, 7, 3], [12, 0, 7, 13], [12, 0, 8, 3], [12, 0, 8, 13], [12, 0, 10, 1], [12, 0, 10, 3], [12, 0, 10, 11], [12, 0, 10, 13], [12, 0, 13, 0], [12, 0, 13, 7], [12, 0, 13, 8], [12, 0, 13, 10], [12, 0, 13, 15], [12, 0, 14, 6], [12, 0, 15, 3], [12, 0, 15, 13], [12, 1, 10, 10], [12, 3, 0, 0], [12, 3, 0, 7], [12, 3, 0, 8], [12, 3, 0, 10], [12, 3, 0, 15], [12, 3, 2, 3], [12, 3, 2, 13], [12, 3, 3, 2], [12, 3, 3, 4], [12, 3, 3, 12], [12, 3, 3, 14], [12, 3, 4, 1], [12, 3, 4, 3], [12, 3, 4, 11], [12, 3, 4, 13], [12, 3, 7, 0], [12, 3, 7, 7], [12, 3, 7, 8], [12, 3, 7, 10], [12, 3, 7, 15], [12, 3, 8, 0], [12, 3, 8, 7], [12, 3, 8, 8], [12, 3, 8, 10], [12, 3, 8, 15], [12, 3, 9, 6], [12, 3, 10, 0], [12, 3, 10, 5], [12, 3, 10, 7], [12, 3, 10, 8], [12, 3, 10, 15], [12, 3, 12, 3], [12, 3, 12, 13], [12, 3, 13, 2], [12, 3, 13, 4], [12, 3, 13, 12], [12, 3, 13, 14], [12, 3, 14, 1], [12, 3, 14, 3], [12, 3, 14, 11], [12, 3, 14, 13], [12, 3, 15, 0], [12, 3, 15, 7], [12, 3, 15, 8], [12, 3, 15, 10], [12, 3, 15, 15], [12, 4, 0, 6], [12, 4, 1, 3], [12, 4, 1, 13], [12, 4, 3, 1], [12, 4, 3, 3], [12, 4, 3, 11], [12, 4, 3, 13], [12, 4, 6, 0], [12, 4, 6, 6], [12, 4, 6, 7], [12, 4, 6, 8], [12, 4, 6, 10], [12, 4, 6, 15], [12, 4, 7, 6], [12, 4, 8, 6], [12, 4, 11, 3], [12, 4, 11, 13], [12, 4, 13, 1], [12, 4, 13, 3], [12, 4, 13, 11], [12, 4, 13, 13], [12, 4, 15, 6], [12, 6, 4, 10], [12, 6, 10, 4], [12, 6, 10, 14], [12, 6, 14, 10], [12, 7, 0, 3], [12, 7, 0, 13], [12, 7, 3, 0], [12, 7, 3, 7], [12, 7, 3, 8], [12, 7, 3, 10], [12, 7, 3, 15], [12, 7, 4, 6], [12, 7, 7, 3], [12, 7, 7, 13], [12, 7, 8, 3], [12, 7, 8, 13], [12, 7, 10, 1], [12, 7, 10, 3], [12, 7, 10, 11], [12, 7, 10, 13], [12, 7, 13, 0], [12, 7, 13, 7], [12, 7, 13, 8], [12, 7, 13, 10], [12, 7, 13, 15], [12, 7, 14, 6], [12, 7, 15, 3], [12, 7, 15, 13], [12, 8, 0, 3], [12, 8, 0, 13], [12, 8, 3, 0], [12, 8, 3, 7], [12, 8, 3, 8], [12, 8, 3, 10], [12, 8, 3, 15], [12, 8, 4, 6], [12, 8, 7, 3], [12, 8, 7, 13], [12, 8, 8, 3], [12, 8, 8, 13], [12, 8, 10, 1], [12, 8, 10, 3], [12, 8, 10, 11], [12, 8, 10, 13], [12, 8, 13, 0], [12, 8, 13, 7], [12, 8, 13, 8], [12, 8, 13, 10], [12, 8, 13, 15], [12, 8, 14, 6], [12, 8, 15, 3], [12, 8, 15, 13], [12, 10, 0, 1], [12, 10, 0, 3], [12, 10, 0, 11], [12, 10, 0, 13], [12, 10, 1, 0], [12, 10, 1, 6], [12, 10, 1, 7], [12, 10, 1, 8], [12, 10, 1, 10], [12, 10, 1, 15], [12, 10, 2, 6], [12, 10, 3, 0], [12, 10, 3, 5], [12, 10, 3, 7], [12, 10, 3, 8], [12, 10, 3, 15], [12, 10, 5, 3], [12, 10, 5, 13], [12, 10, 6, 1], [12, 10, 6, 2], [12, 10, 6, 4], [12, 10, 6, 11], [12, 10, 6, 12], [12, 10, 6, 14], [12, 10, 7, 1], [12, 10, 7, 3], [12, 10, 7, 11], [12, 10, 7, 13], [12, 10, 8, 1], [12, 10, 8, 3], [12, 10, 8, 11], [12, 10, 8, 13], [12, 10, 11, 0], [12, 10, 11, 6], [12, 10, 11, 7], [12, 10, 11, 8], [12, 10, 11, 10], [12, 10, 11, 15], [12, 10, 12, 6], [12, 10, 13, 0], [12, 10, 13, 5], [12, 10, 13, 7], [12, 10, 13, 8], [12, 10, 13, 15], [12, 10, 15, 1], [12, 10, 15, 3], [12, 10, 15, 11], [12, 10, 15, 13], [12, 11, 10, 10], [12, 13, 0, 0], [12, 13, 0, 7], [12, 13, 0, 8], [12, 13, 0, 10], [12, 13, 0, 15], [12, 13, 2, 3], [12, 13, 2, 13], [12, 13, 3, 2], [12, 13, 3, 4], [12, 13, 3, 12], [12, 13, 3, 14], [12, 13, 4, 1], [12, 13, 4, 3], [12, 13, 4, 11], [12, 13, 4, 13], [12, 13, 7, 0], [12, 13, 7, 7], [12, 13, 7, 8], [12, 13, 7, 10], [12, 13, 7, 15], [12, 13, 8, 0], [12, 13, 8, 7], [12, 13, 8, 8], [12, 13, 8, 10], [12, 13, 8, 15], [12, 13, 9, 6], [12, 13, 10, 0], [12, 13, 10, 5], [12, 13, 10, 7], [12, 13, 10, 8], [12, 13, 10, 15], [12, 13, 12, 3], [12, 13, 12, 13], [12, 13, 13, 2], [12, 13, 13, 4], [12, 13, 13, 12], [12, 13, 13, 14], [12, 13, 14, 1], [12, 13, 14, 3], [12, 13, 14, 11], [12, 13, 14, 13], [12, 13, 15, 0], [12, 13, 15, 7], [12, 13, 15, 8], [12, 13, 15, 10], [12, 13, 15, 15], [12, 14, 0, 6], [12, 14, 1, 3], [12, 14, 1, 13], [12, 14, 3, 1], [12, 14, 3, 3], [12, 14, 3, 11], [12, 14, 3, 13], [12, 14, 6, 0], [12, 14, 6, 6], [12, 14, 6, 7], [12, 14, 6, 8], [12, 14, 6, 10], [12, 14, 6, 15], [12, 14, 7, 6], [12, 14, 8, 6], [12, 14, 11, 3], [12, 14, 11, 13], [12, 14, 13, 1], [12, 14, 13, 3], [12, 14, 13, 11], [12, 14, 13, 13], [12, 14, 15, 6], [12, 15, 0, 3], [12, 15, 0, 13], [12, 15, 3, 0], [12, 15, 3, 7], [12, 15, 3, 8], [12, 15, 3, 10], [12, 15, 3, 15], [12, 15, 4, 6], [12, 15, 7, 3], [12, 15, 7, 13], [12, 15, 8, 3], [12, 15, 8, 13], [12, 15, 10, 1], [12, 15, 10, 3], [12, 15, 10, 11], [12, 15, 10, 13], [12, 15, 13, 0], [12, 15, 13, 7], [12, 15, 13, 8], [12, 15, 13, 10], [12, 15, 13, 15], [12, 15, 14, 6], [12, 15, 15, 3], [12, 15, 15, 13], [13, 0, 0, 2], [13, 0, 0, 4], [13, 0, 0, 12], [13, 0, 0, 14], [13, 0, 2, 0], [13, 0, 2, 7], [13, 0, 2, 8], [13, 0, 2, 9], [13, 0, 2, 10], [13, 0, 2, 15], [13, 0, 4, 0], [13, 0, 4, 5], [13, 0, 4, 7], [13, 0, 4, 8], [13, 0, 4, 15], [13, 0, 7, 2], [13, 0, 7, 4], [13, 0, 7, 12], [13, 0, 7, 14], [13, 0, 8, 2], [13, 0, 8, 4], [13, 0, 8, 12], [13, 0, 8, 14], [13, 0, 9, 1], [13, 0, 9, 2], [13, 0, 9, 3], [13, 0, 9, 11], [13, 0, 9, 12], [13, 0, 9, 13], [13, 0, 12, 0], [13, 0, 12, 7], [13, 0, 12, 8], [13, 0, 12, 9], [13, 0, 12, 10], [13, 0, 12, 15], [13, 0, 14, 0], [13, 0, 14, 5], [13, 0, 14, 7], [13, 0, 14, 8], [13, 0, 14, 15], [13, 0, 15, 2], [13, 0, 15, 4], [13, 0, 15, 12], [13, 0, 15, 14], [13, 1, 4, 4], [13, 1, 4, 14], [13, 1, 9, 9], [13, 1, 9, 10], [13, 1, 14, 4], [13, 1, 14, 14], [13, 2, 0, 0], [13, 2, 0, 7], [13, 2, 0, 8], [13, 2, 0, 10], [13, 2, 0, 15], [13, 2, 3, 2], [13, 2, 3, 4], [13, 2, 3, 12], [13, 2, 3, 14], [13, 2, 4, 1], [13, 2, 4, 3], [13, 2, 4, 11], [13, 2, 4, 13], [13, 2, 7, 0], [13, 2, 7, 7], [13, 2, 7, 8], [13, 2, 7, 10], [13, 2, 7, 15], [13, 2, 8, 0], [13, 2, 8, 7], [13, 2, 8, 8], [13, 2, 8, 10], [13, 2, 8, 15], [13, 2, 10, 0], [13, 2, 10, 5], [13, 2, 10, 7], [13, 2, 10, 8], [13, 2, 10, 15], [13, 2, 13, 2], [13, 2, 13, 4], [13, 2, 13, 12], [13, 2, 13, 14], [13, 2, 14, 1], [13, 2, 14, 3], [13, 2, 14, 11], [13, 2, 14, 13], [13, 2, 15, 0], [13, 2, 15, 7], [13, 2, 15, 8], [13, 2, 15, 10], [13, 2, 15, 15], [13, 3, 2, 2], [13, 3, 2, 12], [13, 3, 9, 5], [13, 3, 12, 2], [13, 3, 12, 12], [13, 4, 0, 0], [13, 4, 0, 5], [13, 4, 0, 7], [13, 4, 0, 8], [13, 4, 0, 15], [13, 4, 1, 1], [13, 4, 1, 2], [13, 4, 1, 4], [13, 4, 1, 11], [13, 4, 1, 12], [13, 4, 1, 14], [13, 4, 2, 1], [13, 4, 2, 2], [13, 4, 2, 3], [13, 4, 2, 11], [13, 4, 2, 12], [13, 4, 2, 13], [13, 4, 5, 0], [13, 4, 5, 5], [13, 4, 5, 6], [13, 4, 5, 7], [13, 4, 5, 8], [13, 4, 5, 9], [13, 4, 5, 10], [13, 4, 5, 15], [13, 4, 7, 0], [13, 4, 7, 5], [13, 4, 7, 7], [13, 4, 7, 8], [13, 4, 7, 15], [13, 4, 8, 0], [13, 4, 8, 5], [13, 4, 8, 7], [13, 4, 8, 8], [13, 4, 8, 15], [13, 4, 11, 1], [13, 4, 11, 2], [13, 4, 11, 4], [13, 4, 11, 11], [13, 4, 11, 12], [13, 4, 11, 14], [13, 4, 12, 1], [13, 4, 12, 2], [13, 4, 12, 3], [13, 4, 12, 11], [13, 4, 12, 12], [13, 4, 12, 13], [13, 4, 15, 0], [13, 4, 15, 5], [13, 4, 15, 7], [13, 4, 15, 8], [13, 4, 15, 15], [13, 5, 4, 10], [13, 5, 10, 4], [13, 5, 10, 14], [13, 5, 14, 10], [13, 7, 0, 2], [13, 7, 0, 4], [13, 7, 0, 12], [13, 7, 0, 14], [13, 7, 2, 0], [13, 7, 2, 7], [13, 7, 2, 8], [13, 7, 2, 9], [13, 7, 2, 10], [13, 7, 2, 15], [13, 7, 4, 0], [13, 7, 4, 5], [13, 7, 4, 7], [13, 7, 4, 8], [13, 7, 4, 15], [13, 7, 7, 2], [13, 7, 7, 4], [13, 7, 7, 12], [13, 7, 7, 14], [13, 7, 8, 2], [13, 7, 8, 4], [13, 7, 8, 12], [13, 7, 8, 14], [13, 7, 9, 1], [13, 7, 9, 2], [13, 7, 9, 3], [13, 7, 9, 11], [13, 7, 9, 12], [13, 7, 9, 13], [13, 7, 12, 0], [13, 7, 12, 7], [13, 7, 12, 8], [13, 7, 12, 9], [13, 7, 12, 10], [13, 7, 12, 15], [13, 7, 14, 0], [13, 7, 14, 5], [13, 7, 14, 7], [13, 7, 14, 8], [13, 7, 14, 15], [13, 7, 15, 2], [13, 7, 15, 4], [13, 7, 15, 12], [13, 7, 15, 14], [13, 8, 0, 2], [13, 8, 0, 4], [13, 8, 0, 12], [13, 8, 0, 14], [13, 8, 2, 0], [13, 8, 2, 7], [13, 8, 2, 8], [13, 8, 2, 9], [13, 8, 2, 10], [13, 8, 2, 15], [13, 8, 4, 0], [13, 8, 4, 5], [13, 8, 4, 7], [13, 8, 4, 8], [13, 8, 4, 15], [13, 8, 7, 2], [13, 8, 7, 4], [13, 8, 7, 12], [13, 8, 7, 14], [13, 8, 8, 2], [13, 8, 8, 4], [13, 8, 8, 12], [13, 8, 8, 14], [13, 8, 9, 1], [13, 8, 9, 2], [13, 8, 9, 3], [13, 8, 9, 11], [13, 8, 9, 12], [13, 8, 9, 13], [13, 8, 12, 0], [13, 8, 12, 7], [13, 8, 12, 8], [13, 8, 12, 9], [13, 8, 12, 10], [13, 8, 12, 15], [13, 8, 14, 0], [13, 8, 14, 5], [13, 8, 14, 7], [13, 8, 14, 8], [13, 8, 14, 15], [13, 8, 15, 2], [13, 8, 15, 4], [13, 8, 15, 12], [13, 8, 15, 14], [13, 9, 0, 1], [13, 9, 0, 3], [13, 9, 0, 11], [13, 9, 0, 13], [13, 9, 1, 0], [13, 9, 1, 6], [13, 9, 1, 7], [13, 9, 1, 8], [13, 9, 1, 10], [13, 9, 1, 15], [13, 9, 3, 0], [13, 9, 3, 5], [13, 9, 3, 7], [13, 9, 3, 8], [13, 9, 3, 15], [13, 9, 6, 1], [13, 9, 6, 2], [13, 9, 6, 4], [13, 9, 6, 11], [13, 9, 6, 12], [13, 9, 6, 14], [13, 9, 7, 1], [13, 9, 7, 3], [13, 9, 7, 11], [13, 9, 7, 13], [13, 9, 8, 1], [13, 9, 8, 3], [13, 9, 8, 11], [13, 9, 8, 13], [13, 9, 11, 0], [13, 9, 11, 6], [13, 9, 11, 7], [13, 9, 11, 8], [13, 9, 11, 10], [13, 9, 11, 15], [13, 9, 13, 0], [13, 9, 13, 5], [13, 9, 13, 7], [13, 9, 13, 8], [13, 9, 13, 15], [13, 9, 15, 1], [13, 9, 15, 3], [13, 9, 15, 11], [13, 9, 15, 13], [13, 10, 2, 5], [13, 10, 5, 2], [13, 10, 5, 12], [13, 10, 12, 5], [13, 11, 4, 4], [13, 11, 4, 14], [13, 11, 9, 9], [13, 11, 9, 10], [13, 11, 14, 4], [13, 11, 14, 14], [13, 12, 0, 0], [13, 12, 0, 7], [13, 12, 0, 8], [13, 12, 0, 10], [13, 12, 0, 15], [13, 12, 3, 2], [13, 12, 3, 4], [13, 12, 3, 12], [13, 12, 3, 14], [13, 12, 4, 1], [13, 12, 4, 3], [13, 12, 4, 11], [13, 12, 4, 13], [13, 12, 7, 0], [13, 12, 7, 7], [13, 12, 7, 8], [13, 12, 7, 10], [13, 12, 7, 15], [13, 12, 8, 0], [13, 12, 8, 7], [13, 12, 8, 8], [13, 12, 8, 10], [13, 12, 8, 15], [13, 12, 10, 0], [13, 12, 10, 5], [13, 12, 10, 7], [13, 12, 10, 8], [13, 12, 10, 15], [13, 12, 13, 2], [13, 12, 13, 4], [13, 12, 13, 12], [13, 12, 13, 14], [13, 12, 14, 1], [13, 12, 14, 3], [13, 12, 14, 11], [13, 12, 14, 13], [13, 12, 15, 0], [13, 12, 15, 7], [13, 12, 15, 8], [13, 12, 15, 10], [13, 12, 15, 15], [13, 13, 2, 2], [13, 13, 2, 12], [13, 13, 9, 5], [13, 13, 12, 2], [13, 13, 12, 12], [13, 14, 0, 0], [13, 14, 0, 5], [13, 14, 0, 7], [13, 14, 0, 8], [13, 14, 0, 15], [13, 14, 1, 1], [13, 14, 1, 2], [13, 14, 1, 4], [13, 14, 1, 11], [13, 14, 1, 12], [13, 14, 1, 14], [13, 14, 2, 1], [13, 14, 2, 2], [13, 14, 2, 3], [13, 14, 2, 11], [13, 14, 2, 12], [13, 14, 2, 13], [13, 14, 5, 0], [13, 14, 5, 5], [13, 14, 5, 6], [13, 14, 5, 7], [13, 14, 5, 8], [13, 14, 5, 9], [13, 14, 5, 10], [13, 14, 5, 15], [13, 14, 7, 0], [13, 14, 7, 5], [13, 14, 7, 7], [13, 14, 7, 8], [13, 14, 7, 15], [13, 14, 8, 0], [13, 14, 8, 5], [13, 14, 8, 7], [13, 14, 8, 8], [13, 14, 8, 15], [13, 14, 11, 1], [13, 14, 11, 2], [13, 14, 11, 4], [13, 14, 11, 11], [13, 14, 11, 12], [13, 14, 11, 14], [13, 14, 12, 1], [13, 14, 12, 2], [13, 14, 12, 3], [13, 14, 12, 11], [13, 14, 12, 12], [13, 14, 12, 13], [13, 14, 15, 0], [13, 14, 15, 5], [13, 14, 15, 7], [13, 14, 15, 8], [13, 14, 15, 15], [13, 15, 0, 2], [13, 15, 0, 4], [13, 15, 0, 12], [13, 15, 0, 14], [13, 15, 2, 0], [13, 15, 2, 7], [13, 15, 2, 8], [13, 15, 2, 9], [13, 15, 2, 10], [13, 15, 2, 15], [13, 15, 4, 0], [13, 15, 4, 5], [13, 15, 4, 7], [13, 15, 4, 8], [13, 15, 4, 15], [13, 15, 7, 2], [13, 15, 7, 4], [13, 15, 7, 12], [13, 15, 7, 14], [13, 15, 8, 2], [13, 15, 8, 4], [13, 15, 8, 12], [13, 15, 8, 14], [13, 15, 9, 1], [13, 15, 9, 2], [13, 15, 9, 3], [13, 15, 9, 11], [13, 15, 9, 12], [13, 15, 9, 13], [13, 15, 12, 0], [13, 15, 12, 7], [13, 15, 12, 8], [13, 15, 12, 9], [13, 15, 12, 10], [13, 15, 12, 15], [13, 15, 14, 0], [13, 15, 14, 5], [13, 15, 14, 7], [13, 15, 14, 8], [13, 15, 14, 15], [13, 15, 15, 2], [13, 15, 15, 4], [13, 15, 15, 12], [13, 15, 15, 14], [14, 0, 0, 1], [14, 0, 0, 3], [14, 0, 0, 11], [14, 0, 0, 13], [14, 0, 1, 0], [14, 0, 1, 6], [14, 0, 1, 7], [14, 0, 1, 8], [14, 0, 1, 10], [14, 0, 1, 15], [14, 0, 3, 0], [14, 0, 3, 5], [14, 0, 3, 7], [14, 0, 3, 8], [14, 0, 3, 15], [14, 0, 6, 1], [14, 0, 6, 2], [14, 0, 6, 4], [14, 0, 6, 11], [14, 0, 6, 12], [14, 0, 6, 14], [14, 0, 7, 1], [14, 0, 7, 3], [14, 0, 7, 11], [14, 0, 7, 13], [14, 0, 8, 1], [14, 0, 8, 3], [14, 0, 8, 11], [14, 0, 8, 13], [14, 0, 11, 0], [14, 0, 11, 6], [14, 0, 11, 7], [14, 0, 11, 8], [14, 0, 11, 10], [14, 0, 11, 15], [14, 0, 13, 0], [14, 0, 13, 5], [14, 0, 13, 7], [14, 0, 13, 8], [14, 0, 13, 15], [14, 0, 15, 1], [14, 0, 15, 3], [14, 0, 15, 11], [14, 0, 15, 13], [14, 1, 0, 0], [14, 1, 0, 7], [14, 1, 0, 8], [14, 1, 0, 10], [14, 1, 0, 15], [14, 1, 3, 2], [14, 1, 3, 4], [14, 1, 3, 12], [14, 1, 3, 14], [14, 1, 4, 1], [14, 1, 4, 3], [14, 1, 4, 11], [14, 1, 4, 13], [14, 1, 7, 0], [14, 1, 7, 7], [14, 1, 7, 8], [14, 1, 7, 10], [14, 1, 7, 15], [14, 1, 8, 0], [14, 1, 8, 7], [14, 1, 8, 8], [14, 1, 8, 10], [14, 1, 8, 15], [14, 1, 10, 0], [14, 1, 10, 5], [14, 1, 10, 7], [14, 1, 10, 8], [14, 1, 10, 15], [14, 1, 13, 2], [14, 1, 13, 4], [14, 1, 13, 12], [14, 1, 13, 14], [14, 1, 14, 1], [14, 1, 14, 3], [14, 1, 14, 11], [14, 1, 14, 13], [14, 1, 15, 0], [14, 1, 15, 7], [14, 1, 15, 8], [14, 1, 15, 10], [14, 1, 15, 15], [14, 2, 3, 3], [14, 2, 3, 13], [14, 2, 6, 6], [14, 2, 6, 10], [14, 2, 13, 3], [14, 2, 13, 13], [14, 3, 0, 0], [14, 3, 0, 5], [14, 3, 0, 7], [14, 3, 0, 8], [14, 3, 0, 15], [14, 3, 1, 1], [14, 3, 1, 2], [14, 3, 1, 4], [14, 3, 1, 11], [14, 3, 1, 12], [14, 3, 1, 14], [14, 3, 2, 1], [14, 3, 2, 2], [14, 3, 2, 3], [14, 3, 2, 11], [14, 3, 2, 12], [14, 3, 2, 13], [14, 3, 5, 0], [14, 3, 5, 5], [14, 3, 5, 6], [14, 3, 5, 7], [14, 3, 5, 8], [14, 3, 5, 9], [14, 3, 5, 10], [14, 3, 5, 15], [14, 3, 7, 0], [14, 3, 7, 5], [14, 3, 7, 7], [14, 3, 7, 8], [14, 3, 7, 15], [14, 3, 8, 0], [14, 3, 8, 5], [14, 3, 8, 7], [14, 3, 8, 8], [14, 3, 8, 15], [14, 3, 11, 1], [14, 3, 11, 2], [14, 3, 11, 4], [14, 3, 11, 11], [14, 3, 11, 12], [14, 3, 11, 14], [14, 3, 12, 1], [14, 3, 12, 2], [14, 3, 12, 3], [14, 3, 12, 11], [14, 3, 12, 12], [14, 3, 12, 13], [14, 3, 15, 0], [14, 3, 15, 5], [14, 3, 15, 7], [14, 3, 15, 8], [14, 3, 15, 15], [14, 4, 1, 1], [14, 4, 1, 11], [14, 4, 6, 5], [14, 4, 11, 1], [14, 4, 11, 11], [14, 5, 3, 10], [14, 5, 10, 3], [14, 5, 10, 13], [14, 5, 13, 10], [14, 6, 0, 2], [14, 6, 0, 4], [14, 6, 0, 12], [14, 6, 0, 14], [14, 6, 2, 0], [14, 6, 2, 7], [14, 6, 2, 8], [14, 6, 2, 9], [14, 6, 2, 10], [14, 6, 2, 15], [14, 6, 4, 0], [14, 6, 4, 5], [14, 6, 4, 7], [14, 6, 4, 8], [14, 6, 4, 15], [14, 6, 7, 2], [14, 6, 7, 4], [14, 6, 7, 12], [14, 6, 7, 14], [14, 6, 8, 2], [14, 6, 8, 4], [14, 6, 8, 12], [14, 6, 8, 14], [14, 6, 9, 1], [14, 6, 9, 2], [14, 6, 9, 3], [14, 6, 9, 11], [14, 6, 9, 12], [14, 6, 9, 13], [14, 6, 12, 0], [14, 6, 12, 7], [14, 6, 12, 8], [14, 6, 12, 9], [14, 6, 12, 10], [14, 6, 12, 15], [14, 6, 14, 0], [14, 6, 14, 5], [14, 6, 14, 7], [14, 6, 14, 8], [14, 6, 14, 15], [14, 6, 15, 2], [14, 6, 15, 4], [14, 6, 15, 12], [14, 6, 15, 14], [14, 7, 0, 1], [14, 7, 0, 3], [14, 7, 0, 11], [14, 7, 0, 13], [14, 7, 1, 0], [14, 7, 1, 6], [14, 7, 1, 7], [14, 7, 1, 8], [14, 7, 1, 10], [14, 7, 1, 15], [14, 7, 3, 0], [14, 7, 3, 5], [14, 7, 3, 7], [14, 7, 3, 8], [14, 7, 3, 15], [14, 7, 6, 1], [14, 7, 6, 2], [14, 7, 6, 4], [14, 7, 6, 11], [14, 7, 6, 12], [14, 7, 6, 14], [14, 7, 7, 1], [14, 7, 7, 3], [14, 7, 7, 11], [14, 7, 7, 13], [14, 7, 8, 1], [14, 7, 8, 3], [14, 7, 8, 11], [14, 7, 8, 13], [14, 7, 11, 0], [14, 7, 11, 6], [14, 7, 11, 7], [14, 7, 11, 8], [14, 7, 11, 10], [14, 7, 11, 15], [14, 7, 13, 0], [14, 7, 13, 5], [14, 7, 13, 7], [14, 7, 13, 8], [14, 7, 13, 15], [14, 7, 15, 1], [14, 7, 15, 3], [14, 7, 15, 11], [14, 7, 15, 13], [14, 8, 0, 1], [14, 8, 0, 3], [14, 8, 0, 11], [14, 8, 0, 13], [14, 8, 1, 0], [14, 8, 1, 6], [14, 8, 1, 7], [14, 8, 1, 8], [14, 8, 1, 10], [14, 8, 1, 15], [14, 8, 3, 0], [14, 8, 3, 5], [14, 8, 3, 7], [14, 8, 3, 8], [14, 8, 3, 15], [14, 8, 6, 1], [14, 8, 6, 2], [14, 8, 6, 4], [14, 8, 6, 11], [14, 8, 6, 12], [14, 8, 6, 14], [14, 8, 7, 1], [14, 8, 7, 3], [14, 8, 7, 11], [14, 8, 7, 13], [14, 8, 8, 1], [14, 8, 8, 3], [14, 8, 8, 11], [14, 8, 8, 13], [14, 8, 11, 0], [14, 8, 11, 6], [14, 8, 11, 7], [14, 8, 11, 8], [14, 8, 11, 10], [14, 8, 11, 15], [14, 8, 13, 0], [14, 8, 13, 5], [14, 8, 13, 7], [14, 8, 13, 8], [14, 8, 13, 15], [14, 8, 15, 1], [14, 8, 15, 3], [14, 8, 15, 11], [14, 8, 15, 13], [14, 10, 1, 5], [14, 10, 5, 1], [14, 10, 5, 11], [14, 10, 11, 5], [14, 11, 0, 0], [14, 11, 0, 7], [14, 11, 0, 8], [14, 11, 0, 10], [14, 11, 0, 15], [14, 11, 3, 2], [14, 11, 3, 4], [14, 11, 3, 12], [14, 11, 3, 14], [14, 11, 4, 1], [14, 11, 4, 3], [14, 11, 4, 11], [14, 11, 4, 13], [14, 11, 7, 0], [14, 11, 7, 7], [14, 11, 7, 8], [14, 11, 7, 10], [14, 11, 7, 15], [14, 11, 8, 0], [14, 11, 8, 7], [14, 11, 8, 8], [14, 11, 8, 10], [14, 11, 8, 15], [14, 11, 10, 0], [14, 11, 10, 5], [14, 11, 10, 7], [14, 11, 10, 8], [14, 11, 10, 15], [14, 11, 13, 2], [14, 11, 13, 4], [14, 11, 13, 12], [14, 11, 13, 14], [14, 11, 14, 1], [14, 11, 14, 3], [14, 11, 14, 11], [14, 11, 14, 13], [14, 11, 15, 0], [14, 11, 15, 7], [14, 11, 15, 8], [14, 11, 15, 10], [14, 11, 15, 15], [14, 12, 3, 3], [14, 12, 3, 13], [14, 12, 6, 6], [14, 12, 6, 10], [14, 12, 13, 3], [14, 12, 13, 13], [14, 13, 0, 0], [14, 13, 0, 5], [14, 13, 0, 7], [14, 13, 0, 8], [14, 13, 0, 15], [14, 13, 1, 1], [14, 13, 1, 2], [14, 13, 1, 4], [14, 13, 1, 11], [14, 13, 1, 12], [14, 13, 1, 14], [14, 13, 2, 1], [14, 13, 2, 2], [14, 13, 2, 3], [14, 13, 2, 11], [14, 13, 2, 12], [14, 13, 2, 13], [14, 13, 5, 0], [14, 13, 5, 5], [14, 13, 5, 6], [14, 13, 5, 7], [14, 13, 5, 8], [14, 13, 5, 9], [14, 13, 5, 10], [14, 13, 5, 15], [14, 13, 7, 0], [14, 13, 7, 5], [14, 13, 7, 7], [14, 13, 7, 8], [14, 13, 7, 15], [14, 13, 8, 0], [14, 13, 8, 5], [14, 13, 8, 7], [14, 13, 8, 8], [14, 13, 8, 15], [14, 13, 11, 1], [14, 13, 11, 2], [14, 13, 11, 4], [14, 13, 11, 11], [14, 13, 11, 12], [14, 13, 11, 14], [14, 13, 12, 1], [14, 13, 12, 2], [14, 13, 12, 3], [14, 13, 12, 11], [14, 13, 12, 12], [14, 13, 12, 13], [14, 13, 15, 0], [14, 13, 15, 5], [14, 13, 15, 7], [14, 13, 15, 8], [14, 13, 15, 15], [14, 14, 1, 1], [14, 14, 1, 11], [14, 14, 6, 5], [14, 14, 11, 1], [14, 14, 11, 11], [14, 15, 0, 1], [14, 15, 0, 3], [14, 15, 0, 11], [14, 15, 0, 13], [14, 15, 1, 0], [14, 15, 1, 6], [14, 15, 1, 7], [14, 15, 1, 8], [14, 15, 1, 10], [14, 15, 1, 15], [14, 15, 3, 0], [14, 15, 3, 5], [14, 15, 3, 7], [14, 15, 3, 8], [14, 15, 3, 15], [14, 15, 6, 1], [14, 15, 6, 2], [14, 15, 6, 4], [14, 15, 6, 11], [14, 15, 6, 12], [14, 15, 6, 14], [14, 15, 7, 1], [14, 15, 7, 3], [14, 15, 7, 11], [14, 15, 7, 13], [14, 15, 8, 1], [14, 15, 8, 3], [14, 15, 8, 11], [14, 15, 8, 13], [14, 15, 11, 0], [14, 15, 11, 6], [14, 15, 11, 7], [14, 15, 11, 8], [14, 15, 11, 10], [14, 15, 11, 15], [14, 15, 13, 0], [14, 15, 13, 5], [14, 15, 13, 7], [14, 15, 13, 8], [14, 15, 13, 15], [14, 15, 15, 1], [14, 15, 15, 3], [14, 15, 15, 11], [14, 15, 15, 13], [15, 0, 0, 0], [15, 0, 0, 7], [15, 0, 0, 8], [15, 0, 0, 10], [15, 0, 0, 15], [15, 0, 3, 2], [15, 0, 3, 4], [15, 0, 3, 12], [15, 0, 3, 14], [15, 0, 4, 1], [15, 0, 4, 3], [15, 0, 4, 11], [15, 0, 4, 13], [15, 0, 7, 0], [15, 0, 7, 7], [15, 0, 7, 8], [15, 0, 7, 10], [15, 0, 7, 15], [15, 0, 8, 0], [15, 0, 8, 7], [15, 0, 8, 8], [15, 0, 8, 10], [15, 0, 8, 15], [15, 0, 10, 0], [15, 0, 10, 5], [15, 0, 10, 7], [15, 0, 10, 8], [15, 0, 10, 15], [15, 0, 13, 2], [15, 0, 13, 4], [15, 0, 13, 12], [15, 0, 13, 14], [15, 0, 14, 1], [15, 0, 14, 3], [15, 0, 14, 11], [15, 0, 14, 13], [15, 0, 15, 0], [15, 0, 15, 7], [15, 0, 15, 8], [15, 0, 15, 10], [15, 0, 15, 15], [15, 1, 4, 10], [15, 1, 10, 4], [15, 1, 10, 14], [15, 1, 14, 10], [15, 2, 3, 10], [15, 2, 10, 3], [15, 2, 10, 13], [15, 2, 13, 10], [15, 3, 0, 2], [15, 3, 0, 4], [15, 3, 0, 12], [15, 3, 0, 14], [15, 3, 2, 0], [15, 3, 2, 7], [15, 3, 2, 8], [15, 3, 2, 9], [15, 3, 2, 10], [15, 3, 2, 15], [15, 3, 4, 0], [15, 3, 4, 5], [15, 3, 4, 7], [15, 3, 4, 8], [15, 3, 4, 15], [15, 3, 7, 2], [15, 3, 7, 4], [15, 3, 7, 12], [15, 3, 7, 14], [15, 3, 8, 2], [15, 3, 8, 4], [15, 3, 8, 12], [15, 3, 8, 14], [15, 3, 9, 1], [15, 3, 9, 2], [15, 3, 9, 3], [15, 3, 9, 11], [15, 3, 9, 12], [15, 3, 9, 13], [15, 3, 12, 0], [15, 3, 12, 7], [15, 3, 12, 8], [15, 3, 12, 9], [15, 3, 12, 10], [15, 3, 12, 15], [15, 3, 14, 0], [15, 3, 14, 5], [15, 3, 14, 7], [15, 3, 14, 8], [15, 3, 14, 15], [15, 3, 15, 2], [15, 3, 15, 4], [15, 3, 15, 12], [15, 3, 15, 14], [15, 4, 0, 1], [15, 4, 0, 3], [15, 4, 0, 11], [15, 4, 0, 13], [15, 4, 1, 0], [15, 4, 1, 6], [15, 4, 1, 7], [15, 4, 1, 8], [15, 4, 1, 10], [15, 4, 1, 15], [15, 4, 3, 0], [15, 4, 3, 5], [15, 4, 3, 7], [15, 4, 3, 8], [15, 4, 3, 15], [15, 4, 6, 1], [15, 4, 6, 2], [15, 4, 6, 4], [15, 4, 6, 11], [15, 4, 6, 12], [15, 4, 6, 14], [15, 4, 7, 1], [15, 4, 7, 3], [15, 4, 7, 11], [15, 4, 7, 13], [15, 4, 8, 1], [15, 4, 8, 3], [15, 4, 8, 11], [15, 4, 8, 13], [15, 4, 11, 0], [15, 4, 11, 6], [15, 4, 11, 7], [15, 4, 11, 8], [15, 4, 11, 10], [15, 4, 11, 15], [15, 4, 13, 0], [15, 4, 13, 5], [15, 4, 13, 7], [15, 4, 13, 8], [15, 4, 13, 15], [15, 4, 15, 1], [15, 4, 15, 3], [15, 4, 15, 11], [15, 4, 15, 13], [15, 6, 4, 4], [15, 6, 4, 14], [15, 6, 9, 9], [15, 6, 9, 10], [15, 6, 14, 4], [15, 6, 14, 14], [15, 7, 0, 0], [15, 7, 0, 7], [15, 7, 0, 8], [15, 7, 0, 10], [15, 7, 0, 15], [15, 7, 3, 2], [15, 7, 3, 4], [15, 7, 3, 12], [15, 7, 3, 14], [15, 7, 4, 1], [15, 7, 4, 3], [15, 7, 4, 11], [15, 7, 4, 13], [15, 7, 7, 0], [15, 7, 7, 7], [15, 7, 7, 8], [15, 7, 7, 10], [15, 7, 7, 15], [15, 7, 8, 0], [15, 7, 8, 7], [15, 7, 8, 8], [15, 7, 8, 10], [15, 7, 8, 15], [15, 7, 10, 0], [15, 7, 10, 5], [15, 7, 10, 7], [15, 7, 10, 8], [15, 7, 10, 15], [15, 7, 13, 2], [15, 7, 13, 4], [15, 7, 13, 12], [15, 7, 13, 14], [15, 7, 14, 1], [15, 7, 14, 3], [15, 7, 14, 11], [15, 7, 14, 13], [15, 7, 15, 0], [15, 7, 15, 7], [15, 7, 15, 8], [15, 7, 15, 10], [15, 7, 15, 15], [15, 8, 0, 0], [15, 8, 0, 7], [15, 8, 0, 8], [15, 8, 0, 10], [15, 8, 0, 15], [15, 8, 3, 2], [15, 8, 3, 4], [15, 8, 3, 12], [15, 8, 3, 14], [15, 8, 4, 1], [15, 8, 4, 3], [15, 8, 4, 11], [15, 8, 4, 13], [15, 8, 7, 0], [15, 8, 7, 7], [15, 8, 7, 8], [15, 8, 7, 10], [15, 8, 7, 15], [15, 8, 8, 0], [15, 8, 8, 7], [15, 8, 8, 8], [15, 8, 8, 10], [15, 8, 8, 15], [15, 8, 10, 0], [15, 8, 10, 5], [15, 8, 10, 7], [15, 8, 10, 8], [15, 8, 10, 15], [15, 8, 13, 2], [15, 8, 13, 4], [15, 8, 13, 12], [15, 8, 13, 14], [15, 8, 14, 1], [15, 8, 14, 3], [15, 8, 14, 11], [15, 8, 14, 13], [15, 8, 15, 0], [15, 8, 15, 7], [15, 8, 15, 8], [15, 8, 15, 10], [15, 8, 15, 15], [15, 9, 3, 3], [15, 9, 3, 13], [15, 9, 6, 6], [15, 9, 6, 10], [15, 9, 13, 3], [15, 9, 13, 13], [15, 10, 0, 0], [15, 10, 0, 5], [15, 10, 0, 7], [15, 10, 0, 8], [15, 10, 0, 15], [15, 10, 1, 1], [15, 10, 1, 2], [15, 10, 1, 4], [15, 10, 1, 11], [15, 10, 1, 12], [15, 10, 1, 14], [15, 10, 2, 1], [15, 10, 2, 2], [15, 10, 2, 3], [15, 10, 2, 11], [15, 10, 2, 12], [15, 10, 2, 13], [15, 10, 5, 0], [15, 10, 5, 5], [15, 10, 5, 6], [15, 10, 5, 7], [15, 10, 5, 8], [15, 10, 5, 9], [15, 10, 5, 10], [15, 10, 5, 15], [15, 10, 7, 0], [15, 10, 7, 5], [15, 10, 7, 7], [15, 10, 7, 8], [15, 10, 7, 15], [15, 10, 8, 0], [15, 10, 8, 5], [15, 10, 8, 7], [15, 10, 8, 8], [15, 10, 8, 15], [15, 10, 11, 1], [15, 10, 11, 2], [15, 10, 11, 4], [15, 10, 11, 11], [15, 10, 11, 12], [15, 10, 11, 14], [15, 10, 12, 1], [15, 10, 12, 2], [15, 10, 12, 3], [15, 10, 12, 11], [15, 10, 12, 12], [15, 10, 12, 13], [15, 10, 15, 0], [15, 10, 15, 5], [15, 10, 15, 7], [15, 10, 15, 8], [15, 10, 15, 15], [15, 11, 4, 10], [15, 11, 10, 4], [15, 11, 10, 14], [15, 11, 14, 10], [15, 12, 3, 10], [15, 12, 10, 3], [15, 12, 10, 13], [15, 12, 13, 10], [15, 13, 0, 2], [15, 13, 0, 4], [15, 13, 0, 12], [15, 13, 0, 14], [15, 13, 2, 0], [15, 13, 2, 7], [15, 13, 2, 8], [15, 13, 2, 9], [15, 13, 2, 10], [15, 13, 2, 15], [15, 13, 4, 0], [15, 13, 4, 5], [15, 13, 4, 7], [15, 13, 4, 8], [15, 13, 4, 15], [15, 13, 7, 2], [15, 13, 7, 4], [15, 13, 7, 12], [15, 13, 7, 14], [15, 13, 8, 2], [15, 13, 8, 4], [15, 13, 8, 12], [15, 13, 8, 14], [15, 13, 9, 1], [15, 13, 9, 2], [15, 13, 9, 3], [15, 13, 9, 11], [15, 13, 9, 12], [15, 13, 9, 13], [15, 13, 12, 0], [15, 13, 12, 7], [15, 13, 12, 8], [15, 13, 12, 9], [15, 13, 12, 10], [15, 13, 12, 15], [15, 13, 14, 0], [15, 13, 14, 5], [15, 13, 14, 7], [15, 13, 14, 8], [15, 13, 14, 15], [15, 13, 15, 2], [15, 13, 15, 4], [15, 13, 15, 12], [15, 13, 15, 14], [15, 14, 0, 1], [15, 14, 0, 3], [15, 14, 0, 11], [15, 14, 0, 13], [15, 14, 1, 0], [15, 14, 1, 6], [15, 14, 1, 7], [15, 14, 1, 8], [15, 14, 1, 10], [15, 14, 1, 15], [15, 14, 3, 0], [15, 14, 3, 5], [15, 14, 3, 7], [15, 14, 3, 8], [15, 14, 3, 15], [15, 14, 6, 1], [15, 14, 6, 2], [15, 14, 6, 4], [15, 14, 6, 11], [15, 14, 6, 12], [15, 14, 6, 14], [15, 14, 7, 1], [15, 14, 7, 3], [15, 14, 7, 11], [15, 14, 7, 13], [15, 14, 8, 1], [15, 14, 8, 3], [15, 14, 8, 11], [15, 14, 8, 13], [15, 14, 11, 0], [15, 14, 11, 6], [15, 14, 11, 7], [15, 14, 11, 8], [15, 14, 11, 10], [15, 14, 11, 15], [15, 14, 13, 0], [15, 14, 13, 5], [15, 14, 13, 7], [15, 14, 13, 8], [15, 14, 13, 15], [15, 14, 15, 1], [15, 14, 15, 3], [15, 14, 15, 11], [15, 14, 15, 13], [15, 15, 0, 0], [15, 15, 0, 7], [15, 15, 0, 8], [15, 15, 0, 10], [15, 15, 0, 15], [15, 15, 3, 2], [15, 15, 3, 4], [15, 15, 3, 12], [15, 15, 3, 14], [15, 15, 4, 1], [15, 15, 4, 3], [15, 15, 4, 11], [15, 15, 4, 13], [15, 15, 7, 0], [15, 15, 7, 7], [15, 15, 7, 8], [15, 15, 7, 10], [15, 15, 7, 15], [15, 15, 8, 0], [15, 15, 8, 7], [15, 15, 8, 8], [15, 15, 8, 10], [15, 15, 8, 15], [15, 15, 10, 0], [15, 15, 10, 5], [15, 15, 10, 7], [15, 15, 10, 8], [15, 15, 10, 15], [15, 15, 13, 2], [15, 15, 13, 4], [15, 15, 13, 12], [15, 15, 13, 14], [15, 15, 14, 1], [15, 15, 14, 3], [15, 15, 14, 11], [15, 15, 14, 13], [15, 15, 15, 0], [15, 15, 15, 7], [15, 15, 15, 8], [15, 15, 15, 10], [15, 15, 15, 15]]
    
    psi_list = []
    for init in init_list:
        psi = mpomps_obc(init)
        if psi is not None:
            psi_list.append(psi)
    
    for i in range(len(psi_list)):
        for j in range(len(psi_list)):
            ovlp = psi_list[i].overlap(psi_list[j])
            if abs(ovlp) < 1e-8:
                print(init_list[i], 'is orthogonal to', init_list[j])