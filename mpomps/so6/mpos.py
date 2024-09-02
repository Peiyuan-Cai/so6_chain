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
from so6bbqham import *
from tenpy.algorithms import dmrg
import logging
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True

class KitaevSingleChain():
    def __init__(self, chi, delta, lamb, L, pbc=-1):
        """
        The Single Kitaev chain class. 
        
        The Hamiltonian is in the BdG form, the matrix is written under the Bogoliubov *QUASIHOLE* representation. 

        Args:
            chi (float): variational parameter $\chi$
            delta (float): variational parameter $\delta$
            lamb (float): variational parameter $\lambda$
            L (int): Chain length
            pbc (int, optional): Boundary condition. Defaults to be 0. 0 for OBC, 1 for PBC, -1 for APBC. 
            
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
        20240828: any necessarity to consider parity? I don't think so, we are considering even chain size now. 
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
        elif cons_N == None and cons_S == None:
            print("No symmetry used in site 'sixparton'. ")
            leg = npc.LegCharge.from_trivial(64)
        else:
            raise ValueError("Check your conserve quantities. ")

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
        
        #print('Should be true', np.allclose(Fz@Fy@Fx@Fw@Fv@Fu,JW))
        
        cu = cudag.T; cv = cvdag.T; cw = cwdag.T; cx = cxdag.T; cy = cydag.T; cz = czdag.T; 
        
        ops = dict(id64=id64, JW=JW, 
                   cudag=cudag, cvdag=cvdag, cwdag=cwdag, cxdag=cxdag, cydag=cydag, czdag=czdag, 
                   cu=cu, cv=cv, cw=cw, cx=cx, cy=cy, cz=cz)
        
        Site.__init__(self, leg, names, **ops)
        
class MPOMPS():
    def __init__(self, v, u, **kwargs):
        self.cons_N = kwargs.get("cons_N", None)
        self.cons_S = kwargs.get("cons_S", None)
        self.trunc_params = kwargs.get("trunc_params", dict(chi_max=20) )
        self.pbc = kwargs.get("pbc", -1)
        
        assert v.ndim == 2
        self._V = v
        self._U = u
        assert self._U.shape == self._V.shape
        self.projection_type = kwargs.get("projection_type", "Gutz")

        self.L = self.Llat = u.shape[0] #the length of real sites
        
        self.site = sixparton(self.cons_N, self.cons_S)
        self.init_mps()
    
    def init_mps(self, init=None):
        L = self.L
        if init is None:
            if self.pbc == -1 or self.pbc == 0:
                init = [0] * L #all empty
            if self.pbc == 1:
                init = [1] + [0]*(L-1) #a_u^\dagger \ket{0}_a
        site = self.site
        self.init_psi = MPS.from_product_state([site]*L, init)
        self.psi = self.init_psi.copy()
        self.n_omode = 0
        return self.psi
    
    def get_mpo_trivial(self, v, u, flavor):
        chinfo = self.site.leg.chinfo
        pleg = self.site.leg #physical leg of parton site

        firstleg = npc.LegCharge.from_trivial(1)
        lastleg = npc.LegCharge.from_trivial(1)
        bulkleg = npc.LegCharge.from_trivial(2)
        #legs arrange in order 'wL', 'wR', 'p', 'p*'
        legs_first = [firstleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_bulk = [bulkleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_last = [bulkleg, lastleg, pleg, pleg.conj()]
        
        mpo = []
        L = self.L
        
        op_dict = {'u': ('cudag', 'cu'), 'v': ('cvdag', 'cv'), 'w': ('cwdag', 'cw'), 
                   'x': ('cxdag', 'cx'), 'y': ('cydag', 'cy'), 'z': ('czdag', 'cz')}
        
        t0 = npc.zeros(legs_first, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        i = 0
        if flavor in op_dict:
            cr, an = op_dict[flavor]
            t0[0, 0, :, :] = v[i]*self.site.get_op(cr) + u[i]*self.site.get_op(an)
        t0[0, 1, :, :] = self.site.get_op('JW')
        mpo.append(t0)
        
        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
            ti[0,0,:,:] = self.site.get_op('id64')
            if flavor in op_dict:
                cr, an = op_dict[flavor]
                ti[1, 0, :, :] = v[i]*self.site.get_op(cr) + u[i]*self.site.get_op(an)
            ti[1, 1, :, :] = self.site.get_op('JW')
            mpo.append(ti)
                
        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        tL[0,0,:,:] = self.site.get_op('id64')
        if flavor in op_dict:
            cr, an = op_dict[flavor]
            tL[1, 0, :, :] = v[i]*self.site.get_op(cr) + u[i]*self.site.get_op(an)
        mpo.append(tL)
        
        return mpo
    
    def mpomps_step_1time(self, m, flavor):
        vm = self._V[:,m]
        um = self._U[:,m]
        mps = self.psi
        if self.cons_N is None and self.cons_S is None:
            mpo = self.get_mpo_trivial(vm, um, flavor)
        elif self.cons_N=='Z2' and self.cons_S=='flavor':
            mpo = self.get_mpo_Z2U1(vm, um, flavor)
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
        
        flavorlist = ['u','v','w','x','y','z']
        for m in range(nmode):
            for flavor in flavorlist:
                err, self.psi = self.mpomps_step_1time(m, flavor)
                self.fidelity *= 1-err.eps
                self.chi_max = np.max(self.psi.chi)
                print( "applied the {}-th {} mode, the fidelity is {}, the largest bond dimension is {}. ".format( self.n_omode, flavor, self.fidelity, self.chi_max) )
            self.n_omode += 1

'''
------------------------------------------------------Extra functions---------------------------------------------
'''
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

def GutzwillerProjectionParton2Spin(partonpsi):
    """
    The GP function to project a parton mps onto a spin mps

    sixparton site -> so6 site

    Args:
        partonpsi (tenpy.MPS object): the parton MPS
        
    Returns:
        spinpsi (tenpy.MPS object): the spin MPS
    """
    partonsite = partonpsi.sites[0]
    cons_N, cons_S = partonsite.conserve
    partonleg = partonsite.leg
    
    so6gen, cmn = get_opr_list()
    spinsite = SO6Site(so6gen, cons_N, cons_S)
    spinleg = spinsite.leg
    
    if cons_N == 'Z2' and cons_S == 'flavor':
        qtotal = [0, 0]
    else:
        qtotal = None
        
    projector = npc.zeros([spinleg, partonleg.conj()], qtotal=qtotal, labels=['p','p*'], dtype=partonpsi.dtype)
    projector[0,1] = 1 #0th spin index <=> u parton
    projector[1,2] = 1
    projector[2,3] = 1
    projector[3,4] = 1
    projector[4,5] = 1
    projector[5,6] = 1
    
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
    parser.add_argument("-lx", type=int, default=6)
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=1.)
    parser.add_argument("-lamb", type=float, default=0.)
    parser.add_argument("-D", type=int, default=64)
    parser.add_argument("-pbc", type=int, default=-1)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=0.1666666667)
    parser.add_argument("-verbose", type=int, default=1)
    args = parser.parse_args()
        
    np.random.seed(0)
    
    chi = args.chi
    delta = args.delta
    lamb = args.lamb
    lx = args.lx
    pbc = args.pbc
    D = args.D
    J = args.J
    K = args.K
    
    if pbc == 1 or pbc==-1:
        Econst = (5/3) * K * lx
    elif pbc == 0:
        Econst = (5/3) * K * (lx-1)
    
    print("----------Build single Kitaev chain Hamiltonian----------")
    singlechain = KitaevSingleChain(chi, delta, lamb, lx, pbc)
    singlechain.calc_hamiltonian()
    vmat = singlechain.V
    umat = singlechain.U

    print("----------Build MLWO----------")
    wv, wu = Wannier_Z2(vmat.T, umat.T)

    print("----------MPO-MPS method: MLWO----------")
    params_mpomps = dict(cons_N=None, cons_S=None, trunc_params=dict(chi_max=D), pbc=pbc)
    mpos = MPOMPS(wv, wu, **params_mpomps)
    mpos.run()
    
    print("----------Gutzwiller projection to SO(6) site----------")
    psimlwo = mpos.psi
    gppsimlwo = GutzwillerProjectionParton2Spin(psimlwo)
    
    print("----------SO(6) Spin1 model DMRG---------")
    model_params = dict(cons_N=None, cons_S=None, Lx = lx, pbc=pbc, J=J, K=K, D=216, sweeps=6, verbose=2)
    so6dmrgmodel = BBQJK(model_params)
    psidmrg, Edmrg = so6dmrgmodel.run_dmrg()
    psidmrg2, Edmrg2 = so6dmrgmodel.run_dmrg_orthogonal([psidmrg])
    print("SO(6) DMRG results")
    print("psi1 after DMRG is", psidmrg)
    print("psi2 after DMRG is", psidmrg2)
    print("SO(6) DMRG Energy is", Edmrg, Edmrg2)
    
    print("----------sandwiches----------")
    bbqmpo = so6dmrgmodel.calc_H_MPO()
    print(" ")
    print("the sandwich of projected psimlwo and SO(6) MPO is", bbqmpo.expectation_value(gppsimlwo)+Econst)
    print(" ")
    print("the overlap of psidmrg and gppsimlwo", psidmrg.overlap(gppsimlwo))
    print(" ")
    print("the overlap of psidmrg2 and gppsimlwo", psidmrg2.overlap(gppsimlwo))