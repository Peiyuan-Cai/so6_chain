"""
The MPO-MPS method for SO(3) spin1 chain, with Z(2) symmetry from spin up/dn and U(1) symmetry from x,y,z flavor. 

20240827: this code doesn't work. 
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
    """
    The 3 in 1 parton site for MPO-MPS method, different from the one for DMRG
    
    local physical leg dimension = 8 = 2**3
    empty, single occupancy of(x,y,z), double occupancy, full
    
    this site is a combination of 3 parton sites, and for the MPOS method, there is no need to define operators here
    """
    def __init__(self, cons_N=None, cons_S=None):
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        if cons_N == 'N' and cons_S == 'xyz':
            chinfo = npc.ChargeInfo([1, 1], ['N', 'xyz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[0, 0], [1, -1], [1, 0], [1, 1], [2, -1], [2, 0], [2, 1], [3, 0]])
        elif cons_N == 'Z2' and cons_S == 'xyz':
            chinfo = npc.ChargeInfo([1, 1], ['N', 'xyz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 0], [1, -1], [1, 0], [1, 1], [1, -1], [1, 0], [1, 1], [1, 0]])
        else:
            leg = npc.LegCharge.from_trivial(8)
        
        names = ['empty', 'x', 'y', 'z', 'xy', 'zx', 'yz', 'xyz']
        
        JW = np.diag([1,-1,-1,-1,1,1,1,-1]) #the F matrix
        #the 2x2 operators
        id8 = np.diag([1,1,1,1,1,1,1,1])
        
        cxdag = np.zeros((8,8))
        cxdag[1,0] = 1; cxdag[4,2] = 1; cxdag[5,3] = 1; cxdag[7,6] = 1; 
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
        
        if xyz == -1: #v cxdag + u cz
            qn = [0,-1]
        elif xyz == 0: #v cydag + u cy
            qn = [0, 0]
        elif xyz == 1: #v czdag + u cx
            qn = [0, 1]
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
        if xyz == -1:
            t0[0, 0, 1, 0] = v[0]; t0[0, 0, 4, 2] = v[0]; t0[0, 0, 5, 3] = v[0]; t0[0, 0, 7, 6] = v[0]; #v cxdag
            t0[0, 0, 0, 3] = u[0]; t0[0, 0, 1, 5] = -u[0]; t0[0, 0, 2, 6] = -u[0]; t0[0, 0, 4, 7] = u[0]; #u F cz
        elif xyz == 0:
            t0[0, 0, 2, 0] = v[0]; t0[0, 0, 4, 1] = -v[0]; t0[0, 0, 6, 3] = v[0]; t0[0, 0, 7, 5] = -v[0]; #v cydag F
            t0[0, 0, 0, 2] = u[0]; t0[0, 0, 1, 4] = -u[0]; t0[0, 0, 3, 6] = u[0]; t0[0, 0, 5, 7] = -u[0]; #u F cy
        elif xyz == 1:
            t0[0, 0, 3, 0] = v[0]; t0[0, 0, 5, 1] = -v[0]; t0[0, 0, 6, 2] = -v[0]; t0[0, 0, 7, 4] = v[0]; #v czdag F
            t0[0, 0, 0, 1] = u[0]; t0[0, 0, 2, 4] = u[0]; t0[0, 0, 3, 5] = u[0]; t0[0, 0, 6, 7] = u[0]; #u cx
            
        t0[0, 1, 0, 0] = 1; t0[0, 1, 1, 1] = -1; t0[0, 1, 2, 2] = -1; t0[0, 1, 3, 3] = -1; 
        t0[0, 1, 4, 4] = 1; t0[0, 1, 5, 5] = 1; t0[0, 1, 6, 6] = 1; t0[0, 1, 7, 7] = -1; 
        mpo.append(t0)
        
        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
            ti[0,0,0,0] = 1; ti[0,0,1,1] = 1; ti[0,0,2,2] = 1; ti[0,0,3,3] = 1; 
            ti[0,0,4,4] = 1; ti[0,0,5,5] = 1; ti[0,0,6,6] = 1; ti[0,0,7,7] = 1; 
            if xyz == -1:
                ti[1, 0, 1, 0] = v[i]; ti[1, 0, 4, 2] = v[i]; ti[1, 0, 5, 3] = v[i]; ti[1, 0, 7, 6] = v[i]; 
                ti[1, 0, 0, 3] = u[i]; ti[1, 0, 1, 5] = -u[i]; ti[1, 0, 2, 6] = -u[i]; ti[1, 0, 4, 7] = u[i]; 
            elif xyz == 0:
                ti[1, 0, 2, 0] = v[i]; ti[1, 0, 4, 1] = -v[i]; ti[1, 0, 6, 3] = v[i]; ti[1, 0, 7, 5] = -v[i]; 
                ti[1, 0, 0, 2] = u[i]; ti[1, 0, 1, 4] = -u[i]; ti[1, 0, 3, 6] = u[i]; ti[1, 0, 5, 7] = -u[i]; 
            elif xyz == 1:
                ti[1, 0, 3, 0] = v[i]; ti[1, 0, 5, 1] = -v[i]; ti[1, 0, 6, 2] = -v[i]; ti[1, 0, 7, 4] = v[i]; 
                ti[1, 0, 0, 1] = u[i]; ti[1, 0, 2, 4] = u[i]; ti[1, 0, 3, 5] = u[i]; ti[1, 0, 6, 7] = u[i]; 
            ti[1, 1, 0, 0] = 1; ti[1, 1, 1, 1] = -1; ti[1, 1, 2, 2] = -1; ti[1, 1, 3, 3] = -1; 
            ti[1, 1, 4, 4] = 1; ti[1, 1, 5, 5] = 1; ti[1, 1, 6, 6] = 1; ti[1, 1, 7, 7] = -1; 
            mpo.append(ti)
                
        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        tL[0,0,0,0] = 1; tL[0,0,1,1] = 1; tL[0,0,2,2] = 1; tL[0,0,3,3] = 1; 
        tL[0,0,4,4] = 1; tL[0,0,5,5] = 1; tL[0,0,6,6] = 1; tL[0,0,7,7] = 1; 
        if xyz == -1:
            tL[1, 0, 1, 0] = v[i]; tL[1, 0, 4, 2] = v[i]; tL[1, 0, 5, 3] = v[i]; tL[1, 0, 7, 6] = v[i]; 
            tL[1, 0, 0, 3] = u[i]; tL[1, 0, 1, 5] = -u[i]; tL[1, 0, 2, 6] = -u[i]; tL[1, 0, 4, 7] = u[i]; 
        elif xyz == 0:
            tL[1, 0, 2, 0] = v[i]; tL[1, 0, 4, 1] = -v[i]; tL[1, 0, 6, 3] = v[i]; tL[1, 0, 7, 5] = -v[i]; 
            tL[1, 0, 0, 2] = u[i]; tL[1, 0, 1, 4] = -u[i]; tL[1, 0, 3, 6] = u[i]; tL[1, 0, 5, 7] = -u[i]; 
        elif xyz == 1:
            tL[1, 0, 3, 0] = v[i]; tL[1, 0, 5, 1] = -v[i]; tL[1, 0, 6, 2] = -v[i]; tL[1, 0, 7, 4] = v[i]; 
            tL[1, 0, 0, 1] = u[i]; tL[1, 0, 2, 4] = u[i]; tL[1, 0, 3, 5] = u[i]; tL[1, 0, 6, 7] = u[i]; 
        mpo.append(tL)
        
        return mpo
    
    def mpomps_step_3times(self, m):
        """
        m = 1,2,3,...,3L
        run this function once for applying mpo -1, 0, 1 on psi, totally 3 times
        """
        vm = self._V[:,m]
        um = self._U[:,m]
        xyzlist = [-1, 0, 1]
        mps = self.psi
        for xyz in xyzlist:
            print("applying the {} mode".format(xyz))
            mpo = self.get_mpo_Z2U1(vm, um, xyz)
            for i in range(self.L):
                B = npc.tensordot(mps.get_B(i,'B'), mpo[i], axes=('p','p*'))
                B = B.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
                B.ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
                B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
                B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
                mps._B[i] = B
            mps.compress_svd(self.trunc_params)
        return mps
    
    def mpomps_step_1time(self, m, xyz):
        vm = self._V[:,m]
        um = self._U[:,m]
        mps = self.psi
        mpo = self.get_mpo_Z2U1(vm, um, xyz)
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
        
        xyzlist = [-1, 0, 1]
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
    def __init__(self, cons_N=None):
        cons_S = 'xyz'
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        if cons_N is None and cons_S == 'xyz':
            chinfo = npc.ChargeInfo([1, 1], ['N', 'xyz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, -1], [1, 0], [1, +1]])
        elif cons_N == 'N' and cons_S == 'xyz':
            chinfo = npc.ChargeInfo([1, 1], ['N', 'xyz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, -1], [1, 0], [1, +1]])
        else:
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

class BBQJK(CouplingModel):
    def __init__(self, model_params):
        print(model_params)
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        self.Lx = model_params.get('Lx', 10)
        self.S = model_params.get('S', 1)
        self.bc = model_params.get('bc', 'periodic')
        self.J = model_params.get('J', 1)
        self.K = model_params.get('K', 1/3)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        conserve = model_params.get('conserve', 'parity')
        self.verbose = model_params.get('verbose', 2)
        
        site = Spin1(cons_N=None)
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
            self.add_coupling_term(K, i0, i1, "Szx", "Sxz")
            self.add_coupling_term(K, i0, i1, "Sxz", "Szx")
            self.add_coupling_term(J-K,  i0, i1, "Sxy", "Szy")
            self.add_coupling_term(J-K,  i0, i1, "Syx", "Syz")
            self.add_coupling_term(J-K,  i0, i1, "Syz", "Syx")
            self.add_coupling_term(J-K,  i0, i1, "Szy", "Sxy")
            self.add_coupling_term((J+K)/4,  i0, i1, "Q1", "Q1")
            self.add_coupling_term((J-K)*np.sqrt(3)/4,  i0, i1, "Q1", "Q2")
            self.add_coupling_term((J-K)*np.sqrt(3)/4,  i0, i1, "Q2", "Q1")
            self.add_coupling_term((J*3-K)/4,  i0, i1, "Q2", "Q2")
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

    spin1_site = Spin1(cons_N='N')
    spin1_leg = spin1_site.leg

    #the projection shouldn't change the qns
    if cons_N == 'Z2' and cons_S == 'xyz':
        qtotal = [0, 0]
    else:
        qtotal = [0]

    projector = npc.zeros([spin1_leg, threeparton_leg.conj()], qtotal=qtotal, labels=['p','p*'], dtype=psi.dtype)
    projector[0,1] = 1 #single occupied x parton
    projector[1,2] = 1 #y parton
    projector[2,3] = 1 #z parton
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
    params_mpompsz2u1 = dict(cons_N="Z2", cons_S='xyz', trunc_params=dict(chi_max=D))
    mpos = MPOMPS(vmat, umat, **params_mpompsz2u1)
    mpos.run()

    print("----------Gutzwiller projection to Spin1 site----------")
    psi1 = mpos.psi
    gppsi = GutzwillerProjection2Spin1(psi1)

    print("----------Z2U1 MPO-MPS method: MLWO----------")
    params_mpompsz2u1 = dict(cons_N="Z2", cons_S='xyz', trunc_params=dict(chi_max=D))
    mpos = MPOMPS(wv, wu, **params_mpompsz2u1)
    mpos.run()

    print("----------Gutzwiller projection to Spin1 site----------")
    psimlwo = mpos.psi
    gppsimlwo = GutzwillerProjection2Spin1(psimlwo)

    print("----------SU(3) Spin1 model DMRG---------")
    su3dmrgmodel = BBQJK(model_params)
    #sites2 = su3dmrgmodel.sites
    sites2 = [Spin1(cons_N=None)] * lx
    psi2 = MPS.from_product_state(sites2, [1]*lx, "finite")
    #psi2.norm = 1
    psi2.canonical_form()
    dmrg_params2 = dict(mixer=True, max_E_err=1.e-12 , max_chi = 20)
    eng2 = dmrg.TwoSiteDMRGEngine(psi2, su3dmrgmodel, dmrg_params2)
    E2, psi2 = eng2.run()
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