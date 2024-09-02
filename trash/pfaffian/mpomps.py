import numpy as np
from tenpy.networks.site import Site
import tenpy.linalg.np_conserved as npc
from tenpy.networks import MPS

class onepartontrivial(Site):
    def __init__(self, cons_N=None, cons_S=None):
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        leg = npc.LegCharge.from_trivial(2)
        
        names = ['unoccupied', 'occupied']
        
        id2 = np.diag([1,1])
        
        ops = dict(id=id2)
        
        Site.__init__(self, leg, names, **ops)

class MPOMPSfullham_trivial():
    def __init__(self, v, u, **kwargs):
        self.cons_N = kwargs.get("cons_N", None)
        self.cons_S = kwargs.get("cons_S", None)
        self.trunc_params = kwargs.get("trunc_params", dict(chi_max=20) )
        
        assert v.ndim == 2
        self._V = v
        self._U = u
        assert self._U.shape == self._V.shape
        self.projection_type = kwargs.get("projection_type", "Gutz")
        #self.L = self.Llat = u.shape[1]//3 #for 3 in 1 site, only 1 Kitaev chain should be calculated
        self.L = u.shape[0] #the length of the interleaved sites
        
        self.site = onepartontrivial(self.cons_N, self.cons_S)
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
    
    def get_mpo_trivial(self, v, u):
        pleg = self.site.leg #physical leg, for full hamiltonian, dimension is 2
        firstleg = npc.LegCharge.from_trivial(1)
        lastleg = npc.LegCharge.from_trivial(1)
        bulkleg = npc.LegCharge.from_trivial(2)
        legs_first = [firstleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_bulk = [bulkleg, bulkleg.conj(), pleg, pleg.conj()]
        legs_last = [bulkleg, lastleg.conj(), pleg, pleg.conj()]
        
        mpo = []
        L = self.L
        
        i = 0
        t0 = npc.zeros(legs_first, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        t0[0,0,0,1] = v[0]; t0[0,0,1,0] = u[0]; t0[0,1,0,0] = 1; t0[0,1,1,1] = -1; 
        mpo.append(t0)
        
        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
            ti[0,0,0,0] = 1; ti[0,0,1,1] = 1; 
            ti[1,0,0,1] = v[i]; ti[1,0,1,0] = u[i]; 
            ti[1,1,0,0] = 1; ti[1,1,1,1] = -1; 
            mpo.append(ti)
            
        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        tL[0,0,0,0] = 1; tL[0,0,1,1] = 1; 
        tL[1,0,0,1] = v[i]; tL[1,0,1,0] = u[i]; 
        mpo.append(tL)
        
        return mpo
    
    def mpomps_step(self, m):
        #m = 1,2,...,L counting to the interleaved site number
        vm = self._V[:,m]
        um = self._U[:,m]
        mps = self.psi
        mpo = self.get_mpo_trivial(vm,um)
        for i in range(self.L):
            B = npc.tensordot(mps.get_B(i,'B'), mpo[i], axes=('p','p*'))
            B = B.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
            B.ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
            B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
            B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
            mps._B[i] = B
        err = mps.compress_svd(self.trunc_params)
        return err, mps
    
    def run(self):
        self.fidelity = 1
        if self.n_omode > 0:
            print("initialize the mpo-mps calculation mps")
            self.init_mps(init=None)
            self.n_omode = 0
        nmode = self._U.shape[0]
        print("MPO-MPS application start")

        for m in range(nmode):
            err, self.psi = self.mpomps_step(m)
            self.fidelity *= 1-err.eps
            self.chi_max = np.max(self.psi.chi)
            print( "applied the {}-th mode, the fidelity is {}, the largest bond dimension is {}. ".format( self.n_omode, self.fidelity, self.chi_max) )
            self.n_omode += 1
            
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
            leg = npc.LegCharge.from_qflat(chinfo, [[0, 0], [1, 1], [1, -1], [1, 0], [2, 0], [2, 1], [2, -1], [3, 0]])
        elif cons_N == 'Z2' and cons_S == 'xyz':
            chinfo = npc.ChargeInfo([1, 1], ['Z2', 'xyz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 0], [1, 1], [1, -1], [1, 0], [1, 0], [1, 1], [1, -1], [1, 0]])
        elif cons_N == 'Z2' and cons_S == None:
            chinfo = npc.ChargeInfo([1], ['Z2'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1],[1],[1],[1],[1],[1],[1],[1]])
        else:
            print("No symmetry used in site 'threeparton'. ")
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
        
        xyzlist = [-1, 0, 1]
        for m in range(nmode):
            for xyz in xyzlist:
                err, self.psi = self.mpomps_step_1time(m, xyz)
                self.fidelity *= 1-err.eps
                self.chi_max = np.max(self.psi.chi)
                print( "applied the {}-th {} mode, the fidelity is {}, the largest bond dimension is {}. ".format( self.n_omode, xyz, self.fidelity, self.chi_max) )
            self.n_omode += 1