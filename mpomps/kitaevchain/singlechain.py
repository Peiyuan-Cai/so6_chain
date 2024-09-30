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
from copy import deepcopy
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True

class KitaevSingleChain():
    def __init__(self, chi, delta, lamb, L, pbc):
        """
        The Single Kitaev chain class. 
        
        The Hamiltonian is in the BdG form, the matrix is written under the Bogoliubov *QUASIHOLE* representation. 

        Args:
            chi (float): variational parameter $\chi$
            delta (float): variational parameter $\delta$
            lamb (float): variational parameter $\lambda$
            L (int): Chain length
            pbc (int, optional): Boundary condition. 0 for OBC, 1 for PBC, -1 for APBC. 
            
        Raises:
            Check pbc must be 0:open or 1:periodic or -1:anti-periodic: check your boundary condition input
            
        Notes:
            240827: We should now have three types of boundary conditions namely 0(open), 1(periodic) and -1(anti-periodic), and cancel the 'bc' stuff
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

class singleparton(Site):
    def __init__(self):
        leg = npc.LegCharge.from_trivial(2)

        names = ['0', '1'] #empty and occupied

        #operators
        id = np.eye(2)
        JW = np.array([[1,0],[0,-1]])
        cdag = np.array([[0,0],[1,0]]); c = cdag.T
        ops = dict(id=id, JW=JW, cdag=cdag, c=c)

        Site.__init__(self, leg, names, **ops)

class MPOMPS():
    def __init__(self, v, u, **kwargs):
        self.trunc_params = kwargs.get("trunc_params", dict(chi_max=64) )
        self.pbc = kwargs.get("pbc", -1)

        assert v.ndim == 2
        self._V = v
        self._U = u
        assert self._U.shape == self._V.shape
        self.projection_type = kwargs.get("projection_type", "Gutz")

        self.L = self.Llat = u.shape[0] #the length of real sites
        
        self.site = singleparton()
        self.init_mps()

    def init_mps(self, init=None):
        L = self.L
        if init is None:
            if self.pbc == -1 or self.pbc == 0:
                #init = [1,1] + [0]*(L-2) #even parity
                init = [0] * L #all empty
            if self.pbc == 1:
                init = [1] + [0]*(L-1) #a_{1}^\dagger \ket{0}_a
        print("the initial state is", init)
        site = self.site
        self.init_psi = MPS.from_product_state([site]*L, init)
        self.psi = self.init_psi.copy()
        self.n_omode = 0
        return self.psi
    
    def get_mpo_trivial(self, v, u):
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
        
        t0 = npc.zeros(legs_first, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        i = 0
        cr, an = 'cdag', 'c'
        t0[0, 0, :, :] = v[i]*self.site.get_op(cr) + u[i]*self.site.get_op(an)
        t0[0, 1, :, :] = self.site.get_op('JW')
        mpo.append(t0)

        for i in range(1,L-1):
            ti = npc.zeros(legs_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
            ti[0,0,:,:] = self.site.get_op('id')
            cr, an = 'cdag', 'c'
            ti[1, 0, :, :] = v[i]*self.site.get_op(cr) + u[i]*self.site.get_op(an)
            ti[1, 1, :, :] = self.site.get_op('JW')
            mpo.append(ti)

        i = L-1
        tL = npc.zeros(legs_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype)
        tL[0,0,:,:] = self.site.get_op('id')
        cr, an = 'cdag', 'c'
        tL[1, 0, :, :] = v[i]*self.site.get_op(cr) + u[i]*self.site.get_op(an)
        mpo.append(tL)
        
        return mpo
    
    def mpomps_step_1time(self, m):
        vm = self._V[:,m]
        um = self._U[:,m]
        mps = self.psi
        mpo = self.get_mpo_trivial(vm, um)
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
        
        for m in range(nmode):
            err, self.psi = self.mpomps_step_1time(m)
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=8)
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=1.)
    parser.add_argument("-lamb", type=float, default=0.)
    parser.add_argument("-Dmpos", type=int, default=64)
    parser.add_argument("-pbc", type=int, default=2)
    args = parser.parse_args()

    np.random.seed(0)

    chi = args.chi
    delta = args.delta
    lamb = args.lamb
    lx = args.lx
    pbc = args.pbc
    Dmpos = args.Dmpos

    if pbc == 2:
        print("Generating two ground states by APBC and PBC at a time. ")
        pbc1 = -1
        pbc2 = 1

    print(" ")
    print("APBC case MPOMPS")
    print("----------Build single Kitaev chain Hamiltonian----------")
    singlechain = KitaevSingleChain(chi, delta, lamb, lx, pbc1)
    singlechain.calc_hamiltonian()
    vmat = singlechain.V
    umat = singlechain.U

    print("----------Build MLWO----------")
    wv, wu = Wannier_Z2(vmat.T, umat.T)

    print("----------MPO-MPS method: MLWO----------")
    params_mpomps = dict(trunc_params=dict(chi_max=Dmpos), pbc=pbc1)
    mpos = MPOMPS(wv, wu, **params_mpomps)
    mpos.run()
    psimlwo_apbc = mpos.psi

    print(" ")
    print("PBC case MPOMPS")
    print("----------Build single Kitaev chain Hamiltonian----------")
    singlechain = KitaevSingleChain(chi, delta, lamb, lx, pbc2)
    singlechain.calc_hamiltonian()
    vmat = singlechain.V
    umat = singlechain.U

    print("----------Build MLWO----------")
    wv, wu = Wannier_Z2(vmat.T, umat.T)

    print("----------MPO-MPS method: MLWO----------")
    params_mpomps = dict(trunc_params=dict(chi_max=Dmpos), pbc=pbc2)
    mpos = MPOMPS(wv, wu, **params_mpomps)
    mpos.run()
    psimlwo_pbc = mpos.psi

    dimercheck = 1
    if dimercheck == 1:
        print(" Dimercheck ")
        def mps1_mpo_mps2(mps1, opr_string, mps2):
            assert len(mps1._B) == len(opr_string) == len(mps2._B)
            site = mps1.sites[0]
            L = len(mps1._B)
            temp = npc.tensordot(mps1._B[0].conj(), site.get_op(opr_string[0]), axes=('p*', 'p'))
            left = npc.tensordot(temp, mps2._B[0], axes=('p*', 'p'))
            for _ in range(1, L):
                temp = npc.tensordot(mps1._B[_].conj(), site.get_op(opr_string[_]), axes=('p*', 'p'))
                left = npc.tensordot(left, temp, axes=(['vR*'],["vL*"]))
                left = npc.tensordot(left, mps2._B[_], axes=(['vR','p*'],['vL','p']))
            value = left.to_ndarray()
            return value.reshape(-1)[0]*mps1.norm*mps2.norm
        
        def trnslop_mpo(site, L=2, **kwargs):
            """
            get the MPO of translational operator with given site and length

            output:
                1. list of npc.Arrays
            """
            bc = kwargs.get('bc', 'pbc')    
            assert L>1
            leg = site.leg
            chinfo = leg.chinfo
            zero_div = [0]*chinfo.qnumber
            from tenpy.linalg.charges import LegPipe, LegCharge
            cleg = LegPipe([leg, leg.conj()], sort=False, bunch=False).to_LegCharge()
            nleg = npc.LegCharge.from_qflat(chinfo, [zero_div])
            
            swap = npc.zeros([leg, leg.conj(), leg, leg.conj()], qtotal=zero_div, labels=['p1', 'p1*', 'p2', 'p2*']) 
            for _i in range( site.dim ):
                for _j in range( site.dim ):
                    swap[_j,_i,_i,_j] = 1
                    
            reshaper = npc.zeros([leg, leg.conj(), cleg.conj()], qtotal=zero_div, labels=['p', 'p*', '(p*.p)'] )
            for _i in range( site.dim ):
                for _j in range( site.dim ):          
                    idx = _i* site.dim + _j 
                    reshaper[_i,_j,idx] = 1
                        
            swap = npc.tensordot(reshaper.conj(), swap, axes=((0,1),(0,1)))
            swap.ireplace_labels(['(p.p*)'], ['p1.p1*'])
            swap = npc.tensordot(swap, reshaper.conj(), axes=((1,2),(0,1)))
            swap.ireplace_labels(['(p.p*)'], ['p2.p2*'])
            
            left, right = npc.qr(swap)
                
            left  = npc.tensordot(reshaper, left, axes=((2), (0)))
            left.ireplace_labels([None], ['wR'])
            
            right = npc.tensordot(right, reshaper, axes=((1), (2)))
            right.ireplace_labels([None], ['wL'])
            
            bulk = npc.tensordot(right, left, axes=('p*','p'))

            if bc == 'pbc':
                bt = npc.zeros([nleg, leg, leg.conj()], qtotal=zero_div, labels=['wL', 'p', 'p*',])
                for _i in range( site.dim ):
                    bt[0,_i,_i] = 1
                left = npc.tensordot(bt, left, axes=(('p*', 'p')))
            
                bt = npc.zeros([leg, leg.conj(), nleg.conj()], qtotal=zero_div, labels=['p', 'p*', 'wR'])
                for _i in range( site.dim ):
                    bt[_i,_i,0] = 1
                right = npc.tensordot(right, bt, axes=(('p*', 'p')))  

            return [left] +  [bulk]*(L-2) +  [right] 
        
        def trnslop_mpo_fermion(site, L=2, **kwargs):
            bc = kwargs.get('bc', 'pbc')    
            assert L>1
            leg = site.leg
            chinfo = leg.chinfo
            zero_div = [0]*chinfo.qnumber
            from tenpy.linalg.charges import LegPipe, LegCharge
            cleg = LegPipe([leg, leg.conj()], sort=False, bunch=False).to_LegCharge()
            nleg = npc.LegCharge.from_qflat(chinfo, [zero_div])
            
            swap = npc.zeros([leg, leg.conj(), leg, leg.conj()], qtotal=zero_div, labels=['p1', 'p1*', 'p2', 'p2*']) 
            for _i in range( site.dim ):
                for _j in range( site.dim ):
                    if _i == 1 and _j == 1:
                        swap[_j,_i,_i,_j] = -1
                    else:
                        swap[_j,_i,_i,_j] = 1
                    
            reshaper = npc.zeros([leg, leg.conj(), cleg.conj()], qtotal=zero_div, labels=['p', 'p*', '(p*.p)'] )
            for _i in range( site.dim ):
                for _j in range( site.dim ):
                    idx = _i* site.dim + _j 
                    reshaper[_i,_j,idx] = 1
                        
            swap = npc.tensordot(reshaper.conj(), swap, axes=((0,1),(0,1)))
            swap.ireplace_labels(['(p.p*)'], ['p1.p1*'])
            swap = npc.tensordot(swap, reshaper.conj(), axes=((1,2),(0,1)))
            swap.ireplace_labels(['(p.p*)'], ['p2.p2*'])
            
            left, right = npc.qr(swap)
                
            left  = npc.tensordot(reshaper, left, axes=((2), (0)))
            left.ireplace_labels([None], ['wR'])
            
            right = npc.tensordot(right, reshaper, axes=((1), (2)))
            right.ireplace_labels([None], ['wL'])
            
            bulk = npc.tensordot(right, left, axes=('p*','p'))

            if bc == 'pbc':
                bt = npc.zeros([nleg, leg, leg.conj()], qtotal=zero_div, labels=['wL', 'p', 'p*',])
                for _i in range( site.dim ):
                    bt[0,_i,_i] = 1
                left = npc.tensordot(bt, left, axes=(('p*', 'p')))
            
                bt = npc.zeros([leg, leg.conj(), nleg.conj()], qtotal=zero_div, labels=['p', 'p*', 'wR'])
                for _i in range( site.dim ):
                    bt[_i,_i,0] = 1
                right = npc.tensordot(right, bt, axes=(('p*', 'p')))  

            return [left] +  [bulk]*(L-2) +  [right] 

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
        
        '''
        tpsi1 = deepcopy(gppsimlwo_apbc)
        tpsi1 = apply_mpo(transop, tpsi1)
        tpsi1.canonical_form()
        print(tpsi1.chi)
        print("<projected_pbc|T|projected_apbc> is", psimlwo_pbc.overlap(psimlwo_apbc))
        '''
        
        site = psimlwo_apbc.sites[0]
        transop = trnslop_mpo_fermion(site, lx)

        tpsi1 = deepcopy(psimlwo_apbc)
        tpsi1 = apply_mpo(transop, tpsi1)
        tpsi1.canonical_form()
        print("<unprojected_pbc|T|unprojected_apbc> is", psimlwo_pbc.overlap(tpsi1))

        tpsi1 = deepcopy(psimlwo_pbc)
        tpsi1 = apply_mpo(transop, tpsi1)
        tpsi1.canonical_form()
        print("<unprojected_pbc|T|unprojected_pbc> is", psimlwo_pbc.overlap(tpsi1))

        tpsi1 = deepcopy(psimlwo_apbc)
        tpsi1 = apply_mpo(transop, tpsi1)
        tpsi1.canonical_form()
        print("<unprojected_apbc|T|unprojected_apbc> is", psimlwo_apbc.overlap(tpsi1))