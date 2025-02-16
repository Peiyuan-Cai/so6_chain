"""
The Pfaffian method on SO(4) BBQ chain

The BdG Hamiltonian is of a single Kitaev chain, using fermionic swap gates to tensor product the tensors from different chains. 

Puiyuen 2025.02.16
"""

import numpy as np
from gaussianbdg import *
from tenpy.tools.params import asConfig
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import SpinSite, FermionSite, Site
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge, ChargeInfo
from copy import deepcopy
from tenpy.algorithms import dmrg
from tenpy.models.aklt import AKLTChain
import matplotlib.pyplot as plt
import numpy.linalg as LA
#import mpomps as mpos

def get_so4_opr_list_new():
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    id = np.eye(2)

    Sx = 0.5 * sigmax
    Sy = 0.5 * sigmay
    Sz = 0.5 * sigmaz

    L1 = -np.kron(Sz, id) - np.kron(id, Sz)
    L2 = -np.kron(Sx, id) + np.kron(id, Sx)
    L3 = -np.kron(Sy, id) - np.kron(id, Sy)
    L4 = -np.kron(Sy, id) + np.kron(id, Sy)
    L5 = +np.kron(Sx, id) + np.kron(id, Sx)
    L6 = -np.kron(Sz, id) + np.kron(id, Sz)

    Loprs = [L1, L2, L3, L4, L5, L6]
    Lhatoprs = [
    np.array([
        [-1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
    ]),
    np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ]),
    np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0]
    ]),
    np.array([
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ]),
    np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
    ]),
    np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ]),
    np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0]
    ]),
    np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
]

    coe_list = []
    for a in range(6):
        for b in range(6):
            LiLi = Loprs[a] @ Loprs[b]
            Amat = np.zeros((16, len(Lhatoprs)), dtype=complex)
            B = LiLi.reshape(-1,1)
            for l in range(len(Lhatoprs)):
                Amat[:,l] = Lhatoprs[l].reshape(-1,1)[:,0]
            pcoe = LA.solve(Amat, B)
            coe_list.append(pcoe)

    for i in range(len(coe_list)):
        coe_list[i] = coe_list[i].reshape(16)

    def pvec(a,b):
        return coe_list[6*a+b] #a,b=0,1,2,3,4,5
    
    cmn = np.zeros((16,16), dtype=complex)

    P = dict()
    for a in range(6):
        for b in range(6):
            P[(a,b)] = pvec(a,b)
    
    for m in range(16):
        for n in range(16):
            for a in range(6):
                for b in range(6):
                    cmn[m,n] += P[(a,b)][m] * P[(a,b)][n]

    coe_list_new = []
    for a in range(6):
        Li = Loprs[a]
        Amat = np.zeros((16, len(Lhatoprs)), dtype=complex)
        B = Li.reshape(-1,1)
        for l in range(len(Lhatoprs)):
            Amat[:,l] = Lhatoprs[l].reshape(-1,1)[:,0]
        qcoe = LA.solve(Amat, B)
        coe_list_new.append(qcoe)
    
    for i in range(len(coe_list_new)):
        coe_list_new[i] = coe_list_new[i].reshape(16)

    def qvec(a):
        return coe_list_new[a]
    
    dmn = np.zeros((16,16), dtype=complex)
    
    Q = dict()
    for a in range(6):
        Q[a] = qvec(a)

    for m in range(16):
        for n in range(16):
            for a in range(6):
                dmn[m,n] += Q[a][m] * Q[a][n]

    return Lhatoprs, cmn, dmn

class SO4Site(Site):
    def __init__(self, so4g, cons_N=None, cons_S=None):
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        self.so4g = so4g

        if self.cons_N is None and self.cons_S == 'U1':
            chinfo = npc.ChargeInfo([1,1], ['S','T'])
            leg = npc.LegCharge.from_qflat(chinfo, [[-1,0],[0,-1],[0,1],[1,0]])
        elif self.cons_N == 'Z2' and self.cons_S == None:
            chinfo = npc.ChargeInfo([2], ['Z2'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1],[1],[1],[1]])
        elif self.cons_N is None and self.cons_S is None:
            leg = npc.LegCharge.from_trivial(4)
        
        ops = dict()
        for i in range(len(self.so4g)):
            ops['L{}'.format(i)] = self.so4g[i]

        names = ['1u2u', '1d2u', '1u2d', '1d2d']
        Site.__init__(self, leg, names, **ops)

    def __repr__(self):
        return "trivial site for 16 so4 generators"
    
class BBQJKSO4(CouplingModel):
    def __init__(self, model_params):
        print(model_params)
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        self.Lx = model_params.get('Lx', 12)
        self.S = model_params.get('S', 1)
        self.bc = model_params.get('bc', 'periodic')
        self.J = model_params.get('J', 1)
        self.K = model_params.get('K', 1/4)
        self.verbose = model_params.get('verbose', 2)
        self.D = model_params.get('D', 200)
        self.sweeps = model_params.get('sweeps', 10)
        
        self.so4_generators, self.c_mn, self.d_mn = get_so4_opr_list_new()

        self.cons_N = model_params.get('cons_N', None)
        self.cons_S = model_params.get('cons_S', None)

        site = SO4Site(self.so4_generators, cons_N=self.cons_N, cons_S=self.cons_S)
        self.sites = [site] * self.Lx
        self.lat = Chain(self.Lx, site, bc=self.bc)
        CouplingModel.__init__(self, self.lat, explicit_plus_hc=False)
        self.init_terms(model_params)
        H_MPO = self.calc_H_MPO()
        if model_params.get('sort_mpo_legs', False):
            H_MPO.sort_legcharges()
        MPOModel.__init__(self, self.lat, H_MPO)
        model_params.warn_unused()

    def init_terms(self, model_params):
        J = model_params.get("J", 1.)
        K = model_params.get('K', 1/4)
        for l in range(self.Lx):
            if l < self.Lx - 1:
                i0, i1 = l, (l+1)%self.Lx
            elif l == self.Lx-1 and self.bc == 'periodic':
                i0, i1 = 0, self.Lx-1
                print("periodic terms added")
            else:
                break

            for m in range(16):
                for n in range(16):
                    self.add_coupling_term(J*np.round(self.d_mn[m,n],6), i0, i1, "L"+str(m), "L"+str(n))
                    self.add_coupling_term(K*np.round(self.c_mn[m,n],6), i0, i1, "L"+str(m), "L"+str(n))
    
    def run_dmrg(self, **kwargs):
        mixer      = kwargs.get('mixer', True)
        chi_max    = kwargs.get('chi_max', self.D)
        max_E_err  = kwargs.get('max_E_err', 1e-10)
        max_sweeps = kwargs.get('max_sweeps', self.sweeps)
        min_sweeps = kwargs.get('min_sweeps', min(4, max_sweeps) )
        dmrg_params = dict(mixer=mixer, 
                           trunc_params=dict(chi_max=chi_max),
                           max_E_err=max_E_err, 
                           max_sweeps=max_sweeps,
                           min_sweeps=min_sweeps,
                           verbose=2)

        init = kwargs.get('init', None)
        if init is None:
            N = self.lat.N_sites
            if N%4==0 and N>0:
                init = [0]*(N//4) + [1]*(N//4) + [2]*(N//4) + [3]*(N//4)
            else:
                raise("Check the system size must be integral multiple of 6")
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

class singlechain(bdg):
    """
    The Bogoliubov-de Gennes form of a SINGLE Kitaev chain.
    """
    def __init__(self, chi, d, lamb, Nx, D, pbc):
        self.model = "Kitaev single chain_L{}_chi{}_d{}_lambda{}_D{}".format(Nx, round(chi,6), round(d,6), round(lamb,6), D)
        super().__init__(Nx=Nx, Ny=1, model=self.model, D=D)
        self.t, self.d = round(-chi, 6), round(d, 6)
        self.mu = lamb
        self.dtype = np.float64
        self.pbc = pbc
        self.Nlatt = Nx #the REAL site number
        self.D = D #the cutoff dimension of pfaffian method
        
    def hamiltonian(self):
        N = self.Nlatt
        self.tmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        self.dmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        t, d = self.t, self.d
        mu = self.mu
        print("t=",self.t, "d=",self.d, "mu=",self.mu)
        for i in range(N):
            self.tmat[i, i] = mu/2 #why devided by 2 -> we will add the diagonal terms twice below
        for i in range(N-1):
            self.tmat[i, (i+1)%N] = t 
            self.dmat[i, (i+1)%N] = d 
        self.parity = 1
        if self.pbc:
            parity = - 1 + 2 * ( N % 2 )
            self.parity = parity
            self.tmat[N-1, 0] = t*parity 
            self.dmat[N-1, 0] = d*parity 
        self.tmat += self.tmat.T.conj() #here we doubled the diagonal elements
        self.dmat -= self.dmat.T
        
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])
        self.eig_eng, self.eig_vec = bdgeig_zeromode(self.ham, ref=0.0)
        print("the eig energies", self.eig_eng)
        self.exact_EgsXY = -self.eig_eng[:self.Nlatt].sum()/2 + np.trace(self.tmat)/2 - self.Nlatt*mu/2
        print("the exact energy by em and trt", -self.eig_eng[:self.Nlatt].sum()/2 + np.trace(self.tmat)/2 - self.Nlatt*mu/2)
        self.u, self.v = m2uv(self.eig_vec)
        self.m = uv2m(self.u, self.v)
        print("check the hamiltonian diagonalization: HM=ME, M=[[u,v*];[v,u*]]")
        check_orthonormal(self.ham, self.m)
        self.covariance()
        
class oneparton(Site):
    """
    Site of single parton, occupied or unoccupied, dimension=2
    """
    def __init__(self, cons_N=None):
        self.cons_N = cons_N
        
        if cons_N == None:
            leg = npc.LegCharge.from_trivial(2)
        elif cons_N == 'Z2':
            chinfo = npc.ChargeInfo([2], ['Z2']) #mod 2
            leg = npc.LegCharge.from_qflat(chinfo, [[0],[1]]) #0->unocc, 1->occ

        names = ['unoccupied', 'occupied']
        
        id2 = npc.diag([1,1], leg)
        ops = dict(id=id2)
        
        Site.__init__(self, leg, names, **ops)
        
def fermion_swap_gate_z2(legv, legp):
    #The charges must be set for every leg
    fswap = npc.zeros((legv, legp, legv.conj(),legp.conj()), labels=('fvR','sp','fvL','sp*'))
    for _v in range(legv.block_number):
        cv = legv.charges[_v]*legv.qconj
        pv = 1 - 2*(cv[0]%2)
        qv = legv.get_qindex_of_charges(cv)
        sv = legv.get_slice(qv)
        #print('pv', pv, 'cv', cv, 'qv', qv,'sv', sv)
        for _p in range(legp.block_number):
            cp = legp.charges[_p]*legp.qconj
            pp = 1 - 2*(cp[0]%2)
            qp = legp.get_qindex_of_charges(cp)
            sp = legp.get_slice(_p)
            #print('pp', pp, 'cp', cp, 'qp', qp, 'sp', sp)
            val = pp & pv
            for ip in range(sp.start, sp.stop):
                for iv in range(sv.start, sv.stop):
                    fswap[iv,ip,iv,ip] = val
    return fswap

def combine_four_copies(mps):
    lx = mps.L
    mps4_Bslist = []

    for i in range(lx):
        tw = deepcopy(mps.get_B(i))
        tx = deepcopy(mps.get_B(i))
        ty = deepcopy(mps.get_B(i))
        tz = deepcopy(mps.get_B(i))

        #combining w and x
        legv = tw.get_leg('vR')
        tw.ireplace_label('vL', 'v1L')
        legp = tx.get_leg('p')
        fswap = fermion_swap_gate_z2(legv, legp)
        temp = npc.tensordot(tw, fswap, axes=[['vR'], ['fvL']])
        res = npc.tensordot(temp, tx, axes=[['sp*'], ['p']])

        res = res.combine_legs([['v1L','vL'], ['fvR','vR']], qconj=[+1, -1])
        res.ireplace_labels(['(v1L.vL)', '(fvR.vR)'], ['vL', 'vR'])
        res.legs[res.get_leg_index('vL')] = res.get_leg('vL').to_LegCharge()
        res.legs[res.get_leg_index('vR')] = res.get_leg('vR').to_LegCharge()
        res.itranspose(['vL', 'p', 'sp', 'vR'])

        #combining wx and y
        legv = res.get_leg('vR')
        res.ireplace_label('vL', 'v1L')
        legp = ty.get_leg('p')
        fswap = fermion_swap_gate_z2(legv, legp)
        fswap.ireplace_label('sp', 'ssp')
        temp = npc.tensordot(res, fswap, axes=[['vR'], ['fvL']])
        res = npc.tensordot(temp, ty, axes=[['sp*'], ['p']])

        res = res.combine_legs([['v1L','vL'], ['fvR','vR']], qconj=[+1, -1])
        res.ireplace_labels(['(v1L.vL)', '(fvR.vR)'], ['vL', 'vR'])
        res.legs[res.get_leg_index('vL')] = res.get_leg('vL').to_LegCharge()
        res.legs[res.get_leg_index('vR')] = res.get_leg('vR').to_LegCharge()
        res.itranspose(['vL', 'p', 'sp', 'ssp', 'vR'])

        #combining wxy and z
        legv = res.get_leg('vR')
        res.ireplace_label('vL', 'v1L')
        legp = tz.get_leg('p')
        fswap = fermion_swap_gate_z2(legv, legp)
        fswap.ireplace_label('sp', 'sssp')
        temp = npc.tensordot(res, fswap, axes=[['vR'], ['fvL']])
        res = npc.tensordot(temp, tz, axes=[['sp*'], ['p']])

        res = res.combine_legs([['v1L','vL'], ['fvR','vR']], qconj=[+1, -1])
        res.ireplace_labels(['(v1L.vL)', '(fvR.vR)'], ['vL', 'vR'])
        res.legs[res.get_leg_index('vL')] = res.get_leg('vL').to_LegCharge()
        res.legs[res.get_leg_index('vR')] = res.get_leg('vR').to_LegCharge()
        res.itranspose(['vL', 'p', 'sp', 'ssp', 'sssp', 'vR'])

        mps4_Bslist.append(res)
    
    return mps4_Bslist

def gutzwiller_projection_npcarraylist(npcarraylist):
    gp_npcarraylist = []
    lx = len(npcarraylist)
    so4gen, cmn, dmn = get_so4_opr_list_new()
    
    spin1_site = SO4Site(cons_N='Z2', so4g=so4gen)
    spin1_leg = spin1_site.leg
    parton_site = oneparton(cons_N='Z2')
    parton_leg = parton_site.leg
    
    projector = npc.zeros([spin1_leg, parton_leg.conj(), parton_leg.conj(), parton_leg.conj(), parton_leg.conj()], labels=['p','p*1','p*2','p*3','p*4'], dtype=npcarraylist[0].dtype)
    projector[0,1,0,0,0] = 1
    projector[1,0,1,0,0] = 1
    projector[2,0,0,1,0] = 1
    projector[3,0,0,0,1] = 1
    
    for i in range(lx):
        res = npc.tensordot(projector, npcarraylist[i], axes=(['p*1', 'p*2', 'p*3', 'p*4'], ['p', 'sp', 'ssp', 'sssp']))
        gp_npcarraylist.append(res)
    return gp_npcarraylist

import logging
logging.basicConfig(level=2)
for _ in ['parso.python.diff', 'parso.cache', 'parso.python.diff', 
            'parso.cache', 'matplotlib.font_manager', 'tenpy.tools.cache', 
            'tenpy.algorithms.mps_common', 'tenpy.linalg.lanczos', 'tenpy.tools.params']:
    logging.getLogger(_).disabled = True

if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=8)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=0.5)
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=0.0)
    parser.add_argument("-lamb", type=float, default=1.)
    parser.add_argument("-Dpfaf", type=int, default=12)
    parser.add_argument("-Ddmrg", type=int, default=128)
    parser.add_argument("-pbc", type=int, default=1)
    args = parser.parse_args()
    
    np.random.seed(0)
    
    J, K = round(args.J, 6), round(args.K, 6)
    chi, delta, lamb = round(args.chi, 6), round(args.delta, 6), round(args.lamb, 6)
    lx, Ddmrg, Dpfaf, pbc = args.lx, args.Ddmrg, args.Dpfaf, args.pbc
    
    if pbc == 1:
        bc = 'periodic'
    elif pbc == 0:
        bc = 'open'
    else:
        raise "pbc must be 1(periodic) or 0(open)"
    
    
    print('----------Pfaffian Start---------')
    singlekitaev = singlechain(chi, delta, lamb, lx, Dpfaf, pbc)
    singlekitaev.hamiltonian()
    mpsflat = []
    for i in range(1,lx+1):
        tensortemp = singlekitaev.A_nba(i)
        mpsflat.append(tensortemp)
        
    bflat = deepcopy(mpsflat)
    for i in range(lx):
        if i == 0:
            bflat[i] = np.reshape(bflat[i], (1,2,2))
            bflat[i] = np.transpose(bflat[i], (1,0,2))
        elif i == lx-1:
            bflat[i] = np.reshape(bflat[i], (2,2,1))
    #now bflat is a list of B-tensor in the leg label ['p','vL','vR'], ready for building MPS object
    
    sites = [oneparton(cons_N='Z2')] * lx
    pfaffian_mps = MPS.from_Bflat(sites, bflat)
    pfaffian_mps.canonical_form()
    
    Bs_list = combine_four_copies(pfaffian_mps)
    gp_Bs_list = gutzwiller_projection_npcarraylist(Bs_list)
    
    so4generators, cmn, dmn = get_so4_opr_list_new()
    spin1sites = [SO4Site(cons_N='Z2', so4g=so4generators)] * lx
    gp_pfaffian_mps = MPS.from_product_state(spin1sites, [0]*lx)
    for i in range(lx):
        gp_pfaffian_mps._B[i] = gp_Bs_list[i]
    gp_pfaffian_mps.canonical_form()
    
    print("Gutzwiller projected pfaffian MPS", gp_pfaffian_mps)
    
    
    print("----------DMRG Start---------")
    bbq_model_params = model_params = dict(Lx=lx, bc=bc, J=J, K=K, D=Ddmrg, cons_N='Z2')
    bbqmodel = BBQJKSO4(bbq_model_params)
    start_time1 = time.time()
    dmrg_psi, E = bbqmodel.run_dmrg()
    end_time1 = time.time()
    print("Spin1 site DMRG results")
    print("psi2 after DMRG is", dmrg_psi)
    print("Energy is", E)

    print("----------Pfaffian Boost DMRG Start----------")
    bbq_model_params = model_params = dict(Lx=lx, bc=bc, J=J, K=K, D=Ddmrg, cons_N='Z2')
    bbqmodel = BBQJKSO4(bbq_model_params)
    start_time2 = time.time()
    dmrg_psi2, E2 = bbqmodel.run_dmrg(init=gp_pfaffian_mps)
    end_time2 = time.time()
    print("Pfaffian boosted DMRG results")
    print("pfaffian boosted DMRG is", dmrg_psi2)
    print("Energy is", E2)

    print("----------Measurements----------")
    print("Time taken for DMRG: {:.2f} seconds".format(end_time1 - start_time1))
    print("Time taken for Pfaffian Boosted DMRG: {:.2f} seconds".format(end_time2 - start_time2))
    bbqmpo = bbqmodel.calc_H_MPO()
    print("sandwich of gpmps and BBQJK hamiltonian", bbqmpo.expectation_value(gp_pfaffian_mps))
    print("overlap of pfaffian state and random DMRG result", dmrg_psi.overlap(gp_pfaffian_mps))
    print("overlap of two states", dmrg_psi.overlap(dmrg_psi2))