"""
The Pfaffian method on SO(3) spin-1 chain with bilinear-biquadratic(BBQ) Hamiltonian ver.2

The BdG Hamiltonian is of a single Kitaev chain, using fermionic swap gates to tensor product the tensors from different chains. 

No DMRG is used in the method by default, only no_sym_mpomps method for the comparision. 

classes threeparton(Site) and MPOMPS() are in mpomps.py

Puiyuen 2024.06.19-
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

class Spin1(Site):
    """
    Customized Spin-1 site, local operators are generators of SU(3)
    
    the leg is trivial(3) by default. (20240619)
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
        return "Self defined site Spin1 with good quantum numbers = {}".format([self.cons_N, self.cons_S])
    
class BBQJK(CouplingModel):
    def __init__(self, model_params):
        print(model_params)
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        self.Lx = model_params.get('Lx', 6)
        self.S = model_params.get('S', 1)
        self.bc = model_params.get('bc', 'periodic')
        self.J = model_params.get('J', 1)
        self.K = model_params.get('K', 1/3)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        self.verbose = model_params.get('verbose', 2)
        self.cons_N = model_params.get('cons_N', None)
        self.cons_S = model_params.get('cons_S', None)
        
        site = Spin1(cons_N=self.cons_N, cons_S=self.cons_S)
        self.sites = [site]*self.Lx
        self.lat = Chain(self.Lx, site, bc=self.bc)
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
            elif l == self.Lx-1 and self.bc == 'periodic':
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
            self.add_coupling_term(K*4/3, i0, i1, "Id", "Id") #this is a constant term
    
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

def combine_three_copies(mps):
    """
    combining 3 copies with fermionic swap gates
    
    Inputs:
        1. mps, tempy.MPS object
        
    Outputs:
        1. mps3_Bslist, list of npc.Array
    """
    lx = mps.L
    mps3_Bslist = []
    
    for i in range(lx):
        tx = deepcopy(mps.get_B(i))
        ty = deepcopy(mps.get_B(i))
        tz = deepcopy(mps.get_B(i))
        
        #combining x and y
        legv = tx.get_leg('vR')
        tx.ireplace_label('vL','v1L')
        legp = ty.get_leg('p')
        fswap = fermion_swap_gate_z2(legv, legp)
        temp = npc.tensordot(tx, fswap, axes=(['vR'], ['fvL']))
        res = npc.tensordot(temp, ty, axes=(['sp*'], ['p']))
        
        res = res.combine_legs([['v1L','vL'], ['fvR','vR']], qconj=[+1, -1])
        res.ireplace_labels(['(v1L.vL)', '(fvR.vR)'], ['vL', 'vR'])
        res.legs[res.get_leg_index('vL')] = res.get_leg('vL').to_LegCharge()
        res.legs[res.get_leg_index('vR')] = res.get_leg('vR').to_LegCharge()
        res.itranspose(['vL', 'p', 'sp', 'vR'])
        
        #combining xy and z
        legv = res.get_leg('vR')
        res.ireplace_label('vL', 'v1L')
        legp = tz.get_leg('p')
        fswap = fermion_swap_gate_z2(legv, legp)
        fswap.ireplace_label('sp', 'ssp')
        temp = npc.tensordot(res, fswap, axes=(['vR'], ['fvL']))
        res = npc.tensordot(temp, tz, axes=(['sp*'], ['p']))
        
        res = res.combine_legs([['v1L','vL'], ['fvR','vR']], qconj=[+1, -1])
        res.ireplace_labels(['(v1L.vL)', '(fvR.vR)'], ['vL', 'vR'])
        res.legs[res.get_leg_index('vL')] = res.get_leg('vL').to_LegCharge()
        res.legs[res.get_leg_index('vR')] = res.get_leg('vR').to_LegCharge()
        res.itranspose(['vL', 'p', 'sp', 'ssp', 'vR'])
        mps3_Bslist.append(res)
    
    return mps3_Bslist

def gutzwiller_projection_npcarraylist(npcarraylist):
    """
    Gutzwiller projection function: 
    
               p
         ______|______
         |___________|           projector
           |   |   |
          p*1 p*2 p*3
          (contract)
           p   sp ssp
         __|___|___|__
     vL--|___________|--vR       npcarraylist[i]
     
    contracting the p* and sp legs.  
    
    Input: 
        1. npcarraylist, list of npc.Array, the combined, unprojected xyz mps
    Output:
        1. gp_npcarraylist, list of npc.Array, the projected xyz mps, labeled as 'vL', 'p', 'vR'
    """
    gp_npcarraylist = []
    lx = len(npcarraylist)
    
    spin1_site = Spin1(cons_N='Z2')
    spin1_leg = spin1_site.leg
    parton_site = oneparton(cons_N='Z2')
    parton_leg = parton_site.leg
    
    projector = npc.zeros([spin1_leg, parton_leg.conj(), parton_leg.conj(), parton_leg.conj()], labels=['p','p*1','p*2','p*3'], dtype=npcarraylist[0].dtype)
    projector[0,1,0,0] = 1
    projector[1,0,1,0] = 1
    projector[2,0,0,1] = 1
    
    for i in range(lx):
        res = npc.tensordot(projector, npcarraylist[i], axes=(['p*1', 'p*2', 'p*3'], ['p', 'sp', 'ssp']))
        gp_npcarraylist.append(res)
    return gp_npcarraylist

def test_contractor(tensor1, tensor2):
    t1 = deepcopy(tensor1)
    t2 = deepcopy(tensor2)
    legv = t1.get_leg('vR')
    t1.ireplace_label('vL','v1L')
    legp = t2.get_leg('p')
    fswap = fermion_swap_gate_z2(legv, legp)
    temp = npc.tensordot(t1, fswap, axes=(['vR'], ['fvL']))
    res = npc.tensordot(temp, t2, axes=(['sp*'], ['p']))
    
    res = res.combine_legs([['v1L','vL'], ['fvR','vR']], qconj=[+1, -1])
    res.ireplace_labels(['(v1L.vL)', '(fvR.vR)'], ['vL', 'vR'])
    res.legs[res.get_leg_index('vL')] = res.get_leg('vL').to_LegCharge()
    res.legs[res.get_leg_index('vR')] = res.get_leg('vR').to_LegCharge()
    res.itranspose(['vL', 'p', 'sp', 'vR'])
    return res

def test_3_contractor(tensor): 
    tx = deepcopy(tensor)
    ty = deepcopy(tensor)
    tz = deepcopy(tensor)
        
    #combining x and y
    legv = tx.get_leg('vR')
    tx.ireplace_label('vL','v1L')
    legp = ty.get_leg('p')
    fswap = fermion_swap_gate_z2(legv, legp)
    temp = npc.tensordot(tx, fswap, axes=(['vR'], ['fvL']))
    res = npc.tensordot(temp, ty, axes=(['sp*'], ['p']))
        
    res = res.combine_legs([['v1L','vL'], ['fvR','vR']], qconj=[+1, -1])
    res.ireplace_labels(['(v1L.vL)', '(fvR.vR)'], ['vL', 'vR'])
    res.legs[res.get_leg_index('vL')] = res.get_leg('vL').to_LegCharge()
    res.legs[res.get_leg_index('vR')] = res.get_leg('vR').to_LegCharge()
    res.itranspose(['vL', 'p', 'sp', 'vR'])
        
    #combining xy and z
    legv = res.get_leg('vR')
    res.ireplace_label('vL', 'v1L')
    legp = tz.get_leg('p')
    fswap = fermion_swap_gate_z2(legv, legp)
    fswap.ireplace_label('sp', 'ssp')
    temp = npc.tensordot(res, fswap, axes=(['vR'], ['fvL']))
    res = npc.tensordot(temp, tz, axes=(['sp*'], ['p']))
        
    res = res.combine_legs([['v1L','vL'], ['fvR','vR']], qconj=[+1, -1])
    res.ireplace_labels(['(v1L.vL)', '(fvR.vR)'], ['vL', 'vR'])
    res.legs[res.get_leg_index('vL')] = res.get_leg('vL').to_LegCharge()
    res.legs[res.get_leg_index('vR')] = res.get_leg('vR').to_LegCharge()
    res.itranspose(['vL', 'p', 'sp', 'ssp', 'vR'])
    
    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=6)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=0.)
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=0.98)
    parser.add_argument("-lamb", type=float, default=1.78)
    parser.add_argument("-D", type=int, default=16)
    parser.add_argument("-pbc", type=int, default=1)
    args = parser.parse_args()
    
    np.random.seed(0)
    
    J, K = round(args.J, 6), round(args.K, 6)
    chi, delta, lamb = round(args.chi, 6), round(args.delta, 6), round(args.lamb, 6)
    lx, D, pbc = args.lx, args.D, args.pbc
    
    if pbc == 1:
        bc = 'periodic'
    elif pbc == 0:
        bc = 'open'
    else:
        raise "pbc must be 1(periodic) or 0(open)"
    
    
    print('----------Pfaffian Start---------')
    singlekitaev = singlechain(chi, delta, lamb, lx, D, pbc)
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
    
    #test_tensor = pfaffian_mps.get_B(0)
    #print('test tensor', test_tensor)
    
    #test = test_contractor(test_tensor,test_tensor)
    #print('test result', test)
    
    #test3 = test_3_contractor(test_tensor)
    #print('test 3 result', test3)
    
    Bs_list = combine_three_copies(pfaffian_mps)
    gp_Bs_list = gutzwiller_projection_npcarraylist(Bs_list)
    
    spin1sites = [Spin1(cons_N='Z2')] * lx
    gp_pfaffian_mps = MPS.from_product_state(spin1sites, [0]*lx)
    for i in range(lx):
        gp_pfaffian_mps._B[i] = gp_Bs_list[i]
    gp_pfaffian_mps.canonical_form()
    
    print("Gutzwiller projected pfaffian MPS", gp_pfaffian_mps)
    
    
    print("----------DMRG Start---------")
    bbq_model_params = model_params = dict(Lx=lx, bc=bc, J=J, K=K, cons_N='Z2')
    bbqmodel = BBQJK(bbq_model_params)
    dmrg_psi, E = bbqmodel.run_dmrg()
    print("Spin1 site DMRG results")
    print("psi2 after DMRG is", dmrg_psi)
    print("E2 is", E)
    
    bbqmpo = bbqmodel.calc_H_MPO()
    print("sandwich of gpmps and BBQJK hamiltonian", bbqmpo.expectation_value(gp_pfaffian_mps))