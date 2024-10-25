"""
SO6 chain DMRG code compact version for HPC use. Using U(1)xU(1) symmetry by default

Puiyuen 2401003
"""
import numpy as np
import numpy.linalg as LA
from copy import deepcopy
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import SpinSite, FermionSite, Site
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge, ChargeInfo
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig
import pickle

import time
start_time = time.time()

def get_so4_opr_list():
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
        elif self.cons_N is None and self.cons_S is None:
            leg = npc.LegCharge.from_trivial(4)
        
        ops = dict()
        for i in range(len(self.so4g)):
            ops['L{}'.format(i)] = self.so4g[i]

        names = ['1u2u', '1u2d', '1d2u', '1d2d']
        Site.__init__(self, leg, names, **ops)

    def __repr__(self):
        return "site for 16 so4 generators conserved N={}, S={}".format(self.cons_N, self.cons_S)
    
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
        self.cons_N = model_params.get('cons_N', None)
        self.cons_S = model_params.get('cons_S', 'U1')
        
        self.so4_generators, self.c_mn, self.d_mn = get_so4_opr_list()

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
    
    def run_dmrg_orthogonal(self, gslist, **kwargs):
        mixer      = kwargs.get('mixer', True)
        chi_max    = kwargs.get('chi_max', self.D)
        max_E_err  = kwargs.get('max_E_err', 1e-10)
        max_sweeps = kwargs.get('max_sweeps', self.sweeps)
        min_sweeps = kwargs.get('min_sweeps', min(3, max_sweeps) )
        dmrg_params = dict(mixer=mixer, 
                           trunc_params=dict(chi_max=chi_max),
                           max_E_err=max_E_err, 
                           max_sweeps=max_sweeps,
                           min_sweeps=min_sweeps,
                           verbose=2,
                           orthogonal_to=gslist)

        init = kwargs.get('init', None)
        if init is None:
            N = self.lat.N_sites
            if N%4==0 and N>0:
                init = [0]*(N//4) + [1]*(N//4) + [2]*(N//4) + [3]*(N//4)
            else:
                raise("Check the system size must be integral multiple of 4")
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
        print("DMRG2 Eng = ", E)
        self.psidmrg = psidmrg
        return psidmrg, E
    
if __name__ == "__main__":
    #parsers
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=8)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=1/4)
    parser.add_argument("-D", type=int, default=200)
    parser.add_argument("-pbc", type=int, default=1)
    parser.add_argument("-sweeps", type=int, default=10)
    parser.add_argument("-job", type=str, default='dmrg')
    parser.add_argument("-verbose", type=int, default=1)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.verbose)
    for _ in ['parso.python.diff', 'parso.cache', 'parso.python.diff', 
              'parso.cache', 'matplotlib.font_manager', 'tenpy.tools.cache', 
              'tenpy.algorithms.mps_common', 'tenpy.linalg.lanczos', 'tenpy.tools.params']:
        logging.getLogger(_).disabled = True

    np.random.seed(0)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=10, suppress=True)
    
    J, K = round(args.J, 6), round(args.K, 6)
    lx, D, pbc, sweeps = args.lx, args.D, args.pbc, args.sweeps
    job = args.job
    
    if pbc == 1:
        bc = 'periodic'
    elif pbc == 0:
        bc = 'open'
    else:
        raise "pbc must be 1(periodic) or 0(open)"
    
    #pathmaking
    import os
    homepath  = os.getcwd()
    if os.path.isdir(homepath+'/data/') == False:
        os.mkdir(homepath+'/data/')
    path = homepath + '/data/' + "SO4DMRG_lx{}_J{}_K{}_pbc{}/".format(lx, J, K, pbc)
    if os.path.isdir(path) == False:
        os.mkdir(path)
    fname = path+'psidmrg_job{}_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(job, lx, J, K, pbc, D, sweeps)
    
    model_paras = dict(cons_N=None, cons_S='U1', Lx = lx, bc=bc, J=J, K=K, D=D, sweeps=sweeps, verbose=2)
    so4bbq = BBQJKSO4(model_paras)

    if args.job == 'dmrg':
        print("----------Start Job DMRG----------")
        psi_dmrg, E = so4bbq.run_dmrg()
        print("DMRG results")
        print("DMRG psi", psi_dmrg)
        #DMRG state saving
        with open(fname, 'wb') as f:
            pickle.dump(psi_dmrg, f)

        print("entropy", psi_dmrg.entanglement_entropy())

    if args.job == 'dmrg2':
        print("----------Start Job DMRG2----------")
        print("----------Designed for the MPS point, the second time DMRG----------")
        #load DMRG state
        fname = path+'psidmrg_jobdmrg_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, D, sweeps)
        with open(fname, 'rb') as f:
            gs_dmrg = pickle.load(f)
        print(fname, "state file loaded. ")

        psi_dmrg, E = so4bbq.run_dmrg_orthogonal(gslist=[gs_dmrg])
        print("Second orthogonal DMRG results")
        print("DMRG psi", psi_dmrg)
        print("check orthogonal should be zero", psi_dmrg.overlap(gs_dmrg))
        
        #DMRG state saving
        fname = path+'psidmrg_job{}_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(job, lx, J, K, pbc, D, sweeps)
        with open(fname, 'wb') as f:
            pickle.dump(psi_dmrg, f)
            
        #small measurements along with DMRG, not printing local operators anymore
        print("entropy", psi_dmrg.entanglement_entropy())
        print("<psi2|psi1>", psi_dmrg.overlap(gs_dmrg))
        bbqmpo = so4bbq.calc_H_MPO()
        print("energy difference", np.abs(E - bbqmpo.expectation_value(gs_dmrg)))

    if args.job == 'dimercheck':
        print("----------Start job dimer check----------")
        print("----------Designed for the MPS point, finding two dimerized ground states----------")
        #DMRG state loading
        fname = path+'psidmrg_jobdmrg_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, D, sweeps)
        with open(fname, 'rb') as f:
            psi1 = pickle.load(f)
        fname = path+'psidmrg_jobdmrg2_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, D, sweeps)
        with open(fname, 'rb') as f:
            psi2 = pickle.load(f)

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
        
        print(" ")
        print("translational operator check")
        ovlp = psi2.overlap(psi1)
        print("<psi2|psi1>", ovlp)
        bbqmpo = so4bbq.calc_H_MPO()
        eng1 = bbqmpo.expectation_value(psi1)
        eng2 = bbqmpo.expectation_value(psi2)
        print("<psi1|H|psi1>", eng1)
        print("<psi2|H|psi2>", eng2)
        print("abs energy difference", np.abs(eng1-eng2))
        
        site = psi1.sites[0]
        transop = trnslop_mpo(site, lx)
        
        psi1.canonical_form()
        tpsi1 = deepcopy(psi1)
        tpsi1 = apply_mpo(transop,tpsi1)
        tpsi1.canonical_form()
        T11 = psi1.overlap(tpsi1)
        T21 = psi2.overlap(tpsi1)
        print("<psi1|T|psi1>", T11)
        print("<psi2|T|psi1>", T21)

        #ttpsi1 = apply_mpo(transop,tpsi1)
        #ttpsi1.canonical_form()
        #print("<psi1|TT|psi1>", psi1.overlap(ttpsi1))
        #print("<psi2|TT|psi1>", psi2.overlap(ttpsi1))

        psi2.canonical_form()
        tpsi2 = deepcopy(psi2)
        tpsi2 = apply_mpo(transop, tpsi2)
        tpsi2.canonical_form()
        T12 = psi1.overlap(tpsi2)
        T22 = psi2.overlap(tpsi2)
        print("<psi1|T|psi2>", T12)
        print("<psi2|T|psi2>", T22)
        #ttpsi2 = apply_mpo(transop,tpsi2)
        #ttpsi2.canonical_form()
        #print("<psi1|TT|psi2>", psi1.overlap(ttpsi2))
        #print("<psi2|TT|psi2>", psi2.overlap(ttpsi2))

        T_mat = np.array([[T11, T12], [T21, T22]])
        eigvals, eigvecs = LA.eig(T_mat)
        print("eigvals of T_mat", eigvals)

    end_time = time.time()
    print("runtime", end_time-start_time)