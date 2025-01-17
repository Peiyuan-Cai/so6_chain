"""
SO4 chain DMRG code compact version for HPC use. Use U(1)xU(1) symmetry by default. 

Gutzwiller guided DMRG code for the SO4 chain.

Puiyuen 241025
    1. 241106 No orthogonal in jobmposdmrg2
    2. 241203 save data in data2
    3. 241206 bring the dmrgengine as a self var of BBQJK in order to fetch the sweep data
    4. 241206 min sweep 10, max sweep 20, one won't sweep for too long
    5. 241225 only for the r point
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
        self.init = model_params.get('init', None)
        
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
        #max_E_err  = kwargs.get('max_E_err', 1e-10)
        max_sweeps = kwargs.get('max_sweeps', int(self.sweeps + 10)) #maximum sweep time = 10 + 10
        min_sweeps = kwargs.get('min_sweeps', self.sweeps) #forced to sweep 10 times
        dmrg_params = dict(mixer=mixer, 
                           trunc_params=dict(chi_max=chi_max),
                           #max_E_err=max_E_err, 
                           max_sweeps=max_sweeps,
                           min_sweeps=min_sweeps,
                           verbose=2)

        init = self.init
        if init is None: #no specific initial state input
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
            print("init state loaded", psiinit)
            dmrg_params['mixer'] = False
        elif isinstance(init, list):
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)            
        elif isinstance(init, MPS):
            psiinit = init
        else:
            print("wrong init")

        self.dmrg_engine = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = self.dmrg_engine.run() #update 241206
        print("DMRG Energy = ", E)
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
    
    J, K = round(args.J, 3), round(args.K, 3)
    lx, D, pbc, sweeps = args.lx, args.D, args.pbc, args.sweeps
    job = args.job
    
    if pbc == 1:
        bc = 'periodic'
    elif pbc == 0:
        bc = 'open'
    else:
        raise "pbc must be 1(periodic) or 0(open)"
    
    #determine lambda by K
    K_list = np.arange(0.1, 0.156, 0.004)
    lamb_list_APBC = [1.89, 1.88, 1.87, 1.86, 1.85, 1.84, 1.82, 1.81, 1.80, 1.78, 1.77, 1.75, 1.73, 1.71]
    lamb_list_PBC = [1.94, 1.93, 1.91, 1.9, 1.88, 1.86, 1.85, 1.83, 1.81, 1.79, 1.77, 1.75, 1.73, 1.71]
    
    #pathmaking
    import os
    homepath  = os.getcwd()
    if os.path.isdir(homepath+'/data/') == False:
        os.mkdir(homepath+'/data/')
    path = homepath + '/data/' + "SO4DMRG_lx{}_J{}_K{}_pbc{}/".format(lx, J, K, pbc)
    if os.path.isdir(path) == False:
        os.mkdir(path)
    fname = path+'psidmrg_job{}_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(job, lx, J, K, pbc, D, sweeps)
    errorname = path+'errordata_job{}_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(job, lx, J, K, pbc, D, sweeps)
    
    model_paras = dict(cons_N=None, cons_S='U1', Lx = lx, bc=bc, J=J, K=K, D=D, sweeps=sweeps, verbose=2)
    
    if args.job == 'mposdmrg':
        print("----------Start Job DMRG from MPOMPS initialized----------")
        lamb = lamb_list_APBC[K_list.tolist().index(K)]
        model_paras['init'] = homepath+'/data/so4psimpos_lx{}_delta1.0_lambda{}'.format(lx, lamb)+'/so4psimpos_lx{}_delta1.0_lambda{}_Dmpos{}_APBC'.format(lx, lamb, 1000)
        so4bbq = BBQJKSO4(model_paras)

        psi_dmrg, E = so4bbq.run_dmrg()
        print("DMRG results")
        print("DMRG psi", psi_dmrg)
        errordata = [so4bbq.dmrg_engine.sweep_stats['Delta_E'],so4bbq.dmrg_engine.sweep_stats['max_trunc_err'],so4bbq.dmrg_engine.sweep_stats['max_E_trunc']]
        print("error data", errordata)
        
        #DMRG state saving
        with open(fname, 'wb') as f:
            pickle.dump(psi_dmrg, f)
        print("DMRG state saved at", fname)
        with open(errorname, 'wb') as f:
            pickle.dump(errordata, f)
        print("error data saved at", errorname)
            
        print("entropy", psi_dmrg.entanglement_entropy())
        
    if args.job == 'mposdmrg2':
        #this DMRG2 is orthogonal to the first Gutz guided DMRG result and initialized by the second MPOMPS result (guided by the second Gutzwiller projected MPOMPS)
        print("----------Start Job DMRG2 from MPOMPS initialized----------")
        print("----------Designed for the MPS point, the second time DMRG----------")
        lamb = lamb_list_PBC[K_list.tolist().index(K)]
        model_paras['init'] = homepath+'/data/so4psimpos_lx{}_delta1.0_lambda{}'.format(lx, lamb)+'/so4psimpos_lx{}_delta1.0_lambda{}_Dmpos{}_PBC'.format(lx, lamb, 1000)
        so4bbq = BBQJKSO4(model_paras)

        #load DMRG state
        # fname = path+'psidmrg_jobmposdmrg_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, D, sweeps)
        # with open(fname, 'rb') as f:
        #     gs_dmrg = pickle.load(f)
        # print(fname, "state file loaded. ")

        psi_dmrg, E = so4bbq.run_dmrg()
        print("Second orthogonal DMRG results")
        print("DMRG psi", psi_dmrg)
        # print("check orthogonal should be zero", psi_dmrg.overlap(gs_dmrg))
        errordata = [so4bbq.dmrg_engine.sweep_stats['Delta_E'],so4bbq.dmrg_engine.sweep_stats['max_trunc_err'],so4bbq.dmrg_engine.sweep_stats['max_E_trunc']]
        print("error data", errordata)
        
        #DMRG state saving
        fname = path+'psidmrg_job{}_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}_no_orth'.format(job, lx, J, K, pbc, D, sweeps)
        with open(fname, 'wb') as f:
            pickle.dump(psi_dmrg, f)
        with open(errorname, 'wb') as f:
            pickle.dump(errordata, f)
        print("error data saved at", errorname)
            
        print("entropy", psi_dmrg.entanglement_entropy())
        
        #energy difference check
        # so4mpo = so4bbq.calc_H_MPO()
        # eng1 = so4mpo.expectation_value(gs_dmrg)
        # eng2 = so4mpo.expectation_value(psi_dmrg)
        # print("Energy difference", np.abs(eng2-eng1))
        
        end_time = time.time()
        print("runtime", end_time-start_time)
