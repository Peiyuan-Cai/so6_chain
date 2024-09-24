"""
SO4 chain DMRG code compact version for HPC use. As a copy of so6dmrg.py

Puiyuen 240918-240924

This is a resource killer version, dont use it on HPC. Use so4dmrg.py instead.
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

    return Loprs

class SO4Site(Site):
    def __init__(self, cons_N=None, cons_S=None):
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S

        leg = npc.LegCharge.from_trivial(4)
        L6list = get_so4_opr_list()
        
        ops = dict()
        for i in range(len(L6list)):
            ops['L{}'.format(i)] = L6list[i] #linears
        
        for a in range(6):
            for b in range(6):
                ops['L{}'.format(6+6*a+b)] = L6list[a] @ L6list[b] #quadratics

        names = ['1u2u', '1d2u', '1u2d', '1d2d']
        Site.__init__(self, leg, names, **ops)

    def __repr__(self):
        return "trivial site for 16 so4 generators"
    
class BBQJKSO4(CouplingModel):
    def __init__(self, model_params):
        print(model_params)
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        self.Lx = model_params.get('Lx', 8)
        self.S = model_params.get('S', 1)
        self.bc = model_params.get('bc', 'periodic')
        self.J = model_params.get('J', 1)
        self.K = model_params.get('K', 1/4)
        self.verbose = model_params.get('verbose', 2)
        self.D = model_params.get('D', 200)
        self.sweeps = model_params.get('sweeps', 10)
        

        site = SO4Site(cons_N=None, cons_S=None)
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
        
            for a in range(6):
                self.add_coupling_term(J, i0, i1, "L"+str(a), "L"+str(a))
            
            for a in range(6,42):
                self.add_coupling_term(K, i0, i1, "L"+str(a), "L"+str(a))
    
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
    
if __name__ == "__main__":
    #parsers
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=16)
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
    
    model_paras = dict(cons_N=None, cons_S=None, Lx = lx, bc=bc, J=J, K=K, D=D, sweeps=sweeps, verbose=2)
    so4bbq = BBQJKSO4(model_paras)

    if args.job == 'dmrg':
        print("----------Start Job DMRG----------")
        psi_dmrg, E = so4bbq.run_dmrg()
        print("DMRG results")
        print("DMRG psi", psi_dmrg)

        print("entropy", psi_dmrg.entanglement_entropy())