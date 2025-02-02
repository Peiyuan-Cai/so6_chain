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
    Sp = Sx + 1j * Sy
    Sm = Sx - 1j * Sy

    L1 = -np.kron(Sz, id) - np.kron(id, Sz)
    L2 = -np.kron(Sx, id) + np.kron(id, Sx)
    L3 = -np.kron(Sy, id) - np.kron(id, Sy)
    L4 = -np.kron(Sy, id) + np.kron(id, Sy)
    L5 = +np.kron(Sx, id) + np.kron(id, Sx)
    L6 = -np.kron(Sz, id) + np.kron(id, Sz)

    Loprs = [L1, L2, L3, L4, L5, L6]
    coe_list = []

    for a in range(6):
        for b in range(6):
            LiLi = Loprs[a] @ Loprs[b]
            Amat = np.zeros((16, len(Loprs)), dtype=complex)
            B = LiLi.reshape(-1,1)
            for l in range(len(Loprs)):
                Amat[:,l] = Loprs[l].reshape(-1,1)[:,0]
            pcoe, resi, rank, sing = LA.lstsq(Amat, B, rcond=None)
            if len(resi)!=0 and resi[0]>1e-10:
                Loprs.append(LiLi)
                pcoe = np.append(np.zeros((len(Loprs)-1, 1)),1).reshape(len(Loprs),1)
                coe_list.append(pcoe)
            else:
                coe_list.append(pcoe)

    coe_list_new = []
    for a in range(6):
        for b in range(6):
            LiLi = Loprs[a] @ Loprs[b]
            Amat = np.zeros((16, len(Loprs)), dtype=complex)
            B = LiLi.reshape(-1,1)
            for l in range(len(Loprs)):
                Amat[:,l] = Loprs[l].reshape(-1,1)[:,0]
            pcoe = LA.solve(Amat, B)
            coe_list_new.append(pcoe)
    
    coe_list = coe_list_new

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

    return Loprs, cmn

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
        elif self.cons_N is None and self.cons_S is None:
            leg = npc.LegCharge.from_trivial(4)
        
        ops = dict()
        for i in range(len(self.so4g)):
            ops['L{}'.format(i)] = self.so4g[i]

        names = ['uu', 'ud', 'du', 'dd']
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
        self.pbc = model_params.get('pbc', 1)
        self.J = model_params.get('J', 1)
        self.K = model_params.get('K', 1/4)
        self.verbose = model_params.get('verbose', 2)
        self.D = model_params.get('D', 200)
        self.sweeps = model_params.get('sweeps', 10)
        self.init = model_params.get('init', None)
        
        self.so4_generators, self.c_mn, self.d_mn = get_so4_opr_list_new()
        
        if self.pbc == 0:
            self.bc = 'open'

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
            print('init by MPS input')
        else:
            print("wrong init")

        eng = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = eng.run()
        print("Eng = ", E)
        self.psidmrg = psidmrg
        return psidmrg, E
    
    def run_dmrg_orthogonal(self, gslist, **kwargs):
        """
        gslist is a list of states to projected
        """
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

        init = self.init
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
        print("Eng = ", E)
        self.psidmrg = psidmrg
        return psidmrg, E