import numpy as np
import numpy.linalg as LA
from copy import deepcopy
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import Site, SpinSite
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge, ChargeInfo
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig

'''
---------------------------------------------SO(6) site----------------------------------------------------------
'''
class SO6Site(Site):
    def __init__(self, so6g, cons_N=None, cons_S=None):
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        self.so6g = so6g
        if cons_N == None and cons_S == 'U1':
            chinfo = npc.ChargeInfo([1, 1, 1], ['P', 'Q', 'R'])
            leg = npc.LegCharge.from_qflat(chinfo, [[-1, -2, 0], [-1, 1, 1], [-1, 1, -1], [1, -1, 1], [1, -1, -1], [1, 2, 0]])
        elif cons_N == 'N' and cons_S == 'U1':
            chinfo = npc.ChargeInfo([1, 1, 1, 1], ['FakeN', 'P', 'Q', 'R'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, -1, -2, 0], [1, -1, 1, 1], [1, -1, 1, -1], [1, 1, -1, 1], [1, 1, -1, -1], [1, 1, 2, 0]])
        elif cons_N == 'N' and cons_S == None:
            chinfo = npc.ChargeInfo([2], ['FakeN'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1],[1],[1],[1],[1],[1]])
        else:
            print("No symmetry used in site SU4HalfFillingSite. ")
            leg = npc.LegCharge.from_trivial(6)
        
        ops = dict()
        for i in range(36):
            ops['lambda{}'.format(i)] = self.so6g[i]
        
        names = ['a','b','c','d','e','f'] #for the 6 double-occupied states we defined. C_4^2=6
        Site.__init__(self, leg, names, **ops)

    def __repr__(self):
        return "site with physical basis of half-filling SU(4) fermions with conserve = {}".format([self.cons_N, self.cons_S])

class SU4HalfFillingSite(Site):
    def __init__(self, so6g, cons_N=None, cons_S=None):
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        self.so6g = so6g
        if cons_N == None and cons_S == 'U1':
            chinfo = npc.ChargeInfo([1, 1, 1], ['P', 'Q', 'R'])
            leg = npc.LegCharge.from_qflat(chinfo, [[-1, -2, 0], [-1, 1, 1], [-1, 1, -1], [1, -1, 1], [1, -1, -1], [1, 2, 0]])
        elif cons_N == 'N' and cons_S == 'U1':
            chinfo = npc.ChargeInfo([1, 1, 1, 1], ['FakeN', 'P', 'Q', 'R'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, -1, -2, 0], [1, -1, 1, 1], [1, -1, 1, -1], [1, 1, -1, 1], [1, 1, -1, -1], [1, 1, 2, 0]])
        elif cons_N == 'N' and cons_S == None:
            chinfo = npc.ChargeInfo([2], ['FakeN'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1],[1],[1],[1],[1],[1]])
        else:
            print("No symmetry used in site SU4HalfFillingSite. ")
            leg = npc.LegCharge.from_trivial(6)
        
        ops = dict()
        for i in range(36):
            ops['lambda{}'.format(i)] = self.so6g[i]
        
        names = ['a','b','c','d','e','f'] #for the 6 double-occupied states we defined. C_4^2=6
        Site.__init__(self, leg, names, **ops)

    def __repr__(self):
        return "site with physical basis of half-filling SU(4) fermions with conserve = {}".format([self.cons_N, self.cons_S])
    
'''
--------------------------------------------SU(4) oprs functions-------------------------------------------------
'''
def oprs_on_ket(oprs_original, ket):
    """
    Inputs: 
        1. oprs_original, list of strings, the operators in the middle
        2. ket, list of strings, the ket

    Outputs:
        1. coe, the coefficient, 1 or -1 or 0
        2. ket, list of strings or zero(integer), the result of application of operators
    """

    oprs = deepcopy(oprs_original)
    assert len(oprs) != 0 #there must be at least one operator inside list oprs
    coe = 1

    while len(oprs) != 0:
        opr = oprs[-1]

        if opr.endswith('d'): #creation operator
            opr = opr[:-1] #cut the 'd' in the end
            if any(opr in _ for _ in ket): #if opr is already in ket
                ket = 0
                return 0, ket #return (0,0)
            else: #opr is not in ket
                ket.insert(0,opr)
        else: #annihilation operator
            if any(opr in _ for _ in ket): 
                index = ket.index(opr) #index is the number of particles 'before' opr
                coe *= (-1)**index
                ket.remove(opr)
            else:
                ket = 0
                return 0, ket #return (0,0)
        oprs = oprs[:-1] #cut the operator in oprs after application
    return coe, ket

def get_ket_from_6states(n):
    if n == 1:
        return ['1','2']
    elif n == 2:
        return ['1','3']
    elif n == 3:
        return ['1','4']
    elif n == 4:
        return ['2','3']
    elif n == 5:
        return ['2','4']
    elif n == 6:
        return ['3','4']
    else:
        raise("Out of 6 states. ")

def S_representation_matrix(alpha, beta):
    if type(alpha) != int or type(beta) != int or alpha>4 or alpha<1 or beta>4 or beta<1:
        raise("Check your alpha and beta. They must be 1,2,3,4. ")
    S_mat = np.zeros((6,6))
    for left in range(1,7):
        bra = get_ket_from_6states(left)
        oprs = [str(alpha)+'d', str(beta)]
        oprs.insert(0, bra[0])
        oprs.insert(0, bra[1]) #put the annihilation operators in the front of oprs
        for right in range(1,7):
            ket = get_ket_from_6states(right)
            coe, ket = oprs_on_ket(oprs, ket)
            if ket == []:
                S_mat[left-1, right-1] = coe
            elif ket == 0:
                S_mat[left-1, right-1] = 0
            else:
                raise('something wrong')
    if alpha==beta:
        S_mat -= (1/2)*np.diag([1,1,1,1,1,1])
    return S_mat

def gram_schmidt(A):
    Q, _ = LA.qr(A)
    return Q

def schmidt_to_cartan_subalgebra(S1,S2,S3):
    A = np.array([S1.flatten(), S2.flatten(), S3.flatten()])
    Q = gram_schmidt(A.T).T

    B1 = Q[0].reshape(6, 6)
    B2 = Q[1].reshape(6, 6)
    B3 = Q[2].reshape(6, 6)

    C1 = B1 * np.sqrt(2 / np.trace(B1 @ B1.T))
    C2 = B2 * np.sqrt(2 / np.trace(B2 @ B2.T))
    C3 = B3 * np.sqrt(2 / np.trace(B3 @ B3.T))

    return C1, C2, C3

def get_opr_list():
    su4g = dict()
    for alpha in range(1,5):
        for beta in range(1,5):
            su4g[(alpha,beta)] = S_representation_matrix(alpha,beta)


    so6g = [] #start with 15 generators
    coe_list = [] #the coefficient list, list of str
    for alpha in range(1,5):
        for beta in range(1,5):
            if not(alpha==4 and beta==4):
                so6g.append(S_representation_matrix(alpha,beta))
    C1, C2, C3 = schmidt_to_cartan_subalgebra(S_representation_matrix(1,1), S_representation_matrix(2,2), S_representation_matrix(3,3))
    so6g[0] = C1; so6g[5] = C2; so6g[10] = C3

    for a in range(1,5):
        for b in range(1,5):
            for c in range(1,5):
                for d in range(1,5):
                    SiSi = su4g[(a,b)] @ su4g[(c,d)]
                    Amat = np.zeros((36, len(so6g)))
                    B = SiSi.reshape(-1,1)
                    for l in range(len(so6g)):
                        Amat[:,l] = so6g[l].reshape(-1,1)[:,0]
                    #print("a,b,c,d",a,b,c,d,'shape of equation', Amat.shape, B.shape)
                    pcoe, resi, rank, sing = LA.lstsq(Amat, B, rcond=None)
                    #print("shape of coe", coe.shape)
                    if len(resi)!=0 and resi[0]>1e-10: #no solution
                        so6g.append(SiSi)
                        #print("a,b,c,d",a,b,c,d,"New added to so6g, now we have",len(so6g),'operators. ') #no more output here 240724
                        pcoe = np.append(np.zeros((len(so6g)-1, 1)),1).reshape(len(so6g),1)
                        coe_list.append(pcoe)
                    else:
                        coe_list.append(pcoe)
    
    so6g_new = deepcopy(so6g)

    for i in range(15,36):
        if i == 15:
            so6g_new[i] = np.diag([1,1,1,1,1,1])
        elif i == 16:
            so6g_new[i] = np.diag([2,-1,-1,-1,-1,2])/np.sqrt(6)
        elif i == 20:
            so6g_new[i] = np.diag([0,-1,1,1,-1,0])/np.sqrt(2)
        elif i in {17,18,19,21,22,23,26,31}:
            so6g_new[i] *= 2
        else:
            so6g_new[i] *= np.sqrt(2)

    coe_list_new = []

    for a in range(1,5):
        for b in range(1,5):
            for c in range(1,5):
                for d in range(1,5):
                    SiSi = su4g[(a,b)] @ su4g[(c,d)]
                    Amat = np.zeros((36, len(so6g_new)))
                    B = SiSi.reshape(-1,1)
                    for l in range(len(so6g_new)):
                        Amat[:,l] = so6g_new[l].reshape(-1,1)[:,0]
                    pcoe = LA.solve(Amat, B)
                    coe_list_new.append(pcoe)

    for i in range(len(coe_list_new)):
        coe_list_new[i] = coe_list_new[i].reshape(36)

    def pvec_new(a,b,c,d):
        return coe_list_new[64*(a-1)+16*(b-1)+4*(c-1)+d-1]

    cmn_new = np.zeros((36,36))

    P = dict()
    for a in range(1,5):
        for b in range(1,5):
            for c in range(1,5):
                for d in range(1,5):
                    P[(a,b,c,d)] = pvec_new(a,b,c,d)

    for m in range(36):
        for n in range(36):
            for a in range(1,5):
                for b in range(1,5):
                    for c in range(1,5):
                        for d in range(1,5):
                            cmn_new[m,n] += P[(a,b,c,d)][m] * P[(b,a,d,c)][n]

    return so6g_new, cmn_new

'''
--------------------------------------------SO(6) BBQ Hamiltonian----------------------------------------------
'''
class BBQJK(CouplingModel):
    """
    include cmn in model_params
    """
    def __init__(self, model_params):
        print(model_params)
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        self.Lx = model_params.get('Lx', 6)
        self.S = model_params.get('S', 1)
        self.pbc = model_params.get('pbc', -1)
        self.J = model_params.get('J', 1)
        self.K = model_params.get('K', 1/6)
        self.verbose = model_params.get('verbose', 2)
        self.D = model_params.get('D', 64)
        self.sweeps = model_params.get('sweeps', 6)
        
        #defined as self variables 240716
        self.so6_generators, self.c_mn = get_opr_list()

        site = SU4HalfFillingSite(self.so6_generators, cons_N=None, cons_S=None) #20240829 for the trivial test, we use [None, None] here
        self.sites = [site] * self.Lx
        if self.pbc == 1 or self.pbc==-1:
            self.bc = 'periodic'
        elif self.pbc == 0:
            self.bc = 'open'
        self.lat = Chain(self.Lx, site, bc=self.bc)
        CouplingModel.__init__(self, self.lat, explicit_plus_hc=False)
        self.init_terms()
        H_MPO = self.calc_H_MPO()
        if model_params.get('sort_mpo_legs', False):
            H_MPO.sort_legcharges()
        MPOModel.__init__(self, self.lat, H_MPO)
        model_params.warn_unused()
    
    def init_terms(self):
        J = self.J
        K = self.K
        for l in range(self.Lx):
            if l < self.Lx - 1:
                i0, i1 = l, (l+1)%self.Lx
            elif l == self.Lx-1 and self.bc == 'periodic':
                i0, i1 = 0, self.Lx-1
                print("periodic terms added")
            else:
                break
            self.add_coupling_term(J-2*K,  i0, i1, "lambda0", "lambda0")
            self.add_coupling_term(J-2*K,  i0, i1, "lambda1", "lambda4", plus_hc=True)
            self.add_coupling_term(J-4*K,  i0, i1, "lambda2", "lambda8", plus_hc=True)
            self.add_coupling_term(J-4*K,  i0, i1, "lambda3", "lambda12", plus_hc=True)
            self.add_coupling_term(J-2*K,  i0, i1, "lambda5", "lambda5")
            self.add_coupling_term(J-2*K,  i0, i1, "lambda6", "lambda9", plus_hc=True)
            self.add_coupling_term(J-2*K,  i0, i1, "lambda7", "lambda13", plus_hc=True)
            self.add_coupling_term(J-2*K,  i0, i1, "lambda10", "lambda10")
            self.add_coupling_term(J-2*K,  i0, i1, "lambda11", "lambda14", plus_hc=True)
            
            if np.abs(K) > 1e-6:
                for m in range(36):
                    for n in range(36):
                        if (np.abs(self.c_mn[m,n]) > 1e-10) and not(np.allclose(np.kron(self.so6_generators[m], self.so6_generators[n]), np.zeros((36,36)), atol=1e-10)) and ((m,n) not in {(0,0),(1,4),(4,1),(2,8),(8,2),(3,12),(12,3),(5,5),(6,9),(9,6),(7,13),(13,7),(10,10),(11,14),(14,11),(15,15)}):
                            self.add_coupling_term(K*np.round(self.c_mn[m,n],6),  i0, i1, "lambda"+str(m), "lambda"+str(n)) #no identity here but the energy has shifted
    
    def run_dmrg(self, **kwargs):
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
                           verbose=2)

        init = kwargs.get('init', None)
        if init is None:
            N = self.lat.N_sites
            if N%6==0 and N>0:
                init = [0]*(N//6) + [1]*(N//6) + [2]*(N//6) + [3]*(N//6) + [4]*(N//6) + [5]*(N//6)
            else:
                raise("Check the system size must be integral multiple of 6")
            np.random.shuffle(init)
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)
            psiinit.norm = 1
            psiinit.canonical_form()
        elif isinstance(init, list):
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)            
        elif isinstance(init, MPS):
            psiinit = init
        else:
            print("wrong init")

        if self.pbc == 1 or self.pbc == -1:
            Econst = 5/3 * self.K * self.Lx
        elif self.pbc == 0:
            Econst = 5/3 * self.K * (self.Lx-1)
            
        eng = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = eng.run()
        print("Shifted Energy = ", E+Econst)
        self.psidmrg = psidmrg
        return psidmrg, E+Econst
    
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

        init = kwargs.get('init', None)
        if init is None:
            N = self.lat.N_sites
            if N%6==0 and N>0:
                init = [0]*(N//6) + [1]*(N//6) + [2]*(N//6) + [3]*(N//6) + [4]*(N//6) + [5]*(N//6)
            else:
                raise("Check the system size must be integral multiple of 6")
            np.random.shuffle(init)
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)
            psiinit.norm = 1
            psiinit.canonical_form()
        elif isinstance(init, list):
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)
        elif isinstance(init, MPS):
            psiinit = init
        else:
            print("wrong init")

        if self.pbc == 1 or self.pbc == -1:
            Econst = 5/3 * self.K * self.Lx
        elif self.pbc == 0:
            Econst = 5/3 * self.K * (self.Lx-1)
            
        eng = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = eng.run()
        print("Shifted Energy = ", E+Econst)
        self.psidmrg = psidmrg
        return psidmrg, E+Econst