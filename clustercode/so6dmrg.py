"""
SO6 chain DMRG code compact version for HPC use. 

Puiyuen 240712-240726
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
        elif cons_N == 'N' and cons_S == 'None':
            chinfo = npc.ChargeInfo([2], ['FakeN'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1],[1],[1],[1],[1],[1]])
        else:
            print("No symmetry used in site SO6Site. ")
            leg = npc.LegCharge.from_trivial(6)
        
        ops = dict()
        for i in range(36):
            ops['lambda{}'.format(i)] = self.so6g[i]
        
        names = ['a', 'b', 'c', 'd', 'e', 'f'] #for the 6 double-occupied states we defined. 
        Site.__init__(self, leg, names, **ops)

    def __repr__(self):
        return "site for half-filled parton in 6 basis with conserve = {}".format([self.cons_N, self.cons_S])
    
class BBQJK(CouplingModel):
    """
    include cmn in model_params
    """
    def __init__(self, model_params):
        print(model_params)
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        self.Lx = model_params.get('Lx', 12)
        self.S = model_params.get('S', 1)
        self.bc = model_params.get('bc', 'periodic')
        self.J = model_params.get('J', 1)
        self.K = model_params.get('K', 1/3)
        self.verbose = model_params.get('verbose', 2)
        self.D = model_params.get('D', 64)
        self.sweeps = model_params.get('sweeps', 6)
        
        #defined as self variables 240716
        self.so6_generators, self.c_mn = get_opr_list()

        site = SO6Site(self.so6_generators, cons_N=None, cons_S='U1')
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
            init = [0]*(N//6) + [1]*(N//6) + [2]*(N//6) + [3]*(N//6) + [4]*(N//6) + [5]*(N//6)
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

        if self.bc == 'periodic':
            Econst = 5/3 * K * lx
        elif self.bc == 'open':
            Econst = 5/3 * K * (lx-1)
            
        eng = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = eng.run()
        print("Eng = ", E+Econst)
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
            init = [0]*(N//6) + [1]*(N//6) + [2]*(N//6) + [3]*(N//6) + [4]*(N//6) + [5]*(N//6)
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

        if self.bc == 'periodic':
            Econst = 5/3 * K * lx
        elif self.bc == 'open':
            Econst = 5/3 * K * (lx-1)
            
        eng = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = eng.run()
        print("Eng = ", E+Econst)
        self.psidmrg = psidmrg
        return psidmrg, E+Econst
    
if __name__ == "__main__":
    #parsers
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=12)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=1.)
    parser.add_argument("-D", type=int, default=64)
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
        measure_E_shift = 5/3 * K * lx
    elif pbc == 0:
        bc = 'open'
        measure_E_shift = 5/3 * K * (lx-1)
    else:
        raise "pbc must be 1(periodic) or 0(open)"
    
    #pathmaking
    import os
    homepath  = os.getcwd()
    if os.path.isdir(homepath+'/data/') == False:
        os.mkdir(homepath+'/data/')
    path = homepath + '/data/' + "SO6DMRG_lx{}_J{}_K{}_pbc{}/".format(lx, J, K, pbc)
    if os.path.isdir(path) == False:
        os.mkdir(path)
    fname = path+'psidmrg_job{}_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(job, lx, J, K, pbc, D, sweeps)

    #not global variables anymore 240716
    #so6_generators, c_mn = get_opr_list()
    
    model_paras = dict(cons_N=None, cons_S='U1', Lx = lx, bc=bc, J=J, K=K, D=D, sweeps=sweeps, verbose=2)
    so6bbq = BBQJK(model_paras)
    
    if args.job == 'dmrg':
        print("----------Start Job DMRG----------")
        psi_dmrg, E = so6bbq.run_dmrg()
        print("DMRG results")
        print("DMRG psi", psi_dmrg)
        
        #DMRG state saving
        with open(fname, 'wb') as f:
            pickle.dump(psi_dmrg, f)
            
        #small measurements along with DMRG, not printing local operators anymore
        print("entropy", psi_dmrg.entanglement_entropy())

    if args.job == 'dmrg2':
        print("----------Start Job DMRG2----------")
        print("----------Designed for the MPS point, the second time DMRG----------")
        #load DMRG state
        fname = path+'psidmrg_jobdmrg_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, D, sweeps)
        with open(fname, 'rb') as f:
            gs_dmrg = pickle.load(f)
        print(fname, "state file loaded. ")

        psi_dmrg, E = so6bbq.run_dmrg_orthogonal(gslist=[gs_dmrg])
        print("Second orthogonal DMRG results")
        print("DMRG psi", psi_dmrg)
        print("check orthogonal should be zero", psi_dmrg.overlap(gs_dmrg))
        
        #DMRG state saving
        fname = path+'psidmrg_job{}_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(job, lx, J, K, pbc, D, sweeps)
        with open(fname, 'wb') as f:
            pickle.dump(psi_dmrg, f)
            
        #small measurements along with DMRG, not printing local operators anymore
        print("entropy", psi_dmrg.entanglement_entropy())
    
    if args.job == 'measure':
        print("----------Start Job Measure----------")
        #DMRG state loading
        fname = path+'psidmrg_jobdmrg_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, D, sweeps)
        with open(fname, 'rb') as f:
            psi_dmrg = pickle.load(f)
        print(psi_dmrg)
        
        print("-----energy-----")
        bbqmpo = so6bbq.calc_H_MPO()
        print("The DMRG energy of psi is", bbqmpo.expectation_value(psi_dmrg)+measure_E_shift)
        
        print("-----entropy-----")
        print("The entanglement entropy of psi is", psi_dmrg.entanglement_entropy().tolist()) #printing tolist for preserving commas

        print("-----local operators expectations-----")
        for i in {0,5,10,20}: #not printing 16 16 anymore 240716
            print("i=",i)
            print('expectation value of lambda',i,'is', psi_dmrg.expectation_value("lambda"+str(i)).tolist())

        print("-----spin-spin correlations-----")
        spinspin_corr = np.zeros((lx,lx))
        for (m,n) in {(0,0),(1,4),(4,1),(2,8),(8,2),(3,12),(12,3),(5,5),(6,9),(9,6),(7,13),(13,7),(10,10),(11,14),(14,11)}:
            spinspin_corr += psi_dmrg.correlation_function('lambda'+str(m), 'lambda'+str(n))
        print('spinspin correlation function is', spinspin_corr)
        
        print("-----squared spin-spin correlations-----")
        sqrd_spinspin_corr = np.zeros((lx,lx))
        for m in range(36):
            for n in range(36):
                sqrd_spinspin_corr += so6bbq.c_mn[m,n] * psi_dmrg.correlation_function('lambda'+str(m), 'lambda'+str(n))
        print('sqrd spinspin correlation function is', sqrd_spinspin_corr)

    if args.job == 'measure2':
        print("----------Start Job Measure2----------")
        print("----------Designed for the MPS point, measurement of the second time DMRG----------")
        #DMRG state loading
        fname = path+'psidmrg_jobdmrg2_lx{}_J{}_K{}_pbc{}_D{}_sweeps{}'.format(lx, J, K, pbc, D, sweeps)
        with open(fname, 'rb') as f:
            psi_dmrg = pickle.load(f)
        print(psi_dmrg)
        
        print("-----energy-----")
        bbqmpo = so6bbq.calc_H_MPO()
        print("The DMRG energy of psi is", bbqmpo.expectation_value(psi_dmrg)+measure_E_shift)
        
        print("-----entropy-----")
        print("The entanglement entropy of psi is", psi_dmrg.entanglement_entropy().tolist()) #printing tolist for preserving commas

        print("-----local operators expectations-----")
        for i in {0,5,10,20}: #not printing 16 16 anymore 240716
            print("i=",i)
            print('expectation value of lambda',i,'is', psi_dmrg.expectation_value("lambda"+str(i)).tolist())

        print("-----spin-spin correlations-----")
        spinspin_corr = np.zeros((lx,lx))
        for (m,n) in {(0,0),(1,4),(4,1),(2,8),(8,2),(3,12),(12,3),(5,5),(6,9),(9,6),(7,13),(13,7),(10,10),(11,14),(14,11)}:
            spinspin_corr += psi_dmrg.correlation_function('lambda'+str(m), 'lambda'+str(n))
        print('spinspin correlation function is', spinspin_corr)
        
        print("-----squared spin-spin correlations-----")
        sqrd_spinspin_corr = np.zeros((lx,lx))
        for m in range(36):
            for n in range(36):
                sqrd_spinspin_corr += so6bbq.c_mn[m,n] * psi_dmrg.correlation_function('lambda'+str(m), 'lambda'+str(n))
        print('sqrd spinspin correlation function is', sqrd_spinspin_corr)

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

            #why the ones in bt are located on 0,i,i
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

        print(psi1.overlap(psi2))
        
        """
        from tenpy.networks import MPSEnvironment

        env11 = MPSEnvironment(psi1, psi1)
        env12 = MPSEnvironment(psi1, psi2)
        env21 = MPSEnvironment(psi2, psi1)
        env22 = MPSEnvironment(psi2, psi2)

        envmatlist = dict()
        for l in {0,5,10,20}:
            envmatlist[(l)] = []
            for i in range(lx-1):
                envmat = np.zeros((2,2))
                envmat[0,0] = env11.expectation_value_term([('lambda'+str(l), i), ('lambda'+str(l), i+1)])
                envmat[0,1] = env12.expectation_value_term([('lambda'+str(l), i), ('lambda'+str(l), i+1)])
                envmat[1,0] = env21.expectation_value_term([('lambda'+str(l), i), ('lambda'+str(l), i+1)])
                envmat[1,1] = env22.expectation_value_term([('lambda'+str(l), i), ('lambda'+str(l), i+1)])
                envmatlist[(l)].append(envmat)

        for l in {0,5,10,20}:
            for i in range(lx-1):
                u,v = LA.eig(envmatlist[(l)][i])
                if i%2==0: 
                    print("l=",l,"i=",i,"even-odd envmat is \n",envmatlist[(l)][i],"eigvals",u)
                else:
                    print("l=",l,"i=",i,"odd-even envmat is \n",envmatlist[(l)][i],"eigvals",u)

        phi1 = psi1.add(psi2,1,1)
        phi1.canonical_form()
        phi2 = psi1.add(psi2,1,-1)
        phi2.canonical_form()
        print("-----entropy-----")
        print("The entanglement entropy of psi1+psi2 is", phi1.entanglement_entropy().tolist())
        print("The entanglement entropy of psi1-psi2 is", phi2.entanglement_entropy().tolist())
        """

        site = psi1.sites[0]
        transop = trnslop_mpo(site, lx)
        tpsi1 = deepcopy(psi1)
        tpsi1 = apply_mpo(transop,tpsi1)
        tpsi1.canonical_form()
        print("<psi2|psi1>", psi2.overlap(psi1))
        print("<psi1|T|psi1>", psi1.overlap(tpsi1))
        print("<psi2|T|psi1>", psi2.overlap(tpsi1))

        ttpsi1 = apply_mpo(transop,tpsi1)
        ttpsi1.canonical_form()
        print("<psi1|TT|psi1>", psi1.overlap(ttpsi1))
        print("<psi2|TT|psi1>", psi2.overlap(ttpsi1))