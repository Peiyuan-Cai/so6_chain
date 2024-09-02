"""
The Pfaffian method on SO(3) spin-1 chain with bilinear-biquadratic(BBQ) Hamiltonian

The BdG Hamiltonian is under xyz basis

Puiyuen 2024.06.19-
"""
from os import stat
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
import mpomps as mpos

class BBQModel_SpinSite(CouplingModel):
    """
    The BBQ model written on the build-in SpinSite. For DMRG use. 
    """
    def __init__(self, model_params):
        print(model_params)
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        self.Lx = model_params.get('Lx', 10)
        self.S = model_params.get('S', 1)
        self.bc = model_params.get('bc', 'periodic')
        self.bc_MPS = model_params.get('bc_MPS', 'finite')
        self.conserve = model_params.get('conserve', 'parity')
        self.verbose = model_params.get('verbose', 2)
        self.J = model_params.get('J', 1.)
        self.K = model_params.get('K', 1/3)
        
        site = SpinSite(S=self.S, conserve=self.conserve)
        site.add_op("SxSx", site.multiply_operators(["Sx","Sx"]) )
        site.add_op("SxSy", site.multiply_operators(["Sx","Sy"]) )
        site.add_op("SxSz", site.multiply_operators(["Sx","Sz"]) )
        site.add_op("SySx", site.multiply_operators(["Sy","Sx"]) )
        site.add_op("SySy", site.multiply_operators(["Sy","Sy"]) )
        site.add_op("SySz", site.multiply_operators(["Sy","Sz"]) )
        site.add_op("SzSx", site.multiply_operators(["Sz","Sx"]) )
        site.add_op("SzSy", site.multiply_operators(["Sz","Sy"]) )
        site.add_op("SzSz", site.multiply_operators(["Sz","Sz"]) )
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
        J = model_params.get("J", 1.)
        K = model_params.get("K", 1/3)
        for l in range(self.Lx):
            if l < self.Lx - 1:
                i0, i1 = l, (l+1)%self.Lx
            elif l == self.Lx-1 and self.bc == 'periodic' :
                i0, i1 = 0, self.Lx-1
                print("periodic terms added in SpinSite model")
            else:
                break
            #the bilinear terms
            self.add_coupling_term(J,  i0, i1, "Sx", "Sx")
            self.add_coupling_term(J,  i0, i1, "Sy", "Sy")
            self.add_coupling_term(J,  i0, i1, "Sz", "Sz")

            self.add_coupling_term(K,  i0, i1, "SxSx", "SxSx")
            self.add_coupling_term(K,  i0, i1, "SxSy", "SxSy")
            self.add_coupling_term(K,  i0, i1, "SxSz", "SxSz")
            self.add_coupling_term(K,  i0, i1, "SySx", "SySx")
            self.add_coupling_term(K,  i0, i1, "SySy", "SySy")
            self.add_coupling_term(K,  i0, i1, "SySz", "SySz")
            self.add_coupling_term(K,  i0, i1, "SzSx", "SzSx")
            self.add_coupling_term(K,  i0, i1, "SzSy", "SzSy")
            self.add_coupling_term(K,  i0, i1, "SzSz", "SzSz")
            
class Spin1(Site):
    """
    Customized Spin-1 site, local operators are generators of SU(3)
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
        return "site for spin-1 in trs basis with conserve = {}".format(["N", self.cons_S])
        
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
        self.lat = Chain(self.Lx, site, bc=bc)
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
            elif l == self.Lx-1 and bc == 'periodic':
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
    
class SO3Kitaevchain3(bdg):
    """
    The Bogoliubov-de Gennes form of 3 Kitaev chains, the Hamiltonian matrix is finally under the x,y,z basis
    """
    def __init__(self, chi, d, lamb, Nx, D, pbc):
        self.model = "SO(3)Chain_L{}_chi{}_d{}_lambda{}_D{}".format(Nx, round(chi,6), round(d,6), round(lamb,6), D)
        super().__init__(Nx=Nx, Ny=1, model=self.model, D=D)
        self.t, self.d = round(-chi, 6), round(d, 6)
        self.mu = lamb
        self.dtype = np.float64
        self.pbc = pbc
        self.Nx = Nx #the REAL site number
        self.Nlatt = Nx * 3 #here we defined Nlatt is the number of interleaved indecies
        
    def hamiltonian(self):
        N = self.Nx
        self.tmat = np.zeros((N, N), self.dtype)
        self.dmat = np.zeros((N, N), self.dtype)
        t, d = self.t, self.d
        mu = self.mu
        print("t=",self.t, "d=",self.d, "mu=",self.mu)
        for i in range(N):
            self.tmat[i, i] = mu/2
        for i in range(N-1):
            self.tmat[i, (i+1)%N] = t 
            self.dmat[i, (i+1)%N] = d 
        self.parity = 1
        if self.pbc:
            parity = - 1 + 2 * ( N % 2 )
            self.parity = parity
            self.tmat[N-1, 0] = t*parity 
            self.dmat[N-1, 0] = d*parity 
        self.tmat += self.tmat.T.conj()
        self.dmat -= self.dmat.T
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])

        #now copy 3 times of it
        self.ham = np.kron(self.ham, np.eye(3))
        self.tmat = self.ham[:self.Nlatt, :self.Nlatt]
        self.dmat = self.ham[:self.Nlatt, self.Nlatt:]
        
        '''
        #plot the full hamiltonian
        fig, axs = plt.subplots(1, 2)
        axs[0].matshow(self.tmat)
        axs[0].set_title('tmat')
        axs[1].matshow(self.dmat)
        axs[1].set_title('dmat')
        plt.show()
        '''

        self.eig_eng, self.eig_vec = bdgeig_zeromode(self.ham, ref=0.0)
        print("the eig energies", self.eig_eng)
        #self.exact_gseng = -self.eig_eng[:self.Nlatt].sum()/2 + np.trace(self.tmat)/2 #the exact gsenergy of BdG Ham
        #print("the exact ground state energy is", self.exact_gseng)
        self.u, self.v = m2uv(self.eig_vec)
        self.m = uv2m(self.u, self.v)
        print("check the hamiltonian diagonalization: HM=ME, M=[[u,v*];[v,u*]]")
        check_orthonormal(self.ham, self.m)
        self.covariance()
        
    def hamiltonian_nozeromode(self):
        N = self.Nx
        self.tmat = np.zeros((N, N), self.dtype)
        self.dmat = np.zeros((N, N), self.dtype)
        t, d = self.t, self.d
        mu = self.mu
        print("t=",self.t, "d=",self.d, "mu=",self.mu)
        for i in range(N):
            self.tmat[i, i] = mu/2
        for i in range(N-1):
            self.tmat[i, (i+1)%N] = t 
            self.dmat[i, (i+1)%N] = d 
        self.parity = 1
        if self.pbc:
            parity = - 1 + 2 * ( N % 2 )
            self.parity = parity
            self.tmat[N-1, 0] = t*parity 
            self.dmat[N-1, 0] = d*parity 
        self.tmat += self.tmat.T.conj()
        self.dmat -= self.dmat.T
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])

        #now copy 3 times of it
        self.ham = np.kron(self.ham, np.eye(3))
        self.tmat = self.ham[:self.Nlatt, :self.Nlatt]
        self.dmat = self.ham[:self.Nlatt, self.Nlatt:]

        self.eig_eng, self.eig_vec = LA.eigh(self.ham)
        print("the eig energies", self.eig_eng)
        self.u, self.v = m2uv(self.eig_vec)
        self.m = uv2m(self.u, self.v)
        
def gutzwiller_projection_flat(parton_bflat):
    """
    The Gutzwiller projection function. 
    
    Inputs:
        1. parton_bflat, list of ndarrays, labeled in 'p', 'vL', 'vR'
        
    Outputs:
        1. spin1_bflat, list of ndarrays, labeled in 'p', 'vL', 'vR', ready for the Spin1 site
    """
    l_parton = len(parton_bflat)
    l_spin = l_parton // 3
    spin1_bflat = []
    for j in range(l_spin):
        A1 = parton_bflat[3*j]
        A2 = parton_bflat[3*j + 1]
        A3 = parton_bflat[3*j + 2]
        
        B0 = np.einsum('ab,bc,cd->ad',A1[1,:,:],A2[0,:,:],A3[0,:,:])
        B1 = np.einsum('ab,bc,cd->ad',A1[0,:,:],A2[1,:,:],A3[0,:,:])
        B2 = np.einsum('ab,bc,cd->ad',A1[0,:,:],A2[0,:,:],A3[1,:,:])
        
        left_leg_dimension = np.shape(B0)[0]
        right_leg_dimension = np.shape(B2)[-1]

        B_temp = np.zeros((3,left_leg_dimension,right_leg_dimension))
        B_temp[0,:,:] = B0
        B_temp[1,:,:] = B1
        B_temp[2,:,:] = B2
        
        spin1_bflat.append(B_temp)
    return spin1_bflat

def gutzwiller_projection_npc(parton_mps):
    """
    The Gutzwiller projection function, operating tenpy.MPS objects
    
    Inputs:
        1. parton_mps, tenpy.MPS object
        
    Outputs:
        1. spin1_mps, tenpy.MPS object
    """
    lc = parton_mps.L
    lx = lc // 3
    parton_site = parton_mps.sites[0]
    parton_leg = parton_site.leg
    spin1_site = Spin1()
    spin1_leg = spin1_site.leg
    
    projector = npc.zeros([spin1_leg, parton_leg.conj(), parton_leg.conj(), parton_leg.conj()], labels=['p','p*1','p*2','p*3'], dtype=parton_mps.dtype) #dim: 3,2,2,2
    projector[0,1,0,0] = 1
    projector[1,0,1,0] = 1
    projector[2,0,0,1] = 1
    
    spin1_mps = MPS.from_product_state([spin1_site]*lx, [0]*lx)
    for i in range(lx):
        tp1 = npc.tensordot(projector, parton_mps.get_B(3*i), axes=(['p*1'],['p']))
        tp2 = npc.tensordot(tp1, parton_mps.get_B(3*i+1), axes=(['p*2', 'vR'], ['p', 'vL']))
        tp3 = npc.tensordot(tp2, parton_mps.get_B(3*i+2), axes=(['p*3', 'vR'], ['p', 'vL']))
        spin1_mps.set_B(i, tp3, form=None)
    spin1_mps.canonical_form()
    return spin1_mps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=6)
    parser.add_argument("-theta", type=float, default=np.arctan(1/3))
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=1.)
    parser.add_argument("-lamb", type=float, default=.5)
    parser.add_argument("-D", type=int, default=10)
    parser.add_argument("-pbc", type=int, default=0)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=1/3)
    args = parser.parse_args()
    
    np.random.seed(0)
    
    theta, chi, delta, lamb = round(args.theta, 6), round(args.chi, 6), round(args.delta, 6), round(args.lamb, 6)
    J, K = round(args.J, 6), round(args.K, 6)
    D = args.D
    lx = args.lx #the number of REAL SPIN sites
    lc = np.int64(lx*3) #the number of INTERLEAVED sites
    pbc = args.pbc
    if pbc == 1:
        bc = 'periodic'
    else:
        bc = 'open'
        
    model_params = dict(Lx=lx, theta=theta, bc=bc, J=J, K=K, cons_N=None, cons_S=None)
    threekitaev = SO3Kitaevchain3(chi, delta, lamb, lx, D, pbc)
    su3dmrgmodel = BBQJK(model_params)
    spinsitedmrgmodel = BBQModel_SpinSite(model_params)
    
    print("----------Build-in AKLT----------")
    akltparams = dict(L = lx, J=1, bc = bc, conserve = 'parity')
    aklt = AKLTChain(akltparams)
    psiaklt = aklt.psi_AKLT()
    print("expected open boundary AKLT energy from theory (J=1, K=1/3)", -2/3 *(lx-1)*1)
    print(" ")
    
    print("----------SpinSite model DMRG----------")
    sites1 = spinsitedmrgmodel.sites
    init = [0]*(lx//3)+[1]*(lx//3)+[2]*(lx-lx//3-lx//3)
    np.random.shuffle(init)
    psi1 = MPS.from_product_state(sites1, init, "finite")
    dmrg_params1 = dict(mixer=True, max_E_err=1.e-12 , max_chi = 100)
    eng1 = dmrg.TwoSiteDMRGEngine(psi1, spinsitedmrgmodel, dmrg_params1)
    E1, psi1 = eng1.run()
    print("SpinSite DMRG results")
    print("psi1 after DMRG is", psi1)
    print("E1 is", E1)
    print(" ")
    
    print("----------SU(3) Spin1 model DMRG---------")
    psi2, E2 = su3dmrgmodel.run_dmrg()
    print("SU3 site DMRG results")
    print("psi2 after DMRG is", psi2)
    print("E2 is", E2)
    
    print("----------MPOMPS full hamiltonian----------")
    threekitaev.hamiltonian_nozeromode()
    gpsites = su3dmrgmodel.sites
    su3mpo = su3dmrgmodel.calc_H_MPO()
    
    vmat = threekitaev.v
    umat = threekitaev.u
    print("build mlwo")
    wv, wu = mpos.Wannier_Z2(vmat.T, umat.T)
    params_mpompstrivial = dict(cons_N=None, cons_S=None, trunc_params=dict(chi_max=512))
    mpos = mpos.MPOMPSfullham_trivial(wv, wu, **params_mpompstrivial)
    mpos.run()
    print("GP to spin1 trivial")
    
    bflatmpos = deepcopy(mpos.psi._B)
    for i in range(lc):
        bflatmpos[i] = npc.Array.to_ndarray(bflatmpos[i])
        bflatmpos[i] = np.transpose(bflatmpos[i], (1,0,2)) #reshape into p,vL,vR
        print('reshaped bflatmpos shape', np.shape(bflatmpos[i]))
    gpbflatmpos = gutzwiller_projection_flat(bflatmpos)
    mposgppsi = MPS.from_Bflat(gpsites, gpbflatmpos)
    mposgppsi.canonical_form()
    print(" ")
    print("the sandwich of fullham mpos and SO(3) MPO is", su3mpo.expectation_value(mposgppsi))
    
    print(" ")
    print("overlap of mposgppsi and no-sym spin1 DMRG result", mposgppsi.overlap(psi2))
    
    mposgppsi2 = gutzwiller_projection_npc(mpos.psi)
    print(" ")
    print("the sandwich of fullham mpos2 and SO(3) MPO is", su3mpo.expectation_value(mposgppsi2))
    print(" ")
    print("overlap of mposgppsi2 and no-sym spin1 DMRG result", mposgppsi2.overlap(psi2))
    
    print("----------Pfaffian Process Start----------")
    print("Building the Hamiltonian")
    threekitaev = SO3Kitaevchain3(chi, delta, lamb, lx, D, pbc)
    threekitaev.hamiltonian()
    pmps = [] #list of pfaffian mps
    for i in range(1,lc+1):
        tensortemp = threekitaev.A_nba(i)
        pmps.append(tensortemp)
    
    bflat = deepcopy(pmps)
    bflat[0] = np.reshape(bflat[0], (1,2,2))
    bflat[0] = np.transpose(bflat[0], (1,0,2))
    bflat[-1] = np.reshape(bflat[-1], (2,2,1))
    #the leg order of bflat is now in 'p', 'vL', 'vR', ready for project

    gpmps = gutzwiller_projection_flat(bflat)

    for i in range(lx):
        print("shape of gpmps", np.shape(gpmps[i]))

    gpsites = su3dmrgmodel.sites
    gppsi = MPS.from_Bflat(gpsites,gpmps)
    gppsi.canonical_form()
    su3mpo = su3dmrgmodel.calc_H_MPO()
    print(" ")
    print("the sandwich of projected psi and SO(3) MPO is", su3mpo.expectation_value(gppsi))
    
    print(" ")
    print("overlap of gpmps and no-sym spin1 DMRG result", gppsi.overlap(psi2))