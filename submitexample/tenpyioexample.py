import numpy as np
import scipy as sp
import sys, os
from tenpy.tools.params import asConfig
from tenpy.models.model import CouplingModel, MPOModel
from mpomps_tenpy import Electron
from tenpy.networks.site import Site
from tenpy.models.lattice import Lattice, Chain, Square
from tenpy.algorithms import dmrg
from tenpy.algorithms.truncation import TruncationError
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
import pickle
from matplotlib import pyplot as plt

from mpomps_tenpy import *
from PartonSquare import *


class SpinHalf(Site):
    def __init__(self, cons_N=None, cons_S=None):
        """
        the basis is {empty, up, down, double}
        """
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        if cons_N == 'N' and cons_S == '2*Sz':
            chinfo = npc.ChargeInfo([1, 1], ['N', '2*Sz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 1], [1, -1]])
        elif cons_N == 'N' and cons_S == 'parity':
            chinfo = chinfo = npc.ChargeInfo([1, 2], ['N', 'parity'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 0], [1, 1]])
        elif cons_N == "N":
            chinfo = chinfo = npc.ChargeInfo([1], ['N'])
            leg = npc.LegCharge.from_qflat(chinfo, [1, 1])
        elif cons_N == "Z2" and cons_S == '2*Sz':
            chinfo = chinfo = npc.ChargeInfo([1, 1], ['N', '2*Sz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 1], [1, -1]])
        elif cons_N == "Z2" and cons_S == 'parity':
            chinfo = chinfo = npc.ChargeInfo([1, 2], ['N', 'parity'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 0], [1, 1]])
        elif cons_N == 'Z2':
            chinfo = chinfo = npc.ChargeInfo([1], ['N'])
            leg = npc.LegCharge.from_qflat(chinfo, [1,1])
        else:
            leg = npc.LegCharge.from_trivial(2)

        JW = np.diag([-1, -1])
        JWu = np.diag([-1,  1])
        JWd = np.diag([1, -1])
        Sp = np.array([[0.        , 1.],
                       [0.        , 0.]])
        Sm = Sp.T
        Sz = np.diag([0.5, -0.5])
        Sx = (Sp + Sm)/2
        iSy = (Sp - Sm)/2
        Sy = -1j*(Sp - Sm)/2


        Nu = np.diag([1., 0.])
        Nd = np.diag([0., 1.])

        if cons_S == "2*Sz":
            ops = dict(JW=JW, JWu=JWu, JWd=JWd,
                       Sp=Sp, Sm=Sm, Sz=Sz, 
                       Nu=Nu, Nd=Nd)
        else:
            ops = dict(JW=JW, JWu=JWu, JWd=JWd,
                       Sp=Sp, Sm=Sm, Sz=Sz, 
                       Sx=Sx, Sy=Sy, iSy=iSy,
                       X=2*Sx, Y=2*Sy, Z=2*Sz,
                       Nu=Nu, Nd=Nd)
        names = ['up', 'down']
        Site.__init__(self, leg, names, **ops)

    def __repr__(self):
        """Debug representation of self."""
        return "site for spin-1/2  with conserve = {}".format(["N", self.cons_S])

def mps1_mps2(mps1, mps2):
    assert len(mps1._B) == len(mps2._B)
    L = len(mps1._B)
    left = npc.tensordot(mps1._B[0].conj(), mps2._B[0], axes=('p*', 'p'))
    for _ in range(1, L):
        left = npc.tensordot(left, mps1._B[_].conj(), axes=(['vR*'],["vL*"]))
        left = npc.tensordot(left, mps2._B[_], axes=(['vR','p*'],['vL','p']))
    value = left.to_ndarray()
    return value.reshape(-1)[0]

def mps1_mpo_mps2(mps1, mpo, mps2):
    assert len(mps1._B) == len(mpo) == len(mps2._B)
    L = len(mps1._B)
    temp = npc.tensordot(mps1._B[0].conj(), mpo[0], axes=('p*', 'p'))
    left = npc.tensordot(temp, mps2._B[0], axes=('p*', 'p'))
    for _ in range(1, L):
        temp = npc.tensordot(mps1._B[_].conj(), mpo[_], axes=('p*', 'p'))
        left = npc.tensordot(left, temp, axes=(['vR*', 'wR'],["vL*", 'wL']))
        left = npc.tensordot(left, mps2._B[_], axes=(['vR','p*'],['vL','p']))
    value = left.to_ndarray()
    return value.reshape(-1)[0]*mps1.norm*mps2.norm

def Electron2SpinHalf(mps):
    elect = mps.sites[0]
    spin = SpinHalf(*elect.conserve)
    projector = npc.zeros( [spin.leg, elect.leg.conj()], qtotal=[0,0], labels=['p', 'p*'], dtype=mps.dtype )
    projector[0, 1] = 1
    projector[1, 2] = 1 
    L = mps.L 
    gp_psi = MPS.from_product_state([spin]*L, [0]*L)
    for _ in range(L):
        t1 = npc.tensordot(mps._B[_], projector, axes=(['p'],['p*']))
        gp_psi.set_B(_, t1, form=None)
    gp_psi.canonical_form()
    return gp_psi

def trnslop_mpo(site, L=2, **kwargs):
    bc = kwargs.get('bc', 'pbc')
    
    assert L>1
    leg = site.leg
    chinfo = leg.chinfo
    zero_div = [0]*chinfo.qnumber
    from tenpy.linalg.charges import LegPipe, LegCharge
    # cleg = LegPipe([leg, leg.conj()]).to_LegCharge()
    cleg = npc.LegCharge.from_qflat(chinfo, [[0,0],[0,-2],[0,2],[0,0]])
    nleg = npc.LegCharge.from_qflat(chinfo, [zero_div])
    
    swap = npc.zeros([leg, leg.conj(), leg, leg.conj()], qtotal=zero_div, labels=['p1', 'p1*', 'p2', 'p2*']) 
    swap[0,0,0,0]=1
    swap[0,1,1,0]=1
    swap[1,0,0,1]=1
    swap[1,1,1,1]=1
    
    reshaper = npc.zeros([leg, leg.conj(), cleg.conj()], qtotal=zero_div, labels=['p', 'p*', '(p*.p)'] )
    reshaper[0,0,0] = 1;  reshaper[1,0,1] = 1;  reshaper[0,1,2] = 1; reshaper[1,1,3] = 1
    
    swap = npc.tensordot(reshaper.conj(), swap, axes=((0,1),(0,1)))
    swap.ireplace_labels(['(p.p*)'], ['p1.p1*'])
    swap = npc.tensordot(swap, reshaper.conj(), axes=((1,2),(0,1)))
    swap.ireplace_labels(['(p.p*)'], ['p2.p2*'])
    
    left, right = npc.qr(swap)
    # print(left.to_ndarray(), right.to_ndarray(),)
    
    left  = npc.tensordot(reshaper, left, axes=((2), (0)))
    left.ireplace_labels([None], ['wR'])
    
    right = npc.tensordot(right, reshaper, axes=((1), (2)))
    right.ireplace_labels([None], ['wL'])
    
    bulk = npc.tensordot(right, left, axes=('p*','p'))
    if bc == 'pbc':
        bt = npc.zeros([nleg, leg, leg.conj()], qtotal=zero_div, labels=['wL', 'p', 'p*',])
        bt[0,0,0] = 1; bt[0,1,1]=1
        left = npc.tensordot(bt, left, axes=(('p*', 'p')))
    
        bt = npc.zeros([leg, leg.conj(), nleg.conj()], qtotal=zero_div, labels=['p', 'p*', 'wR'])
        bt[0,0,0] = 1; bt[1,1,0]=1
        right = npc.tensordot(right, bt, axes=(('p*', 'p')))
    
    return [left] +  [bulk]*(L-2) +  [right] 

def apply_mpo(mpo, mps, i0=0):
    L = len(mpo)
    for i in range( i0, i0+L ):
        B = npc.tensordot(mps.get_B(i, 'B'), mpo[i-i0], axes=('p', 'p*'))
        B = B.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
        B.ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
        B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
        B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
        mps._B[i] = B#.itranspose(('vL', 'p', 'vR'))
    return mps

def calc_rdc_2mps(mps1, mps2, i=0):
    left = npc.ones( (mps1.get_B(0).legs[0], mps2.get_B(0).legs[0].conj()), labels=['vR', 'vR*'] )
    for idx in range(i):
        left = npc.tensordot(mps1.get_B(idx), left, axes=('vL', 'vR'))
        left = npc.tensordot(left, mps2.get_B(idx).conj(), axes=(('p', 'vR*'), ('p*', 'vL*')) )
    return left        

def calc_entanglement_spectrum_yshift(psi, lx, ly, i=None):
    if i is None:
        i = (lx//2)*ly
    site = psi.sites[0]
    swap = trnslop_mpo(site, ly)
    psis = psi.copy()
    for i0 in range(i//ly):
        psis = apply_mpo(swap, psis, i0=i0*ly)
    left = calc_rdc_2mps(psi, psis, i=i)
    l1, l2 = left.legs
    ees = []
    for _v1 in range(l1.block_number):
        cv1 = l1.charges[_v1]*l1.qconj
        qv1 = l1.get_qindex_of_charges(cv1)
        sv1 = l1.get_slice(qv1)
        cv2 = np.array(cv1)*-1
        qv2 = l2.get_qindex_of_charges(cv2)
        # print(cv1, cv2, qv1, qv2)
        block = left.get_block([qv1,qv2])
        eigval = np.linalg.eigvals(block)
        ees.append( (cv2, eigval ) )
    return ees    

    
class HeisenbergJ1J2Square(CouplingModel):

    def __init__(self, model_params):
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        Lx = model_params.get('Lx', 8)
        Ly = model_params.get('Ly', 4)
        self.Lx = Lx
        self.Ly = Ly
        bc = model_params.get('bc', 'periodic')
        bc_MPS = model_params.get('bc_MPS', 'finite')
        cons_S = model_params.get('cons_S', "2*Sz")
        self.cons_S = cons_S

        site = SpinHalf(cons_N='Z2', cons_S=cons_S)
        self.site = site
        self.lat = Square(Lx, Ly, site, bc=['open', bc], bc_MPS=bc_MPS)
        CouplingModel.__init__(self, self.lat, explicit_plus_hc=False)

        self.init_terms(model_params)

        H_MPO = self.calc_H_MPO()
        if model_params.get('sort_mpo_legs', False):
            H_MPO.sort_legcharges()
        MPOModel.__init__(self, self.lat, H_MPO)
        model_params.warn_unused()

    def _xy2lable(self, ix, iy):
        Lx, Ly = self.Lx, self.Ly
        unit_cell = (ix % Lx) * Ly + (iy % Ly)
        return unit_cell
           
    def init_terms(self, model_params):
        self.t1 = model_params.get('t1', 1.0)
        self.t2 = model_params.get('t2', 0.5)
        self.tx = model_params.get('tx', 0.05)

        self.nn1st = []
        self.nn2nd = []
        self.nnx3s = []

        lx, ly = self.Lx, self.Ly
        for _x in range(lx):
            for _y in range(ly):
                id0    = self._xy2lable(_x, _y)
                idpx   = self._xy2lable(_x+1, _y)
                idpy   = self._xy2lable(_x, _y+1)
                idpxpy = self._xy2lable(_x+1, _y+1)
                idmxpy = self._xy2lable(_x-1, _y+1)

                '''
                t1
                '''
                if _y == ly-1:
                    self.nn1st.append((idpy, id0))
                    self.add_coupling_term(    self.t1, idpy, id0, 'Sz', 'Sz', op_string='Id', plus_hc=False)
                    self.add_coupling_term(0.5*self.t1, idpy, id0, 'Sp', 'Sm', op_string='Id', plus_hc=True)
                else:
                    self.nn1st.append((id0, idpy))
                    self.add_coupling_term(    self.t1, id0, idpy, 'Sz', 'Sz', op_string='Id', plus_hc=False)
                    self.add_coupling_term(0.5*self.t1, id0, idpy, 'Sp', 'Sm', op_string='Id', plus_hc=True)

                if _x < lx - 1:
                    self.nn1st.append((id0, idpx))
                    self.add_coupling_term(    self.t1, id0, idpx, 'Sz', 'Sz', op_string='Id', plus_hc=False)
                    self.add_coupling_term(0.5*self.t1, id0, idpx, 'Sp', 'Sm', op_string='Id', plus_hc=True)
                '''
                t2
                '''
                if _y == ly-1 and _x == 0:
                    pass
                elif _y == ly-1 and _x !=0:
                    self.nn2nd.append((idmxpy, id0))
                    self.add_coupling_term(    self.t2, idmxpy, id0, 'Sz', 'Sz', op_string='Id', plus_hc=False)
                    self.add_coupling_term(0.5*self.t2, idmxpy, id0, 'Sp', 'Sm', op_string='Id', plus_hc=True)
                elif _x == 0 and _y != ly-1:
                    pass
                else:
                    self.nn2nd.append((idmxpy, id0))
                    self.add_coupling_term(    self.t2, idmxpy, id0, 'Sz', 'Sz', op_string='Id', plus_hc=False)
                    self.add_coupling_term(0.5*self.t2, idmxpy, id0, 'Sp', 'Sm', op_string='Id', plus_hc=True)

                if _y == ly-1 and _x == lx-1:
                    pass
                elif _y == ly-1 and _x != lx-1:
                    self.nn2nd.append((id0, idpxpy))
                    self.add_coupling_term(    self.t2, id0, idpxpy, 'Sz', 'Sz', op_string='Id', plus_hc=False)
                    self.add_coupling_term(0.5*self.t2, id0, idpxpy, 'Sp', 'Sm', op_string='Id', plus_hc=True)
                elif _x == lx-1 and _y != ly-1:
                    pass
                else:
                    self.nn2nd.append((id0, idpxpy))
                    self.add_coupling_term(    self.t2, id0, idpxpy, 'Sz', 'Sz', op_string='Id', plus_hc=False)
                    self.add_coupling_term(0.5*self.t2, id0, idpxpy, 'Sp', 'Sm', op_string='Id', plus_hc=True)
                '''
                tx
                '''
                si, sj, sk = self._xy2lable(_x, _y), self._xy2lable(_x, _y+1), self._xy2lable(_x+1, _y)
                if _x != lx-1:
                    self.add_chiral_term([si, sj, sk])
                    self.nnx3s.append( [si, sj, sk] )

                si, sj, sk = self._xy2lable(_x, _y), self._xy2lable(_x+1, _y), self._xy2lable(_x, _y-1)
                if _x != lx-1:
                    self.add_chiral_term([si, sj, sk])
                    self.nnx3s.append( [si, sj, sk] )

                si, sj, sk = self._xy2lable(_x, _y), self._xy2lable(_x, _y-1), self._xy2lable(_x-1, _y)
                if _x != 0:
                    self.add_chiral_term([si, sj, sk])
                    self.nnx3s.append( [si, sj, sk] )

                si, sj, sk = self._xy2lable(_x, _y), self._xy2lable(_x-1, _y), self._xy2lable(_x, _y+1)
                if _x != 0:
                    self.add_chiral_term([si, sj, sk])
                    self.nnx3s.append( [si, sj, sk] )

        print('1st nn of {}'.format(len(self.nn1st)), self.nn1st)
        print('2nd nn of {}'.format(len(self.nn2nd)), self.nn2nd)
        print('chiral nn of {}'.format(len(self.nnx3s)), self.nnx3s)

    def add_chiral_term(self, ijk, tx=None):
        ijk = np.array(ijk)
        p   = np.argsort(ijk)
        if tx is None:
            tx = self.tx
        opls = [[1j, 'Sp', 'Sm', 'Sz'], [-1j, 'Sp', 'Sz', 'Sm'], [1j, 'Sz', 'Sp', 'Sm']]
        p = np.argsort(ijk)
        for opl in opls:
            self.add_multi_coupling_term(tx*opl[0]/2, ijk[p], np.array(opl[1:])[p], op_string=['Id','Id'], plus_hc=True)

    def plot_lat(self, ax=None):
        if ax is None:
            ax = plt.subplot(111)
        self.lat.plot_basis(ax)
        self.lat.plot_coupling(ax)
        self.lat.plot_bc_identified(ax)
        self.lat.plot_sites(ax)
        self.lat.plot_order(ax)
        plt.show()

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
            init = [0]*(N//2)+[1]*(N//2)
            np.random.shuffle(init)
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)
            psiinit.norm = 1
            psiinit.canonical_form()
            # print("init total particle number", psiinit.expectation_value('Ntot').sum() )
            print("init total Sz number", psiinit.expectation_value('Sz').sum() )
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-Lx", type=int, default=12)
    parser.add_argument("-Ly", type=int, default=4)    
    parser.add_argument("-bcy",type=int, default=1)
    parser.add_argument("-t2", type=float, default=0.5)
    parser.add_argument("-tx", type=float, default=0.05)
    parser.add_argument("-U",  type=float, default=10.0)
    parser.add_argument("-chi",     type=int, default=5)
    parser.add_argument("-Sweeps",  type=int, default=5)
    parser.add_argument("-init",    type=str, default='i')
    parser.add_argument("-init2",   type=str, default='i')
    parser.add_argument("-verbose", type=int, default=1)
    parser.add_argument("-cmprss", type=str, default="SVD")
    parser.add_argument("-job", type=str, default='dmrg')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.verbose)
    for _ in ['parso.python.diff', 'parso.cache', 'parso.python.diff', 
              'parso.cache', 'matplotlib.font_manager', 'tenpy.tools.cache', 
              'tenpy.algorithms.mps_common', 'tenpy.linalg.lanczos', 'tenpy.tools.params']:
        logging.getLogger(_).disabled = True

    homepath  = os.getcwd()
    if os.path.isdir(homepath+'/data/') == False:
        os.mkdir(homepath+'/data/')
    path = homepath + '/data/' + "HeisenbergSquare_t1t2tx_Ly_{}_Lx_{}/".format(args.Ly, args.Lx)
    if os.path.isdir(path) == False:
        os.mkdir(path)

    model_params = dict(Ly=args.Ly, Lx=args.Lx, t2=args.t2, tx=args.tx)
    model = HeisenbergJ1J2Square(model_params)
    if args.job == 'dmrg': 
        if args.init == 'rand':
            initname = 'rand'
            psidmrg, E = model.run_dmrg(chi_max=args.chi, max_sweeps=args.Sweeps, init=None)
        else:
            with open(args.init, 'rb') as f:
                psii = pickle.load(f)
            if psii.sites[0].__repr__()[:16] == 'site for hubbard':
                psii = Electron2SpinHalf(psii)
            initname = args.init.split('/')[-1]
            psidmrg, E = model.run_dmrg(chi_max=args.chi, max_sweeps=args.Sweeps, init=psii)
        print('energy = ', E)

        fname = path+'DMRG_Psi_{}_t_{}_U_{}_D_{}'.format(initname, args.t2, args.tx, args.chi)        
        with open (fname, 'wb') as f:
            pickle.dump(psidmrg, f)
    if args.job == 'eng':
        with open(args.init, 'rb') as f:
            psi1 = pickle.load(f)
        if psi1.sites[0].__repr__()[:16] == 'site for hubbard':
            psi1 = Electron2SpinHalf(psi1)
        print( 'load psi from ', args.init)
        energy = model.H_MPO.expectation_value(psi1)
        print("energy = ", energy)
    if args.job == '2eng':
        with open(args.init, 'rb') as f:
            psi1 = pickle.load(f)
        if psi1.sites[0].__repr__()[:16] == 'site for hubbard':
            psi1 = Electron2SpinHalf(psi1)
        print( 'load psi from ', args.init)

        model_params = dict(Ly=args.Ly, Lx=args.Lx, t2=0.5, tx=0.0)
        model = HeisenbergJ1J2Square(model_params)
        energy = model.H_MPO.expectation_value(psi1)
        print("part1 energy = ", energy)

        model_params = dict(Ly=args.Ly, Lx=args.Lx, t1=0.0, t2=0.0, tx=1)
        model = HeisenbergJ1J2Square(model_params)
        energy = model.H_MPO.expectation_value(psi1)
        print("part2 energy = ", energy)

    if args.job == 'ovlp':
        with open(args.init, 'rb') as f:
            psi1 = pickle.load(f)
        if psi1.sites[0].__repr__()[:16] == 'site for hubbard':
            psi1 = Electron2SpinHalf(psi1)

        with open(args.init2, 'rb') as f:
            psi2 = pickle.load(f)
        if psi2.sites[0].__repr__()[:16] == 'site for hubbard':
            psi2 = Electron2SpinHalf(psi2)
        ovlp = psi1.overlap(psi2)
        print( 'overlap = ', ovlp, ' with abs = ', abs(ovlp) )
    if args.job == 'mpomps':
        params_spin = dict(lx=args.Lx, ly=args.Ly, bcy=args.bcy, bcx=0, t=1)
        model_qsl = SU2PiSquare(params_spin)
        fu = model_qsl.calc_wannier_state()
        params_mpomps = dict(cons_N="N", cons_S="2*Sz", trunc_params=dict(chi_max=args.chi))
        eng_spin = MPOMPSU1(fu, **params_mpomps)
        eng_spin.run()
        gpsi_spin = gutzwiller_projection(eng_spin.psi)
        psii = Electron2SpinHalf(gpsi_spin)
        fname = path+'SU2Piflux_WY_{}_D_{}'.format(args.bcy, args.chi)        
        with open (fname, 'wb') as f:
            pickle.dump(psii, f)


        