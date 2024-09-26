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

print(" Dimercheck ")
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=6)
    args = parser.parse_args()
    
    lx = args.lx
    
    import os
    homepath  = os.getcwd()
    
    fname = homepath + '/psimpos_lx{}_1'.format(lx)
    with open(fname, 'rb') as f:
        psiapbc = pickle.load(f)
    fname = homepath + '/psimpos_lx{}_2'.format(lx)
    with open(fname, 'rb') as f:
        psipbc = pickle.load(f)
        
    site = psiapbc.sites[0]
    transop = trnslop_mpo(site, lx)
    tpsi = deepcopy(psiapbc)
    tpsi = apply_mpo(transop, tpsi)
    tpsi.canonical_form()
    print('<pbc|T|apbc> = ', psipbc.overlap(tpsi))
    
    tpsi = deepcopy(psiapbc)
    tpsi = apply_mpo(transop, tpsi)
    tpsi.canonical_form()
    print('<apbc|T|apbc> = ', psiapbc.overlap(tpsi))
    
    tpsi = deepcopy(psipbc)
    tpsi = apply_mpo(transop, tpsi)
    tpsi.canonical_form()
    print('<pbc|T|pbc> = ', psipbc.overlap(tpsi))