from itertools import chain, permutations
from os import TMP_MAX, stat
from typing import OrderedDict
import numpy as np
import scipy as sp
import numpy.linalg as LA
from scipy import linalg
from scipy import sparse
import sys, os
from pfaffian import pfaffian 
import pickle

def roundprint(a, n=4):
    print( np.round(a,n) )

def get_bmrotation(n):
    sy = -np.array([[0,1],[-1,0]])
    return np.kron(np.eye(n//2), sy )

def sqrtmat(m):
    s,v,d = np.linalg.svd(m)
    v = np.diag( np.sqrt(v) )
    return s @ v @ d

def TakagiFactorization(A):
    s,v,d = np.linalg.svd(A)
    # v = np.diag( np.sqrt(v) )
    z = s.T @ d.conj().T
    q = sqrtmat(z)
    ua = s @ q.conj() @ np.diag( np.sqrt(v) )
    roundprint( ua @ ua.T - A)
    return ua, 

def asymeig(v):
    ve, vs = np.linalg.eig(v)
    u = np.zeros( vs.shape )
    dege = v.shape[0]
    for i in range(dege//2):
        u[:,i*2+0] = vs[:,i].real * np.sqrt(2)
        u[:,i*2+1] = vs[:,i].imag * np.sqrt(2)
    return u, u.T @ v @ u

def bloch_messiah(u, v, tol=1e-8):
    du, ubar, cu = np.linalg.svd(u)
    p = np.argsort(ubar)
    ubar = ubar[p]; du, cu = du[:,p], cu[p,:]
    rank_u = np.sum(ubar < 1-tol)
    # print("ubar", ubar)
    ubar = ubar[:rank_u]
    du = du[:, :rank_u]
    cu = cu[:rank_u, :]
    dv, v_snglr, cv = np.linalg.svd(v.conj())
    # print("v_snglr:", v_snglr)
    # print("du", du.shape)
    # print("v", v.shape)
    # print("cu", cu.shape)
    vbar_t = du.T @ v @ cu.conj().T
    count_occ = np.sum( v_snglr > 1-tol )
    d = np.zeros(du.shape, du.dtype); 
    c = np.zeros(cu.shape, cu.dtype)
    d[:, :count_occ] = dv[:, :count_occ]; 
    c[:count_occ, :] = cv[:count_occ, :].conj()
    vp = []
    vbar = np.diag( [1.]*count_occ + [0.]*(rank_u - count_occ) )
    iu = count_occ
    while (iu < rank_u):
        uval = ubar[iu]
        dege = np.sum( np.abs(ubar - uval) < tol )
        # dege = 2
        # print(dege)
        if dege == 2:
            if vbar_t[iu, iu+1] < 0:
                d[:, iu]   = du[:, iu+1].copy()
                d[:, iu+1] = du[:, iu].copy()
                c[iu,  :]  = cu[iu+1, :].copy()
                c[iu+1,:]  = cu[iu, :].copy()
            else:
                d[:, iu:iu+2] = du[:, iu:iu+2].copy()
                c[iu:iu+2, :] = cu[iu:iu+2, :].copy()
            vbar[iu, iu+1] = abs(vbar_t[iu, iu+1])
            vbar[iu+1, iu] = -abs(vbar_t[iu, iu+1])
            tempv = (d[:, iu]) @ v @ (c[iu+1,:]).T.conj()
            phi = np.diag( [ np.sqrt(vbar[iu, iu+1]/tempv) ]*2 )
            d[:, iu:iu+2] = d[:, iu:iu+2] @ phi 
            c[iu:iu+2, :] = phi.conj() @ c[iu:iu+2, :]
        else:
            td = du[:, iu:iu+dege]
            tc = cu[iu:iu+dege, :]
            tempv = (td).T.conj() @ v.conj() @ (tc).T
            tempu = td.T.conj() @ u @ tc.T.conj()
            # print("trmpv")
            # roundprint(tempv)
            # print("trmpu")
            # roundprint(tempv.T + tempv)
            # roundprint(tempv.conj().T @ tempv)
            ua, tempv = asymeig(tempv)
            phi = []
            for i in range(dege//2):
                phi += [ np.sqrt(abs(tempv[i*2, i*2+1])/tempv[i*2, i*2+1])  ]*2
            phi = np.diag( phi )
            td = td @ ua @ phi.conj()
            tc = phi @ ua.T @ tc 
            tempv = (td).T.conj() @ v.conj() @ (tc).T
            # print("temp")
            # roundprint(tempv)
            vbar[iu:iu+dege, iu:iu+dege] = tempv.real
            d[:, iu:iu+dege] = td
            c[iu:iu+dege, :] = tc
        vp.append( vbar[iu, iu+1] )
        iu += dege
    # checkvbar = np.linalg.norm( d.T @ v @ c.T.conj() - vbar)
    # if checkvbar > 1e-12:
    #     print("ubar")
    #     roundprint( ( d.T.conj() @ u @ c.T.conj() ) )
    #     print("vbar")
    #     roundprint( d.T @ v @ c.T.conj() )
    #     print("true vbar")
    #     roundprint( vbar )
    #     print("checkvbar", checkvbar)
    #     raise
    return np.diag(ubar), vbar, d, c, np.prod(vp)

def check_orthonormal(ham, m):
    check = np.allclose(m.conj().T @ m, np.eye(m.shape[0]))
    if not check:
        print("check orthonormal: ", check)
        print( sparse.csr_matrix( np.round( m.conj().T @ m - np.eye(m.shape[0]), 8) ) )
        raise "check orthonormal fails"
        # print( np.round( m.conj().T @ m, 6) )
    test = m.conj().T @ ham @ m
    # print(np.round( test.diagonal(), 4))
    check = np.allclose(test, np.diag(test.diagonal()) )
    if not check:
        print("check eigen dcmpstn: ",  check)
        print(np.round( test.diagonal(), 4))
        raise "check eigen dcmpstn fails"

def uv2m(u,v):
    return np.block( [[u, v.conj()], [v, u.conj()]] )

def m2uv(m, order=None):
    na = m.shape[0]//2
    if order is None:
        order = [i for i in range(na)]
    return m[:na, order].copy(), m[na:, order].copy()

def state2order(na, sa):
    ordera = [i for i in range(na)]
    for s in sa:
        ordera[s] = ordera[s] + na
    return ordera

def check_diagonal(a):
    check = np.allclose(a, np.diag( np.diag(a) ) )
    if check:
        print( "it is diagonal"  )
    else:
        print( sparse.csr_matrix( np.round( a - np.diag( np.diag(a) ), 8) ) )

def majoranabasis(n):
    invsqrt2 = 1/np.sqrt(2)
    i = np.eye(n)*invsqrt2
    return np.block( [[i, -1j*i], [i, 1j*i]] )

def majoranabasis2(n):
    invsqrt2 = 1/np.sqrt(2)
    i = np.array([[1, -1j], [1, 1j]])*invsqrt2
    return np.kron(np.eye(n), i)

# def findmajorana(ham, ref=0.5):
#     na = ham.shape[0]//2
#     hamp = ham - np.eye(na*2)*ref
#     t, d = hamp[:na, :na], hamp[:na,na:]
#     # e, psi = sp.linalg.eig(t+d)
#     _,e,u = sp.linalg.svd(t+d)
#     # print("e",e)
#     p = np.argsort(np.abs(e))
#     u = u[p[0],:]
#     _,e,v = sp.linalg.svd(t-d)
#     p = np.argsort(np.abs(e))
#     v = v[p[0],:]
#     s1 = np.block([u+v, u-v])/2
#     s2 = np.block([u-v, u+v])/2
#     return s1, s2

# def bdgeig_zeromode(ham, tol=1e-14, ref=0.5):
#     na = ham.shape[0]//2
#     eig_eng, eig_vec = np.linalg.eigh(ham  - np.diag([ref]*na*2))
#     orderu = np.array([na-i-1 for i in range(na)])
#     orderd = orderu.tolist(); orderd.reverse(); #np.array(
#     eig_eng = np.block( [-eig_eng[:na][orderu], eig_eng[:na][orderu]] ) + ref
#     eig_vec = np.block( [eig_vec[:,na:][:,orderd], eig_vec[:,:na][:,orderu]] )
#     eig_eng = (eig_vec.conj().T @ ham @ eig_vec).diagonal()
#     if abs(eig_eng[0] - ref ) < tol:
#         psi1, psi2 = findmajorana(ham, ref)
#         eig_vec[:,0], eig_vec[:,na] = psi1, psi2
#     u, v = m2uv(eig_vec)
#     eig_vec = uv2m(u, v)
#     return eig_eng, eig_vec

def findmajorana(ham, ref=0.5):
    na = ham.shape[0]//2
    hamp = ham - np.eye(na*2)*ref
    if hamp.dtype == np.complex128 or hamp.dtype == np.complex64:
        # print("hey!!! complex!!")
        w = majoranabasis(na)
        
        xhamp = w.conj().T @ hamp @ w
        theta = ( xhamp.conj().T @ xhamp ).real
        eD, vD = sp.linalg.eigh(theta)
        # print(eD)
        # vD = (vD + vD.conj())/np.sqrt(2)
        # vD, r = np.linalg.qr(vD)
        # print(r)
        w2 = majoranabasis2(na)
        vvD = w @ vD @ w2.conj().T
        s1 = vvD[:,:1].T
        s2 = vvD[:,1:2].T
        # print(s1.conj().T @ s2)
    else:
        tmat, dmat = hamp[:na, :na], hamp[:na,na:]

        _, e, u = sp.linalg.svd(tmat+dmat)
        num_0mode = max(1, np.sum( np.abs(e) < 1e-12 ) )
        p = np.argsort(np.abs(e))
        e = e[p]
        u = u[p[:num_0mode],:]
    
        _,e,v = sp.linalg.svd(tmat-dmat)
        # e, v = sp.linalg.eig(tmat-dmat)
        p = np.argsort(np.abs(e))
        e = e[p]
        # print("e",e)
        v = v[p[:num_0mode],:]
        phaseuv = u @ v.conj().T#.conj().T
        ps, pv, pd = np.linalg.svd(phaseuv)
        num_1mode = max(1, np.sum( np.abs(pv) > 1-1e-12 ))
        u = ( ps.conj().T @ u )[:num_1mode, :]
        v = ( pd @ v )[:num_1mode,:]
        s1 = np.block([(u), (u-u)])
        s2 = np.block([(u-u).conj(), (u).conj()])
        # print( abs(s1 @ s2.T.conj()) )
        # print( s1 @ s1.conj().T )
        # print( s2 @ s2.conj().T )
        # print( u.conj() @ v )
        # print( v.conj() @ u )
    return s1, s2

def bdgeig_zeromode(ham, tol=1e-14, ref=0.5):
    na = ham.shape[0]//2
    eig_eng, eig_vec = np.linalg.eigh(ham  - np.diag([ref]*na*2))
    # print(eig_eng )
    orderu = np.array([na-i-1 for i in range(na)])
    orderd = orderu.tolist(); orderd.reverse(); #np.array(
    eig_eng = np.block( [-eig_eng[:na][orderu], eig_eng[:na][orderu]] ) + ref
    eig_vec = np.block( [eig_vec[:,na:][:,orderd], eig_vec[:,:na][:,orderu]] )
    eig_eng = (eig_vec.conj().T @ ham @ eig_vec).diagonal()
    eig_eng = eig_eng.real
    # print("eig", eig_eng)
    # print(eig_eng[0])
    # print(eig_eng[:2])
    if abs(eig_eng[0] - ref ) < tol:
        print("zero mode!!!")
        psi1, psi2 = findmajorana(ham, ref)
        # print(psi1.conj() @ psi2.T)
        num_0zero = psi1.shape[0]
        eig_vec[:,:num_0zero], eig_vec[:,na:na+num_0zero] = psi1.T, psi2.T
    u, v = m2uv(eig_vec)
    eig_vec = uv2m(u, v)
    return eig_eng, eig_vec

def ubar2lambda(ubar):
    x = ubar.diagonal()
    lmbda = np.power( 1 -x*x, 1/4  )
    return np.diag(lmbda)

def ubar2skew(ubar, tol=1e-6):
    skewu = np.zeros(ubar.shape, ubar.dtype)
    ubar = ubar.diagonal()
    iu0 = 0
    # print(ubar)
    while iu0 < skewu.shape[0]:
        if abs(ubar[iu0]) < tol:
            iu0 += 1
        else:
            skewu[iu0, iu0+1] = ubar[iu0]
            skewu[iu0+1, iu0] = -ubar[iu0]
            iu0 += 2
    return skewu

class bdg(object):
    def __init__(self, Nx=2, Ny=2, model='', D=10):
        self.Nx = Nx
        self.Ny = Ny
        self.nf = 2
        self.Nlatt = self.Nx * self.Ny 
        self.N = self.Nlatt * 2
        self.model = model
        self.nfill = self.N//self.nf
        self.D = D
        self.path = "./gaussiandata/"+self.model
        self.dtype = np.float64
        self.pbc = False
        self.eigtol = 1e-8
        self.bhmtol = 1e-8
        self.ifdynamic = 1
        self.ifparallel = 0
        self.mfpath = "./MFdata/"+self.model+"/"
        try:
            os.mkdir(self.mfpath)
        except:
            pass

    def hamiltonian(self):
        N = self.Nlatt
        self.tmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        self.dmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        t, d = self.t, self.d
        mu = self.mu
        for i in range(N):
            self.tmat[i, i] = mu/4 #np.random.rand()
        
        for i in range(N-1):
            self.tmat[i, (i+1)%N] = t #np.random.rand()
            self.dmat[i, (i+1)%N] = d #np.random.rand()
            # self.tmat[i, (i+2)%N] = 0.1 #np.random.rand()
            # self.dmat[i, (i+2)%N] = -0.1 #np.random.rand()
        self.parity = 1
        if self.pbc:
            parity = -1 + 2 * ( N % 2 )
            self.parity = parity
            self.tmat[N-1, 0] = t*parity #np.random.rand()
            self.dmat[N-1, 0] = d*parity #np.random.rand()
        # self.tmat = np.random.rand(N,N)
        # self.dmat = np.random.rand(N,N)
        self.tmat += self.tmat.T.conj()
        self.dmat -= self.dmat.T
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])
        self.eig_eng, self.eig_vec = bdgeig_zeromode(self.ham, ref=0.0, tol=self.eigtol)
        self.u, self.v = m2uv(self.eig_vec)
        self.m = uv2m(self.u, self.v)
        print("check the hamiltonian diagonalization: HM=ME, M=[[u,v*];[v,u*]]")
        check_orthonormal(self.ham, self.m)
        self.covariance()


    def covariance(self):
        self.cor11 = self.v.conj() @ self.v.T
        self.cor12 = self.v.conj() @ self.u.T
        self.cor21 = - self.cor12.conj()
        self.cor22 = np.eye(self.cor11.shape[0]) - self.cor11.conj()
        self.cov_mat = np.block([[self.cor11, self.cor12],
                                 [self.cor21, self.cor22]])
        return self.cov_mat

    def reduced_cov(self, na, ia=0):
        if na > 0:
            N = self.Nlatt
            cor11, cor12 = self.cov_mat[ia:na, ia:na], self.cov_mat[ia:na, N+ia:N+na]
            cor21, cor22 = self.cov_mat[N+ia:N+na, ia:na], self.cov_mat[N+ia:N+na, N+ia:N+na]
            self.rdc_cov = np.block([[cor11, cor12],[cor21, cor22]])
            self.rdc_E, self.rdc_V = bdgeig_zeromode(self.rdc_cov, ref=0.5, tol=self.eigtol)
        if na == 0:
            self.rdc_cov = np.zeros((0,0), dtype=self.dtype)
            self.rdc_E = np.zeros(shape=(0), dtype=self.dtype)
            self.rdc_V = np.zeros(shape=(0,0), dtype=self.dtype)
        print("check the correlation matrix diagonalization: RM=ME, M=[[u,v*];[v,u*]]")
        print("segmet = {} <----> {}, L_p:".format(ia, na))
        check_orthonormal(self.rdc_cov, self.rdc_V)
        return self.rdc_E, self.rdc_V

    def compute_rdc_K(self, i0, tol=1e-1):
        self.reduced_cov(i0, 0)
        print(self.rdc_E)
        u = self.rdc_V[:i0, :].reshape(-1, self.Ny, i0*2)
        v = self.rdc_V[i0:, :].reshape(-1, self.Ny, i0*2)
        # u = self.rdc_V[:, :i0].reshape(self.Ny, -1, i0*2)
        # v = self.rdc_V[:, i0:].reshape(self.Ny, -1, i0*2)
        # print(u.shape)
        self.rdc_K = []
        for i in range(i0*2):
            vk = np.zeros(self.Ny)
            uk = np.zeros(self.Ny)
            for ly in range(i0//6):
                if np.max(np.abs(v[ly, :, i])) > 1e-5:
                    vk += np.abs( sp.fft.fft( v[ly, :, i], self.Ny, axis=0) )
                if np.max(np.abs(u[ly, :, i])) > 1e-5:
                    uk += np.abs( sp.fft.fft( u[ly, :, i], self.Ny, axis=0) )
            # vk = np.abs( sp.fft.fft(v[:, :, i], self.Ny, axis=1) ).sum(axis=0); 
            # uk = np.abs( sp.fft.fft(u[:, :, i], self.Ny, axis=1) ).sum(axis=0); 
            k_inreal = (np.abs( vk ) > tol) + ( np.abs(uk) > tol)    
            if np.sum(k_inreal) == 1:
                # m = 
                self.rdc_K.append(np.argmax(k_inreal)/self.Ny)
            else:
                print(k_inreal)
                print( vk )
                print( uk )
                raise "momentum is not a good quanutm number!"
        self.rdc_K = np.array(self.rdc_K)
        # print(self.rdc_K)

    def compute_schmidt_K(self, state, i0):
        # states = [i+i0 for i in range(i0)] + state
        
        return np.sum(self.rdc_K[state])%1

    def saveuv(self, fname):
        np.save(fname+"u", self.u)
        np.save(fname+"v", self.v)
        np.save(fname+"eng", self.eig_eng)
        np.save(fname+"ham", self.ham)

    def loaduv(self, fname):
        self.u = np.load(fname+"u"+".npy")
        self.v = np.load(fname+"v"+".npy")
        self.eig_eng = np.load(fname+"eng"+".npy")
        self.ham = np.load(fname+"ham"+".npy")
        self.m = uv2m(self.u, self.v)
        self.eig_vec = self.m
        print("check the hamiltonian diagonalization: HM=ME, M=[[u,v*];[v,u*]]")
        check_orthonormal(self.ham, self.m)
        self.covariance()

    def saverdc(self, fname):
        for i in range(0, self.Nlatt+1):
            self.reduced_cov(i)
            np.save(fname+"rdc_E"+str(i), self.rdc_E)
            np.save(fname+"rdc_V"+str(i), self.rdc_V)

    def loadrdc(self, fname, i=0):
        rdc_E = np.load(fname+"rdc_E"+str(i)+".npy")        
        rdc_V = np.load(fname+"rdc_V"+str(i)+".npy")        
        return rdc_E, rdc_V

    def saveBhMssh(self, fname):
        d = 2
        tol = self.bhmtol
        for nb in range(0, self.Nlatt+1):
            rdc_E, rdc_V = self.reduced_cov(nb)
            Db = min(self.D, d**(nb) )
            states = self.truncation(Db, rdc_E[:nb])
            print("state_b of length = {}:\n".format( len(states) ))
            ubbars, vbbars, dbs = [], [], []
            for ib, sb in enumerate( states ):
                ub, vb = m2uv( rdc_V, state2order(nb, sb) )
                self.ubbar, self.vbbar, self.db, cb, vpb = bloch_messiah(ub, vb, tol)
                ubbars.append( self.ubbar.copy() )
                vbbars.append( self.vbbar.copy() )
                dbs.append( self.db.copy() )
            with open(fname+"ubar"+str(nb), "wb") as fl:
                pickle.dump(ubbars, fl)
            with open(fname+"vbar"+str(nb), "wb") as fl:
                pickle.dump(vbbars, fl)
            with open(fname+"d"+str(nb), "wb") as fl:
                pickle.dump(dbs, fl)
    

    def loadBhMssh(self, fname, nb=0):
        with open(fname+"ubar"+str(nb), "rb") as fl:
            ubars = pickle.load(fl)
        with open(fname+"vbar"+str(nb), "rb") as fl:
            vbars = pickle.load(fl)
        with open(fname+"d"+str(nb), "rb") as fl:
            ds = pickle.load(fl)
        return ubars, vbars, ds

    def test(self, na, sb, sa):
        nb = na
        rdc_Ea, rdc_Va = self.reduced_cov(na)
        ua, va = m2uv(rdc_Va, state2order(na, sa) )        
        self.uabar, self.vabar, self.da, self.ca, self.vpa = bloch_messiah(ua, va)
        rdc_Eb, rdc_Vb = self.reduced_cov(nb)
        ub, vb = m2uv(rdc_Vb, state2order(nb, sb) )
        self.ubbar, self.vbbar, self.db, self.cb, self.vpb = bloch_messiah(ub, vb)
        print("ubar:\n",np.round(self.uabar,4))
        print("vbar:\n",np.round(self.vabar,4))
        print( self.pfffnovlp2() )
        print( self.pfffnovlp() )

    def truncation(self, D, rdc_E):
        # print(rdc_E)
        n_orbital = len( rdc_E )
        states = []
        coes = []
        if 2**n_orbital < D+1:
            n_state = n_orbital
            for i in range(2**n_state):
                temp = ('0'*n_state+bin(i)[2:])[-n_state:]
                state = [i for i in range(n_state) if temp[i]=='1']
                coe = np.prod([rdc_E[i] for i in state])
                coe *= np.prod([1-rdc_E[i] for i in range(n_state) if temp[i]=='0'] )
                # print(coe)
                if abs(coe) > 1e-8:
                    coes.append(coe)
                    states.append( state )
        else:
            n_state = int( np.log2(D) )
            fix_state = [i for i in range(n_state,n_orbital) if rdc_E[i]>0.5]
            fix_coe =  np.prod([rdc_E[i] for i in fix_state])
            for i in range(2**n_state):
                temp = ('0'*n_state+bin(i)[2:])[-n_state:]
                state = [i for i in range(n_state) if temp[i]=='1']
                coe = fix_coe * np.prod([rdc_E[i] for i in state])
                coe *= np.prod( [1-rdc_E[i] for i in range(n_state) if temp[i]=='0'] )
                if abs(coe) > 1e-12:
                    coes.append(coe)
                    states.append(state + fix_state )
        self.coes = coes
        self.staets = states
        return states

    def pfffnovlp(self):
        # nb = vb.shape[1]; na = va.shape[1]
        ubbar, vbbar, db, vpb = self.ubbar, self.vbbar, self.db, self.vpb
        uabar, vabar, da, vpa = self.uabar, self.vabar, self.da, self.vpa
        # ubbar, vbbar, db, cb, vpb = bloch_messiah(ub, vb)
        # uabar, vabar, da, ca, vpa = bloch_messiah(ua, va)
        vavb11, vavb12 = vbbar.T @ ubbar, vbbar.T @ db.conj().T @ da @ vabar
        vavb21, vavb22 = -vabar.T @ da.T @ db.conj() @ vbbar, uabar.T @ vabar
        rdc_v = np.block([[vavb11, vavb12], 
                          [vavb21, vavb22]])
        rdc_v = rdc_v * ( np.abs(rdc_v) > 1e-12 )
        # check = (rdc_v + rdc_v.T).max()
        # if check > 1e-14:
        #     print("check scew:", check)
        #     # print(ua.shape)
        #     print("vabar:\n", vabar)
        #     print("uabar:\n", uabar.diagonal())
        #     # print(ub.shape)
        #     print("vbbar:\n", vbbar)
        #     print("ubbar:\n", ubbar.diagonal())
        #     print(rdc_v)
        norm = 1/np.prod(vpa)/np.prod(vpb)
        lsb = vbbar.shape[0]; lsa = vabar.shape[0]
        lsx = lsb
        sign = ( 1 -  2 * ( (lsx*(lsx-1)//2) % 2 ) )
        if rdc_v.shape[0] == 0:
            pfffn = 1
        elif lsb %2 != lsa %2:
            pfffn = 0
        else:
            pfffn = pfaffian(rdc_v, "H")
        # print(norm, pfffn)
        return norm*pfffn*sign

    def pfffnovlp2(self):
        ubbar, vbbar, db = self.ubbar, self.vbbar, self.db
        uabar, vabar, da = self.uabar, self.vabar, self.da
        lsb = vbbar.shape[0]; lsa = vabar.shape[0]
        lsx = lsb
        sign = ( 1 -  2 * ( (lsx*(lsx-1)//2) % 2 ) )
        if lsb+lsa == 0:
            pfffn = 1
        elif lsb %2 != lsa %2:
            pfffn = 0
        else:
            # roundprint( uabar )
            # print('vabar:')
            # roundprint( vabar )
            lmdbb = ubar2lambda(ubbar)
            lmdba = ubar2lambda(uabar)
            vavb11, vavb12 =  -ubar2skew(ubbar), lmdbb @ db.conj().T @ da @ lmdba
            vavb21, vavb22 =  -lmdba @ da.T @ db.conj() @ lmdbb , ubar2skew(uabar)
            rdc_v = np.block([[vavb11, vavb12], 
                              [vavb21, vavb22]])
            # print("rdc_v")
            # roundprint(rdc_v,3)            
            # print(rdc_v.shape)
            # print("ubar:")
            # roundprint(ubbar)
            # print("da:")
            # roundprint(da)
            pfffn = pfaffian(rdc_v, "H")
        return pfffn*sign

    def A_nba(self, na):
        print( "******************BdG--->MPS at site {}******************".format(na) )
        assert na > 0 and na<self.Nlatt+1
        if self.ifdynamic:
            return self.A_nba_dynamic(na)
        else:
            return self.A_nba_static(na)

    def A_nba_dynamic(self, na):
        d = 2
        tol = self.bhmtol
        dtype = self.dtype
        nb = na - 1
        if self.ifparallel:
            print("parallel")
            self.rdc_Eb, rdc_Vb = self.loadrdc(self.mfpath, nb)
            self.rdc_Ea, rdc_Va = self.loadrdc(self.mfpath, na)
        else:
            self.rdc_Eb, rdc_Vb = self.reduced_cov(nb)
            self.rdc_Ea, rdc_Va = self.reduced_cov(na)        
        # rdc_Va = rdc_Va.T.conj()
        Db = min(self.D, d**(nb) )
        Da = min(self.D, d**(na) )
        states_b = self.truncation(Db, self.rdc_Eb[:nb])
        print("state_b of length = {}:\n".format( len(states_b) ), states_b)
        states_a = self.truncation(Da, self.rdc_Ea[:na])
        if na == self.Nlatt:
            states_a = [[i for i in range(self.Nlatt)]]
        print("state_a of length = {}:\n".format( len(states_a) ), states_a)
        Db = min( Db, len(states_b) )
        Da = min( Da, len(states_a) )
        A = np.zeros([d, Db, Da], dtype=self.dtype)
        # print(A.shape, "\n", len(states_b), "\n", len(states_a))
        id = 0
        for ib, sb in enumerate( states_b ):
            ub, vb = m2uv( rdc_Vb, state2order(nb, sb) )
            # print(rdc_Vb)
            self.ubbar, self.vbbar, self.db, self.cb, self.vpb = bloch_messiah(ub, vb, tol)
            self.db =  np.vstack( (self.db, np.zeros(self.db.shape[1],dtype) ) )
            for ia, sa in  enumerate( states_a ):
                ua, va = m2uv(rdc_Va, state2order(na, sa) )
                self.uabar, self.vabar, self.da, self.ca, self.vpa = bloch_messiah(ua, va, tol)
                A[id, ib, ia] = self.pfffnovlp2() #* sign
        id = 1
        for ib, sb in enumerate( states_b ):
            ub, vb = m2uv(rdc_Vb, state2order(nb, sb) )
            # ubx = np.block([[np.zeros((1,1)),     np.zeros((1,nb))],
            #                [np.zeros((nb,1)), ub]])
            # vbx = np.block([[np.ones((1,1)),     np.zeros((1,nb))],
            #                [np.zeros((nb,1)), vb]])
            # ubx = np.block([[ub,     np.zeros((nb,1))],
            #                [np.zeros((1,nb)), np.zeros((1,1))]])
            # vbx = np.block([[vb,     np.zeros((nb,1))],
            #                 [np.zeros((1,nb)), np.ones((1,1))]])
            # self.ubbar, self.vbbar, self.db, self.cb, self.vpb = bloch_messiah(ubx, vbx)

            # self.ub = ub
            # self.vb = vb
            self.ubbar, self.vbbar, self.db, self.cb, self.vpb = bloch_messiah(ub, vb)
            dim = self.ubbar.shape[0]
            self.ubbar = np.block([[self.ubbar,        np.zeros((dim,1),dtype)],
                                    [np.zeros((1,dim),dtype), np.zeros((1,1),dtype)]])
            self.vbbar = np.block([[self.vbbar,        np.zeros((dim,1),dtype)],
                                    [np.zeros((1,dim),dtype), np.ones((1,1),dtype)]])
            self.db = np.block([[self.db,           np.zeros((self.db.shape[0],1),dtype)],
                                [np.zeros((1,self.db.shape[1]),dtype), np.ones((1,1),dtype)]])
            self.cb = np.block([[self.cb,           np.zeros((self.cb.shape[0],1),dtype)],
                                [np.zeros((1,self.cb.shape[1]),dtype), np.ones((1,1),dtype)]])
            for ia, sa in  enumerate( states_a ):
                ua, va = m2uv(rdc_Va, state2order(na, sa) )
                self.uabar, self.vabar, self.da, self.ca, self.vpa = bloch_messiah(ua, va)
                # sign = 1 - 2 * ( dim %2 )
                A[id, ib, ia] = self.pfffnovlp2() #* sign
        if na == self.Nlatt:
            A = A.reshape(2,-1)
        elif na == 1:
            A = A.reshape(2,-1)
        return A

    def A_nba_static(self, na):
        d = 2
        nb = na - 1
        dtype = self.dtype
        ubbars, vbbars, dbs = self.loadBhMssh(self.mfpath, nb)
        uabars, vabars, das = self.loadBhMssh(self.mfpath, na)
        Db = len(ubbars)
        Da = len(uabars)
        A = np.zeros([d, Db, Da], dtype=self.dtype)
        # print(A.shape, "\n", len(states_b), "\n", len(states_a))
        id = 0
        for ib in range(Db):
            self.ubbar, self.vbbar, self.db = ubbars[ib], vbbars[ib], dbs[ib]
            self.db =  np.vstack( (self.db, np.zeros(self.db.shape[1],dtype) ) )
            for ia in range(Da):
                self.uabar, self.vabar, self.da = uabars[ia], vabars[ia], das[ia]
                A[id, ib, ia] = self.pfffnovlp2()
        id = 1
        for ib in range(Db):
            self.ubbar, self.vbbar, self.db = ubbars[ib], vbbars[ib], dbs[ib]
            dim = self.ubbar.shape[0]
            self.ubbar = np.block([[self.ubbar,        np.zeros((dim,1),dtype)],
                                    [np.zeros((1,dim),dtype), np.zeros((1,1),dtype)]])
            self.vbbar = np.block([[self.vbbar,        np.zeros((dim,1),dtype)],
                                    [np.zeros((1,dim),dtype), np.ones((1,1),dtype)]])
            self.db = np.block([[self.db,           np.zeros((self.db.shape[0],1),dtype)],
                                [np.zeros((1,self.db.shape[1]),dtype), np.ones((1,1),dtype)]])
            for ia in range(Da):
                self.uabar, self.vabar, self.da = uabars[ia], vabars[ia], das[ia]
                A[id, ib, ia] = self.pfffnovlp2()
        if na == self.Nlatt:
            A = A.reshape(2,-1)
        elif na == 1:
            A = A.reshape(2,-1)
        return A

    def get_MPS(self):
        mps = []
        for i in range(1,self.Nlatt+1):
            mps.append( self.A_nba(i) )
        return mps

class MajoranaHubbardTest(bdg):
    def __init__(self, Nx, Ny, D, pbcx=0, pbcy=0):
        self.t = 1
        pbcx = int( pbcx )
        self.model = "MajoranaHubbard_Nx{}_Ny{}_pbcx{}_pbcy{}_D{}".format(Nx, Ny, pbcx, pbcy, D)
        super().__init__(Nx=Nx, Ny=Ny, model=self.model, D=D)
        self.path = "./gaussiandata/"+self.model
        self.dtype = np.complex128
        # self.dtype = np.float64
        self.pbcy = pbcy
        self.pbcx = pbcx
        
    def hamiltonian(self):
        N = self.Nlatt
        Nx, Ny = self.Nx, self.Ny
        self.tmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        self.dmat = np.zeros((self.Nlatt, self.Nlatt), self.dtype)
        t = self.t
        for i in range(N):
            self.tmat[i, i] = t/2
        for nx in range(Nx):
            for ny in range(Ny):
                i0 = ny + nx * Ny
                iy = (ny+1) % Ny + nx * Ny
                ix = ny + ( (nx+1) % Nx ) * Ny
                if ny < self.Ny - 1:
                    self.tmat[i0, iy] += - t/2  
                    self.dmat[i0, iy] += - t/2
                elif abs(self.pbcy) > 0:
                    self.tmat[i0, iy] += - t/2 * self.pbcy 
                    self.dmat[i0, iy] += - t/2 * self.pbcy
                if nx < self.Nx - 1:
                    self.dmat[i0, ix] += -t*1j    
                elif abs(self.pbcx) > 0:
                    self.dmat[i0, ix] += -t*1j * self.pbcx
        self.tmat += self.tmat.T.conj()
        self.dmat -= self.dmat.T
        self.ham = np.block([[self.tmat, self.dmat],[-self.dmat.conj(), -self.tmat.conj()]])
        # self.eig_eng, self.eig_vec = np.linalg.eigh(self.ham)
        self.eig_eng, self.eig_vec = bdgeig_zeromode(self.ham, ref=0.0)
        self.eig_eng = self.eig_eng.real
        # print(self.eig_eng)
        self.u, self.v = m2uv(self.eig_vec)
        self.m = uv2m(self.u, self.v)
        print("check the hamiltonian diagonalization: HM=ME, M=[[u,v*];[v,u*]]")
        check_orthonormal(self.ham, self.m)
        self.covariance()

if __name__ == "__main__":
    from gaussianstate import MPS as MPS2
    m = MajoranaHubbardTest(4, 4, 16, 0, 1)
    m.hamiltonian()
    tensor2 = m.A_nba(1)
