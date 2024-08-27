import numpy as np
import scipy as sp
from scipy import sparse

def hc(m):
    """
    Hermitian conjugate of a matrix
    """
    return m.conj().T

def vu2m(v, u):
    """
    Block matrices U and V together to get the Bogoliubov matrix M
    """
    return np.block( [[v, u.conj()], [u, v.conj()]] )

def m2vu(m, order = None):
    """
    Take matrices V and U out of the Bogoliubov matrix M. 

    No ordering of V and U is applied by default

    return: V, U
    """
    na = m.shape[0]//2
    if order is None:
        order = [i for i in range(na)]
    return m[:na, order].copy(), m[na:, order].copy()

def majoranabasis(n):
    #return  1/2(M) \otimes I_N
    i = np.eye(n) / np.sqrt(2)
    return np.block([[i, -1j*i], [i, 1j*i]])

def majoranabasis2(n):
    #return  I_N \otimes 1/2(M)
    i = np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)
    return np.kron(np.eye(n), i)

def check_orthonormal(ham, m):
    """
    Check if matirx m is in orthonormal form

    Inputs: 
        1. ham::ndarray, the Hamiltonian matrix
        2. m::ndarray, the matrix to diagonalize the ham
    """
    check = np.allclose(hc(m) @ m, np.eye(m.shape[0]))

    if check == False:
        print("check orthonormal: ", check)
        print(sparse.csr_matrix(np.round(hc(m) @ m - np.eye(m.shape[0]), 8)))
        raise "Check orthonomal false. "
    
    test = hc(m) @ ham @ m

    check = np.allclose(test, np.diag(test.diagonal()))

    if check == False:
        print("check eigen decomposition: ",  check)
        print(np.round( test.diagonal(), 4))
        raise "check eigen decomposition false. "

def findmajorana(ham, ref=0.5):
    """
    Find Majorana representation of given Hamiltonian

    """
    na = ham.shape[0]//2
    hamp = ham - np.eye(na*2)*ref

    if hamp.dtype == np.complex128 or hamp.dtype == np.complex64:
        w = majoranabasis(na)
        
        xhamp = w.conj().T @ hamp @ w
        theta = ( xhamp.conj().T @ xhamp ).real
        eD, vD = sp.linalg.eigh(theta)
        w2 = majoranabasis2(na)
        vvD = w @ vD @ w2.conj().T
        s1 = vvD[:,:1].T
        s2 = vvD[:,1:2].T

    else:
        tmat, dmat = hamp[:na, :na], hamp[:na,na:]

        _, e, u = sp.linalg.svd(tmat+dmat)
        num_0mode = max(1, np.sum( np.abs(e) < 1e-12 ) )
        p = np.argsort(np.abs(e))
        e = e[p]
        u = u[p[:num_0mode],:]
    
        _,e,v = sp.linalg.svd(tmat-dmat)
        p = np.argsort(np.abs(e))
        e = e[p]
        v = v[p[:num_0mode],:]
        phaseuv = u @ v.conj().T#.conj().T
        ps, pv, pd = np.linalg.svd(phaseuv)
        num_1mode = max(1, np.sum( np.abs(pv) > 1-1e-12 ))
        u = ( ps.conj().T @ u )[:num_1mode, :]
        v = ( pd @ v )[:num_1mode,:]
        s1 = np.block([(u), (u-u)])
        s2 = np.block([(u-u).conj(), (u).conj()])
    return s1, s2

def bdgeig_zeromode(ham, tol=1e-14, ref=0.5):
    """
    Eigen solver containing zeromode detector. 
    
    output:
        1. eig_eng::ndarray, the reordered eigenvalues, [+1, +2, +3, +4, -1, -2, -3, -4]
        2. eig_vec::ndarray, the corresponding eigenvectors
    """
    na = ham.shape[0]//2
    eig_eng, eig_vec = np.linalg.eigh(ham - np.diag([ref]*na*2)) #we often have ref=0
    
    orderu = np.array([na-i-1 for i in range(na)]) #orderu = array([na-1, na-2, ..., 0])
    orderd = orderu.tolist()
    orderd.reverse() #list with reversed order of order u i.e. [0, 1, 2, ... , na-1]
    eig_eng = np.block( [eig_eng[:na][orderu], -eig_eng[:na][orderu]] ) + ref #[-1, -2, -3, -4, +1, +2, +3, +4]
    eig_vec = np.block( [eig_vec[:,:na][:,orderu], eig_vec[:,na:][:,orderd]] )
    eig_eng = (eig_vec.conj().T @ ham @ eig_vec).diagonal()
    eig_eng = eig_eng.real
    if abs(eig_eng[0] - ref ) < tol:
        print("zero mode!!!")
        psi1, psi2 = findmajorana(ham, ref)
        num_0zero = psi1.shape[0]
        eig_vec[:,:num_0zero], eig_vec[:,na:na+num_0zero] = psi1.T, psi2.T
    v, u = m2vu(eig_vec)
    eig_vec = vu2m(v, u)
    return eig_eng, eig_vec

def bdgeig(ham, tol=1e-14):
    """
    BdG Hamiltonian Eigen solver
    """
    na = ham.shape[0]//2
    eig_eng, eig_vec = np.linalg.eigh(ham)
    orderu = np.array([na-i-1 for i in range(na)]) #orderu = array([na-1, na-2, ..., 0])
    orderd = orderu.tolist()
    orderd.reverse() #list with reversed order of order u i.e. [0, 1, 2, ... , na-1]
    eig_eng = np.block( [eig_eng[:na][orderu], -eig_eng[:na][orderu]] ) #[-1, -2, -3, -4, +1, +2, +3, +4]
    eig_vec = np.block( [eig_vec[:,:na][:,orderu], eig_vec[:,na:][:,orderd]] )
    eig_eng = (eig_vec.conj().T @ ham @ eig_vec).diagonal()
    return eig_eng, eig_vec

