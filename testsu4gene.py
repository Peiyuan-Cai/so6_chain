import numpy as np
from copy import deepcopy
import numpy.linalg as LA

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
    """
    Input n is the a index, the 6 states of half filling. 
    """
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
    
def get_ket_from_6states_from_order(n, arrange_order):
    """
    Inputs:
        1. n, int, the 2-occupied states, 1 to 6 for a to f
        2. arrange_order, list of str, the arrange order of '1234'
        
    Output:
        1. ket, str
    """
    
    if n == 1:
        return [arrange_order[0], arrange_order[1]]
    elif n == 2:
        return [arrange_order[0], arrange_order[2]]
    elif n == 3:
        return [arrange_order[0], arrange_order[3]]
    elif n == 4:
        return [arrange_order[1], arrange_order[2]]
    elif n == 5:
        return [arrange_order[1], arrange_order[3]]
    elif n == 6:
        return [arrange_order[2], arrange_order[3]]
    else:
        raise("Out of 6 states. ")
    
def S_representation_matrix(alpha, beta):
    """
    There are totally 16 generators(actually 15) for SU(4), given by S_{\alpha\beta} = c_\alpha^\dagger c_\beta - \frac{1}{2} \delta_{\alpha\beta}
    
    Inputs:
        1. alpha, int, from 1 to 4, the real fermion index
        2. beta, int, from 1 to 4, the real fermion index
        
    Output:
        1. S_mat, ndarray, the 6-dimensional representation of SU(4) generator of given alpha and beta
    """
    if type(alpha) != int or type(beta) != int or alpha>4 or alpha<1 or beta>4 or beta<1:
        raise("Check your alpha and beta. They must be 1,2,3,4. ")
    
    S_mat = np.zeros((6,6))
    
    for left in range(1,7): #left and the right below are 6-state index from a to f
        bra = get_ket_from_6states(left)
        oprs = [str(alpha)+'d', str(beta)]
        oprs.insert(0, bra[0])
        oprs.insert(0, bra[1]) #put the annihilation operators in the front of oprs
        for right in range(1,7):
            ket_ori = get_ket_from_6states(right)
            ket = deepcopy(ket_ori)
            coe, ket = oprs_on_ket(oprs, ket)
            print('left',left,'right',right,'oprs',oprs,'ket',ket_ori,'ket after',ket,'coe',coe)
            if ket == []:
                S_mat[left-1, right-1] = coe
            elif ket == 0:
                S_mat[left-1, right-1] = 0
            else:
                raise('something wrong')
        print(" ")
    if alpha==beta:
        S_mat -= (1/2)*np.diag([1,1,1,1,1,1])
    return S_mat

def S_representation_matrix_ordering(alpha, beta, arrange_order):
    if type(alpha) != int or type(beta) != int or alpha>4 or alpha<1 or beta>4 or beta<1:
        raise("Check your alpha and beta. They must be 1,2,3,4. ")
    S_mat = np.zeros((6,6))
    for left in range(1,7):
        bra = get_ket_from_6states_from_order(left)
        oprs = [str(alpha)+'d', str(beta)]
        oprs.insert(0, bra[0])
        oprs.insert(0, bra[1])
        for right in range(1,7):
            ket = get_ket_from_6states_from_order(right)
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

if __name__ == "__main__":
    
    print('S11', S_representation_matrix(1,1))
    
    """
    list_of_generators = []
    norm_of_generators = []

    for alpha in range(1,5):
        for beta in range(1,5):
            list_of_generators.append(S_representation_matrix(alpha,beta))
            norm_of_generators.append(LA.norm(list_of_generators[-1]))
            
    list_of_generators.pop()
    norm_of_generators.pop()

    print('norm of oprs', norm_of_generators)
    """
    
    carton = []
    for a in range(1,4):
        carton.append(S_representation_matrix(a,a))
    so6g = dict()
    for a in range(1,5):
        for b in range(1,5):
            so6g[(a,b)] = S_representation_matrix(a,b)

    ham1 = np.zeros( (36,36) )
    for a in range(1,5):
        for b in range(a,5):
            if a!=b:
                ham1 += np.kron( so6g[(a,b)], so6g[(b,a)] )
                ham1 += np.kron( so6g[(b,a)], so6g[(a,b)] )
            elif a == b:
                ham1 += np.kron( so6g[(a,a)], so6g[(a,a)] )
    ham1 = ham1.reshape((6,6,6,6))

    so6g[(1,1)] = np.diag([-1,-1,-1, 1, 1, 1])/np.sqrt(3)
    so6g[(2,2)] = np.diag([-2, 1, 1,-1,-1, 2])/np.sqrt(6)
    so6g[(3,3)] = np.diag([ 0,-1, 1,-1, 1, 0])/np.sqrt(2)
    ham2 = np.zeros( (36,36) )
    for a in range(1,5):
        for b in range(a,5):
            if a!=b :
                ham2 += np.kron( so6g[(a,b)], so6g[(b,a)] )
                ham2 += np.kron( so6g[(b,a)], so6g[(a,b)] )
            elif a == b and a < 4:
                ham2 += np.kron( so6g[(a,a)], so6g[(a,a)] )
    ham2 = ham2.reshape((6,6,6,6))

    print(np.allclose(ham1,ham2))

    for a in range(6):
        for b in range(6):
            for c in range(6):
                for d in range(6):
                    if abs(ham1[a,b,c,d]) > 1e-8:
                        print(a,b,c,d, ham1[a, b, c, d] )