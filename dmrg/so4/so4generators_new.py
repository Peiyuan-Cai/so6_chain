import numpy as np
import numpy.linalg as LA
from copy import deepcopy

sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])
id = np.eye(2)

Sx = 0.5 * sigmax
Sy = 0.5 * sigmay
Sz = 0.5 * sigmaz

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

def get_ket_from_4states(n):
    if n == 1:
        return ['1']
    elif n == 2:
        return ['2']
    elif n == 3:
        return ['3']
    elif n == 4:
        return ['4']
    else:
        raise("Out of 4 states. ")
    
def S_representation_matrix(alpha, beta):
    if type(alpha) != int or type(beta) != int or alpha>4 or alpha<1 or beta>4 or beta<1:
        raise("Check your alpha and beta. They must be 1,2,3,4. ")
    S_mat = np.zeros((4,4))
    for left in range(1,5):
        bra = get_ket_from_4states(left)
        oprs = [str(alpha)+'d', str(beta)]
        oprs.insert(0, bra[0])
        for right in range(1,5):
            ket = get_ket_from_4states(right)
            coe, ket = oprs_on_ket(oprs, ket)
            if ket == []:
                S_mat[left-1, right-1] = coe
            elif ket == 0:
                S_mat[left-1, right-1] = 0
            else:
                raise('something wrong')
    if alpha==beta:
        S_mat -= (1/2)*np.diag([1,1,1,1])
    return S_mat

for a in range(1,5):
    for b in range(1,5):
        print("S",a,b,"matrix is")
        print(S_representation_matrix(a,b))