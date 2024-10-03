import numpy as np
import numpy.linalg as LA
import scipy.linalg as spLA

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

print("L1", L1)
print("L2", L2)
print("L3", L3)
print("L4", L4)
print("L5", L5)
print("L6", L6)

efac = np.exp(1j*np.pi/4)
U = (1/np.sqrt(2))*np.array([[efac, 0, 0, -efac],
              [np.conjugate(efac), 0, 0, np.conjugate(efac)],
              [0, -np.conjugate(efac), np.conjugate(efac), 0],
              [0, efac, efac, 0]])

print("check U is unitary", np.allclose(np.conjugate(U).T, LA.inv(U)))

Loprs = [L1, L2, L3, L4, L5, L6]
c=0
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

print(len(Loprs))

print(len(coe_list))
print(coe_list[0])

Loprs_new = []
for m in range(len(Loprs)):
    Loprs_new.append(U @ Loprs[m] @ np.conjugate(U).T)
    print("the",m,"th new L operator")
    print(Loprs_new[m])

eigvals, eigvecs = spLA.eigh(Loprs_new[0], Loprs_new[5])
print(eigvecs)

for m in range(len(Loprs)):
    Loprs_new.append(U @ Loprs[m] @ np.conjugate(U).T)
    print("the",m,"th newnew L operator")
    print(np.conjugate(eigvecs).T @ Loprs_new[m] @ eigvecs)