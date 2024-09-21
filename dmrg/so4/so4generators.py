import numpy as np

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