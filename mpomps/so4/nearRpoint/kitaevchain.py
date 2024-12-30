import numpy as np
import matplotlib.pyplot as plt

def Kitaev1d(L, bc, **kwargs):
    t = kwargs.get('t', 1.0)
    d = kwargs.get('d', 1.0)
    h = kwargs.get('h', 0.0)
    T = np.zeros((L,L))
    D = np.zeros((L,L))
    for _ in range(L-1):
        T[_, _+1] = -t
        T[_, _]   =  h/2
        D[_, _+1] = +d
    T[L-1, 0] = -t *bc
    D[L-1, 0] = +d *bc
    T[L-1, L-1] = h/2
    T = T + T.conj().T
    D = D - D.T
    ham = np.block(([[T,D],[D.conj().T, -T.conj()]]))
    e, v = np.linalg.eigh(ham)
    return e[:L].sum()

hs = np.linspace(1.,2,5)
Ls = np.arange(4,20)*4
for h in hs:
    gaps = []
    for L in Ls:
        ep = Kitaev1d(L, 1, h=h)
        em = Kitaev1d(L, -1, h=h)
        gap = np.abs(ep-em)
        gaps.append(gap)
    plt.plot(Ls, gaps, 's', label='h={}'.format(np.round(h,4)))
plt.plot(Ls, 0.1*np.exp(-0.138*Ls))
plt.yscale('log')
plt.legend()
plt.show()