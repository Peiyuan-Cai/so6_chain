"""
The color map of the SO(4) model near the R point

Puiyuen
    2024.12.22: Created, run it inside the nearRpoint folder
"""
import matplotlib.pyplot as plt
from bdgpack import *
from so4bbqham import *
import argparse
import pickle
import os

if __name__ == '__main__':
    #parsers
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lx", type=int, default=8)
    parser.add_argument("-chi", type=float, default=1.)
    parser.add_argument("-delta", type=float, default=1.)
    parser.add_argument("-Dmpos", type=int, default=256)
    parser.add_argument("-J", type=float, default=1.)
    parser.add_argument("-K", type=float, default=1.)
    args = parser.parse_args()

    chi = args.chi
    delta = args.delta
    lx = args.lx
    Dmpos = args.Dmpos

    homepath = os.getcwd()
    if os.path.isdir(homepath + '/data/') == False:
        os.mkdir(homepath + '/data/')

    lamb_list = np.arange(1.50, 2.00, 0.02)
    lamb_list = np.round(lamb_list, 2)
    K_list = np.arange(0.0, 0.02, 0.001)
    K_list = np.round(K_list, 3)

    #calculate expectations of HJ and HK
    HJ_value_list_apbc = []
    HK_value_list_apbc = []
    HJ_value_list_pbc = []
    HK_value_list_pbc = []

    for lamb in lamb_list:
        path = homepath + '/data/' + 'so4psimpos_lx{}_delta{}_lambda{}/'.format(lx, delta, lamb)
        psi_apbc = pickle.load(open(path + 'so4psimpos_lx{}_delta{}_lambda{}_Dmpos{}_APBC'.format(lx, delta, lamb, Dmpos), 'rb'))
        psi_pbc = pickle.load(open(path + 'so4psimpos_lx{}_delta{}_lambda{}_Dmpos{}_PBC'.format(lx, delta, lamb, Dmpos), 'rb'))
        H_J = HJ(model_params=dict(Lx = lx, cons_S='U1'))
        H_K = HK(model_params=dict(Lx = lx, cons_S='U1'))
        HJ_value_list_apbc.append(H_J.H_MPO.expectation_value(psi_apbc))
        HK_value_list_apbc.append(H_K.H_MPO.expectation_value(psi_apbc))
        HJ_value_list_pbc.append(H_J.H_MPO.expectation_value(psi_pbc))
        HK_value_list_pbc.append(H_K.H_MPO.expectation_value(psi_pbc))
    
    #draw colormap of K and lambda
    energy_colormap_apbc = np.zeros((len(lamb_list), len(K_list)))
    energy_colormap_pbc = np.zeros((len(lamb_list), len(K_list)))
    for idk, K in enumerate(K_list):
        for idl, lamb in enumerate(lamb_list):
            energy_colormap_apbc[idl, idk] = HJ_value_list_apbc[idl] + K * HK_value_list_apbc[idl]
            energy_colormap_pbc[idl, idk] = HJ_value_list_pbc[idl] + K * HK_value_list_pbc[idl]
            
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(energy_colormap_apbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]])
    ax[0].set_title('APBC')
    ax[0].set_xlabel('K')
    ax[0].set_ylabel('lambda')
    ax[1].imshow(energy_colormap_pbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]])
    ax[1].set_title('PBC')
    ax[1].set_xlabel('K')
    ax[1].set_ylabel('lambda')
    plt.savefig(homepath + 'so4psimpos_lx{}_Dmpos{}_energy_colormap.png'.format(lx, Dmpos))