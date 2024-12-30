"""
The color map of the SO(4) model near the R point

Puiyuen
    2024.12.22: Created, run it inside the nearRpoint folder
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    parser.add_argument("-init", type=int, default=0) #1 for given energy
    args = parser.parse_args()

    chi = args.chi
    delta = args.delta
    lx = args.lx
    Dmpos = args.Dmpos

    homepath = os.getcwd()
    if os.path.isdir(homepath + '/data/') == False:
        os.mkdir(homepath + '/data/')

    lamb_list = np.arange(1.50, 2.00, 0.01)
    lamb_list = np.round(lamb_list, 2)
    K_list = np.arange(0.0, 0.2, 0.004)
    K_list = np.round(K_list, 3)

    if args.init == 0:
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

        #save colormap by pickle
        pickle.dump(energy_colormap_apbc, open(homepath + '/so4psimpos_lx{}_Dmpos{}_energy_colormap_apbc'.format(lx, Dmpos), 'wb'))
        pickle.dump(energy_colormap_pbc, open(homepath + '/so4psimpos_lx{}_Dmpos{}_energy_colormap_pbc'.format(lx, Dmpos), 'wb'))
    elif args.init == 1:
        energy_colormap_apbc = pickle.load(open(homepath + '/so4psimpos_lx{}_Dmpos{}_energy_colormap_apbc'.format(lx, Dmpos), 'rb'))
        energy_colormap_pbc = pickle.load(open(homepath + '/so4psimpos_lx{}_Dmpos{}_energy_colormap_pbc'.format(lx, Dmpos), 'rb'))
            
    # fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    # ax[0].imshow(energy_colormap_apbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]])
    # ax[0].set_title('APBC')
    # ax[0].set_xlabel('K')
    # ax[0].set_ylabel('lambda')
    # ax[1].imshow(energy_colormap_pbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]])
    # ax[1].set_title('PBC')
    # ax[1].set_xlabel('K')
    # ax[1].set_ylabel('lambda')
    # plt.colorbar(ax[0].imshow(energy_colormap_apbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]]), ax=ax[0])
    # plt.colorbar(ax[1].imshow(energy_colormap_pbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]]), ax=ax[1])
    # plt.savefig(homepath + '/so4psimpos_lx{}_Dmpos{}_energy_colormap.pdf'.format(lx, Dmpos))
    # plt.show()
    
    
    # fig = plt.figure(figsize=(25, 10))
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')

    # K_mesh, lamb_mesh = np.meshgrid(K_list, lamb_list)
    # surf1 = ax1.plot_surface(K_mesh, lamb_mesh, energy_colormap_apbc, cmap='viridis')
    # ax1.set_title('APBC')
    # ax1.set_xlabel('K')
    # ax1.set_ylabel('lambda')
    # ax1.set_zlabel('Energy')
    # fig.colorbar(surf1, ax=ax1)
    
    # K_mesh, lamb_mesh = np.meshgrid(K_list, lamb_list)
    # surf2 = ax2.plot_surface(K_mesh, lamb_mesh, energy_colormap_pbc, cmap='viridis')
    # ax2.set_title('PBC')
    # ax2.set_xlabel('K')
    # ax2.set_ylabel('lambda')
    # ax2.set_zlabel('Energy')
    # fig.colorbar(surf2, ax=ax2)
    
    # plt.savefig(homepath + '/so4psimpos_lx{}_Dmpos{}_energy_colormap.pdf'.format(lx, Dmpos))
    # plt.show()
    
    
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))

    ax[0].imshow(energy_colormap_apbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]])
    ax[0].set_title('APBC')
    ax[0].set_xlabel('K')
    ax[0].set_ylabel('lambda')

    num_rows, num_cols = energy_colormap_apbc.shape
    min_energy_values_apbc = []
    min_lambda_values_apbc = []
    for col in range(num_cols):
        col_energy = energy_colormap_apbc[:, col]
        min_index = np.argmin(col_energy)
        min_energy = col_energy[min_index]
        min_lambda = lamb_list[min_index]
        min_energy_values_apbc.append(min_energy)
        min_lambda_values_apbc.append(min_lambda)
        
    print("APBC 最小值点数据:")
    for k, min_energy, min_lambda in zip(K_list, min_energy_values_apbc, min_lambda_values_apbc):
        print(f"K: {k}, 最小值: {min_energy}, lambda: {min_lambda}")

    for col in range(num_cols):
        x = (K_list[col] + K_list[col + 1]) / 2 if col < num_cols - 1 else K_list[-1]
        y = min_lambda_values_apbc[col]
        ax[0].scatter(x, y, c='r', marker='o')

    ax[1].imshow(energy_colormap_pbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]])
    ax[1].set_title('PBC')
    ax[1].set_xlabel('K')
    ax[1].set_ylabel('lambda')

    min_energy_values_pbc = []
    min_lambda_values_pbc = []
    for col in range(num_cols):
        col_energy = energy_colormap_pbc[:, col]
        min_index = np.argmin(col_energy)
        min_energy = col_energy[min_index]
        min_lambda = lamb_list[min_index]
        min_energy_values_pbc.append(min_energy)
        min_lambda_values_pbc.append(min_lambda)
        
    # 打印 PBC 最小值点的数据
    print("PBC 最小值点数据:")
    for k, min_energy, min_lambda in zip(K_list, min_energy_values_pbc, min_lambda_values_pbc):
        print(f"K: {k}, 最小值: {min_energy}, lambda: {min_lambda}")

    for col in range(num_cols):
        x = (K_list[col] + K_list[col + 1]) / 2 if col < num_cols - 1 else K_list[-1]
        y = min_lambda_values_pbc[col]
        ax[1].scatter(x, y, c='r', marker='o')

    plt.colorbar(ax[0].imshow(energy_colormap_apbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]]), ax=ax[0])
    plt.colorbar(ax[1].imshow(energy_colormap_pbc, aspect='auto', cmap='viridis', extent=[K_list[0], K_list[-1], lamb_list[-1], lamb_list[0]]), ax=ax[1])

    plt.savefig(homepath + '/so4psimpos_lx{}_Dmpos{}_energy_colormap.pdf'.format(lx, Dmpos))
    plt.show()