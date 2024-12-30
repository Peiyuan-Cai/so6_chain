"""
Energy plot version 2

Puiyuen 241206
    fit the energy difference with exponential function
"""
import numpy as np
import numpy.linalg as LA
from copy import deepcopy
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import SpinSite, FermionSite, Site
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO, MPOEnvironment
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge, ChargeInfo
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig
import pickle
import matplotlib.pyplot as plt
from so4bbqham import *
import scipy.linalg as spLA
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

lxlist = [20,24,28,32,36,40] #for the critical side, we don't fit N=40
Klist = [0.251, 0.252, 0.253, 0.254, 0.255, 0.256, 0.257, 0.258, 0.259, 0.26]
Klist_2 = [0.251, 0.252, 0.253, 0.254, 0.255] #the critical side, identical number 2
Klist_1 = [0.256, 0.257, 0.258, 0.259, 0.26] #the deeper critical side, identical number 1
colors = ['r', 'g', 'b', 'm', 'c']  
engdifflist_2 = [] #critical side
D = 4000

E_error_2_dmrg1 = [[-1.3812e-09, -7.1054e-15, -4.6185e-14, 3.5527e-14, 3.5527e-15, 4.9738e-14],
                   [1.7764e-15, 1.0658e-14, 7.1054e-15, -4.6185e-14, -1.3500e-13, 3.1974e-14],
                   [1.9540e-14, 1.0658e-14, -2.4869e-14, -3.5527e-14, -4.9738e-14, 1.1724e-13],
                   [2.1316e-14, -9.2371e-14, -1.0658e-14, -1.1013e-13, 2.8422e-14, -1.3500e-13],
                   [1.2434e-14, -2.1316e-14, 1.4211e-14, 1.0658e-14, -7.1054e-15, 3.9080e-14]]
trunc_error_2_dmrg1 = [[6.1304e-16, 4.7606e-18, 3.6555e-18, 3.5564e-18, 4.3113e-18, 3.8211e-18],
                       [4.0087e-16, 5.0156e-17, 4.9425e-17, 4.0894e-17, 4.8099e-17, 3.7933e-17],
                       [1.6266e-16, 2.1516e-16, 2.2161e-16, 2.1825e-16, 1.7582e-16, 1.5643e-16],
                       [4.6545e-16, 5.8807e-16, 6.6204e-16, 6.6663e-16, 5.0948e-16, 5.2737e-16],
                       [1.2229e-15, 1.3132e-15, 1.2946e-15, 1.5370e-15, 1.0508e-15, 1.2012e-15]]
e_error_2_dmrg1 = [[3.3751e-14, 1.4211e-14, 1.0658e-14, 6.3949e-14, 2.8422e-14, 5.6843e-14],
                   [2.6645e-14, 1.0658e-14, 6.0396e-14, 4.9738e-14, 3.5527e-14, 6.7502e-14],
                   [1.7764e-14, 3.9080e-14, 3.1974e-14, 7.4607e-14, 7.4607e-14, 6.7502e-14],
                   [3.1974e-14, 2.4869e-14, 1.0658e-14, 2.8422e-14, 3.5527e-14, 3.1974e-14],
                   [2.3093e-14, 3.9080e-14, 6.3949e-14, 7.8160e-14, 3.9080e-14, 1.2434e-13]]

E_error_2_dmrg2 = [[-1.7764e-15, 3.1974e-14, -7.1054e-15, 1.4211e-14, 2.8422e-14, 7.1054e-14],
                   [-7.1054e-15, -4.6185e-14, -1.2079e-13, 4.2633e-14, 4.2633e-14, 6.7502e-14],
                   [3.1974e-14, -5.6843e-14, 1.7764e-14, 1.4211e-14, -1.4211e-13, -6.3949e-14],
                   [-5.3291e-15, -2.4869e-14, -1.7764e-14, -5.6843e-14, 2.1316e-14, -2.1316e-13],
                   [1.7764e-14, 4.9738e-14, -4.9738e-14, 2.1316e-14, 3.5527e-15, -3.5527e-15]]
trunc_error_2_dmrg2 = [[4.2744e-18, 4.6244e-18, 3.6091e-18, 3.7144e-18, 3.5784e-18, 3.3492e-18],
                       [3.9229e-17, 4.8012e-17, 4.7858e-17, 3.9374e-17, 3.4688e-17, 3.6468e-17],
                       [1.5417e-16, 2.0280e-16, 1.8408e-16, 1.8221e-16, 1.5775e-16, 1.6008e-16],
                       [4.4736e-16, 5.7498e-16, 5.9879e-16, 6.2731e-16, 5.0547e-16, 5.0704e-16],
                       [1.0588e-15, 1.2274e-15, 1.4294e-15, 1.5348e-15, 1.2594e-15, 1.1717e-15]]
e_error_2_dmrg2 = [[3.5527e-14, 2.8422e-14, 1.7764e-14, 4.2633e-14, 1.7764e-14, 4.9738e-14],
                   [1.7764e-14, 2.8422e-14, 2.4869e-14, 6.3949e-14, 2.4869e-14, 4.9738e-14],
                   [4.4409e-14, 7.8160e-14, 4.2633e-14, 5.6843e-14, 5.6843e-14, 4.6185e-14],
                   [2.8422e-14, 5.3291e-14, 1.4211e-14, 1.4211e-14, 3.1974e-14, 3.9080e-14],
                   [3.1974e-14, 5.3291e-14, 3.5527e-14, 6.7502e-14, 6.0396e-14, 6.3949e-14]]
trunc_error_2_dmrg1 = np.array(trunc_error_2_dmrg1)
trunc_error_2_dmrg2 = np.array(trunc_error_2_dmrg2)
e_error_2_dmrg1 = np.array(e_error_2_dmrg1)
e_error_2_dmrg2 = np.array(e_error_2_dmrg2)
E_error_2_dmrg1 = np.array(E_error_2_dmrg1)
E_error_2_dmrg2 = np.array(E_error_2_dmrg2)

trunc_error_2 = trunc_error_2_dmrg1 + trunc_error_2_dmrg2
e_error_2 = e_error_2_dmrg1 + e_error_2_dmrg2
E_error_2 = abs(E_error_2_dmrg1) + abs(E_error_2_dmrg2)

trunc_error_2 = trunc_error_2.tolist()
e_error_2 = e_error_2.tolist()
E_error_2 = E_error_2.tolist()

import os
homepath = os.getcwd()

#energy reading
eng_list_1_fname = homepath + '/englist1_D{}'.format(D) #identical number 1 for K=0.256 to 0.260
with open(eng_list_1_fname, 'rb') as f:
    engdifflist_1 = pickle.load(f)

eng_list_2_fname = homepath + '/englist2_D{}'.format(D) #2 for K=0.251 to 0.255
with open(eng_list_2_fname, 'rb') as f:
    engdifflist_2 = pickle.load(f)

error_list_1_fname = homepath + '/errorlist1_D{}'.format(D) #1 for K=0.256 to 0.260
with open(error_list_1_fname, 'rb') as f:
    errorlist_1 = pickle.load(f)
E_error_1 = errorlist_1[0]
trunc_error_1 = errorlist_1[1]
e_error_1 = errorlist_1[2]

#fitting
# 定义指数函数模型
def exponential_func(x, A, a):
    return A * np.exp(-a * x)

line_styles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5))]

# 对Klist_1中的每条线进行拟合并绘制拟合结果
fig, ax = plt.subplots(figsize=(15, 10))
A_fit_list = []
a_fit_list = []
for idx, ki in enumerate(Klist_2):
    y_data = [engdifflist_2[Klist_2.index(ki)][i] for i in range(len(lxlist))]
    # 使用curve_fit进行拟合
    popt, pcov = curve_fit(exponential_func, lxlist, y_data)
    A_fit, a_fit = popt

    color = colors[idx]
    line_style_data = line_styles[0]
    line_style_fit = line_styles[1]

    # 生成拟合曲线的x值（和原始数据x范围一致方便绘制对比）
    x_fit = np.linspace(min(lxlist), max(lxlist), 100)
    # if ki == 0.24:
    #     A_fit = 0.5105
    #     a_fit = 0.4369
    y_fit = exponential_func(x_fit, A_fit, a_fit)

    #save A_fit and a_fit
    A_fit_list.append(A_fit)
    a_fit_list.append(a_fit)

    # 绘制原始数据和拟合曲线，设置不同线条样式
    ax.errorbar(lxlist, y_data, yerr=E_error_2[Klist_2.index(ki)][0:6], fmt='o' + line_style_data, capsize=3, label='K={}, D={}'.format(ki, D), color=color)
    #ax.errorbar(lxlist, y_data, fmt='o' + line_style_data, capsize=3, label='K={}, D={}'.format(ki, D), color=color)
    #ax.plot(x_fit, y_fit, label='Fit for K={}, A={}, a={}'.format(np.round(ki,3), np.round(A_fit, 3), np.round(a_fit, 3)),color=color, linestyle=line_style_fit)
    
for idx, ki in enumerate(Klist_1):
    y_data = [engdifflist_1[Klist_1.index(ki)][i] for i in range(len(lxlist))]
    # 使用curve_fit进行拟合
    popt, pcov = curve_fit(exponential_func, lxlist, y_data)
    A_fit, a_fit = popt

    color = colors[idx]
    line_style_data = line_styles[0]
    line_style_fit = line_styles[1]

    # 生成拟合曲线的x值（和原始数据x范围一致方便绘制对比）
    x_fit = np.linspace(min(lxlist), max(lxlist), 100)
    y_fit = exponential_func(x_fit, A_fit, a_fit)

    #save A_fit and a_fit
    A_fit_list.append(A_fit)
    a_fit_list.append(a_fit)

    # 绘制原始数据和拟合曲线，设置不同线条样式
    ax.errorbar(lxlist, y_data, yerr=E_error_1[Klist_1.index(ki)][0:6], fmt='x' + line_style_data, capsize=3, label='K={}, D={}'.format(ki, D), color=color)
    #ax.errorbar(lxlist, y_data, fmt='x' + line_style_data, capsize=3, label='K={}, D={}'.format(ki, D), color=color)
    #ax.plot(x_fit, y_fit, label='Fit for K={}, A={}, a={}'.format(np.round(ki,3), np.round(A_fit, 3), np.round(a_fit, 3)),color=color, linestyle=line_style_fit)

ax.set_title('$|E_1-E_2|$ (log-linear scale)')
ax.set_xlabel('$N$')
ax.set_ylabel('$|E_1-E_2|$')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))  # 设置y轴对数刻度显示格式
ax.legend(loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('edplotgn{}_log_linear_1_error_fit.pdf'.format(D))

#plot A_fit and a_fit with K
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(Klist, A_fit_list, 'o-', label='A_fit')
ax.plot(Klist, a_fit_list, 'x-', label='a_fit')
ax.set_title('A_fit and a_fit with K')
ax.set_xlabel('K')
ax.set_ylabel('A_fit/a_fit')
ax.legend(loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('A_fit_a_fit_K.pdf')