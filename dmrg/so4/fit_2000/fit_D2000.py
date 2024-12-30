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
import matplotlib as mpl
import matplotlib.cm as cm
# 配置matplotlib使用LaTeX渲染文本（需确保系统已安装LaTeX相关环境）
mpl.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'

lxlist = [24,28,32,36,40]
Klist = [0.24, 0.241, 0.242, 0.243, 0.244, 0.245, 0.246, 0.247, 0.248, 0.249]
Klist_1 = [0.245, 0.246, 0.247, 0.248, 0.249] #the dimer side
Klist_2 = [0.24, 0.241, 0.242, 0.243, 0.244] #the deeper dimer side
#Klist_2 = [0.251, 0.253, 0.255] #the critical side
engdifflist_2 = [] #critical side
D = 2000

trunc_error_1_dmrg1 = [[2.3652e-14, 3.1591e-14, 2.9448e-14, 2.5563e-14, 2.9443e-14, 2.9337e-14],
                       [1.1787e-14, 1.4496e-14, 1.5045e-14, 1.1747e-14, 1.3313e-14, 1.3076e-14],
                       [4.7961e-15, 6.3199e-15, 6.2886e-15, 5.0823e-15, 4.8514e-15, 4.8131e-15],
                       [1.4861e-15, 1.4113e-15, 1.5552e-15, 1.3223e-15, 1.1685e-15, 1.1662e-15],
                       [2.7567e-16, 2.0343e-16, 1.8800e-16, 1.3215e-16, 1.1638e-16, 1.0830e-16]]
e_error_1_dmrg1 = [[2.1494e-13, 2.9843e-13, 3.1974e-13, 2.8422e-13, 2.6290e-13, 3.1619e-13],
                   [1.6698e-13, 1.7053e-13, 1.6342e-13, 1.3145e-13, 1.7764e-13, 1.5277e-13],
                   [4.4409e-14, 7.4607e-14, 8.1712e-14, 7.8160e-14, 7.1054e-14, 8.1712e-14],
                   [4.9738e-14, 6.7502e-14, 1.3145e-13, 7.1054e-14, 9.9476e-14, 1.2434e-13],
                   [3.0198e-14, 3.9080e-14, 4.2633e-14, 5.6843e-14, 6.0396e-14, 3.5527e-14]]
E_error_1_dmrg1 = [[4.2633e-14, -3.1974e-13, -6.8923e-13, -2.8422e-14, -3.2685e-12, -3.5698e-11],
                   [2.1316e-14, 2.8422e-14, -7.6739e-13, -1.9185e-13, -5.7625e-12, -1.6140e-11],
                   [-1.0658e-14, -2.8422e-14, -1.3145e-12, -1.3145e-13, -1.2307e-11, -1.2967e-11],
                   [-1.4211e-14, -6.3949e-14, 3.1974e-14, -2.1672e-13, -1.8368e-12, -8.2814e-12],
                   [-8.8818e-15, -1.7764e-14, 7.1054e-15, -1.7408e-13, -1.6342e-13, -8.7397e-13]]

trunc_error_2_dmrg1 = [[4.3075e-14, 2.1952e-16, 1.7352e-16, 1.6045e-16, 1.3132e-16, 8.6299e-17],
                       [2.1926e-15, 1.7865e-15, 1.6069e-15, 1.5572e-15, 1.4897e-15, 1.1401e-15],
                       [7.1281e-15, 7.4986e-15, 7.7145e-15, 6.9549e-15, 6.4449e-15, 3.6542e-15],
                       [2.2214e-14, 2.2016e-14, 2.4448e-14, 1.9002e-14, 1.5245e-14, 8.3607e-15],
                       [4.3075e-14, 6.2662e-14, 4.6404e-14, 4.1312e-14, 4.2372e-14, 1.8082e-14]]
e_error_2_dmrg1 = [[3.7303e-13, 2.8422e-14, 3.5527e-14, 3.9080e-14, 4.2633e-14, 6.7502e-14],
                   [5.8620e-14, 7.4607e-14, 7.8160e-14, 7.1054e-14, 9.2371e-14, 9.2371e-14],
                   [6.3949e-14, 1.1724e-13, 1.1369e-13, 1.2079e-13, 1.1724e-13, 1.0658e-13],
                   [1.9007e-13, 2.0606e-13, 1.9540e-13, 2.5935e-13, 1.8829e-13, 1.5987e-13],
                   [3.7303e-13, 4.6896e-13, 3.5527e-13, 3.7303e-13, 3.9790e-13, 1.9895e-13]]
E_error_2_dmrg1 = [[2.3093e-14, -3.5527e-15, -5.3291e-14, -4.2633e-14, -4.4054e-13, -6.2172e-13],
                   [1.9540e-14, -3.5527e-15, -1.3145e-13, -4.0501e-13, -1.5987e-12, -5.0164e-12],
                   [3.9080e-14, -2.6290e-13, -7.4607e-14, -8.3844e-13, -1.4669e-11, -4.2885e-11],
                   [1.7764e-14, -7.1054e-14, -5.0449e-13, -1.3145e-13, -4.2881e-12, -1.2619e-11],
                   [2.8422e-14, -6.3594e-13, -1.8119e-13, -4.7606e-13, -6.3828e-11, -3.0234e-12]]

trunc_error_1_dmrg2 = [[2.36e-14, 3.0170e-14, 3.2053e-14, 3.3224e-14, 2.6930e-14, 2.4879e-14],
                       [1.18e-14, 1.4013e-14, 1.4468e-14, 1.3793e-14, 1.2192e-14, 1.1303e-14],
                       [5.75e-15, 5.3470e-15, 5.3631e-15, 5.0120e-15, 4.5145e-15, 4.2773e-15],
                       [1.72e-15, 1.3777e-15, 1.3432e-15, 1.1298e-15, 1.0276e-15, 9.1235e-16],
                       [2.38e-16, 2.2110e-16, 1.7445e-16, 1.1174e-16, 1.0567e-16, 8.1869e-17]]
e_error_1_dmrg2 = [[2.17e-13, 2.8777e-13, 3.4106e-13, 2.8066e-13, 2.5935e-13, 3.0198e-13],
                   [1.46e-13, 1.4211e-13, 1.7764e-13, 1.3145e-13, 1.7408e-13, 1.7408e-13],
                   [6.22e-14, 7.1054e-14, 9.2371e-14, 4.6185e-14, 5.3291e-14, 7.1054e-14],
                   [3.38e-14, 8.5265e-14, 9.2371e-14, 1.0658e-13, 7.4607e-14, 1.3145e-13],
                   [5.86e-14, 7.1054e-14, 4.9738e-14, 5.6843e-14, 6.3949e-14, 7.8160e-14]]
E_error_1_dmrg2 = [[-1.7337e-12, -8.5265e-14, -1.1013e-13, -8.1002e-13, -7.0742e-11, -6.3423e-11],
                   [-1.0658e-14, -1.2434e-13, -8.1712e-14, -5.2225e-13, -5.0676e-11, -3.8362e-11],
                   [4.2633e-14, 1.4211e-14, -4.9738e-14, -3.5882e-13, -3.3118e-11, -1.5454e-12],
                   [3.5527e-15, 3.9080e-14, -1.4211e-14, 3.1974e-14, -8.5265e-12, -1.9007e-12],
                   [1.0658e-14, 1.0658e-14, 6.3949e-14, 2.8422e-14, -3.3396e-13, -3.1264e-12]]

trunc_error_2_dmrg2 = [[2.94e-16, 2.1613e-16, 1.6915e-16, 1.2375e-16, 1.0687e-16, 6.3633e-17],
                       [2.37e-15, 2.1076e-15, 1.8121e-15, 1.3350e-15, 1.1615e-15, 7.8021e-16],
                       [9.1275e-15, 7.3324e-15, 8.0013e-15, 6.1844e-15, 5.2525e-15, 4.4171e-15],
                       [2.5002e-14, 2.4164e-14, 2.2557e-14, 1.9223e-14, 1.7035e-14, 8.0177e-15],
                       [5.61e-14, 8.7595e-14, 5.4303e-14, 4.3669e-14, 3.6515e-14, 1.8088e-14]]
e_error_2_dmrg2 = [[4.80e-14, 1.7764e-14, 2.8422e-14, 2.1316e-14, 5.6843e-14, 7.4607e-14],
                   [5.51e-14, 5.6843e-14, 8.8818e-14, 1.0303e-13, 6.3949e-14, 8.8818e-14],
                   [8.3489e-14, 1.2790e-13, 8.5265e-14, 1.1724e-13, 1.4921e-13, 7.8160e-14],
                   [1.9362e-13, 1.7764e-13, 1.7764e-13, 1.7764e-13, 2.1672e-13, 1.4211e-13],
                   [4.32e-13, 5.4712e-13, 4.4409e-13, 4.1211e-13, 4.1922e-13, 1.9185e-13]]
E_error_2_dmrg2 = [[-1.5987e-14, 2.4869e-14, -1.2790e-13, -7.8160e-14, -1.5206e-12, -1.1369e-11],
                   [7.1054e-15, 7.1054e-15, -2.3448e-13, -4.5475e-13, -2.4514e-12, -6.4375e-12],
                   [-2.8422e-14, -2.1316e-14, -1.2434e-13, -2.0322e-12, -1.0168e-11, -5.3291e-13],
                   [5.3291e-14, -1.3856e-13, -1.2790e-13, -2.9772e-12, -1.3767e-11, -9.5888e-12],
                   [-2.1316e-14, -3.5527e-15, -2.9132e-13, -5.6097e-11, -5.2466e-11, -1.1532e-11]]

trunc_error_2_dmrg1 = np.array(trunc_error_2_dmrg1)
trunc_error_2_dmrg2 = np.array(trunc_error_2_dmrg2)
e_error_2_dmrg1 = np.array(e_error_2_dmrg1)
e_error_2_dmrg2 = np.array(e_error_2_dmrg2)
E_error_2_dmrg1 = np.array(E_error_2_dmrg1)
E_error_2_dmrg2 = np.array(E_error_2_dmrg2)

trunc_error_1_dmrg1 = np.array(trunc_error_1_dmrg1)
trunc_error_1_dmrg2 = np.array(trunc_error_1_dmrg2)
e_error_1_dmrg1 = np.array(e_error_1_dmrg1)
e_error_1_dmrg2 = np.array(e_error_1_dmrg2)
E_error_1_dmrg1 = np.array(E_error_1_dmrg1)
E_error_1_dmrg2 = np.array(E_error_1_dmrg2)

trunc_error_2 = trunc_error_2_dmrg1 + trunc_error_2_dmrg2
e_error_2 = e_error_2_dmrg1 + e_error_2_dmrg2
E_error_2 = abs(E_error_2_dmrg1) + abs(E_error_2_dmrg2)

trunc_error_1 = trunc_error_1_dmrg1 + trunc_error_1_dmrg2
e_error_1 = e_error_1_dmrg1 + e_error_1_dmrg2
E_error_1 = abs(E_error_1_dmrg1) + abs(E_error_1_dmrg2)

trunc_error_2 = trunc_error_2.tolist()
e_error_2 = e_error_2.tolist()
E_error_2 = E_error_2.tolist()

trunc_error_1 = trunc_error_1.tolist()
e_error_1 = e_error_1.tolist()
E_error_1 = E_error_1.tolist()

import os
homepath = os.getcwd()

#energy reading
eng_list_1_fname = homepath + '/englist1_D{}'.format(D) #1 for K=0.245 to 0.249
with open(eng_list_1_fname, 'rb') as f:
    engdifflist_1 = pickle.load(f)

eng_list_2_fname = homepath + '/englist2_D{}'.format(D) #2 for K=0.24 to 0.244
with open(eng_list_2_fname, 'rb') as f:
    engdifflist_2 = pickle.load(f)

error_list_2_fname = homepath + '/errorlist2_D{}'.format(D)
with open(error_list_2_fname, 'rb') as f:
    errorlist_2 = pickle.load(f)
E_error_2 = errorlist_2[0]
trunc_error_2 = errorlist_2[1]
e_error_2 = errorlist_2[2]

#fitting
# 定义指数函数模型
def exponential_func(x, A, a):
    return A * np.exp(-a * x)

line_styles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5))]
# colors = ['r', 'g', 'b', 'm', 'c']
# colors2 = ['y', 'k', 'orange', 'purple', 'brown']
cmap = cm.get_cmap('cool')
# 生成颜色列表，根据数据点数量生成对应的渐变色
combined_Klist = Klist_1 + Klist_2
# 生成颜色列表，从颜色映射中选取相对较深、辨识度高的部分区间（这里示例选取后半部分区间，可根据实际调整）
colors = [cmap(i / len(combined_Klist)) for i in range(len(combined_Klist))]
colors = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#33BBBB', '#00BFFF', '#882255', '#AA4499']
colors1 = colors[0:5]
colors2 = colors[5:10]

# 对Klist_1中的每条线进行拟合并绘制拟合结果
fig, ax = plt.subplots(figsize=(15, 10))
A_fit_list = []
a_fit_list = []
for idx, ki in enumerate(Klist_2):
    y_data = [engdifflist_2[Klist_2.index(ki)][i+1] for i in range(len(lxlist))]
    # 使用curve_fit进行拟合
    popt, pcov = curve_fit(exponential_func, lxlist, y_data)
    A_fit, a_fit = popt

    color = colors[idx]
    line_style_data = line_styles[0]
    line_style_fit = line_styles[1]

    # 生成拟合曲线的x值（和原始数据x范围一致方便绘制对比）
    x_fit = np.linspace(min(lxlist), max(lxlist), 100)
    if ki == 0.24:
        A_fit = 0.5105
        a_fit = 0.4369
    y_fit = exponential_func(x_fit, A_fit, a_fit)

    #save A_fit and a_fit
    A_fit_list.append(A_fit)
    a_fit_list.append(a_fit)

    # 绘制原始数据和拟合曲线，设置不同线条样式
    ax.errorbar(np.array(lxlist) * a_fit, y_data, yerr=E_error_2[Klist_2.index(ki)][1:6], fmt='o' + line_style_data, capsize=3, label='$K$={}'.format(ki), color=color) #times a_fit for data collapse
    #ax.plot(x_fit, y_fit, label='Fit for $K$={}, $A$={}, $a$={}'.format(np.round(ki,3), np.round(A_fit, 3), np.round(a_fit, 3)), color=color, linestyle=line_style_fit)
    #ax.plot(x_fit, y_fit, color=color, linestyle=line_style_fit)
    
for idx, ki in enumerate(Klist_1):
    y_data = [engdifflist_1[Klist_1.index(ki)][i+1] for i in range(len(lxlist))]
    # 使用curve_fit进行拟合
    popt, pcov = curve_fit(exponential_func, lxlist, y_data)
    A_fit, a_fit = popt

    color = colors2[idx]
    line_style_data = line_styles[0]
    line_style_fit = line_styles[1]

    # 生成拟合曲线的x值（和原始数据x范围一致方便绘制对比）
    x_fit = np.linspace(min(lxlist), max(lxlist), 100)
    y_fit = exponential_func(x_fit, A_fit, a_fit)

    #save A_fit and a_fit
    A_fit_list.append(A_fit)
    a_fit_list.append(a_fit)

    # 绘制原始数据和拟合曲线，设置不同线条样式
    ax.errorbar(np.array(lxlist) * a_fit, y_data, yerr=E_error_1[Klist_1.index(ki)][1:6], fmt='o' + line_style_data, capsize=3, label='$K={}$'.format(ki), color=color) #times a_fit for data collapse
    #ax.plot(x_fit, y_fit, label='Fit for $K$={}, $A$={}, $a$={}'.format(np.round(ki,3), np.round(A_fit, 3), np.round(a_fit, 3)), color=color, linestyle=line_style_fit)
    #ax.plot(x_fit, y_fit, color=color, linestyle=line_style_fit)

#ax.set_title('$|E_1-E_2|$ with Fit (log-linear scale)')
ax.set_xlabel('$a(K)L$', fontsize=20)
ax.set_ylabel('$\Delta E = |E_1-E_2|$', fontsize=20)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))  # 设置y轴对数刻度显示格式
#ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.legend(loc='upper right', ncol=2, fontsize=15)

ax_sub = fig.add_axes([0.12,0.12,0.3,0.3])
ax_sub.plot(Klist, A_fit_list, 'o-', label='$A(K)$')
ax_sub.plot(Klist, a_fit_list, 'x-', label='$a(K)$')
#ax_sub.set_title('$A(K)$ and $a(K)$')
# ax_sub.set_xlabel('$K$', fontsize=15)
# ax_sub.set_ylabel('$A(K)$ or $a(K)$', fontsize=15)
# 在右上角添加LaTeX公式，通过调整坐标位置来放置在合适的地方，这里的坐标是坐标轴坐标体系下的值
# 你可以根据实际图形效果调整x和y的值来改变公式位置
ax_sub.text(0.31, 0.11, r'$\Delta E = A \exp(-aL)$', fontsize=25, ha='right', va='top', transform=ax.transAxes)
ax_sub.tick_params(axis='x', labelsize=13)
ax_sub.tick_params(axis='y', labelsize=13)
ax_sub.legend(loc='best')

plt.tight_layout()
plt.savefig('edplotgn{}_log_linear_1_error_fit.pdf'.format(D))

#plot A_fit and a_fit with K
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(Klist, A_fit_list, 'o-', label='A_fit')
ax.plot(Klist, a_fit_list, 'x-', label='a_fit')
ax.set_title('A_fit and a_fit with K')
ax.set_xlabel('K')
ax.set_ylabel('A_fit/a_fit')
#ax.legend(loc='lower left', frameon=False, facecolor='none', edgecolor='none', ncol=2, bbox_to_anchor=(1.1, 1.0))
ax.legend(loc='best', fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_facecolor('#ffffff')
plt.tight_layout()
plt.savefig('A_fit_a_fit_K.pdf')