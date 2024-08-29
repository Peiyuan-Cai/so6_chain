import itertools
import numpy as np
from copy import deepcopy

# 定义字符列表
characters = ['u', 'v', 'w', 'x', 'y', 'z']

# 存放组合结果的列表
combinations = []

# 生成所有长度为1到6的组合
for i in range(1, len(characters) + 1):
    for combo in itertools.combinations(characters, i):
        # 将组合转换为字符串并添加到结果列表中
        combinations.append(''.join(combo))

print(combinations)
#print(len(combinations))
'''
def flavor_qn(combination):
    qn_map = {'u': -5/2, 'v': -3/2, 'w': -1/2, 'x': 1/2, 'y': 3/2, 'z': 5/2}
    totalqn = 0
    for char in combination:
        totalqn += qn_map[char]
    return totalqn

flavorqn = [0] #the first state is empty, and the qn is 0
for i in range(len(combinations)):
    flavorqn.append(flavor_qn(combinations[i]))

print(flavorqn)
print(len(flavorqn))

leglist0 = [[0,0]]; leglist1 = []; leglist2 = []
for i in range(len(flavorqn)):
    leglist1.append([1, flavorqn[i]])
    leglist2.append([flavorqn[i]])
for i in range(len(flavorqn)-1):
    leglist0.append([len(combinations[i]), flavorqn[i+1]])
'''

'''
print('leglist0')
print(leglist0)
print(len(leglist0))
print('leglist1')
print(leglist1)
print(len(leglist1))
print('leglist2')
print(leglist2)
print(len(leglist2))
'''
JW = np.eye(64)
for i in range(1,64):
    if len(combinations[i-1]) %2 == 1:
        JW[i,i] = -1
    
def fmatrix(flavor, basis):
    flist = [1]
    print("length of basis", len(basis))
    for i in range(1,len(basis)):
        if flavor in basis[i]:
            flist.append(-1)
        else:
            flist.append(1)
    fmat = np.diag(flist)
    print(flist)
    return fmat

def adaggermatrix(flavor, basis):
    basislength = len(basis) #for SO(6) case, it's 64
    adaggermatrixform = np.zeros((basislength,basislength))
    for l in range(basislength):
        if basis[l] == 'empty':
            setL = set()
        else:
            setL = set(basis[l])
        for r in range(basislength):
            if basis[r] == 'empty':
                setR = set()
            else:
                setR = set(basis[r])
            
            if (flavor in setL) and (flavor not in setR):
                diff = setL - setR
                listdiff = list(diff)
                
                if len(setL)-len(setR)==1 and len(listdiff) == 1 and listdiff[0] == flavor:
                    adaggermatrixform[l,r] = 1
                    print('l=',setL,'r=',setR)
    return adaggermatrixform

name = ['empty'] + combinations

fm = fmatrix('v',name)
audag = adaggermatrix('v',name)
print(audag)