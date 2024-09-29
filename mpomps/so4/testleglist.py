import numpy as np
import itertools

characters = ['x']
combinations = []

for i in range(1, len(characters) + 1):
    for combo in itertools.combinations(characters, i):
        combinations.append(''.join(combo))

print(combinations)
#print(len(combinations))

JW = np.eye(2**len(characters))
for i in range(1,2**len(characters)):
    if len(combinations[i-1]) %2 == 1:
        JW[i,i] = -1
    
def fmatrix(flavor, basis):
    flist = [1]
    #print("length of basis", len(basis))
    for i in range(1,len(basis)):
        if flavor in basis[i]:
            flist.append(-1)
        else:
            flist.append(1)
    fmat = np.diag(flist)
    #print('flist of flavor', flavor, 'is', flist)
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
                    #print('l=',setL,'r=',setR)
    return adaggermatrixform

name = ['empty'] + combinations

print(" ")
print("cxdag")
adag = adaggermatrix('x', name)
cdag = adag
for i in range(cdag.shape[0]):
    for j in range(cdag.shape[0]):
        if cdag[i,j] != 0:
            print('left',name[i],'right',name[j],'entry',cdag[i,j])