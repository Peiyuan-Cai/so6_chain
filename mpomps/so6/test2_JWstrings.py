import numpy as np


#archieved 20240829 17:36

cxdag = np.zeros((8,8))
cxdag[1,0] = 1; cxdag[4,2] = 1; cxdag[5,3] = 1; cxdag[7,6] = 1; 
cydag = np.zeros((8,8))
cydag[2,0] = 1; cydag[4,1] = 1; cydag[6,3] = 1; cydag[7,5] = 1; 
czdag = np.zeros((8,8))
czdag[3,0] = 1; czdag[5,1] = 1; czdag[6,2] = 1; czdag[7,4] = 1; 
JW = np.diag([1,-1,-1,-1,1,1,1,-1])
Fx = np.diag([1,-1,1,1,-1,-1,1,-1])
Fy = np.diag([1,1,-1,1,-1,1,-1,-1])

print('cxdag F', cxdag@JW)
print('cydag F', cydag@JW)
print('cydag Fx', cydag@Fx)
print('czdag F', czdag@JW)
print('czdag Fy Fx', czdag@Fy@Fx)
