import numpy as np
import time


matrix1=np.zeros((100,80))
matrix2=np.zeros((80,100))
for i in range(100):
	for j in range(80):
		matrix1[i][j]=i
		matrix2[j][i]=j

time1 = time.clock()
for i in range(100000):
	result = np.dot(matrix1,matrix2)

time2 = time.clock()
print time2 - time1

