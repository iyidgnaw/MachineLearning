import numpy as np
time_size = 3
def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output

#sum(tensor[i]*vector[i])  return a matrix
def tensor_mul(tensor, vector):
	product = 0
	print np.shape(vector)
	for i in range(len(tensor)):
		product+=tensor[i]*vector[i].reshape(1,1)
	return product

def updateTm(Tt, dMi,time_interval):
	for i in range(len(Tt)):
		Tt[i] += (dMi*time_interval[i])
	return Tt

#x[i] = sum(matrix*tensor[i])   return x  (a vector)
def mul_add(matrix, tensor):
	x = np.zeros((1,time_size))

	for i in range(len(tensor)):
		print x
		matrix2 = tensor[i]
		print "matrix2"
		print matrix2
		result_matrix = matrix2*matrix
		print result_matrix
		for j in range(len(result_matrix)):
			for k in range(len(result_matrix[0])):
				x[0][i] += result_matrix[j][k]
	return x

#test tensor mul right
tensor = np.array([[[1,2,3],[2,3,4],[5,9,7]],[[3,2,4],[5,6,4],[3,2,6]],[[3,5,7],[4,5,7],[4,8,6]]])
vector = np.array([[1,2,3]])
vector = vector[0,:].reshape(time_size,1)
dm = np.array([[2,5,4],[3,5,1],[2,4,7]])
print tensor
print dm
# dTm = updateTm(tensor, dm, vector)
# print tensor_mul(tensor,vector)

mul_ad = mul_add(dm,tensor)
print mul_ad