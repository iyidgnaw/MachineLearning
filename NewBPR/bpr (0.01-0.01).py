__author__ = 'wangdiyi'
#test/train = 1:1
#data set : music

import numpy as np
import random
import math
user_size=1000
item_size=14238
hidden_size=10

lamda = 0.01
learn_rate=0.01

train_matrix=np.zeros((user_size,item_size))
test_matrix=np.zeros((user_size,item_size))#Initiation
user_matrix=np.random.randn(user_size, hidden_size)*0.5
item_matrix=np.random.randn(item_size, hidden_size)*0.5


def load_file(name,target):#load data function
	data=open(name,'r')
	lines=data.readlines()
	for line in lines:
		record=line.split('	')
		row=int(record[0])
		col=int (record[1])
		target[row-1][col-1]=1
	return target

train_matrix=load_file('train.txt',train_matrix)#input the data to the matrix
test_matrix=load_file('test.txt',test_matrix)



def fanshu(vector):
	fanshu=0
	for i in xrange(len(vector)):
		fanshu+=vector[i]*vector[i]
		fanshu = math.sqrt(fanshu)
	return fanshu

def train(user_matrix,item_matrix):
	sumnumber=0
	for i in xrange(user_size):						#each user
		for j in xrange(item_size):       			#each item in user
			if train_matrix[i][j]==1:
				negy =  random.randint(0, item_size-1)
				while not((negy != j)&(train_matrix[i][negy]==0)):
					negy =  random.randint(0, item_size-1)
				Xij = np.dot(user_matrix[i],item_matrix[j])-np.dot(user_matrix[i], item_matrix[negy])
				sumnumber+=Xij				
				mid=-np.exp(-Xij)/(1+np.exp(-Xij))
				tmp_user = user_matrix[i]
				user_matrix[i] += -learn_rate*(mid*(item_matrix[j]-item_matrix[negy])+lamda*user_matrix[i])
				item_matrix[j] += -learn_rate*(mid*user_matrix[i]+lamda*item_matrix[j])
				item_matrix[negy] += -learn_rate*(-mid*user_matrix[i]+lamda*item_matrix[negy])
	
	print sumnumber

	return user_matrix, item_matrix

def predict(user_matrix, item_matrix):
	predict_matrix = np.zeros((user_size,item_size))
	for i in xrange(user_size):
		for j in xrange(item_size):
			predict_matrix[i][j] = np.dot(user_matrix[i][:],item_matrix[j][:])
	return predict_matrix

def evaluate(predict_matrix, test_matrix):
	counter = 0
	preat5 = 0.0
	rank_index = np.argsort(predict_matrix, axis=1) #ordered by row small->big return index
	rank_index = rank_index[:, -50:np.shape(rank_index)[1]]
	for user in xrange(user_size):
		rank_index2 = []
		t = 0
		for i in xrange(50):
			x = user
			y = rank_index[user][49-i]
			if train_matrix[x][y]!=1:
				t += 1
				rank_index2.append(y)
			if t > 20:
				break

		for y in rank_index2:
			x = user
			if test_matrix[x][y] == 1:
				counter += 1
				
	preat20 = counter/(20.0*user_size)
	print "prec%f"%preat20
	return preat20

for i in xrange(1000):
	print "iter %i"%i
	train(user_matrix,item_matrix)
	predict_matrix = predict(user_matrix, item_matrix)
	preat20 = evaluate(predict_matrix, test_matrix)
	print "pre@5 %f"%preat20

