import numpy as np
import random
import math
user_size=943
item_size=1682
rating_matrix=np.zeros((user_size,item_size))
test_matrix=np.zeros((user_size,item_size))#Initiation

def load_file(name,target):#load data function
	data=open(name,'r')
	lines=data.readlines()
	for line in lines:
		record=line.split('	')
		row=int(record[0])
		col=int (record[1])
		rate=int(record[2])
		target[row-1][col-1]=rate
	return target

rating_matrix=load_file('train.txt',rating_matrix)#input the data to the matrix
test_matrix=load_file('test.txt',test_matrix)


def overall_avg(target,key):
	total=0.0
	if key=='Train':
		count=80000
	else:
		count=20000
	for i in range(user_size):
		for j in range(item_size):
			total+=target[i][j]
	avg=total/count
	return avg

def user_bias(target,avg):
	biaslist=[]
	totalcount=0
	for i in range(user_size):
		count=0
		total=0.0
		for j in target[i]:
			if j!=0:
				count+=1
				total+=j
		totalcount+=count
		useravg=total/count
		userbias=useravg-avg
		biaslist.append(userbias)
	print totalcount
	return biaslist

def item_bias(target,avg):
	biaslist=[]
	totalcount=0
	for i in range(item_size):
		count=0
		total=0
		for j in range(user_size):
			if target[j][i]!=0:
				count+=1
				total+=target[j][i]
		totalcount+=count		
		if count!=0:
			itemavg=total/count
			itembias=itemavg-avg
		else:
			itembias=0
		biaslist.append(itembias)
	return biaslist


def predict(user_matrix, item_matrix):
	predict = np.zeros((user_size,item_size))
	for i in range(user_size):
		for j in range(item_size):
			# predict[i][j] = np.dot(user_matrix[i][:],item_matrix[j][:])+userbias[i]+itembias[j]
			predict[i][j] = np.dot(user_matrix[i][:],item_matrix[j][:])
	return predict


def evaluate(predict, test_matrix):
	counter = 0
	sum = 0
	for i in range(user_size):
		for j in range(item_size):
			if((predict[i][j]>0) & (test_matrix[i][j]>0)):
				counter += 1
				sum += math.pow(test_matrix[i][j] - predict[i][j], 2)

	return math.sqrt(sum/counter)


# user_matrix = np.random.randn(943, 10)
# item_matrix = np.random.randn(item_size, 10)
# predict_matrix = predict(user_matrix, item_matrix)
# evalu = evaluate(predict_matrix, test_matrix)
# print predict_matrix
# print evalu
