__author__ = 'lizk'
import numpy as np
import random
import math
from collections import *


user_size=943
item_size=1682
hidden_size=10
itemlamda = 2
userlamda = 2
itembiaslamda = 10
# lamda=2                         #2 0.01
learn_rate=0.01

rating_matrix=np.zeros((user_size,item_size))
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
		rate=int(record[2])
		target[row-1][col-1]=rate
	return target

rating_matrix=load_file('train.txt',rating_matrix)#input the data to the matrix
test_matrix=load_file('test.txt',test_matrix)

def fanshu(vector):
	fanshu=0
	for i in range(len(vector)):
		fanshu+=vector[i]*vector[i]
		fanshu = math.sqrt(fanshu)
	return fanshu

def overall_avg(target,key):
	total=0.0
	if key=='Train':
		count=50000
	else:
		count=20000
	count = 0
	for i in range(user_size):
		for j in range(item_size):
			if target[i][j] != 0:
				total += target[i][j]
				count += 1
	avg = total/count
	return avg

def user_bias(target,avg):
	biaslist=[]
	totalcount=0
	for i in range(user_size):
		count=1
		total=0.0
		for j in target[i]:
			if j!=0:
				count+=1
				total+=j-avg
		totalcount+=count
		userbias=total/(count)
		# userbias=useravg-avg
		biaslist.append(userbias)
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
			itemavg=total/(count)
			itembias=itemavg-avg
		else:
			itembias=0
		biaslist.append(itembias)
	return biaslist

def dataselect(rating_matrix):
	nonezeroinfo=[]
	for i in range (user_size):
		list1=[]
		for j in range(item_size):
			if rating_matrix[i][j]>0:
				list1.append(j)
		nonezeroinfo.append(list1)
	
	return nonezeroinfo


def train(user_matrix,item_matrix,nonezeroinfo):
	avg=overall_avg(rating_matrix,"Train")
	userbias=user_bias(rating_matrix,avg)
	itembias=item_bias(rating_matrix,avg)
	for i in range(user_size):						#each user
		for j in range(item_size):       			#each item in user
			if rating_matrix[i][j]>3:
				negy =  random.randint(0, len(nonezeroinfo[i])-1)
				negy= nonezeroinfo[i][negy]
				while not((negy != j)&(rating_matrix[i][negy]<4)):
					negy =  random.randint(0, item_size-1)
				minus_Xij = np.dot(user_matrix[i], item_matrix[negy]) +itembias[negy]- np.dot(user_matrix[i],item_matrix[j])-itembias[j]
				if (minus_Xij<-5):
					minus_Xij=-5
				if (minus_Xij>0):
					minus_Xij=0
				# print "minus"
				# print minus_Xij
				tmp_user = user_matrix[i]
				user_matrix[i] += learn_rate*(item_matrix[negy]-item_matrix[j])*np.exp(minus_Xij)/(1+np.exp(minus_Xij))+learn_rate*userlamda*user_matrix[i]
				item_matrix[j] += -learn_rate*tmp_user*np.exp(minus_Xij)/(1+np.exp(minus_Xij))+learn_rate*itemlamda*item_matrix[j]
				item_matrix[negy] += learn_rate*tmp_user*np.exp(minus_Xij)/(1+np.exp(minus_Xij))+learn_rate*itemlamda*item_matrix[negy]
				itembias[j] += learn_rate*itembiaslamda*itembias[j]
				itembias[negy] += learn_rate*itembiaslamda*itembias[negy]

	return user_matrix, item_matrix

def predict(user_matrix, item_matrix):
	avg=overall_avg(test_matrix,"Test")
	userbias=user_bias(test_matrix,avg)
	itembias=item_bias(test_matrix,avg)
	predict_matrix = np.zeros((user_size,item_size))
	for i in range(user_size):
		for j in range(item_size):
			predict_matrix[i][j] = np.dot(user_matrix[i][:],item_matrix[j][:])+itembias[j]
	return predict_matrix

def evaluate(predict_matrix, test_matrix):
	counter = 0
	preat5 = 0.0
	rank_index = np.argsort(predict_matrix, axis=1) #ordered by row small->big return index
	# print "rank"
	# print rank_index
	# print np.shape(rank_index)[1]
	rank_index = rank_index[:, -30:np.shape(rank_index)[1]]
	# print "rank_index"
	# print rank_index
	for user in range(user_size):
		rank_index2 = []
		t = 0
		for i in range(30):
			x = user
			y = rank_index[user][29-i]
			if rating_matrix[x][y]<4:
				t += 1
				rank_index2.append(y)
			if t > 4:
				break
		for y in rank_index2:
			x = user
			if test_matrix[x][y] > 3:
				counter += 1
	preat5 = counter/(5.0*user_size)
	print "prec%f"%preat5
	return preat5




nonezeroinfo=dataselect(rating_matrix)
for i in range(100):
	print "iter %i"%i
	train(user_matrix,item_matrix,nonezeroinfo)
	predict_matrix = predict(user_matrix, item_matrix)
	preat5 = evaluate(predict_matrix, test_matrix)
	print "pre@5 %f"%preat5


# for i in range(user_size):
# 	countpo=0
# 	countneg=0
# 	for j in range(item_size):
# 		if rating_matrix[i][j]>3:
# 			countpo += 1
# 		if test_matrix[i][j]>3:
# 			countneg += 1
# 	print countpo, countneg


# count3=0
# count5=0
# for i in range(user_size):
# 	count3=0
# 	count5=0
# 	print "user %i"%i
# 	for j in range(item_size):
# 		if test_matrix[i][j]>3:
# 			count5 += 1
# 		elif test_matrix[i][j]>0:
# 			count3 += 1
# 	print count3, count5
