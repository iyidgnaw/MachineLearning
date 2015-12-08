import numpy as np
import random
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


	