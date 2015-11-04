__author__ = 'wangdiyi'
import numpy as np
import pickle
import random
import time


ISOTIMEFORMAT='%Y-%m-%d %X'

#read
f1 = open("./data_NextBasket.txt", "rb")
data = pickle.load(f1)
f1.close()
# data.get(3)
f1 = open("./list_cust4List.txt", "rb")
listcust = pickle.load(f1)
f1.close()
f1 = open("./list_product_id.txt", "rb")
product_id = pickle.load(f1)
f1.close()
print len(listcust)


user_size = len(listcust)
hidden_size = 20
learning_rate = 5e-2
goods_size = 1559 
itert = 0
top = 50
# model parameters
u = np.random.randn(hidden_size, user_size)*0.5 # input to hidden
t = np.random.randn(hidden_size, goods_size)*0.5 # one-hot to embedding


def lossfunction(customer,user):
	pos=[]
	x = np.zeros((goods_size,1))
	xn = np.zeros((goods_size,1))

	for i in range(len(customer)-1):
		for j in range(len(customer[i])):
			pos.append(customer[i][j])
	pos=list(set(pos))
	for i in pos:
		x[i-1][0]=1
	neg=negasamp(pos)
	for i in neg:
		xn[i-1][0]=1
	loss=np.dot(np.dot(user.T,t),x)-np.dot(np.dot(user.T,t),xn)
	du, dt= np.zeros_like(user), np.zeros_like(t)
	dt=np.dot(user,x.T)-np.dot(user,xn.T)
	du=np.dot(t,x)-np.dot(t,xn)
	return loss , du ,dt

def negasamp(pos):
	list3 = []
	list2 = product_id
	for m in pos:
		list2 = filter(lambda a: a != m, list2)
	list3=random.sample(list2, len(pos))
	return list3

def predict(customer,user):

	right = 0
	result=[]
	for z in range(len(customer)-1, len(customer)):
		targets = customer[z]
		valuet = np.dot(user.T,t)
		valuet=valuet.argsort()
		for s in range(top):
			result.append(valuet[0][len(product_id)-s-1])
		right=len(list(set(targets) & set(result)))
	return right

while True:
	itert += 1
	#Train
	right=0
	avrloss=0
	print "This is iter %d"%itert
	time0=time.clock()
	for i in range(len(listcust)-1):
		customer = data[listcust[i]]
		if i%500==0:
			print "Training customer %d"%i
		user = np.zeros((hidden_size,1))
		for j in range(hidden_size):
			user[j][0] = u[j][i]
		loss, du, dt=lossfunction(customer,user)
		t+=dt*learning_rate
		for j in range(hidden_size):
			u[j][i]+= du[j][0]*learning_rate

		avrloss+=loss
	avrloss=avrloss/len(listcust)
	print avrloss

	for i in range(len(listcust)-1):
		customer = data[listcust[i]]
		user = np.zeros((hidden_size,1))
		for j in range(hidden_size):
			user[j][0] = u[j][i]	
		right+=predict(customer,user)
	print right





