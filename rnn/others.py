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
hidden_size =10
learning_rate = 5e-2
goods_size = 1559 
itert = 0
top=50
# model parameters
u = np.random.randn(hidden_size, user_size)*0.5 # input to hidden
t = np.random.randn(hidden_size, goods_size)*0.5 # one-hot to embedding

def function(x):
	result=np.log(1+np.exp(-x))
	return result


def lossfunction(x,xn,user,t):
	minus=np.dot(np.dot(user.T,t),x)-np.dot(np.dot(user.T,t),xn)
	loss =function(minus)
	du, dt= np.zeros_like(user), np.zeros_like(t)
	dtmid=np.dot(user,x.T)-np.dot(user,xn.T)
	dt=(np.exp(minus)*dtmid)/(1+np.exp(minus))
	dumid=np.dot(t,x)-np.dot(t,xn)
	du=(np.exp(minus)*dumid)/(1+np.exp(minus))
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
	number=0
	avrloss=0
	right=0
	print "This is iter %d"%itert
	time0=time.clock()



	for i in range(len(listcust)-1):
		customer = data[listcust[i]]
		#hint 
		if i%500==0:
			print "Training customer %d"%i
		#prepare the user 	
		user = np.zeros((hidden_size,1))
		for j in range(hidden_size):
			user[j][0] = u[j][i]
		
		#prepare the positive set and negative set
		pos=[]
		for i in range(len(customer)-1):
			for j in range(len(customer[i])):
				pos.append(customer[i][j])
		pos=list(set(pos))
		neg=negasamp(pos)
		for i in range(len(pos)):
			number+=1
			x = np.zeros((goods_size,1))
			xn = np.zeros((goods_size,1))
			x[pos[i]-1][0]=1
			xn[neg[i]-1][0]=1
			loss, du, dt=lossfunction(x,xn,user,t)
			t-=dt*learning_rate
			avrloss+=loss
			for j in range(hidden_size):
				u[j][i]-= du[j][0]*learning_rate
	print avrloss/number



	for i in range(len(listcust)-1):
		customer = data[listcust[i]]
		user = np.zeros((hidden_size,1))
		for j in range(hidden_size):
			user[j][0] = u[j][i]	
		right+=predict(customer,user)
	print right





