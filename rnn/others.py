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
learning_rate = 1e-3
goods_size = 1559 
itert = 0
top = 50
bias = 0
# model parameters
u = np.random.randn(user_size, hidden_size)*0.5 # input to hidden
t = np.random.randn(hidden_size, goods_size)*0.5 # one-hot to embedding
print u

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
	print np.shape(x),np.shape(xn),np.shape(user),np.shape(t)
	print user
	loss=np.dot(np.dot(user,t),x)-np.dot(np.dot(user,t),xn)
	du, dt= np.zeros_like(user), np.zeros_like(t)
	dt=np.dot(user.T,x.T)-np.dot(user.T,xn.T)
	du=np.dot(t,x)-np.dot(t,xn)
	return loss , du ,dt



def negasamp(pos):
	list3 = []
	list2 = product_id
	for m in pos:
		list2 = filter(lambda a: a != m, list2)
	list3=random.sample(list2, len(pos))
	return list3




def predict(customer, u, w, t):

	right = 0
	hl = np.zeros((hidden_size, 1))
	x = np.zeros((goods_size,1)) # encode in 1-of-k representation
	xt = np.zeros((goods_size,1)) # encode in 1-of-k representation
	# rank = np.zeros((20,2))
	allrank = [[0]*2 for row in range(len(product_id))]

	for q in range(len(customer)-1):
		inputs = customer[q]
		for c in inputs:
			x[c-1][0] = 1
		h = sigmoid(np.dot(np.dot(u,t.T),x) + np.dot(w,hl)) # hidden state
		hl=h
	for z in range(len(customer)-1, len(customer)):
		targets = customer[z]
		a=0
		for r in product_id:
			xt[r-1][0] = 1
			valuet = np.dot(np.dot(xt.T,t),h)
			xt = np.zeros((goods_size,1))
			allrank[a][0] = r
			allrank[a][1] = valuet
			a+=1
		allrank.sort(key=lambda x:x[1])
			 
		for b in targets:
			for s in range(top):

				if b == allrank[len(product_id)-s-1][0]:
					right += 1
					break
	return right

while True:
	itert += 1
	#Train
	print "This is iter %d"%itert
	time0=time.clock()
	for i in range(len(listcust)-1):
		customer = data[listcust[i]]
		user = np.zeros((1,hidden_size))
		for j in range(hidden_size):
			user[0][j] = u[i][j]
		loss, du, dt=lossfunction(customer,user)
		
	
	





