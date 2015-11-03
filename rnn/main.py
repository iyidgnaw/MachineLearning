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
product_id=product_id[0:500]
f1.close()
print len(listcust)



hidden_size = 10
learning_rate = 1e-3
goods_size = 1559 
itert = 0
top = 50
bias = 0
# model parameters
u = np.random.randn(hidden_size, hidden_size)*0.5 # input to hidden
w = np.random.randn(hidden_size, hidden_size)*0.5 # hidden to hidden
t = np.random.randn(goods_size, hidden_size)*0.5 # one-hot to embedding


def sigmoid(x):                  #sigmoid function
	return 1.0/(1+np.exp(-x))

def lossFun(inputs, targets, negtargets, hprev,itert)                    :#loss function    everybasket
	loss = 0
	midh = 0
	midt = 0
	hl = np.copy(hprev)
	x = np.zeros((goods_size,1)) 
	bias=50/len(targets)

  # forward pass
	for k in inputs:
		x[k-1][0] = 1
	h = sigmoid(np.dot(np.dot(u,t.T),x) + np.dot(w,hl)) # hidden state
	for n in targets:                 #calculate the loss
		xt = np.zeros((goods_size,1))
		xt[n-1][0] = 1
		loss += bias*np.log(sigmoid(np.dot(np.dot(xt.T,t),h)))
		midh += bias*(1-sigmoid(np.dot(np.dot(xt.T,t),h)))*np.dot(xt.T,t).T
		midt += bias*(1-sigmoid(np.dot(np.dot(xt.T, t), h))) * np.dot(xt,h.T)

	for g in negtargets:
		xn = np.zeros((goods_size,1))
		xn[g-1][0] = 1
		loss += np.log(1 - sigmoid(np.dot(np.dot(xn.T,t),h)))
		midh -= sigmoid(np.dot(np.dot(xn.T, t), h))*np.dot(xn.T,t).T
		midt -= sigmoid(np.dot(np.dot(xn.T, t), h)) * np.dot(xn,h.T)

	du, dw, dt= np.zeros_like(u), np.zeros_like(w), np.zeros_like(t)
	dw = np.dot(midh*h*(1-h),hl.T)
	du = np.dot(midh*h*(1-h), np.dot(t.T,x).T)         #x how to choose   x x+1
	dt += np.dot(np.dot(u.T,midh*h*(1-h)),x.T).T+midt
	hl = h

	return loss, du, dw, dt, hl

def negasamp(targets):
	negtargets = []
	list2 = product_id
	negtargets=random.sample(list2, 80)
	for m in targets:
		negtargets = filter(lambda a: a != m, negtargets)
	negtargets = negtargets[0:50]
	return negtargets

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
	basketnum=0
	avrloss=0
	preloss=0
	print "This is iter %d"%itert
	time0=time.clock()
	for i in range(len(listcust)-1):
		customer = data[listcust[i]]
		if i%500==0:
			print "Training customer %d"%i
		hprev = np.zeros((hidden_size, 1))
		for j in range(len(customer)-1):
			inputs = customer[j]
			targets = customer[j+1]
			negtargets = negasamp(targets)
			basketnum+=1
			loss, du, dw, dt, hprev= lossFun(inputs, targets, negtargets, hprev,itert)
			avrloss+=loss
	
			for param, dparam in zip([u, w, t],[du, dw, dt]):
				param += learning_rate * dparam # adagrad update
	avrloss=avrloss/basketnum
	if avrloss>preloss:
		learning_rate=learning_rate*1.02
	else:
		learning_rate=learning_rate*0.98
	preloss=avrloss
	print "The average loss is :%f"%avrloss
	time1=time.clock()
	print "Training cost :%f second"%(time1-time0)



	if itert%1==0:
	#predict/test
		rightpredict=0
		for p in range(len(listcust)-1):	
			customer = data[listcust[p]]
			rightpredict += predict(customer, u, w, t)
			
		time2=time.clock()
		print "Total right is :%d"%rightpredict
		print "Predict cost  %f seconds"%(time2-time1)


















