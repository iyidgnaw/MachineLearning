__author__ = 'wangdiyi'
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import pickle
import random
import time


ISOTIMEFORMAT='%Y-%m-%d %X'
print "Start time is :"
print time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) ) 
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



# hyperparameters
hidden_size = 5# size of hidden layer of neurons
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
	total=0
	mid = 0
	midn = 0
	midt = 0
	right=0
	valuetrue=0
	valuepredict=0
	hl = np.copy(hprev)
	x = np.zeros((goods_size,1)) 
	xt= np.zeros((goods_size,1)) # encode in 1-of-k representation
	bias=50/len(targets)

  # forward pass
	for i in inputs:
		x[i-1][0] = 1
	h = sigmoid(np.dot(np.dot(u,t.T),x) + np.dot(w,hl)) # hidden state
	if itert%5==0:
		allrank = [[0]*2 for row in range(len(product_id))]
		a=0
		for i in product_id:
			xt[i-1][0] = 1
			valuet = np.dot(np.dot(xt.T,t),h)
			xt = np.zeros((goods_size,1))
			allrank[a][0] = i
			allrank[a][1] = valuet
			a+=1
		allrank.sort(key=lambda x:x[1])       
		for i in targets:
			for j in range(top):
				if i == allrank[len(product_id)-j-1][0]:
					right += 1
					break
		for i in targets:
			xt[i-1][0] = 1
			valuet = np.dot(np.dot(xt.T,t),h)
			valuetrue+=valuet
			xt = np.zeros((goods_size,1))
		valuetrue=valuetrue/len(targets)
		for j in range(len(targets)):
			valuepredict+=allrank[len(product_id)-j-1][1]
		valuepredict=valuepredict/len(targets)
	for i in targets:                 #calculate the loss
		xt = np.zeros((goods_size,1))
		xt[i-1][0] = 1
		loss += bias*np.log(sigmoid(np.dot(np.dot(xt.T,t),h)))
	for i in negtargets:
		xn = np.zeros((goods_size,1))
		xn[i-1][0] = 1
		loss += np.log(1 - sigmoid(np.dot(np.dot(xn.T,t),h)))


	du, dw, dt = np.zeros_like(u), np.zeros_like(w), np.zeros_like(t)



	for i in targets:                   #loss to hide
		xt = np.zeros((goods_size,1))
		xt[i-1][0] = 1
		# mid += 1-sigmoid(np.dot((np.dot(np.dot(x.T, t), h)), np.dot(x.T,t))).T
		mid += bias*(1-sigmoid(np.dot(np.dot(xt.T,t),h)))*np.dot(xt.T,t).T
		midt += bias*(1-sigmoid(np.dot(np.dot(xt.T, t), h))) * np.dot(xt,h.T)

		# print np.shape(mid)  #1, 1559
	for i in negtargets:
		xn= np.zeros((goods_size,1))
		xn[i-1][0] = 1
		mid -= sigmoid(np.dot(np.dot(xn.T, t), h))*np.dot(xn.T,t).T
		midn += sigmoid(np.dot(np.dot(xn.T, t), h)) * np.dot(xn,h.T)
	dw = np.dot(mid*h*(1-h),hl.T)
	du = np.dot(mid*h*(1-h), np.dot(t.T,x).T)         #x how to choose   x x+1
	dt += np.dot(np.dot(u.T,mid*h*(1-h)),x.T).T

	dt +=midt-midn
	# dt = np.dot(np.dot((1 - sigmoid(np.dot(np.dot(x.T,t), h))),x),h.T) + midn + \
	#      np.dot(np.dot(mid.T*hl*(1-hl), u.T), x.T)
	hl = h
	if itert%5==0:
		total=len(targets)
	return loss, du, dw, dt, hl,total,right,valuetrue,valuepredict


def negasamp(targets):
	negtargets = []
	list2 = product_id
	negtargets=random.sample(list2, 80)
	for i in targets:
		negtargets = filter(lambda a: a != i, negtargets)
	negtargets = negtargets[0:50]
	return negtargets




def predict(customer, u, w, t):

	right = 0
	hl = np.zeros((hidden_size, 1))
	x = np.zeros((goods_size,1)) # encode in 1-of-k representation
	xt = np.zeros((goods_size,1)) # encode in 1-of-k representation
	# rank = np.zeros((20,2))
	allrank = [[0]*2 for row in range(len(product_id))]

	for j in range(len(customer)-1):
		inputs = customer[j]
		for i in inputs:
			x[i-1][0] = 1
		h = sigmoid(np.dot(np.dot(u,t.T),x) + np.dot(w,hl)) 
		hl=h# hidden state
	for j in range(len(customer)-1, len(customer)):
		targets = customer[j]
		a=0
		for i in product_id:
			xt[i-1][0] = 1
			valuet = np.dot(np.dot(xt.T,t),h)
			xt = np.zeros((goods_size,1))
			allrank[a][0] = i
			allrank[a][1] = valuet
			a+=1
		allrank.sort(key=lambda x:x[1])
		     
		for i in targets:
			for j in range(top):

				if i == allrank[len(product_id)-j-1][0]:
					right += 1
					
					break
	
	
		
	return right

while True:
	right = 0
	rightpredict=0
	itert += 1
	basketnum=0
	total=0
	avrloss=0
	preloss=0
	valuetruemid=0
	valuepredictmid=0
	valuetrue=0
	valuepredict=0
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
			loss, du, dw, dt, hprev ,totalmid,rightmid,valuetruemid,valuepredictmid= lossFun(inputs, targets, negtargets, hprev,itert)
			avrloss+=loss
			valuepredict+=valuepredictmid
			valuetrue+=valuetruemid
			total+=totalmid
			right+=rightmid
			for param, dparam in zip([u, w, t],[du, dw, dt]):
				param += learning_rate * dparam # adagrad update
	avrloss=avrloss/basketnum
	valuetrue=valuetrue/basketnum
	valuepredict=valuepredict/basketnum
	if avrloss>preloss:
		learning_rate=learning_rate*1.05
	else:
		learning_rate=learning_rate*0.95
	preloss=avrloss
	print "The average loss is :%f"%avrloss
	if itert%5==0:
		print total,right
		print "The average value of next basket on trainset is : %f"% valuetrue
		print "The average value of predict basket on trainset is : %f"% valuepredict  
	time1=time.clock()
	print "Training cost :%f second"%(time1-time0)
	if itert%1==0:
		for p in range(len(listcust)-1):
			customer = data[listcust[p]]
			rightmid = predict(customer, u, w, t)
			rightpredict+=rightmid
	strright=str(rightpredict)+" "
	result=open("result.txt", "a")
	result.write(strright)
	pickle.dump(u,open("resultu.txt", "w"))
	pickle.dump(w,open("resultw.txt", "w"))
	pickle.dump(t,open("resultt.txt", "w"))
	time2=time.clock()
	print "Total right is :%d"%rightpredict
	print "Predict cost  %f seconds"%(time2-time1)







