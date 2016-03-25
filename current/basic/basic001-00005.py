# -*- coding: utf-8 -*
__author__ = 'lizk'
import numpy as np
import random
import json
import pickle
import sys

old_settings = np.seterr(all='print')



all_cart = []
data = open('../user_cart.json', 'r')

lines = data.readlines()
for line in lines:
	line1 = json.loads(line)
	all_cart.append(line1)


user_list=[]
itemid_list = []
behavior_list = []
recall = {}
recallatx = {}
hit = {}
for i in range(20):
	hit[i+1] = 0
	recall[i+1] = 0
i=0
for line in open("../mobile_time.csv"):
	userid, artid, month, day, hour, time_sub = line.split(",")
	userid = int(userid)
	artid = int(artid)
	time_sub = int(time_sub)
	i += 1
	user_list.append(int(userid))
	itemid_list.append(int(artid))


user_id = list(set(user_list))
product_id = list(set(itemid_list))
user_size = len(user_id)
product_size = len(product_id)
print user_size, product_size
learning_rate = 0.05
lamda_pos = 0.001
# lamda = 0.001
# lamda_unique =0.001
lamda = 0.001
lamda_unique = 0.001
hidden_size = 10
#tensor:hidden_size*hidden_size*time_size
neg_num = 1
u = np.random.randn(hidden_size, hidden_size)*0.5
x = np.random.randn(product_size, hidden_size)*0.5
hprev = np.zeros((1, hidden_size))
w = np.random.randn(hidden_size, hidden_size)*0.5



def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output

def negative(user_cart):
	negtargets = {}
	for item in user_cart:
		negtarget=[]
		while True:
			negtarget=random.sample(allrecord, neg_num)
			if item != negtarget:
				negtargets[item] = negtarget
				break
	return negtargets

def pre(all_cart):
	dictiontrain = {}
	dictiontest = {}
	for i in xrange(len(all_cart)):
			user_cart_train = []
			user_cart_test = []
			behavior_list = all_cart[i]
			if len(behavior_list)<10:
				continue
			behavior_train = behavior_list[0:int(0.8*len(behavior_list))]
			behavior_test = behavior_list[int(0.8*len(behavior_list)):len(behavior_list)]
			for behavior in behavior_train:
				user_cart_train.append(behavior[0])
			for behavior in behavior_test:
				user_cart_test.append(behavior[0])
			dictiontest[i]=user_cart_test
			dictiontrain[i]=user_cart_train
	return dictiontest,dictiontrain


def train(user_cart,u ,x ,w):

	hl = np.copy(hprev)
	dhlist=[]
	hiddenlist=[]
	midlist=[]#sigmoid(bi)*(1-sigmoid(bi))
	sumdu= 0
	sumdw= 0

#BPR
	dh1= np.copy(hprev)#dh for the back process
	user_neg = negative(user_cart)#dictioanry/negative samples for each id in user_cart
	loss = 0
	for i in xrange(len(user_cart)-1):

		#neglist=user_neg[user_cart[i]]  ->  i+1
		neg=user_neg[user_cart[i+1]][0]#list for negative samples for the id
		item = x[user_cart[i+1]-1,:].reshape(1,hidden_size)#positive sample's vector
		item1= x[user_cart[i]-1,:].reshape(1,hidden_size)#current input vector
		neg_item =  x[neg-1,:].reshape(1,hidden_size)
		hiddenlist.append(hl)
		b = np.dot(item1, u)+ np.dot(hl, w)
		np.clip(b, -15,15, out=b)
		mid=sigmoid(b)*(1-sigmoid(b))
		midlist.append(mid)

		h = sigmoid(b)

		Xi_j = item.T-neg_item.T

		Xij = np.dot(h, Xi_j)


		if Xij>10:
			Xij = 10
		elif Xij<-10:
			Xij = -10
		loss+=Xij

		dneg=(1-sigmoid(Xij))*h
		np.clip(dneg, -5, 5, out=dneg)
		x[neg-1,:] += -learning_rate*(dneg.reshape(hidden_size,)+lamda*x[neg-1,:])

		ditem=-(1-sigmoid(Xij))*h+lamda_pos*item
		x[user_cart[i+1]-1,:] += -learning_rate*(ditem.reshape(hidden_size,))


		hl = h
		dhlist.append(-(1-sigmoid(Xij))*(item-neg_item))#save the dh for each bpr step

#BPTT
	for i in range(len(user_cart)-1)[::-1]:
		item = x[user_cart[i]-1,:].reshape(1,hidden_size)
		hnminus2=hiddenlist[i]
		dh=dhlist[i]+dh1

		sumdu += np.dot(item.T,dh*midlist[i])+lamda*u

		sumdw += np.dot(hnminus2.T,dh*midlist[i])+lamda*w

		dx=np.dot(dh*midlist[i],u.T)
		x[user_cart[i]-1,:] += -learning_rate*(dx.reshape(hidden_size,)+lamda_pos*x[user_cart[i]-1,:])

		dh1 = np.dot(dh*midlist[i],w.T)
	u += -learning_rate*sumdu
	w += -learning_rate*sumdw
	return u,x,loss,w



def predict(dictiontrain,dictiontest,allresult):
	relevant = 0.0
	for i in range(20):
		hit[i+1] = 0
		recall[i+1] = 0
	# num = 0
	for n in dictiontest.keys():
		train = dictiontrain[n]
		test = dictiontest[n]
		hl = np.copy(hprev)
		for item_id in train:
			item = x[item_id-1]
			b = np.dot(item, u)+ np.dot(hl, w)
			np.clip(b, -15, 15, out=b)
			h = sigmoid(b)
			hl = h
		for j in xrange(len(test)-1):

			relevant += 1
			item=x[test[j]-1]
			b = np.dot(item, u)+ np.dot(h, w)
			h = sigmoid(b)
			predict_matrix = np.dot(h,x.T)
			rank = np.argpartition(predict_matrix[0],-20)[-20:]
			rank = rank[np.argsort(predict_matrix[0][rank])]
			rank_index_list = list(reversed(list(rank)))
			if test[j+1]-1 in rank_index_list:
				index = rank_index_list.index(test[j+1]-1)
				hit[index+1] += 1
		# num +=1
		# if num%100==0:
		# 	print num

	for i in range(20):
		for j in range(20-i):
			recall[20-j] += hit[i+1]
	for i in range(20):
		recallatx[i+1] = recall[i+1]/relevant

	print relevant
	print recall
	print recallatx
	return recall,recallatx


def savefunction(learning_rate,lamda,u,w,x,recall,recallatx):
	result = open('result.txt','a')
	hint1 = "learningrate = %f"%learning_rate
	hint2 = "lamda=%f"%lamda
	list1 = [hint1,hint2,recall,recallatx,u,w,x]
	pickle.dump(list1,result)
	result.close()
	return 


allrecord=[]
for i in xrange(len(all_cart)):
	user_cart = all_cart[i]
	for behavior in user_cart:
		allrecord.append(behavior[0])
print "learningrate = %f"%learning_rate
print "lamda=%f"%lamda
iter = 0
dictiontest,dictiontrain = pre(all_cart)

while (iter<150):
	allresult=[]
	f_handler = open('result0005-0001.txt','a')
	sys.stdout=f_handler
	print "Iter %d"%iter
	print "Training..."
	sumloss=0
	# num = 0
	for i in dictiontrain.keys():
		user_cart = dictiontrain[i]
		u,x,loss,w=train(user_cart,u,x,w)
		sumloss+=loss
		# num +=1
		# if num%100==0:
		# 	print num
	print "begin predict"
	print sumloss

	recall,recallatx = predict(dictiontrain,dictiontest,allresult)
	f_handler.close()
	iter += 1
	if iter%1==0:
		savefunction(learning_rate,lamda,u,w,x,recall,recallatx)

