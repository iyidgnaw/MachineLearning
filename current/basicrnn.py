# -*- coding: utf-8 -*
__author__ = 'lizk'
import numpy as np
import random
import json
import sys
old_settings = np.seterr(all='print')

all_cart = []
data = open('subuser_cart.json', 'r')
lines = data.readlines()
for line in lines:
	line1 = json.loads(line)
	all_cart.append(line1)

user_list=[]
itemid_list = []
behavior_list = []
for line in open("submobile_time.csv"):
	userid, artid, month, day, hour, time_sub = line.split(",")
	userid = int(userid)
	artid = int(artid)
	time_sub = int(time_sub)
	user_list.append(int(userid))
	itemid_list.append(int(artid))




recall = {}
recallatx = {}
hit = {}
for i in range(20):
	hit[i+1] = 0
	recall[i+1] = 0

user_id = list(set(user_list))
product_id = list(set(itemid_list))
user_size = len(user_id)
product_size = len(product_id)
print user_size, product_size

learning_rate = 0.001
lamda_pos = 0.01
lamda = 0.01
lamda_unique = 0.01
hidden_size = 10
neg_num = 1
u = np.random.randn(hidden_size, hidden_size)*0.5
w = np.random.randn(hidden_size, hidden_size)*0.5
x = np.random.randn(product_size, hidden_size)*0.5
hprev = np.zeros((1, hidden_size))


def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output

def negative(user_cart):
	negtargets = {}
	for item in user_cart:
		negtarget=[]
		negtarget=random.sample(allrecord, neg_num)
		negtargets[item] = negtarget
	return negtargets



def train(user_cart,u ,x ,w):
	hl = np.copy(hprev)
	dhlist=[]
	hiddenlist=[]
	midlist=[]#sigmoid(bi)*(1-sigmoid(bi))
	sumdu= 0
	sumdw= 0
	dh1= np.copy(hprev)#dh for the back process
	user_neg = negative(user_cart)#dictioanry/negative samples for each id in user_cart
	loss = 0
	for i in xrange(len(user_cart)-1):
		neg=user_neg[user_cart[i+1]][0]#list for negative samples for the id
		item = x[user_cart[i+1]-1,:].reshape(1,10)#positive sample's vector
		item1= x[user_cart[i]-1,:].reshape(1,10)#current input vector
		neg_item =  x[neg-1,:].reshape(1,hidden_size)#the average vector for 50 negative sample
		hiddenlist.append(hl)

		b = np.dot(item1, u)+ np.dot(hl, w)
		np.clip(b, -15,15, out=b)
		mid=sigmoid(b)*(1-sigmoid(b))
		midlist.append(mid)

		h = sigmoid(b)
		Xi_j = item.T - neg_item.T
		Xij = np.dot(h, Xi_j)
		if Xij>10:
			Xij = 10
		elif Xij<-10:
			Xij = -10
		loss+=Xij

		dneg=(1-sigmoid(Xij))*h
		np.clip(dneg, -5, 5, out=dneg)
		x[neg-1,:]+=-learning_rate*(dneg.reshape(10,)+lamda*x[neg-1,:])

		ditem=-(1-sigmoid(Xij))*h+lamda_pos*item
		x[user_cart[i+1]-1,:] += -learning_rate*(ditem.reshape(10,))
		hl = h
		dhlist.append(-(1-sigmoid(Xij))*(item-neg_item))#save the dh for each bpr step
	for i in range(len(user_cart)-1)[::-1]:
		item= x[user_cart[i]-1,:].reshape(1,10)
		hnminus2=hiddenlist[i]
		dh=dhlist[i]+dh1
		sumdu+=np.dot(item.T,dh*midlist[i])+lamda*u
		sumdw+=np.dot(hnminus2.T,dh*midlist[i])+lamda*w
		dx=np.dot(dh*midlist[i],u.T)
		np.clip(dx, -5, 5, out=dx)
		dh1=np.dot(dh*midlist[i],w.T)
		x[user_cart[i]-1,:] += -learning_rate*(dx.reshape(10,)+lamda_pos*x[user_cart[i]-1,:])
	for dparam in [ sumdu, sumdw]:
		np.clip(dparam, -5, 5, out=dparam)
	u-=learning_rate*sumdu
	w-=learning_rate*sumdw
	return u,w,x,loss


def predict(all_cart,allresult):
	relevant = 0.0

	for i in xrange(20):
		hit[i+1] = 0
		recall[i+1] = 0
	for n in xrange(len(all_cart)):
		behavior_list = all_cart[n]
		user_cart = []
		for behavior in behavior_list:
			user_cart.append(behavior[0])
		if len(user_cart)<10:
			continue
		i = 0
		hl = np.copy(hprev)
		for item_id in user_cart:
			item = x[item_id-1]
			b = np.dot(item, u)+ np.dot(hl, w)
			np.clip(b, -15, 15, out=b)
			h = sigmoid(b)
			i += 1
			hl = h
			if i>int(len(user_cart)*0.8):
				break
		for j in xrange(i,len(user_cart)-1):

			relevant += 1
			item=x[user_cart[j]-1]
			b = np.dot(item, u)+ np.dot(h, w)
			h = sigmoid(b)
			predict_matrix = np.dot(h, x.T)
			rank_index = np.argsort(predict_matrix, axis=1) #ordered by row small->big return index
			rank_index = rank_index[:, -20:np.shape(rank_index)[1]]
			rank_index_list = list(reversed(list(rank_index[0])))
			for k in list(rank_index[0]):
				allresult.append(k)
			if user_cart[j+1]-1 in rank_index_list:
				index = rank_index_list.index(user_cart[j+1]-1)
				hit[index+1] += 1


	for i in xrange(20):
		for j in xrange(20-i):
			recall[20-j] += hit[i+1]
	for i in xrange(20):
		recallatx[i+1] = recall[i+1]/relevant

	print relevant
	print recall
	print recallatx
	return


allrecord=[]
for i in xrange(len(all_cart)):
	user_cart = all_cart[i]
	for behavior in user_cart:
		allrecord.append(behavior[0])
iter = 0
while True:
	allresult=[]
	f_handler=open("resultbasic.txt",'a')
	sys.stdout=f_handler
	print "Iter %d"%iter
	print "Training..."
	sumloss=0
	for i in xrange(len(all_cart)):
		user_cart = []
		behavior_list = all_cart[i]
		if len(behavior_list)<10:
			continue
		behavior_list = behavior_list[0:int(0.8*len(behavior_list))]
		for behavior in behavior_list:
			user_cart.append(behavior[0])
		u,w,x,loss=train(user_cart,u,x,w)
		sumloss+=loss
	print "begin predict"
	print sumloss

	predict(all_cart,allresult)
	f_handler.close()
	iter += 1


