__author__ = 'lizk'
import numpy as np
import random
import math
import xlrd
import json
import time
from collections import Counter

all_cart = []
data = open('user_cart.json', 'r')
lines = data.readlines()
for line in lines:
	line1 = json.loads(line)
	all_cart.append(line1)
data1 = xlrd.open_workbook('data.xlsx')
data1 = data1.sheets()[0]
user_id = list(set(data1.col_values(0)))
product_id = list(set(data1.col_values(1)))
user_size, product_size = len(user_id), len(product_id)
print user_size, product_size
learning_rate = 0.01
lamda = 0.01
hidden_size = 10
u = np.random.randn(hidden_size, hidden_size)*0.5
w = np.random.randn(hidden_size, hidden_size)*0.5
x = np.random.randn(product_size, hidden_size)*0.5
hprev = np.zeros((1, 10))

def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output



#return a list including len(user_cart) negative items
def negative(user_cart):
	negtargets = {}
	list2 = product_id
	for item in user_cart:
		negtarget=[]
		negtarget=random.sample(list2, 50)
		negtargets[item] = negtarget
	return negtargets

def detchange(x,y):
	suminx=len(x)*len(x[0])
	sumchange=0
	for i in xrange(len(x)):
		for j in xrange(len(x[i])):
			if x[i][j]!=y[i][j]:
				sumchange+=1

	print "changes:",sumchange
	return

def avg_negitem(negitem):
	total =np.zeros((10,1))
	for item in negitem:
		# print np.shape(total),np.shape(x[item-1])
		total += x[item-1].reshape(10,1)
	avg = total/50
	avg = avg.reshape(1,10)
	return avg

def train(user_cart,u ,x ,w):
	hl = np.copy(hprev)
	dhlist=[]
	hiddenlist=[]
	dxilist=[]
	dxilist.append(hprev)
	midlist=[]#sigmoid(bi)*(1-sigmoid(bi))
	sumdu= 0
	sumdw= 0
	dh1= np.copy(hprev)#dh for the back process
	user_neg = negative(user_cart)#dictioanry/negative samples for each id in user_cart
	loss = 0
	for i in xrange(len(user_cart)-1):
		neglist=user_neg[user_cart[i]]#list for negative samples for the id
		item = x[user_cart[i+1]-1,:].reshape(1,10)#positive sample's vector
		item1= x[user_cart[i]-1,:].reshape(1,10)#current input vector
		neg_item = avg_negitem(neglist)#the average vector for 50 negative sample
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


		for p in range(len(neglist)):#update the negative samples' vector
			dneg=(1-sigmoid(Xij))*h*0.02
			np.clip(dneg, -5, 5, out=dneg)
			x[neglist[p]-1,:]+=-learning_rate*(dneg.reshape(10,)+lamda*x[neglist[p]-1,:])


		ditem=-(1-sigmoid(Xij))*h+lamda*item
		np.clip(ditem, -5, 5, out=ditem)
		dxilist.append(ditem)
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
		dxilist[i]+=dx
		dh1=np.dot(dh*midlist[i],w.T)

	for dparam in [ sumdu, sumdw]:
		np.clip(dparam, -5, 5, out=dparam)
	u-=learning_rate*sumdu
	w-=learning_rate*sumdw
	for i in xrange(len(user_cart)):
		x[user_cart[i]-1,:]+=-learning_rate*(dxilist[i].reshape(10,)+lamda*x[user_cart[i]-1,:])
	return u,w,x,loss



def predict(all_cart,allresult):
	relevant = 0.0
	hit=0
	difference=0
	for n in xrange(len(all_cart)):
		user_cart=all_cart[n]
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
			# real=predict_matrix[0][user_cart[j+1]-1]

			# top1=np.sort(predict_matrix, axis=1)
			# top1=top1[0][-1]
			# difference+=real-top1
			# rank=predict_matrix.tolist()[0]

			rank_index = np.argsort(predict_matrix, axis=1) #ordered by row small->big return index
			rank_index = rank_index[:, -10:np.shape(rank_index)[1]]
			# print user_cart[j+1]
			# print rank_index[0]
			for k in list(rank_index[0]):
				allresult.append(k)
			if user_cart[j+1]-1 in list(rank_index[0]):
				hit+=1


	print difference
	print relevant
	print hit
	return

for iter in xrange(100):
	allresult=[]
	print "Iter %d"%iter
	print "Training..."
	sumloss=0
	for i in xrange(len(all_cart)):
		user_cart = all_cart[i]
		user_cart = user_cart[0:int(0.8*len(user_cart))]
		if len(user_cart)<10:
			continue
		u,w,x,loss=train(user_cart,u,x,w)
		sumloss+=loss

	print sumloss



	predict(all_cart,allresult)
	print Counter(allresult)



