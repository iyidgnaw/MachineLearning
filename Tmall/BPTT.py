__author__ = 'lizk'
import numpy as np
import random
import math
import xlrd
import json


all_cart=[]
data= open('user_cart.json', 'r')
lines=data.readlines()
for line in lines:
	line1=json.loads(line)
	all_cart.append(line1)
data1 = xlrd.open_workbook('data.xlsx')
data1 = data1.sheets()[0]
user_id = list(set(data1.col_values(0)))
product_id = list(set(data1.col_values(1)))
user_size, product_size = len(user_id), len(product_id)

learning_rate = 0.01
lamda=0.1
hidden_size = 10
u=np.random.randn(hidden_size, hidden_size)*0.01
p=np.random.randn(product_size, hidden_size)*0.01
w=np.random.randn(hidden_size, hidden_size)*0.01
x=np.random.randn(product_size, hidden_size)*0.01
hprev = np.zeros((1, 10))

def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output
def sigmoid_output_to_derivative(x):
	output = sigmoid(x)*(1-sigmoid(x))
	return output

#return a list including len(user_cart) negative items
def negative(user_cart):
	negtargets = []
	list2 = product_id
	negtargets=random.sample(list2, 2*len(user_cart))
	for m in list(set(user_cart)):
		negtargets = filter(lambda a: a != m, negtargets)
	negtargets = negtargets[0:len(user_cart)]
	return negtargets

def train(user_cart,x, u ,p ,w,userid):
	Xi, h, b, Xj, Xij = {}, {}, {}, {}, {}
	h[-1] = np.copy(hprev)
	user_neg = negative(user_cart)
	loss = 0
	for t in xrange(len(user_cart)):
		Xi[t] = x[user_cart[t]-1,:].reshape(1,10)
		b[t] = np.dot(Xi[t], u)+ np.dot(h[t-1], w)
		h[t] = sigmoid(b[t])
		Xj[t] = x[user_neg[t]-1,:].reshape(1,10)
		Xij[t] = np.dot((h[t]+p[userid]), ((Xi[t].T)-(Xj[t].T)))
		if Xij[t]>10:
			Xij[t] = 10
		elif Xij[t]<-10:
			Xij[t] = -10
		loss += Xij[t]

	dhnext = np.zeros_like(h[0])
	du, dw = np.zeros_like(u), np.zeros_like(w)
	for t in reversed(xrange(len(user_cart))):

		dXij = -(1-sigmoid(Xij[t]))
		dh = dXij*(Xi[t] - Xj[t]) + dhnext
		dp = dXij*(Xi[t] - Xj[t])
		#


		dxi = dXij*h[t]
		dxj = -dXij*h[t]

		du += np.dot(Xi[t].T, dh)*sigmoid_output_to_derivative(b[t])

		dw += np.dot(h[t-1].T, dh)*sigmoid_output_to_derivative(b[t])
		dxi += np.dot(dh*sigmoid_output_to_derivative(b[t]),u.T)




		dhnext = np.dot((dh * (sigmoid(b[t])*(1-sigmoid(b[t])))), w.T)
		for dparam in [dxi, dxj]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

		print "bbbbbbbb"
		# print Xi
		print user_cart[t]
		print x[user_cart[t]-1]
		print np.shape(Xi[t]),np.shape(x[user_cart[t]-1,1:10]),np.shape(x),np.shape(x[:,1])
		print x[1]
		x[user_cart[t]-1,1:10] += Xi[t]        #left
		print Xi[t]
		print "ddddd"

		x[user_cart[t]-1] -= learning_rate*(dxi +lamda*(Xi[t].reshape(1,10)))
		print "ccccccc"
		x[user_neg[t]-1,:] -= learning_rate*(dxj +lamda*x[user_neg[t]-1].reshape(1,10))
		print "aaaaaaaaa"
	for dparam in [du, dw, dp]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

	return loss, du, dw, dp, x



def predict(all_cart,u ,p ,w):
	# print u, w

	reat1 = 0.0
	reat2 = 0.0
	reat5 = 0.0
	reat10 = 0.0
	relevant = 0.0
	for n in range(len(all_cart)):
		user_cart=all_cart[n]
		i = 0
		hl = np.copy(hprev)
		userid = n





		for item_id in user_cart:
			item = x[item_id-1]
			hid_input = np.dot(item, u)+ np.dot(hl, w)
			# print "hid_input"
			# print hid_input
			h = sigmoid(hid_input)
			i += 1
			hl = h
			# print "h"
			# print h
			if i>int(len(user_cart)*0.8):
				break






		h=np.copy(hprev)
		for j in range(i, len(user_cart)-1):
			relevant += 1
			predict_matrix = np.dot(h+p[userid], x.T)
			item=x[user_cart[j]-1]
			hid_input = np.dot(item, u)+ np.dot(h, w)
			h = sigmoid(hid_input)
			# print "user_cart[j]-1 %d"%(user_cart[j]-1)
			# print "item"
			# print item
			# print "h"
			# print h
			# print "real_v"
			# print v[user_cart[j]]
			# print np.dot(h,v[user_cart[j+1]-1])

			rank_index = np.argsort(predict_matrix, axis=1) #ordered by row small->big return index
			rank_index = rank_index[:, -10:np.shape(rank_index)[1]]
			# print "max_v"
			# print v[rank_index[0][-1]]
			# print np.dot(h, v[rank_index[0][-1]])
			# print "ranklist"
			# print rank_index
			if rank_index[0][-1] == user_cart[j+1]:
				reat1 += 1
				reat2 += 1
				reat5 += 1
				reat10 += 1
				continue
			if rank_index[0][-2] == user_cart[j+1]:
				reat2 += 1
				reat5 += 1
				reat10 += 1
				continue
			for k in rank_index[0,-5:-2]:
				if k == user_cart[j+1]:
					reat5 += 1
					reat10 += 1
				
			for k in rank_index[0,-10:-5]:
				if k == user_cart[j+1]:
					reat10 += 1
				
	recall_1 = reat1/relevant
	recall_2 = reat2/relevant
	recall_5 = reat5/relevant
	recall_10 = reat10/relevant

	print recall_1, recall_2, recall_5, recall_10








for iter in range(1000):
	print "Iter %d"%iter
	print "Training..."
	sumloss=0
	hiddensave=[]
	for i in range(len(all_cart)):
		user_cart = all_cart[i]
		userid = i
		try:
			user_cart = user_cart[0:int(0.8*len(user_cart))]

			loss, du, dw, dp, x = train(user_cart,x, u ,p ,w,userid)

			print du, dw, dp, x
			u -= learning_rate*(du+lamda*u)
			w -= learning_rate*(dw+lamda*w)
			p -= learning_rate*(dw+lamda*w)
			sumloss+=loss
		except:
			continue
	print sumloss
	predict(all_cart,u ,p ,w)
















