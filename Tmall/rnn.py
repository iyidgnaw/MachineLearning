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
v=np.random.randn(product_size, hidden_size)*0.01
w=np.random.randn(hidden_size, hidden_size)*0.01
x=np.random.randn(product_size, hidden_size)*0.01
hprev = np.zeros((1, 10))
print v

def sigmoid(x):
	output = 1/(1+np.exp(-x))
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

def train(user_cart,u ,v ,w):

	hl = np.copy(hprev)
	user_neg = negative(user_cart)
	i = 0
	loss = 0
	for item_id in user_cart:
		item = x[item_id-1,:].reshape(1,10)
		hid_input = np.dot(item, u)+ np.dot(hl, w)
		h = sigmoid(hid_input)
		item_neg_id = user_neg[i]
		i += 1
		Vi_j = v[item_id-1,:].reshape(1,10).T - v[item_neg_id-1,:].reshape(1,10).T
		Xij = np.dot(h, Vi_j)

		if Xij>10:
			Xij = 10
		elif Xij<-10:
			Xij = -10
		loss += Xij                                    #plus Regulation
		dXij = -(1-sigmoid(Xij))
		dh = dXij*(v[item_id-1,:].reshape(1,10) - v[item_neg_id-1,:].reshape(1,10))
		db = dh*sigmoid(hid_input)*(1-sigmoid(hid_input))
		dx = np.dot(db, u.T) + lamda*x[item_id-1, :].reshape(1, 10)
		dvi = dXij*h + lamda*v[item_id-1,:].reshape(1,10)

		dvj = -dXij*h + lamda*v[item_neg_id-1,:].reshape(1,10)
		du = np.dot(x[item_id-1,:].reshape(1,10).T, db) + lamda*u
		dw = np.dot(hl.T, db) + lamda*w

		for dparam in [dx, dvi, dvj, du, dw]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		x[item_id-1] -= learning_rate * dx[0]
		v[item_id-1] -= learning_rate * dvi[0]
		v[item_neg_id-1] -= learning_rate * dvj[0]
		u -= learning_rate * du
		w -= learning_rate * dw
		hl = h

	return u,v,w,x,h,loss


def predict(all_cart,u ,v ,w):
	print u, w

	reat1 = 0.0
	reat2 = 0.0
	reat5 = 0.0
	reat10 = 0.0
	relevant = 0.0
	for n in range(len(all_cart)):
		user_cart=all_cart[n]
		i = 0
		hl = np.copy(hprev)





		# for item_id in user_cart:
		# 	item = x[item_id-1]
		# 	hid_input = np.dot(item, u)+ np.dot(hl, w)
		# 	# print "hid_input"
		# 	# print hid_input
		# 	h = sigmoid(hid_input)
		# 	i += 1
		# 	hl = h
		# 	# print "h"
		# 	# print h
		# 	if i>int(len(user_cart)*0.8):
		# 		break
		# # print "i=%d"%i






		h=np.copy(hprev)
		for j in range(i, len(user_cart)-1):
			relevant += 1
			predict_matrix = np.dot(h, v.T)
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
		try:
			user_cart = user_cart[0:int(0.8*len(user_cart))]
			u,v,w,x,h,loss=train(user_cart,u ,v ,w)
			sumloss+=loss
		except:
			continue
	print sumloss
	predict(all_cart,u ,v ,w)
















