__author__ = 'lizk'
import numpy as np
import random
import math
import xlrd

train_data = xlrd.open_workbook('train.xlsx')
test_data = xlrd.open_workbook('test.xlsx')
train_table = train_data.sheets()[0]
test_table = train_data.sheets()[0]
user_id = list(set(train_table.col_values(0)))
product_id = list(set(train_table.col_values(1)))
user_size, product_size = len(user_id), len(product_id)

learning_rate = 0.1
lamda=0.01
hidden_size = 10
u=np.random.randn(hidden_size, hidden_size)*0.5
v=np.random.randn(hidden_size, product_size)*0.5
w=np.random.randn(hidden_size, hidden_size)*0.5
x=np.random.randn(product_size, hidden_size)*0.5
hprev = np.zeros((hidden_size, 1))

def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output

#return a list including len(user_cart) negative items
def negative(user_cart):
	negtargets = []
	list2 = product_id
	negtargets=random.sample(list2, 80)
	for m in user_cart:
		negtargets = filter(lambda a: a != m, negtargets)
	negtargets = negtargets[0:len(user_cart)]
	return negtargets

def train(user_cart,u ,v ,w):
	hl = np.copy(hprev)
	user_neg = negative(user_cart)
	i = 0
	loss = 0
	for item_id in user_cart:
		item = x[item_id]
		hid_input = np.dot(item, u)+ np.dot(hl, w)
		h = sigmoid(hid_input)
		item_neg_id = user_neg[i]
		i += 1
		Vi_j = v[:, item_id] - v[:, item_neg_id]
		Xij = np.dot(h, Vi_j)
		loss += Xij                                    #plus Regulation
		dXij = -(1-sigmoid(Xij))
		dh = dXij*Vi_j
		db = dXij*dh*sigmoid(hid_input)*(1-sigmoid(hid_input))

		dx = np.dot(db,u)+lamda*np.abs(x[item_id])
		dvi = np.dot(h, dXij) + lamda*np.abs(v[item_id])
		dvj = -np.dot(h, dXij) + lamda*np.abs(v[item_neg_id])
		du = np.dot(x[item_id], db) + lamda*np.abs(u)
		dw = np.dot(h, db) + lamda*np.abs(w)

		x[item_id] -= learning_rate * dx
		v[item_id] -= learning_rate * dvi
		v[item_neg_id] -= learning_rate * dvj
		u -= learning_rate * du
		w -= learning_rate * dw
		hl = h
	return u,v,w,x,h,loss


def predict(all_cart):
	reat1 = 0
	reat2 = 0
	reat5 = 0
	reat10 = 0
	relevant = 0
	for user_cart in all_cart:
		i = 0
		hl = np.copy(hprev)
		for item_id in user_cart:
			item = x[item_id]
			hid_input = np.dot(item, u)+ np.dot(hl, w)
			h = sigmoid(hid_input)
			i += 1
			hl = h
			if i>int(len(user_cart)*0.8):
				break
		for j in range(i, len(user_cart)):
			relevant += 1
			predict_matrix = np.dot(h, v)
			rank_index = np.argsort(predict_matrix, axis=1) #ordered by row small->big return index
			rank_index = rank_index[:, -10:np.shape(rank_index)[1]]
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
			for i in rank_index[:,-5:-2]:
				reat5 += 1
				reat10 += 1
				continue
			if i in rank_index[:,-10:-5]:
				reat10 += 1
				continue
	recall_1 = reat1/relevant
	recall_2 = reat2/relevant
	recall_5 = reat5/relevant
	recall_10 = reat10/relevant

	print recall_1, recall_2, recall_5, recall_10








for iter in range(1000):
	print "Iter %d"%iter
	print "Training..."
	hiddensave=[]
	for i in range(len(all_cart)):
		user_cart = all_cart[i]
		user_cart = user_cart[0:int(0.8*len(user_cart))]
		u,v,w,x,h,loss=train(user_cart,u ,v ,w)


	predict(all_cart)
















