__author__ = 'lizk'
import numpy as np
import random
import math
import xlrd

<<<<<<< HEAD
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

#return a list including len(user_cart) negative items
def negative(user_cart):
	negtargets = []
	list2 = product_id
	negtargets=random.sample(list2, 80)
	for m in user_cart:
		negtargets = filter(lambda a: a != m, negtargets)
	negtargets = negtargets[0:len(user_cart)]
	return negtargets


for iter in range(1000):
	print "Iter %d"%iter
	print "Training..."
	hiddensave=[]
	for i in range(len(all_cart)):
		user_cart = all_cart[i]
		user_cart = user_cart[0:int(0.8*len(user_cart))]
		u,v,w,x,h,loss=train(user_cart,u ,v ,w)
=======
data = xlrd.open_workbook('data.xlsx')

data = data.sheets()[0]
user_data = list(set(data.col_values(0)))
item_data = list(set(data.col_values(1)))
user_size, product_size = len(user_data), len(item_data)
previous=1.0
all_cart=[]
user_cart=[]
print type(data.col_values(0)[0])
print user_size,product_size
print type(previous)
for i in range(len(data.col_values(1))):
    if data.col_values(0)[i]!=previous:
        all_cart.append(user_cart)
        previous=data.col_values(0)[i]
        user_cart=[]
    else:
        user_cart.append(data.col_values(1)[i])
print all_cart[0]




# learning_rate = 0.1
# lamda=0.01
# hidden_size = 10
# u=np.random.randn(hidden_size, hidden_size)*0.5
# v=np.random.randn(hidden_size, product_size)*0.5
# w=np.random.randn(hidden_size, hidden_size)*0.5
# x=np.random.randn(hidden_size, hidden_size)*0.5
#
#
# def train(user_cart,u,v,w):
#
#
#
#
#
#     return du,dv,dw,dus,h
#
# for iter in range(1000):
#     print "Iter %d"%iter
#     print "Training..."
#     hiddensave=[]
#     for i in range(len(all_cart)):
#         user_cart=all_cart[i]
#         du,dv,dw,dus,h=train(user_cart)
#         for param, dparam in zip([u, w, v,user_cart],[du, dw, dv,dus]):
# 	 			param += -learning_rate * dparam
#         all_cart[i]=user_cart
#         hiddensave.append(h)
>>>>>>> f7aeecb6654f488d3c6462486c7c93cc3f6431fc















