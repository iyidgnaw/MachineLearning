__author__ = 'lizk'
import numpy as np
import random
import math
import xlrd

data = xlrd.open_workbook('t_alibaba_data.xlsx')
table = data.sheets()[0]
origin_user_id = list(set(table.col_values(0)))
origin_product_id = list(set(table.col_values(1)))
user_size, product_size = len(origin_user_id), len(origin_product_id)
print 'data has %d users, %d products.' % (user_size, product_size)
user_ori2re = { ch:i for i,ch in enumerate(origin_user_id) }
user_list = { i:ch for i,ch in enumerate(origin_user_id) }
product_ori2re = { ch:i for i,ch in enumerate(origin_product_id) }
product_list = { i:ch for i,ch in enumerate(origin_product_id) }
print user_list
print product_list


learning_rate = 0.1
lamda=0.01
hidden_size = 10
u=np.random.randn(hidden_size, hidden_size)*0.5
v=np.random.randn(hidden_size, product_size)*0.5
w=np.random.randn(hidden_size, hidden_size)*0.5
x=np.random.randn(hidden_size, hidden_size)*0.5


def train(user_cart,u,v,w):





    return du,dv,dw,dus,h

for iter in range(1000):
    print "Iter %d"%iter
    print "Training..."
    hiddensave=[]
    for i in range(len(all_cart)):
        user_cart=all_cart[i]
        du,dv,dw,dus,h=train(user_cart)
        for param, dparam in zip([u, w, v,user_cart],[du, dw, dv,dus]):
	 			param += -learning_rate * dparam
        all_cart[i]=user_cart
        hiddensave.append(h)















