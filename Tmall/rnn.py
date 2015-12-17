__author__ = 'lizk'
import numpy as np
import random
import math
import xlrd

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















