__author__ = 'lizk'
import numpy as np
import random
import math
import xlrd

train_data = xlrd.open_workbook('train.xlsx')
test_data = xlrd.open_workbook('test.xlsx')
train_table = train_data.sheets()[0]
test_table = train_data.sheets()[0]
origin_user_id = list(set(table.col_values(0)))
origin_product_id = list(set(table.col_values(1)))
user_size, product_size = len(origin_user_id), len(origin_product_id)




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















