__author__ = 'lizk'
import numpy as np
import random
import math
import xlrd
import pickle

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
	if not(i%100):
		print i
f1 = open("F:\\pythonPro\\MachineLearning\\Tmalluser_cart.txt", "wb")
pickle.dump(all_cart, f1)
f1.close()