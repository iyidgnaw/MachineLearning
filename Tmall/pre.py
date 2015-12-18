__author__ = 'lizk'
import numpy as np
import random
import math
import xlrd
import json

data = xlrd.open_workbook('data.xlsx')

data = data.sheets()[0]
user_data = list(set(data.col_values(0)))
item_data = list(set(data.col_values(1)))
user_size, product_size = len(user_data), len(item_data)
previous=1.0
user_cart=[]
for i in range(len(data.col_values(1))):
	if i%1000==0:
		print "user %d"%previous
	if data.col_values(0)[i]!=previous:
		json.dump(user_cart, open('user_cart.json', 'a'))
		fp=open('user_cart.json','a')
		fp.write('\n')
		fp.close()
		previous=data.col_values(0)[i]
		user_cart=[]
	else:
		user_cart.append(data.col_values(1)[i])

