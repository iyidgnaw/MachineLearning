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
user_re2ori = { i:ch for i,ch in enumerate(origin_user_id) }
product_ori2re = { ch:i for i,ch in enumerate(origin_product_id) }
product_re2ori = { i:ch for i,ch in enumerate(origin_product_id) }

alpha = 0.1
input_dim = 1
hidden_dim = 10
output_dim = 1





