# -*- coding: utf-8 -*
__author__ = 'lizk'
import numpy as np
import random
import json
import pickle
import sys

all_cart = []
data = open('./data1.json', 'r')
lines = data.readlines()
for line in lines:
	line1 = json.loads(line)
	all_cart.append(line1)
def pre(all_cart):
	dictiontrain = {}
	dictiontest = {}
	for i in xrange(len(all_cart)):
			user_cart = all_cart[i]
			if len(user_cart)<10:
				continue
			user_cart_train = user_cart[0:int(0.8*len(user_cart))]
			user_cart_test = user_cart[int(0.8*len(user_cart)):len(user_cart)]
			dictiontest[i]=user_cart_test
			dictiontrain[i]=user_cart_train
	return dictiontest,dictiontrain
dictiontest,dictiontrain = pre(all_cart)
relevant = 0.0
hit = 0.0
for i in dictiontest.keys():
	for j in range(len(dictiontest[i])-1):
		relevant += 1
		if dictiontest[i][j] ==dictiontest[i][j+1] :
			hit +=1

value = hit / relevant
print value
