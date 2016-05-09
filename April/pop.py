# -*- coding: utf-8 -*
__author__ = 'lizk'
import numpy as np
import random
import math
import json
from collections import Counter


USER_SIZE = 1904            
ITEM_SIZE = 1157            
TOP = 20
ITEM_LIST = []
TEST_SET = []
RECALL_MAX = {}     
for i in range(1, 21):
	RECALL_MAX[i] = 0

def load_file():        #load data function
	global TEST_SET,ITEM_LIST
	list1 = []
	train=open('./MF/bpr_train.txt','r')
	lines=train.readlines()
	for line in lines:
		record=line.split(' ')
		col=int (record[1])
		list1.append(col)
	result = Counter(list1)
	ITEM_LIST = [346,361,338,913,295,314,1,78,555,649,1110,1073,688,592,404,23,259,585,504,458]
	test = open('./MF/bpr_test.json','r')
	lines = test.readlines()
	for line in lines:
		line1 = json.loads(line)
		TEST_SET.append(line1)
	return 


def predict():
	global TEST_SET,ITEM_LIST
	print "Predicting..."
	relevant = 0.0          
	hit = {}                
	recall = {}             
	recallatx = {}          # RecallAtN/relevant
	
	for i in range(TOP):
		hit[i+1] = 0
		recall[i+1] = 0

	for i in range (1904):
		for item in TEST_SET[i]:
			relevant += 1
			if item in ITEM_LIST:
				index = ITEM_LIST.index(item)
				hit[index+1] += 1
	
	for i in range(20):
		for j in range(20-i):
			recall[20-j] += hit[i+1]
	for i in range(20):
		recallatx[i+1] = recall[i+1]/relevant

	print relevant
	print recall
	print recallatx
	return recall, recallatx

def save_max(result):
	'''
	result:list
	保存result[n]最大的result
	'''
	global  RECALL_MAX
	for i in range(1, 21) :
		RECALL_MAX[i] = max(RECALL_MAX[i], result[i])

	print "Best Result"
	print RECALL_MAX
	print

load_file()

recall, recallatx = predict()

	
	


