# -*- coding: utf-8 -*
__author__ = 'lizk'
import numpy as np
import random
import math
import json
USER_SIZE = 1904			
ITEM_SIZE = 1157			
HIDDEN_SIZE = 10
TOP = 20
LEARNING_RATE = 0.1 			
LAMBDA = 0.01 			
TEST_SET = []
TRAIN_MATRIX=np.zeros((USER_SIZE,ITEM_SIZE))
USER_MATRIX=np.random.randn(USER_SIZE, HIDDEN_SIZE)*0.5
ITEM_MATRIX=np.random.randn(ITEM_SIZE, HIDDEN_SIZE)*0.5
RECALL_MAX = {}		
for i in range(1, 21):
	RECALL_MAX[i] = 0

def load_file():		#load data function
	global TRAIN_MATRIX,TEST_SET
	train=open('bpr_train.txt','r')
	lines=train.readlines()
	for line in lines:
		record=line.split(' ')
		row=int(record[0])
		col=int (record[1])
		TRAIN_MATRIX[row-1][col-1]=1
	test = open('bpr_test.json','r')
	lines = test.readlines()
	for line in lines:
		line1 = json.loads(line)
		TEST_SET.append(line1)
	return 

	



def train():
	global USER_MATRIX,ITEM_MATRIX
	loss=0
	for i in range(USER_SIZE):						
		for j in range(ITEM_SIZE):       			
			if TRAIN_MATRIX[i][j]==1:
				negy =  random.randint(0, ITEM_SIZE-1)
				while (negy == j):
					negy =  random.randint(0, ITEM_SIZE-1)
				Xij = np.dot(USER_MATRIX[i],(ITEM_MATRIX[j]- ITEM_MATRIX[negy]))
				loss+=Xij				
				mid=-np.exp(-Xij)/(1+np.exp(-Xij))
				tmp_user = USER_MATRIX[i]
				USER_MATRIX[i] += -LEARNING_RATE*(mid*(ITEM_MATRIX[j]-ITEM_MATRIX[negy])+LAMBDA*USER_MATRIX[i])
				ITEM_MATRIX[j] += -LEARNING_RATE*(mid*USER_MATRIX[i]+LAMBDA*ITEM_MATRIX[j])
				ITEM_MATRIX[negy] += -LEARNING_RATE*(-mid*USER_MATRIX[i]+LAMBDA*ITEM_MATRIX[negy])
	print "Loss of this iter is :"
	print loss

	return USER_MATRIX, ITEM_MATRIX

def predict():
	global USER_MATRIX,ITEM_MATRIX,TEST_SET
	print "Predicting..."
	relevant = 0.0 			
	hit = {}				
	recall = {}				
	recallatx = {}			# RecallAtN/relevant
	
	for i in range(TOP):
		hit[i+1] = 0
		recall[i+1] = 0

	RATING_MATRIX = np.dot(USER_MATRIX,ITEM_MATRIX.T)
	for i in range(len(RATING_MATRIX)):
		rank = np.argpartition(RATING_MATRIX[i], -TOP)[-TOP:]
		rank = rank[np.argsort(RATING_MATRIX[i][rank])]
		rank_index_list = list(reversed(list(rank)))
		for item in TEST_SET[i]:
			relevant += 1
			if item-1 in rank_index_list:
				index = rank_index_list.index(item-1)
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
for i in range(100):
	print "iter %i"%i
	train()
	recall, recallatx = predict()
	save_max(recallatx)
	
	


