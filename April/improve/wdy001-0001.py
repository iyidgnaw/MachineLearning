# -*- coding: utf-8 -*
__author__ = 'lizk'
import numpy as np
import random
import json
import pickle
import sys
import time

USER_SIZE = 1904			# 总用户数
ITEM_SIZE = 1157			# 总商品种数
HIDDEN_SIZE = 10			# hidden layer的维度
LEARNING_RATE = 0.01 		# 学习速率
LAMBDA = 0.001 				# 惩罚系数
TOP = 20 					# recall取前Top个

# Random Initiation
U = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE)*0.5
X = np.random.randn(ITEM_SIZE, HIDDEN_SIZE)*0.5
WPLIST = []
for i in range(5):
	w = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE)*0.5
	WPLIST.append(w)
WKLIST = []
for i in range(5):
	w = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE)*0.5
	WKLIST.append(w)

# # Initiate from files
# FILE = open('bestparameter.txt','rb')
# PARA = pickle.load(FILE)
# FILE.close()
# U = PARA[1]
# WKLIST = PARA[2]
# WPLIST = PARA[3]
# X = PARA[4]


H_ZERO = np.zeros((1, HIDDEN_SIZE))

DATAFILE = '../data/user_cart_delta.json'
Pastrecall = {}
Pastrecall[10]=0
NEG_NUM = 20
ITEM_TRAIN = {}
ITEM_TEST = {}
SPLIT = 0.9


def sigmoid(x):

	output = 1.0/(1.0+np.exp(-x))
	return output


def pre_data():
	"""
	读取数据 初始化ITEM_TRAIN ITEM_TEST
	"""

	global ITEM_TRAIN
	global ITEM_TEST
	global SPLIT
	global DATAFILE

	all_cart = []
	data = open(DATAFILE, 'r')
	lines = data.readlines()
	for line in lines:
		line1 = json.loads(line)
		all_cart.append(line1)

	for i in xrange(len(all_cart)):
			item_train = []
			time_train = []
			item_test = []
			time_test = []
			behavior_list = all_cart[i]
			behavior_train = behavior_list[0:int(SPLIT*len(behavior_list))]
			behavior_test = behavior_list[int(SPLIT*len(behavior_list)):]
			for behavior in behavior_train:
				item_train.append(behavior[0])		# behavior中第一个位置为商品编号
				time_train.append(behavior[1])
			for behavior in behavior_test:
				item_test.append(behavior[0])
				time_test.append(behavior[1])
			ITEM_TRAIN[i] = [item_train,time_train]
			ITEM_TEST[i] = [item_test,time_test]




def train(user_cart,time_cart):
	global U, WKLIST,WPLIST, X

	dhlist = []							# bpr中对h的导数
	hiddenlist = []						# 记录[1,T]状态hidden layer (不包括1的上一个状态的hidden layer)
	midlist = []							# BPTT中传到第一层的导数 sigmoid(bi)*(1-sigmoid(bi))
	hl = np.copy(H_ZERO)				# 初始化last hidden layer	
	sumdu = 0 							# 记录对于每一个用户BPTT中u、w总	更新量
	sumdwp = []
	for i in range(5):
		sumdwp.append(0)
	loss = 0
	# BPR
	dh1 = np.copy(H_ZERO)					# dh for the back process
	for i in xrange(len(user_cart)-1):
		# 对于要预测的item进行负采样
		# itemlist = range(ITEM_SIZE)
		# neglist = random.sample(itemlist,NEG_NUM)
		# item_neg = avg_neg(neglist)
		neg = random.randint(1, ITEM_SIZE)
		while user_cart[i+1] == neg:
			neg = random.randint(1, ITEM_SIZE)

		item_pos = X[user_cart[i+1]-1, :].reshape(1, HIDDEN_SIZE)		# positive sample's vector
		item_curt = X[user_cart[i]-1, :].reshape(1, HIDDEN_SIZE)		# current input vector
		item_neg = X[neg-1, :].reshape(1, HIDDEN_SIZE)			# negative sample's vector
		Wp=WPLIST[time_cart[i]]
		Wk=WKLIST[time_cart[i+1]]

		# 计算状态t的h、dh
		b = np.dot(item_curt, U) + np.dot(hl, Wp)
		h = sigmoid(b)
		



		xi_j = np.dot(Wk,(item_pos.T-item_neg.T))
		xij = np.dot(h, xi_j)
		loss += xij
		# 若为tmp = sigmoid(-Xij) 则LEARNING_RATE和LAMBDA为负
		tmp = -(1 - sigmoid(xij))

		hiddenlist.append(h)
		mid = h * (1 - h)
		midlist.append(mid)	
		dhlist.append(np.dot(tmp * (item_pos - item_neg),Wk.T))		# save the dh for each bpr step

		# 计算对于负样本的导数 并更新负样本的vector
		dneg = np.dot(-tmp * h,Wk) + LAMBDA * item_neg
		X[neg-1, :] += -LEARNING_RATE * (dneg.reshape(HIDDEN_SIZE, ))
		
		# 计算对于正样本的导数 并更新正样本的vector
		ditem = np.dot(tmp * h,Wk) + LAMBDA * item_pos
		X[user_cart[i+1]-1, :] += -LEARNING_RATE * (ditem.reshape(HIDDEN_SIZE,))

		dWk = np.dot(tmp*h.T,(item_pos - item_neg))
		WKLIST[time_cart[i+1]] += -LEARNING_RATE*(dWk+LAMBDA*Wk)	
		# 更新last hidden layer
		hl = h

	# BPTT
	for i in range(len(user_cart) - 1)[::-1]:
		item = X[user_cart[i] - 1, :].reshape(1, HIDDEN_SIZE)
		hnminus2 = hiddenlist[i]
		dh = dhlist[i] + dh1
		Wp=WPLIST[time_cart[i]]

		sumdu += np.dot(item.T, dh * midlist[i])
		dWp = np.dot(hnminus2.T,dh*midlist[i])
		sumdwp[time_cart[i]] +=dWp
		# 更新输入的样本
		dx = np.dot(dh * midlist[i], U.T)
		X[user_cart[i]-1, :] += -LEARNING_RATE*(dx.reshape(HIDDEN_SIZE, ) + LAMBDA * X[user_cart[i]-1, :])

		dh1 = np.dot(dh * midlist[i], Wp.T)
	U += -LEARNING_RATE * (sumdu + LAMBDA * U)
	for i in range(5):
		WPLIST[i]+= -LEARNING_RATE*(sumdwp[i]+LAMBDA*WPLIST[i])
	return loss


def predict():
	relevant = 0.0 			# 所预测的总次数
	hit = {}				# 第n个位置所命中的个数
	recall = {}				# 前n个位置所命中的总数
	recallatx = {}			# RecallAtN/relevant
	
	for i in range(TOP):
		hit[i+1] = 0
		recall[i+1] = 0
	
	for n in ITEM_TEST.keys():
		train = ITEM_TRAIN[n][0]
		train_time = ITEM_TRAIN[n][1]
		test = ITEM_TEST[n][0]
		test_time = ITEM_TEST[n][1]
		hl = np.copy(H_ZERO)
		h = np.copy(H_ZERO)
		# 计算需要预测的状态对应的hidden layer
		
		for i in range(len(train)):
			item = X[train[i]-1]
			w = WPLIST[train_time[i]]
			b = np.dot(item, U) + np.dot(hl, w)
			h = sigmoid(b)
			hl = h
		# 预测
		for j in xrange(len(test)):
			relevant += 1
			Wk = WKLIST[test_time[j]]
			predict_matrix = np.dot(h,np.dot(Wk, X.T))
			rank = np.argpartition(predict_matrix[0], -TOP)[-TOP:]
			rank = rank[np.argsort(predict_matrix[0][rank])]
			rank_index_list = list(reversed(list(rank)))

			if test[j]-1 in rank_index_list:
				index = rank_index_list.index(test[j]-1)
				hit[index+1] += 1

			item = X[test[j] - 1]
			Wp = WPLIST[test_time[j]]
			b = np.dot(item, U) + np.dot(h, Wp)
			h = sigmoid(b)

	for i in range(20):
		for j in range(20-i):
			recall[20-j] += hit[i+1]
	for i in range(20):
		recallatx[i+1] = recall[i+1]/relevant

	print relevant
	print recall
	print recallatx
	return recall





def basic_info():
	print "LEARNING_RATE = %f" % LEARNING_RATE
	print "LAMBDA = %f" % LAMBDA


def learn():
	global Pastrecall
	ite = 0
	while (ite<=800):
		f_handler = open('4_7_result_001-0001.txt','a')
		sys.stdout=f_handler	
		print "Iter %d" % ite
		print "Training..."
		sumloss = 0
		for i in ITEM_TRAIN.keys():
			user_cart = ITEM_TRAIN[i][0]
			time_cart =ITEM_TRAIN[i][1]
			loss = train(user_cart,time_cart)
			sumloss += loss
		print "begin predict"
		print sumloss
		recall = predict()
		# if recall[10]>Pastrecall[10]:
		# 	result = open('bestparameter.txt','w')
		# 	list1 = [recall,U,WKLIST,WPLIST,X]
		# 	pickle.dump(list1,result)
		# 	result.close()
		# 	Pastrecall = recall
		f_handler.close()
		ite += 1


def main():

	basic_info()
	pre_data()
	learn()


if __name__ == '__main__':
	main()
