# -*- coding: utf-8 -*
__author__ = 'lizk'
import numpy as np
import random
import math

import xlrd
import json
import pickle
from collections import Counter
from mail import *
import sys
# f_handler=open('resultmobile_10.txt', 'w')
# sys.stdout=f_handler
old_settings = np.seterr(all='print')

f1 = open("timetointerval","r")     # 把时间间隔小时映射到人为分号的时间间隔
timetointerval = pickle.load(f1)
f1.close()

all_cart = []
data = open('user_cart.json', 'r')
lines = data.readlines()
for line in lines:
	line1 = json.loads(line)
	all_cart.append(line1)


# data1 = xlrd.open_workbook('musicRNN.csv')
# data1 = data1.sheets()[0]
user_list=[]
itemid_list = []
behavior_list = []
recall = {}
recallatx = {}
hit = {}
for i in range(20):
	hit[i+1] = 0
	recall[i+1] = 0
i=0
for line in open("mobile_time.csv"):
	userid, artid, month, day, hour, time_sub = line.split(",")
	userid = int(userid)
	artid = int(artid)
	time_sub = int(time_sub)
	i += 1
	user_list.append(int(userid))
	itemid_list.append(int(artid))


user_id = list(set(user_list))
product_id = list(set(itemid_list))

user_size = len(user_id)
product_size = len(product_id)
print user_size, product_size
learning_rate = 0.01
lamda_pos = 0.001
# lamda = 0.001
# lamda_unique = 0.001
lamda = 0.05
lamda_unique = 0.05
hidden_size = 10
#tensor:hidden_size*hidden_size*time_size
time_size = 5
interval_types = 11
neg_num = 1
u = np.random.randn(hidden_size, hidden_size)*0.5
# w = np.random.randn(hidden_size, hidden_size)*0.5
x = np.random.randn(product_size, hidden_size)*0.5
Tt =np.random.randn(time_size,hidden_size, hidden_size)*0.5
hprev = np.zeros((1, hidden_size))
time_interval = np.random.randn(interval_types, time_size)*0.5


def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output

#sum(tensor[i]*vector[i])  return a matrix
def tensor_mul(tensor, vector):
	product = 0
	for i in range(len(tensor)):
		product+=tensor[i]*vector[i].reshape(1,1)
	return product

def updateTm(Tt, dMi,time_interval):
	for i in range(len(Tt)):
		# if Tt[i][0][0]>1:
		# 	print Tt[i]
		Tt[i] += -learning_rate*(dMi*time_interval[i]+lamda*Tt[i])
		# print "Tt[i]"
		# print Tt[i]
	return Tt


# x[i] = sum(matrix*tensor[i])   return x  (a vector)
def mul_add(matrix, tensor):
	x = np.zeros((1,time_size))

	for i in range(len(tensor)):
		matrix2 = tensor[i]
		result_matrix = matrix2*matrix
		for j in range(len(result_matrix)):
			for k in range(len(result_matrix[0])):
				x[0][i] += result_matrix[j][k]

	return x




#return a list including len(user_cart) negative items
def negative(user_cart):
	negtargets = {}
	list2 = product_id
	for item in user_cart:
		negtarget=[]
		while True:
			negtarget=random.sample(allrecord, neg_num)
			if item != negtarget:
				negtargets[item] = negtarget
				break
	return negtargets

def detchange(x,y):
	suminx=len(x)*len(x[0])
	sumchange=0
	for i in xrange(len(x)):
		for j in xrange(len(x[i])):
			if x[i][j]!=y[i][j]:
				sumchange+=1

	print "changes:",sumchange
	return

def avg_negitem(negitem):
	total =np.zeros((hidden_size,1))
	for item in negitem:
		# print np.shape(total),np.shape(x[item-1])
		total += x[item-1].reshape(hidden_size,1)
	avg = total/neg_num
	avg = avg.reshape(1,hidden_size)
	return avg

def train(user_cart,time_cart,u ,x ,Tt, time_interval,thisisiter):

	hl = np.copy(hprev)
	dhlist=[]
	hiddenlist=[]
	dxilist=[]
	dxilist.append(hprev)
	midlist=[]#sigmoid(bi)*(1-sigmoid(bi))
	dMilist = []
	wlist = []
	sumdu= 0


	dh1= np.copy(hprev)#dh for the back process
	user_neg = negative(user_cart)#dictioanry/negative samples for each id in user_cart
	loss = 0
	for i in xrange(len(user_cart)-1):

		#neglist=user_neg[user_cart[i]]  ->  i+1
		neglist=user_neg[user_cart[i+1]]#list for negative samples for the id
		item = x[user_cart[i+1]-1,:].reshape(1,hidden_size)#positive sample's vector
		item1= x[user_cart[i]-1,:].reshape(1,hidden_size)#current input vector
		neg_item = avg_negitem(neglist)#the average vector for 50 negative sample
		hiddenlist.append(hl)

		interval_typenow=timetointerval[time_cart[i]]
		interval_typenext = timetointerval[time_cart[i+1]]
		w = tensor_mul(Tt, time_interval[interval_typenow,:].reshape(time_size,1))      #h->h now  相当于公式里的M(t)
		wlist.append(w)
		Mi = tensor_mul(Tt, time_interval[interval_typenext,:].reshape(time_size,1))    #h->h next 相当于公式里的Mi

		b = np.dot(item1, u)+ np.dot(hl, w)
		np.clip(b, -15,15, out=b)
		mid=sigmoid(b)*(1-sigmoid(b))
		midlist.append(mid)

		h = sigmoid(b)
		Xi_j = np.dot(Mi, item.T-neg_item.T)
		# Xi_j = np.dot(item,Mi).T - np.dot(neg_item,Mi).T
		Xij = np.dot(h, Xi_j)
		# print "Xij"
		# print Xij

		if Xij>10:
			Xij = 10
		elif Xij<-10:
			Xij = -10
		loss+=Xij


		for p in range(len(neglist)):#update the negative samples' vector
			dneg=np.dot((1-sigmoid(Xij))*h,Mi)/neg_num
			np.clip(dneg, -5, 5, out=dneg)
			x[neglist[p]-1,:] += -learning_rate*(dneg.reshape(hidden_size,)+lamda*x[neglist[p]-1,:])


		ditem=np.dot(-(1-sigmoid(Xij))*h,Mi)+lamda_pos*item
		# print "ditem"
		# print ditem
		x[user_cart[i+1]-1,:] += -learning_rate*(ditem.reshape(hidden_size,))
		# print "0"
		# print item.T
		# print neg_item.T
		# print "1"
		# print item.T-neg_item.T
		# print "2"
		# print h
		# print "-(1-sigmoid(Xij)"
		# print -(1-sigmoid(Xij))
		dMi = np.dot((-(1-sigmoid(Xij))*h.T),(item-neg_item))
		# dMi = np.dot((item.T-neg_item.T)*(-(1-sigmoid(Xij))),h)
		dMilist.append(dMi)
		# print "3"
		# print dMi

		# dTm = tensor_mul(dMi,time_interval[interval_typenext,:].reshape(time_size,1))
		#
		# Tt += dTm+lamda*Tt

		Tt = updateTm(Tt, dMi,time_interval[interval_typenext,:].reshape(time_size,1))
		dtime_interval = mul_add(dMi, Tt)
		#lamda for time_interval should be larger???
		left = dtime_interval.reshape(time_size,)
		right = (time_interval[interval_typenext,:]-1)/learning_rate-lamda*time_interval[interval_typenext,:]

		time_interval[interval_typenext,:] += -learning_rate*(dtime_interval.reshape(time_size,)+lamda*time_interval[interval_typenext,:])



		np.clip(ditem, -5, 5, out=ditem)
		dxilist.append(ditem)
		hl = h
		dhlist.append(np.dot(-(1-sigmoid(Xij))*(item-neg_item),Mi.T))#save the dh for each bpr step



#layer1


	for i in range(len(user_cart)-1)[::-1]:
		item = x[user_cart[i]-1,:].reshape(1,hidden_size)
		hnminus2=hiddenlist[i]
		dh=dhlist[i]+dh1
		# sumdu+=np.dot(item.T,dh*midlist[i])+lamda*u
		sumdu += np.dot(item.T,dh*midlist[i])+lamda*u
		dMt = np.dot(hnminus2.T,dh*midlist[i])
		interval_typenow=timetointerval[time_cart[i]]


		Tt = updateTm(Tt, dMt,time_interval[interval_typenow,:].reshape(time_size,1))

		dtime_interval = mul_add(dMt, Tt)
		time_interval[interval_typenow,:] += -learning_rate*(dtime_interval.reshape(time_size,)+lamda_unique*time_interval[interval_typenow,:])

		dx=np.dot(dh*midlist[i],u.T)
		# print dx
		np.clip(dx, -5, 5, out=dx)
		dxilist[i]+=dx

		# dh1=np.dot(dh*midlist[i],tensor_mul(Tt, time_interval[interval_typenow,:].reshape(time_size,1)).T)
		dh1 = np.dot(dh*midlist[i],wlist[i].T)
		x[user_cart[i]-1,:] += -learning_rate*(dx.reshape(hidden_size,)+lamda_pos*x[user_cart[i]-1,:])

		# for dparam in [ sumdu]:
		# 	np.clip(dparam, -5, 5, out=dparam)
	u += -learning_rate*sumdu




	# for i in xrange(len(user_cart)):
	# 	# freq=fre[user_cart[i]]
	# 	# learn=learning_rate/freq
	# 	x[user_cart[i]-1,:] += -learning_rate*(dxilist[i].reshape(10,)+lamda_pos*x[user_cart[i]-1,:])
	return u,x,loss,Tt, time_interval



def predict(all_cart,allresult):
	relevant = 0.0
	difference=0

	for i in range(20):
		hit[i+1] = 0
		recall[i+1] = 0
	for n in xrange(len(all_cart)):
		behavior_list = all_cart[n]
		user_cart = []
		time_cart = []
		for behavior in behavior_list:
			user_cart.append(behavior[0])
			time_cart.append(behavior[1])
		if len(user_cart)<10:
			continue
		i = 0
		hl = np.copy(hprev)
		for item_id in user_cart:
			interval_typenow = timetointerval[time_cart[i]]
			item = x[item_id-1]
			w = tensor_mul(Tt, time_interval[interval_typenow,:].reshape(time_size,1))
			b = np.dot(item, u)+ np.dot(hl, w)
			np.clip(b, -15, 15, out=b)
			h = sigmoid(b)
			i += 1
			hl = h
			if i>int(len(user_cart)*0.8):
				break
		for j in xrange(i,len(user_cart)-1):

			relevant += 1
			item=x[user_cart[j]-1]

			interval_typenow = timetointerval[time_cart[j]]
			w = tensor_mul(Tt, time_interval[interval_typenow,:].reshape(time_size,1))
			b = np.dot(item, u)+ np.dot(h, w)
			h = sigmoid(b)
			interval_typenext = timetointerval[time_cart[j+1]]
			Mnext = tensor_mul(Tt, time_interval[interval_typenext,:].reshape(time_size,1))
			predict_matrix = np.dot(h, np.dot(Mnext,x.T))

			rank_index = np.argsort(predict_matrix, axis=1) #ordered by row small->big return index
			rank_index = rank_index[:, -20:np.shape(rank_index)[1]]
			rank_index_list = list(reversed(list(rank_index[0])))
			for k in list(rank_index[0]):
				allresult.append(k)
			if user_cart[j+1]-1 in rank_index_list:
				index = rank_index_list.index(user_cart[j+1]-1)
				hit[index+1] += 1


	for i in range(20):
		for j in range(20-i):
			recall[20-j] += hit[i+1]
	for i in range(20):
		recallatx[i+1] = recall[i+1]/relevant

	print relevant
	print recall
	print recallatx
	return

allrecord=[]
for i in xrange(len(all_cart)):
	user_cart = all_cart[i]
	for behavior in user_cart:
		allrecord.append(behavior[0])
fre=Counter(allrecord)
print fre
print "learningrate = 0.01 lamda=0.001"
iter = 0
while True:
	allresult=[]
	print "Iter %d"%iter
	print "Training..."
	sumloss=0
	print "len(all_cart)"
	print len(all_cart)
	thisisiter=0
	for i in xrange(len(all_cart)):
		user_cart = []
		time_cart = []
		behavior_list = all_cart[i]
		if len(behavior_list)<10:
			continue
		behavior_list = behavior_list[0:int(0.8*len(behavior_list))]
		for behavior in behavior_list:
			user_cart.append(behavior[0])
			time_cart.append(behavior[1])
		u,x,loss,Tt, time_interval=train(user_cart, time_cart, u,x,Tt,time_interval, thisisiter)
		thisisiter += 1
		sumloss+=loss
		if i%100==0:
			print i
	print "begin predict"
	print sumloss

	predict(all_cart,allresult)
	# print Counter(allresult)
	iter += 1

# mail('./resultmobile_10.txt')
