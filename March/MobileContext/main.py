# -*- coding: utf-8 -*
__author__ = 'lizk'
import numpy as np
import random
import json
import pickle
import sys

old_settings = np.seterr(all='print')

f1 = open("timeInDay","r")     # 映射到每一天的7个时间段
timeInDay = pickle.load(f1)
f1.close()

all_cart = []
data = open('user_cart.json', 'r') #[itemid, hour, weekday]
lines = data.readlines()
for line in lines:
	line1 = json.loads(line)
	all_cart.append(line1)


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
for line in open("mobile_short.csv"):
	userid, itemid, month, day, hour = line.split(",")
	userid = int(userid)
	itemid = int(itemid)
	i += 1
	user_list.append(int(userid))
	itemid_list.append(int(itemid))


user_id = list(set(user_list))
product_id = list(set(itemid_list))
user_size = len(user_id)
product_size = len(product_id)
print user_size, product_size
learning_rate = 0.01
lamda_pos = 0.001
# lamda = 0.001
# lamda_unique =0.001
lamda = 0.001
lamda_unique = 0.001
hidden_size = 10
#tensor:hidden_size*hidden_size*time_size
interval_types = 7
neg_num = 1
w = np.random.randn(hidden_size, hidden_size)*0.5
x = np.random.randn(product_size, hidden_size)*0.5
hprev = np.zeros((1, hidden_size))
Udlist = []
Uwlist = []
for i in range (7):
	ud = np.random.randn(hidden_size, hidden_size)*0.5
	uw = np.random.randn(hidden_size, hidden_size)*0.5
	Udlist.append(ud)
	Uwlist.append(uw)


def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output

def negative(user_cart):
	negtargets = {}
	for item in user_cart:
		negtarget=[]
		while True:
			negtarget=random.sample(allrecord, neg_num)
			if item != negtarget:
				negtargets[item] = negtarget
				break
	return negtargets




def train(user_cart, daytime_cart, weektime_cart, x, Udlist,Uwlist ,w):

	hl = np.copy(hprev)
	dhlist=[]
	# hits=0
	hiddenlist=[]
	midlist=[]#sigmoid(bi)*(1-sigmoid(bi))
	sumdUd=[]
	sumdUw=[]
	sumdw=0
	for i in range(7):
		sumdUd.append(0)
		sumdUw.append(0)

#BPR
	dh1= np.copy(hprev)#dh for the back process
	user_neg = negative(user_cart)#dictioanry/negative samples for each id in user_cart
	loss = 0
	for i in xrange(len(user_cart)-1):

		#neglist=user_neg[user_cart[i]]  ->  i+1
		neg=user_neg[user_cart[i+1]][0]#list for negative samples for the id
		item = x[user_cart[i+1]-1,:].reshape(1,hidden_size)#positive sample's vector
		item1= x[user_cart[i]-1,:].reshape(1,hidden_size)#current input vector
		neg_item =  x[neg-1,:].reshape(1,hidden_size)
		hiddenlist.append(hl)
		daytime_typenext=timeInDay[daytime_cart[i+1]]
		daytime_typenow=timeInDay[daytime_cart[i]]
		Utd = Udlist[daytime_typenow]
		Utw = Uwlist[weektime_cart[i]]
		b = np.dot(item1, np.dot(Utd,Utw))+ np.dot(hl, w)
		np.clip(b, -15,15, out=b)
		mid=sigmoid(b)*(1-sigmoid(b))
		midlist.append(mid)

		h = sigmoid(b)
		Uid = Udlist[daytime_typenext]
		Uiw = Uwlist[weektime_cart[i+1]]
		Xi_j = np.dot(np.dot(Uid,Uiw),item.T-neg_item.T)
		Xij = np.dot(h, Xi_j)


		if Xij>10:
			Xij = 10
		elif Xij<-10:
			Xij = -10
		loss+=Xij
		
		dLx = -(1-sigmoid(Xij))
		
		dneg=np.dot(-dLx*h,np.dot(Uid,Uiw))
		np.clip(dneg, -5, 5, out=dneg)
		x[neg-1,:] += -learning_rate*(dneg.reshape(hidden_size,)+lamda*x[neg-1,:])

		ditem=np.dot(dLx*h,np.dot(Uid,Uiw))+lamda_pos*item
		x[user_cart[i+1]-1,:] += -learning_rate*(ditem.reshape(hidden_size,))

		dUid = np.dot(np.dot((dLx*h.T),(item-neg_item)),Uiw.T)
		sumdUd[daytime_typenext] +=-learning_rate*(dUid+lamda*Uid)

		dUiw = dLx*np.dot(np.dot(h,Uid).T,(item-neg_item))
		sumdUw[daytime_typenext] +=-learning_rate*(dUiw+lamda*Uiw)
		
		hl = h
		dhlist.append(np.dot(-(1-sigmoid(Xij))*(item-neg_item),np.dot(Uid, Uiw).T))#save the dh for each bpr step

#BPTT
	for i in range(len(user_cart)-1)[::-1]:
		item = x[user_cart[i]-1,:].reshape(1,hidden_size)
		daytime_typenow = timeInDay[daytime_cart[i]]
		Utd = Udlist[daytime_typenow]
		Utw = Uwlist[weektime_cart[i]]
		hnminus1=hiddenlist[i]
		dh=dhlist[i]+dh1

		sumdw+= np.dot(hnminus1.T,dh*midlist[i])+lamda*w

		dUtd = np.dot(np.dot(item.T,dh*midlist[i]),Utw.T)
		sumdUd[daytime_typenow] += -learning_rate*(dUtd+lamda*Utd)

		dUtw = np.dot(np.dot(item,Utd).T,dh*midlist[i])
		sumdUw[weektime_cart[i]] += -learning_rate*(dUtw+lamda*Utw)

		dx=np.dot(dh*midlist[i],np.dot(Utd,Utw))
		x[user_cart[i]-1,:] += -learning_rate*(dx.reshape(hidden_size,)+lamda_pos*x[user_cart[i]-1,:])

		dh1 = np.dot(dh*midlist[i],w.T)
	
	w += -learning_rate*sumdw
	for i in range(7):
		Udlist[i]+=sumdUd[i]
		Uwlist[i]+=sumdUw[i]
	return w,x,loss, Udlist,Uwlist



def predict(all_cart,allresult):
	relevant = 0.0
	for i in range(20):
		hit[i+1] = 0
		recall[i+1] = 0
	for n in xrange(len(all_cart)):
		behavior_list = all_cart[n]
		user_cart = []
		daytime_cart = []
		weektime_cart = []
		for behavior in behavior_list:
			user_cart.append(behavior[0])
			daytime_cart.append(behavior[1])
			weektime_cart.append(behavior[2])
		if len(user_cart)<10:
			continue
		i = 0
		hl = np.copy(hprev)
		for item_id in user_cart:
			interval_typenow = timeInDay[daytime_cart[i]]
			item1 = x[item_id-1]
			Utd =Udlist[interval_typenow]
			Utw = Uwlist[weektime_cart[i]]
			b = np.dot(item1, np.dot(Utd,Utw))+ np.dot(hl, w)
			np.clip(b, -15, 15, out=b)
			h = sigmoid(b)
			i += 1
			hl = h
			if i>int(len(user_cart)*0.8):
				break
		for j in xrange(i,len(user_cart)-1):

			relevant += 1
			item1=x[user_cart[j]-1]
			interval_typenow = timeInDay[daytime_cart[j]]
			Utd =Udlist[interval_typenow]
			Utw = Uwlist[weektime_cart[i]]
			b = np.dot(item1, np.dot(Utd,Utw))+ np.dot(hl, w)
			h = sigmoid(b)
			interval_typenext = timeInDay[daytime_cart[j+1]]
			Uid = Udlist[interval_typenext]
			Uiw = Uwlist[weektime_cart[i+1]]
			predict_matrix = np.dot(np.dot(h,np.dot(Uid,Uiw)),x.T)
			rank_index = np.argsort(predict_matrix, axis=1)
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
print "learningrate = %f"%learning_rate
print "lamda=%f"%lamda
iter = 0
while True:
	allresult=[]
	# f_handler=open('result_001-0001.txt','a')
	# sys.stdout=f_handler
	print "Iter %d"%iter
	print "Training..."
	sumloss=0
	for i in xrange(len(all_cart)):
		user_cart = []
		daytime_cart = []
		weektime_cart = []
		behavior_list = all_cart[i]
		if len(behavior_list)<10:
			continue
		behavior_list = behavior_list[0:int(0.8*len(behavior_list))]
		for behavior in behavior_list:
			user_cart.append(behavior[0])
			daytime_cart.append(behavior[1])
			weektime_cart.append(behavior[2])
		w,x,loss, Udlist,Uwlist=train(user_cart, daytime_cart, weektime_cart, x, Udlist,Uwlist ,w)
		sumloss+=loss
	print "begin predict"
	print sumloss

	predict(all_cart,allresult)
	# f_handler.close()
	iter += 1


