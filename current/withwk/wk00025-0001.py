import json
import pickle
import sys

old_settings = np.seterr(all='print')

f1 = open("../timetointerval","r")     # 把时间间隔小时映射到人为分号的时间间隔
timetointerval = pickle.load(f1)
f1.close()

all_cart = []
data = open('../user_cart.json', 'r')
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
for line in open("../mobile_time.csv"):
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
# lamda_unique =0.001
lamda = 0.001
lamda_unique = 0.001
hidden_size = 10
interval_types = 11
neg_num = 1
u = np.random.randn(hidden_size, hidden_size)*0.5
x = np.random.randn(product_size, hidden_size)*0.5
hprev = np.zeros((1, hidden_size))
time_interval = []
for i in range (interval_types):
	w = np.random.randn(hidden_size, hidden_size)*0.5
	time_interval.append(w)


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
def pre(all_cart):
	dictiontrain = {}
	dictiontest = {}
	for i in xrange(len(all_cart)):
			user_cart_train = []
			time_cart_train = []
			user_cart_test = []
			time_cart_test = []
			behavior_list = all_cart[i]
			if len(behavior_list)<10:
				continue
			behavior_train = behavior_list[0:int(0.8*len(behavior_list))]
			behavior_test = behavior_list[int(0.8*len(behavior_list)):len(behavior_list)]

			for behavior in behavior_train:
				user_cart_train.append(behavior[0])
				time_cart_train.append(behavior[1])
			for behavior in behavior_test:
				user_cart_test.append(behavior[0])
				time_cart_test.append(behavior[1])
			dictiontest[i]=[user_cart_test,time_cart_test]
			dictiontrain[i]=[user_cart_train,time_cart_train]
	return dictiontest,dictiontrain



def train(user_cart,time_cart,u ,x ,time_interval):

	hl = np.copy(hprev)
	dhlist=[]
	hiddenlist=[]
	midlist=[]#sigmoid(bi)*(1-sigmoid(bi))
	sumdu= 0
	sumdw=[]
	for i in range(11):
		sumdw.append(0)

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
		interval_typenext=timetointerval[time_cart[i+1]]
		interval_typenow=timetointerval[time_cart[i]]
		Wp = time_interval[interval_typenow]
		Wk = time_interval[interval_typenext]
		b = np.dot(item1, u)+ np.dot(hl, Wp)
		np.clip(b, -15,15, out=b)
		mid=sigmoid(b)*(1-sigmoid(b))
		midlist.append(mid)

		h = sigmoid(b)
		
		Xi_j = np.dot(Wk,item.T-neg_item.T)
		Xij = np.dot(h, Xi_j)


		if Xij>10:
			Xij = 10
		elif Xij<-10:
			Xij = -10
		loss+=Xij

		dneg=np.dot((1-sigmoid(Xij))*h,Wk)
		np.clip(dneg, -5, 5, out=dneg)
		x[neg-1,:] += -learning_rate*(dneg.reshape(hidden_size,)+lamda*x[neg-1,:])

		ditem=np.dot(-(1-sigmoid(Xij))*h,Wk)+lamda_pos*item
		x[user_cart[i+1]-1,:] += -learning_rate*(ditem.reshape(hidden_size,))

		dWk = np.dot((-(1-sigmoid(Xij))*h.T),(item-neg_item))
		sumdw[interval_typenext] +=-learning_rate*(dWk+lamda*Wk)

		hl = h
		dhlist.append(np.dot(-(1-sigmoid(Xij))*(item-neg_item),Wk.T))#save the dh for each bpr step

#BPTT
	for i in range(len(user_cart)-1)[::-1]:
		item = x[user_cart[i]-1,:].reshape(1,hidden_size)
		hnminus2=hiddenlist[i]
		dh=dhlist[i]+dh1

		sumdu += np.dot(item.T,dh*midlist[i])+lamda*u

		dWp = np.dot(hnminus2.T,dh*midlist[i])
		interval_typenow = timetointerval[time_cart[i]]
		Wp = time_interval[interval_typenow]
		sumdw[interval_typenow] += -learning_rate*(dWp+lamda*Wp)

		dx=np.dot(dh*midlist[i],u.T)
		x[user_cart[i]-1,:] += -learning_rate*(dx.reshape(hidden_size,)+lamda_pos*x[user_cart[i]-1,:])

		dh1 = np.dot(dh*midlist[i],Wp.T)
	u += -learning_rate*sumdu
	for i in range(11):
		time_interval[i]+=sumdw[i]
	return u,x,loss, time_interval



def predict(dictiontrain,dictiontest,allresult):
	relevant = 0.0
	for i in range(20):
		hit[i+1] = 0
		recall[i+1] = 0
	for n in dictiontest.keys():
		train_user_cart = dictiontrain[n][0]
		train_time_cart = dictiontrain[n][1]
		test_user_cart = dictiontest[n][0]
		test_time_cart = dictiontest[n][1]
		i = 0
		hl = np.copy(hprev)
		for item_id in train_user_cart:
			interval_typenow = timetointerval[train_time_cart[i]]
			item = x[item_id-1]
			w =time_interval[interval_typenow]
			b = np.dot(item, u)+ np.dot(hl, w)
			np.clip(b, -15, 15, out=b)
			h = sigmoid(b)
			i += 1
			hl = h
			
		for j in xrange(len(test_user_cart)-1):

			relevant += 1
			item=x[test_user_cart[j]-1]
			interval_typenow = timetointerval[test_time_cart[j]]
			w = time_interval[interval_typenow]
			b = np.dot(item, u)+ np.dot(h, w)
			h = sigmoid(b)
			interval_typenext = timetointerval[test_time_cart[j+1]]
			Wk = time_interval[interval_typenext]
			predict_matrix = np.dot(h,np.dot(Wk,x.T))
			rank = np.argpartition(predict_matrix[0],-20)[-20:]
			rank = rank[np.argsort(predict_matrix[0][rank])]
			rank_index_list = list(reversed(list(rank)))
			if test_user_cart[j+1]-1 in rank_index_list:
				index = rank_index_list.index(test_user_cart[j+1]-1)
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
dictiontest,dictiontrain = pre(all_cart)

while True:
	allresult=[]	
	f_handler = open('result001-0001.txt','a')
	sys.stdout=f_handler	
	print "Iter %d"%iter
	print "Training..."
	sumloss=0
	for i in dictiontrain.keys():
		user_cart = dictiontrain[i][0]
		time_cart = dictiontrain[i][1]
		u,x,loss, time_interval=train(user_cart, time_cart, u,x,time_interval)
		sumloss+=loss

	print "begin predict"
	print sumloss

	predict(dictiontrain,dictiontest,allresult)
	f_handler.close()


	iter += 1
