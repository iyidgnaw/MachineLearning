__author__ = 'wangdiyi'
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import pickle
import random


#read
f1 = open("./data_NextBasket.txt", "rb")
data = pickle.load(f1)
f1.close()
# data.get(3)
f1 = open("./data_idList.txt", "rb")
listfre = pickle.load(f1)
f1.close()
f1 = open("./data_custmTrain.txt", "rb")
listcust = pickle.load(f1)
f1.close()
print len(listcust)

# hyperparameters
hidden_size = 20 # size of hidden layer of neurons
learning_rate = 1e-1
goods_size = 1559

# model parameters
u = np.random.randn(hidden_size, hidden_size)*0.01 # input to hidden
w = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
t = np.random.randn(goods_size, hidden_size)*0.01 # one-hot to embedding


def sigmoid(x):                  #sigmoid function
    return 1.0/(1+np.exp(-x))


def lossFun(inputs, targets, negtargets, hprev)                    :#loss function    everybasket
    loss = 0
    mid = 0
    midn = 0
    midt = 0
    hl = np.copy(hprev)
    x = np.zeros((goods_size,1)) # encode in 1-of-k representation


  # forward pass
    for i in inputs:
        x[i-1][0] = 1
    h = sigmoid(np.dot(np.dot(u,t.T),x) + np.dot(w,hl)) # hidden state
    for i in targets:                 #calculate the loss
        xt = np.zeros((goods_size,1))
        xt[i-1][0] = 1
        loss += np.log(sigmoid(np.dot(np.dot(xt.T,t),h)))
    for i in negtargets:
        xn = np.zeros((goods_size,1))
        xn[i-1][0] = 1
        loss += np.log(1 - sigmoid(np.dot(np.dot(xn.T,t),h)))
    print loss


    du, dw, dt = np.zeros_like(u), np.zeros_like(w), np.zeros_like(t)



    for i in targets:                   #loss to hide
        xt = np.zeros((goods_size,1))
        xt[i-1][0] = 1
        # mid += 1-sigmoid(np.dot((np.dot(np.dot(x.T, t), h)), np.dot(x.T,t))).T
        mid += (1-sigmoid(np.dot(np.dot(xt.T,t),h)))*np.dot(xt.T,t).T
        midt += (1-sigmoid(np.dot(np.dot(xt.T, t), h))) * np.dot(xt,h.T)

        # print np.shape(mid)  #1, 1559
    for i in negtargets:
        xn= np.zeros((goods_size,1))
        xn[i-1][0] = 1
        mid -= sigmoid(np.dot(np.dot(xn.T, t), h))*np.dot(xn.T,t).T
        midn += sigmoid(np.dot(np.dot(xn.T, t), h)) * np.dot(xn,h.T)
    dw = np.dot(mid*h*(1-h),hl.T)
    du += np.dot(mid*h*(1-h), np.dot(t.T,x).T)         #x how to choose   x x+1
    dt += np.dot(np.dot(u.T,mid*h*(1-h)),x.T).T

    dt +=midt-midn
    # dt = np.dot(np.dot((1 - sigmoid(np.dot(np.dot(x.T,t), h))),x),h.T) + midn + \
    #      np.dot(np.dot(mid.T*hl*(1-hl), u.T), x.T)
    hl = h
    return loss, du, dw, dt, hl


def negasamp(targets):
    list2 = listfre
    for i in targets:
        list2 = filter(lambda a: a != i, list2)
    # print targets
    # print list2
    negtargets = []
    for i in range(50):
        negtargets.append(random.choice(list2))
    return negtargets

for i in range(len(listcust)-1):
    customer = data[listcust[i]]
    hprev = np.zeros((hidden_size, 1))

    print "customer"
    print i
    print customer

    for j in range(len(customer)-1):

        inputs = customer[j]
        targets = customer[j+1]
        negtargets = negasamp(targets)
        loss, du, dw, dt, hprev = lossFun(inputs, targets, negtargets, hprev)
    # for j in range(len(inputs)-1):
    #     # print "basket"
    #     # print j
    #     loss, du, dw, dt, hprev = lossFun(inputs, targets, negtargets, hprev)
    	for param, dparam in zip([u, w, t],
                             [du, dw, dt]):
        	param += learning_rate * dparam # adagrad update
    if i%50==0:
	hint="This is round %d"%i
	print hint
	print u
	print w
	print t
		
