__author__ = 'wangdiyi'
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import xlrd
import xlwt

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
learning_rate = 1e-1

# model parameters
u = np.random.randn(hidden_size, goods_size)*0.01 # input to hidden
w = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
t = np.random.randn(goods_size, goods_size)*0.01 # one-hot to embedding
def sigmoid(x):#sigmoid function
	return 1.0/(1+np.exp(-x))


def lossFun(inputs, targets, negtargets,hprev):#loss function 
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
	x= []
	hl=hprev
	loss = 0
	mid = 0
  # forward pass
	for i in inputs:
		x = np.zeros((goods_size,1)) # encode in 1-of-k representation
		x[i][0] = 1
		h = sigmoid(np.dot(np.dot(u,t),x) + np.dot(w,hl)) # hidden state
	for i in targets:#calculate the loss 
		x = np.zeros((goods_size,1))
		x[i][0] = 1
		loss+=np.log(sigmoid(np.dot(np.dot(x.T,t),h)))
	for i in negtargets:
		x = np.zeros((goods_size,1))
		x[i][0] = 1
		loss+=np.log(sigmoid(np.dot(np.dot(x.T,t),h)))


	du, dw ,dt= np.zeros_like(u), np.zeros_like(w), np.zeros_like(t)
	for i in targets:#loss to hide
		x = np.zeros((goods_size,1))
		x[i][0] = 1
		mid+=(1-sigmoid(np.dot(np.dot(x.T,t).h)))*np.dot(x.T,t)
	for i in negtargets:
		x = np.zeros((goods_size,1))
		x[i][0] = 1
		mid+=sigmoid(np.dot(np.dot(x.T,t).h))*np.dot(x.T,t)	
	dw=np.dot(mid*h*(1-h).T,hl.T)	
		
	return loss, du, dw, dt, h


def negasamp(targets):
	list2=listfre
	for i in targets:
		list2.remove(i)
	negtargets=[]
	for i in range(50):
		negtargets.append(list2[i])
	return negtargets
	

for i in range(len(data)-1):
	

