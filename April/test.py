import random
import numpy as np

ITEM_SIZE =1000
HIDDEN_SIZE = 10
def avg_neg(neglist):
	Sum = np.zeros((1, HIDDEN_SIZE))
	for i in neglist:
		Sum +=X[i-1, :].reshape(1, HIDDEN_SIZE)
		print Sum
	return Sum/20


X = np.random.randn(ITEM_SIZE, HIDDEN_SIZE)*0.5
itemlist = range(ITEM_SIZE)
neglist = random.sample(itemlist,20)
item_neg = avg_neg(neglist)
print item_neg
