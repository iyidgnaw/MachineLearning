# -*- coding: utf-8 -*
import numpy as np
import random
import math
USER_SIZE = 10			# 总用户数
ITEM_SIZE = 5			# 总商品种数
HIDDEN_SIZE = 5

LEARNING_RATE = 0.1 		# 学习速率
LAMBDA = 0.001 				# 惩罚系数


USER_MATRIX=np.random.randn(USER_SIZE, HIDDEN_SIZE)*0.5
ITEM_MATRIX=np.random.randn(ITEM_SIZE, HIDDEN_SIZE)*0.5

RATING = np.dot(USER_MATRIX,ITEM_MATRIX.T)
rank = np.argpartition(RATING[0], -3)[-3:]
rank = rank[np.argsort(RATING[0][rank])]
rank_index_list = list(reversed(list(rank)))
print RATING
print RATING[0]
print rank_index_list
