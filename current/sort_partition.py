import numpy as np

x = np.random.randn(1000, 10)
hprev = np.ones((1, 10))
predict_matrix = np.dot(hprev,x.T)

rank_index = np.argsort(predict_matrix, axis=1)
rank_index = rank_index[:, -20:np.shape(rank_index)[1]]
rank_index_list = list(reversed(list(rank_index[0])))
print rank_index_list


rank = np.argpartition(predict_matrix[0],-20)[-20:]
rank = rank[np.argsort(predict_matrix[0][rank])]
rank_index_list = list(reversed(list(rank)))
print rank_index_list