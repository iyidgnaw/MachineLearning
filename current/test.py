from multiprocessing import Process
import os
import time
import numpy as np

def test1():
	print 'process id:', os.getpid()
	x = np.random.randn(1000, 10)
	hprev = np.ones((1, 10))
	predict_matrix = np.dot(hprev,x.T)
	
def test2():
	x = np.random.randn(1000, 10)
	hprev = np.ones((1, 10))
	predict_matrix = np.dot(hprev,x.T)

	
	

	
t1 = time.clock()
if __name__ == '__main__':
	test1()
	queue = []
	for i in range(20):
		p = Process(target=test1)
		queue.append(p)
	for i in range(20):
		queue[i].start()
	for i in range(20):
		queue[i].join()
	print "done"
t2 =time.clock()
print t2-t1

t1 = time.clock()
print 'process id:', os.getpid()
for i in range(20):
	test2()
print "done"
t2 =time.clock()
print t2-t1
