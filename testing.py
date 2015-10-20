from svmutil import *

y, x = svm_read_problem('./trains')
ys,xs = svm_read_problem('./tests')
dict={}
for i in range(-10,11):
	for j in range(-10,11):
		c="-c "+str(2**i)
		g=" -g "+str(2**j)
		param = c+g
		print param
		m = svm_train(y,x,param)
		p_label, p_acc, p_val = svm_predict(ys,xs, m)
		dict[param]=p_acc
accuracy=0
result=''
for i in dict.keys():
	if dict[i][0]>accuracy:
		result=i
		accuracy=dict[i][0]
print result,accuracy