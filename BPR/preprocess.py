data=open('./u.data')
lines=data.readlines()
train=open('train.txt','a')
test=open('test.txt','a')
count=0
for line in lines:
	count+=1
	if count<=50000:
		train.write(line)
	else:
		test.write(line)
