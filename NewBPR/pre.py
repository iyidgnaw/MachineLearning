#Get the number of item
f3 = open("data.txt",'r')
test = open("test.txt",'a')
train = open("train.txt",'a')
lines = f3.readlines()
dic={}
for i in xrange(1000):
	dic[i+1]=[]
for line in lines:
	record = line.split("	")
	dic[int(record[0])].append(line)
for i in xrange(1000):
	for j in xrange(len(dic[i+1])):
		if j<len(dic[i+1])/2:
			test.write(dic[i+1][j])
		else:
			train.write(dic[i+1][j])

