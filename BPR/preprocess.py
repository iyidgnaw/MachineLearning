data=open('./up3.txt','r')
order=open('./order.txt','a')
lines=data.readlines()
down3=open('./down3.txt','r')
lines1=down3.readlines()
train=open('train.txt','a')
test=open('test.txt','a')
count=0
dic={}
for i in range(943):
	dic[i+1]=[]
for line in lines:
	record = line.split("	")
	dic[int(record[0])].append(line)
for i in range(943):
	for j in range(len(dic[i+1])):
		if j<len(dic[i+1])/2:
			test.write(dic[i+1][j])
		else:
			train.write(dic[i+1][j])
for i in lines1:
	train.write(i)

