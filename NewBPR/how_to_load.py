import pickle
f1 = open("long_to_short.txt", "r")
f2 = open("10_data.txt",'r')
f3 = open("data.txt",'a')
dic = pickle.load(f1)
lines = f2.readlines()
count=0
for line in lines:
	record = line.split('	')
	longid=record[2]
	count+=1
	string = str(record[0].split('_')[1].split(' ')[0])+"	"+str(dic[longid])+"\n"
	f3.write(string)
print count
f3.close()