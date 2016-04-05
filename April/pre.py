from datetime import *
from collections import Counter
output = open('delta.csv','a')
data = open('./clothes.csv', 'r')
lines = data.readlines()
pastuserid = "0" 
pastdate = 0
deltalist = []
def judge(delta):
	if delta ==0:
		return 0
	elif delta ==1:
		return 1
	elif delta <4:
		return 2
	elif delta<9:
		return 3
	else:
		return 4
for line in lines:
	splitlist = line.split(',')
	userid = splitlist[0]
	year = int(splitlist[2])
	month = int(splitlist[3])
	day = int(splitlist[4])
	x = date(year,month,day)
	if userid!=pastuserid:
		pastuserid=userid
		pastdate = x
		newline = line.strip() +","+str(0)+"\n"
		deltalist.append(0)
		# output.write(newline)

	else:
		delta = x-pastdate
		num = judge(delta.days)
		deltalist.append(num)
		newline = line.strip()+","+str(num)+"\n"
		# output.write(newline)
		pastuserid=userid
		pastdate = x
print Counter(deltalist)



