import random 
import json
#cut tthe sub data
#file1 = open('user_cart.json','r')
#lines = file1.readlines()
#file2 = open('subuser_cart.json','w')
#for i in range(300):
#	file2.write(lines[i])
#file1.close()
#file2.close()
#file2 = open ('subuser_cart.json','r')
#print len(file2.readlines())
alldata = []
data = open('subuser_cart.json','r')
lines = data.readlines()
for line in lines:
	line1 = json.loads(line)
	alldata.append(line1)
number = 0
for i in alldata:
	number += len(i)

file1 = open('mobile_time.csv','r')
lines = file1.readlines()
file2 = open('submobile_time.csv','w')
for i in range(number):
	file2.write(lines[i])
file1.close()
file2.close()
data.close()



































	
