import json
from collections import Counter
user_cart = []
previous = 1
i = 0
#
# with open('user_cart.json','a') as f:
# 	for line in open("mobile_time.csv"):
# 		# print line
# 		userid, itemid, month, day, hour, time_sub = line.split(",")
# 		userid = int(userid)
# 		itemid = int(itemid)
# 		time_sub = int(time_sub)
# 		i+=1
# 		if i%1000==0:
# 			print i
# 		if int(userid)!=previous:
# 			json.dump(user_cart, f)
# 			f.write('\n')
# 			previous = int(userid)
# 			user_cart=[]
# 		else:
# 			user_cart.append([itemid, time_sub])
# f.close()

all_cart = []
data = open('user_cart.json','r')
lines = data.readlines()
for line in lines:
	i+=1
	line1 = json.loads(line)
	print line1
	for behavior in line1:
		print behavior[0], behavior[1]
	if i >10:
		break

#
# 	all_cart.append(line1)
# allrecord = []
# for i in xrange(len(all_cart)):
# 	user_cart1 = all_cart[i]
# 	for i in user_cart1:
# 		allrecord.append(i)
# fre = Counter(allrecord)
# print fre

