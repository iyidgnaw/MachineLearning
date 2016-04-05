# import json
# from collections import Counter
# user_cart = []
# previous = 1378
# i = 0
#
# with open('user_cart_basic.json','a') as f:
# 	for line in open("data_resetwithTime.csv"):
# 		# print line
# 		userid, itemid, year, month, day = line.split(",")
# 		userid = int(userid)
# 		itemid = int(itemid)
# 		year = int(year)
# 		month = int(month)
# 		day = int(day)
# 		i+=1
# 		if i%1000==0:
# 			print i
# 		if int(userid)!=previous:
# 			json.dump(user_cart, f)
# 			f.write('\n')
# 			previous = int(userid)
# 			user_cart=[]
# 			user_cart.append([itemid])
# 		else:
# 			user_cart.append([itemid])
# 	json.dump(user_cart, f)
#
# f.close()

# all_cart = []
# data = open('user_cart.json','r')
# lines = data.readlines()
# for line in lines:
# 	i+=1
# 	line1 = json.loads(line)
# 	print line1
# 	for behavior in line1:
# 		print behavior[0], behavior[1]
# 	if i >10:
# 		break

#
# 	all_cart.append(line1)
# allrecord = []
# for i in xrange(len(all_cart)):
# 	user_cart1 = all_cart[i]
# 	for i in user_cart1:
# 		allrecord.append(i)
# fre = Counter(allrecord)
# print fre


import json, calendar
from collections import Counter
user_cart = []
previous = 0
i = 0


with open('user_cart_delta.json','a') as f:
	for line in open("delta.csv"):
		# print line
		userid, itemid, year, month, day, delta = line.split(",")
		userid = int(userid)
		itemid = int(itemid)
		delta = int(delta)
		
		i+=1
		if i%1000==0:
			print i
		if int(userid)!=previous:
			json.dump(user_cart, f)
			f.write('\n')
			previous = int(userid)
			user_cart=[]
			user_cart.append([itemid, delta])
		else:
			user_cart.append([itemid, delta])
	json.dump(user_cart, f)

f.close()

