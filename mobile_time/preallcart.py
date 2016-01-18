import pickle,csv

userid_list = []
itemid_list = []
# i = 0
# for line in open("mobile.csv"):
# 	userid, itemid, month, day, hour = line.split(",")
# 	if i>0:
# 		userid_list.append(userid)
# 		itemid_list.append(itemid)
# 	i += 1

# userid_unique = list(set(userid_list))
# itemid_unique = list(set(itemid_list))

# userlong_to_short = {ch:i for i, ch in enumerate(userid_unique)}
# itemlong_to_short = {ch:i for i, ch in enumerate(itemid_unique)}

# usershort_to_long = {i:ch for i, ch in enumerate(userid_unique)}
# itemshort_to_long = {i:ch for i, ch in enumerate(itemid_unique)}

# f1 = open("userlong_to_short","wb")
# pickle.dump(userlong_to_short,f1)
# f1.close()

# f1 = open("itemlong_to_short","wb")
# pickle.dump(itemlong_to_short,f1)
# f1.close()

# f1 = open("usershort_to_long","wb")
# pickle.dump(usershort_to_long,f1)
# f1.close()

# f1 = open("itemshort_to_long","wb")
# pickle.dump(itemshort_to_long,f1)
# f1.close()

f1 = open("itemlong_to_short","r")
itemlong_to_short = pickle.load(f1)
f1.close()

f1 = open("userlong_to_short","r")
userlong_to_short = pickle.load(f1)
f1.close()


dataCSV = open('mobile_short.csv', 'w')
writer = csv.writer(dataCSV, delimiter = ',')
# csvfile = file('mobile_short.csv', 'wb')
# writer = csv.writer(csvfile)
# writer.writerow(['user', 'item', 'month', 'day', 'hour'])
i = 0
for line in open("mobile.csv"):
	i += 1
	if i>1:
		userid, itemid, month, day, hour = line.split(",")
		user = str(userlong_to_short[userid]+1)
		item = str(itemlong_to_short[itemid]+1)

		data = [userlong_to_short[userid]+1, itemlong_to_short[itemid]+1, month, day, int(hour)]
		# print type(data)
		writer.writerows([[userlong_to_short[userid]+1, itemlong_to_short[itemid]+1, month, day, int(hour)]])
dataCSV.close()