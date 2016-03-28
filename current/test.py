import json


user_list=[]
itemid_list = []
for line in open("./mobile_time.csv"):
	userid, artid, month, day, hour, time_sub = line.split(",")
	userid = int(userid)
	artid = int(artid)
	time_sub = int(time_sub)
	user_list.append(int(userid))
	itemid_list.append(int(artid))
product_id = list(set(itemid_list))
print product_id
print len(product_id)