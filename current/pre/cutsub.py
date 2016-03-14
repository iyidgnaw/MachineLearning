import json

f = open("subuser_cart.json","a")
pastuserid = 2483
user_cart = []
id_list = []
diction = {}
data = open("submobile_time1.csv","r")
lines = data.readlines()
for line in lines:
	userid, artid, month, day, hour, time_sub = line.split(",")
	artid = int(artid)
	userid = int(userid)	
	time_sub = int(time_sub)
	record = []
	record.append(artid)
	record.append(time_sub)
	if userid != pastuserid:
		pastuserid = userid
		json.dump(user_cart,f)
		f.write("\n")
		user_cart = []
	user_cart.append(record)

f.close()
