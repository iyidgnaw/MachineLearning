import json

f = open("subuser_cart.json","a")
f1 = open("submobile_time.csv","a")
pastuserid = 2483
user_cart = []
data = open("mobile_time.csv")
lines = data.readlines()
for i in range(101394):
	line = lines[i]
	f1.write(line)
	userid, artid, month, day, hour, time_sub = line.split(",")
	userid = int(userid)
	artid = int(artid)
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
f1.close()