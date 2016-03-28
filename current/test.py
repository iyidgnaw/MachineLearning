for line in open("../mobile_time.csv"):
	userid, artid, month, day, hour, time_sub = line.split(",")
	userid = int(userid)
	artid = int(artid)
	time_sub = int(time_sub)
	i += 1
	user_list.append(int(userid))
	itemid_list.append(int(artid))