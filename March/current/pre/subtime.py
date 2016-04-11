import datetime, csv
from collections import Counter
# dataCSV = open('mobile_time.csv', 'w')
# writer = csv.writer(dataCSV, delimiter = ',')
# time_sub = 0
# time_sub_hour=0
# i=0
# lastuser = 2483
# for line in open("mobile_short.csv"):
# 	i+=1
# 	userid, itemid, month, day, hour = line.split(",")
# 	month = int(month)
# 	day = int(day)
# 	hour = int(hour)
# 	time2 =datetime.datetime(2012,month,day,hour,0,0)

# 	if not(i==1):
# 		time_sub = (time2-time1)
# 		time_sub_hour = time_sub.days*24+time_sub.seconds/3600
# 	if userid!=lastuser:
# 		time_sub_hour=0

# 	writer.writerows([[userid, itemid, month, day, hour, time_sub_hour]])
# 	time1 =time2
# 	lastuser = userid

time_sub = []
for line in open("mobile_time.csv"):
	userid, itemid, month, day, hour, sub = line.split(",")
	month = int(month)
	day = int(day)
	hour = int(hour)
	sub = int(sub)
	time_sub.append(sub)
fre = Counter(time_sub)
print fre

