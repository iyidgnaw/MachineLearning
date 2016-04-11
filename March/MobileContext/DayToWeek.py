import calendar

year = 2014
for line in open("mobile_short.csv"):
	userid, itemid, month, day, hour = line.split(",")
	month = int(month)
	day = int(day)

	weekday = calendar.weekday(year, month, day)