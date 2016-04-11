import pickle

timetointerval = {}
for i in range(462):
	if i < 4:
		timetointerval[i]=i
	if i>=4 or i <= 6:
		timetointerval[i]=4
	if i>=7 or i <= 12:
		timetointerval[i]=5
	if i>=13 or i <= 24:
		timetointerval[i]=6
	if i>=25 or i <= 48:
		timetointerval[i]=7
	if i>=49 or i <= 72:
		timetointerval[i]=8
	if i>=73 or i <= 168:
		timetointerval[i]=9
	if i>168:
		timetointerval[i]=10
print "fi"
f1 = open("timetointerval","wb")
pickle.dump(timetointerval,f1)
f1.close()
