import json
all_cart = []
itemid_list = []
data = open('./data.json', 'r')

lines = data.readlines()
for line in lines:
	line1 = json.loads(line)
	all_cart.append(line1)
for i in all_cart:
	itemid_list.extend(i)
list1 = list(set(itemid_list))
diction = {}
for i in range(len(list1)):
	diction[list1[i]] = i+1

data1 = open('t_alibaba_data.csv','r')
output = open('data1.txt','a')
lines = data1.readlines()
past_target = 0
newlist = []
for line in lines:
	target = line.split(',')
	if past_target!=int(target[0]):
		output.write(str(newlist)+'\n')
		newlist=[]
		oldid = int(target[1])
		newid = diction[oldid]

		newlist.append(newid)
		past_target = int(target[0])
	else:
		oldid = int(target[1])
		newid = diction[oldid]
		newlist.append(newid)
		past_target = int(target[0])		
