import os   
import json

def getlist(dir):
	list = os.listdir(dir)
	return list

def seperate(name,bool,target):
	if bool==True:
		for i in name:
			content="+1"
			line=i.strip('\n').strip(',')
			js=json.loads(line)
			for j in range(1,len(js.keys())):
				content+=" %d:"%(j)+str(js[js.keys()[j]])
			content+="\n"
			eventname=js["eventname"].encode('utf-8')
			dir=getlist(target)
			if eventname in dir:
				train.write(content)
			else:
				test.write(content)

	else:
		for i in name:
			content="-1"
			line=i.strip('\n').strip(',')
			js=json.loads(line)
			for j in range(1,len(js.keys())):
				content+=" %d:"%(j)+str(js[js.keys()[j]])
			content+="\n"
			eventname=js["eventname"].encode('utf-8')
			dir=getlist(target)
			if eventname in dir:
				train.write(content)
			else:
				test.write(content)
	return


url1="./rumor/rumor_weibo_train"
url2="./nonrumor/nonrumor_weibo_train"
train=open("train.txt",'a')
test=open("test.txt",'a')
rumor=open("./rumor/featureresult.json",'r')
nonrumor=open("./nonrumor/featureresult.json",'r')
seperate(rumor,True,url1)
seperate(nonrumor,False,url2)