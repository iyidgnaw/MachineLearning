import os   
import json

def getlist(dir):
	list = os.listdir(dir)
	return list

def seperate(name,bool):
	if bool==True:
		for i in name:
			content="+1"
			line=i.strip('\n').strip(',')
			js=json.loads(line)
			for j in range(1,len(js.keys())):
				content+=" %d:"%(j)+str(js[js.keys()[j]])
			#content+="\n"
			print content
			
	else:
		for i in name:
			content="-1"
			line=i.strip('\n').strip(',')
			js=json.loads(line)
			for j in range(1,len(js.keys())):
				content+=" %d:"%(j)+str(js[js.keys()[j]])
			#content+="\n"
			print content
	return



train=open("train.txt",'w')
test=open("test.txt",'w')
rumor=open("./rumor/featureresult.json",'r')
nonrumor=open("./nonrumor/featureresult.json",'r')
seperate(rumor,True)