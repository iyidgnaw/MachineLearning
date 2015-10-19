__author__ = 'wangdiyi'
import json
import os
filename=os.listdir(r'./7')
test= open('./7.txt', 'w')
for i in range(len(filename)):
    if i==0:
        continue
    else:
        name="./7/"+filename[i]
        file=open(name,'r')
        for eachLine in file:
            line=eachLine.strip().decode('utf-8')
            line=line.strip(',')
            js=json.loads(line)
            for i in range(len(js["comment"])):
                text=js["comment"][i]["content"]
                print text
                print type(text)
                text=text.encode('utf-8')
                print type(text)
                print text
                test.write(text)