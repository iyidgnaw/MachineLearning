__author__ = 'wangdiyi'
f=open('./Gowalla.txt')
numBeijing=0
numLondon=0
numTokyo=0
numNewyork=0
numLos=0
Beijing = open('./beijing.txt', 'w')
London= open('./london.txt', 'w')
Tokyo= open('./tokyo.txt', 'w')
Newyork= open('./newyork.txt', 'w')
Los= open('./los.txt', 'w')
LocationOfBeijing=[39.786039,40.39287,116.2315,116.554603]
LocationOfLondon=[51.288189,51.690081,-0.504274,0.208464]
LocationOfTokyo=[35.615590,35.798471,139.503364,139.895439]
LocationOfNewyork=[40.596516,40.822670,-74.038733,-73.745878]
LocationOfLos=[33.700053,34.326131,-118.632832,-118.146687]
for eachLine in f:
    try:
        str= eachLine.split('\t')
        if (float(LocationOfBeijing[0]<str[2])<LocationOfBeijing[1] and LocationOfBeijing[2]<float(str[3])<LocationOfBeijing[3]):
            numBeijing=numBeijing+1
            Beijing.write(eachLine)
        if (float(LocationOfLondon[0]<str[2])<LocationOfLondon[1] and LocationOfLondon[2]<float(str[3])<LocationOfLondon[3]):
            numLondon=numLondon+1
            London.write(eachLine)
        if (float(LocationOfTokyo[0]<str[2])<LocationOfTokyo[1] and LocationOfTokyo[2]<float(str[3])<LocationOfTokyo[3]):
            numTokyo=numTokyo+1
            Tokyo.write(eachLine)
        if (float(LocationOfNewyork[0]<str[2])<LocationOfNewyork[1] and LocationOfNewyork[2]<float(str[3])<LocationOfNewyork[3]):
            numNewyork=numNewyork+1
            Newyork.write(eachLine)
        if (float(LocationOfLos[0]<str[2])<LocationOfLos[1] and LocationOfLos[2]<float(str[3])<LocationOfLos[3]):
            numLos=numLos+1
            Los.write(eachLine)
    except:
        print "There is an error!"
print numBeijing
print numLondon
print numTokyo
print numNewyork
print numLos