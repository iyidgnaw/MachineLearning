import pickle
basic = open('resultbasic.txt','rb')
wdy = open('resultwdy.txt','rb')
basic = pickle.load(basic)
print basic
wdy = pickle.load(wdy)
print wdy
