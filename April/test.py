import pickle

wdy = open('./data/his_5000.pkl','rb')
wdy = pickle.load(wdy)
print wdy
