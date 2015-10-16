import pickle
f1 = open("./result.txt", "rb")
product_id = pickle.load(f1)
f1.close()
print product_id