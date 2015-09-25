__author__ = 'lizk'
import MySQLdb as mdb
import sys
import pickle

con = None

try:

    con = mdb.connect('localhost', 'root',
        'lzk199445', 'test');

    cur = con.cursor()
    cur.execute("SELECT * FROM data_order")

    rows = cur.fetchall()
    numrows = int(cur.rowcount)


    basketdic = {}
    basket = []
    k = 0
    j = 0
    for i in range(numrows):

        # if i > 0:
        #      if rows[i-1][1] != rows[i][1]:
        #         # print "date"
        #         # print rows[i][1]                           #date
        #         i = i + 1
        #         basket[i].append([])
        #
        #         if rows[i-1][2] != rows[i][2]:                  #customer
        #             # print "customer"
        #             # print rows[i][2]
        #             basketdic[rows[i][2]] = basket
        #             basket = []
        #             i = 0
        if i == 0:
            basket.append([])
            basket[k].append(int(rows[i][0]))
            continue

        if rows[i-1][2] != rows[i][2]:                  #customer
            basketdic[int(rows[i-1][2])] = basket
            basket = []
            k = 0
            basket.append([])
            basket[k].append(int(rows[i][0]))

            continue

        if rows[i-1][1] != rows[i][1]:                         #date
            k = k + 1
            basket.append([])

        print rows[i][0]
        basket[k].append(int(rows[i][0]))
    basketdic[int(rows[i][2])] = basket

    # save dict
    f1 = open("F:\\pythonPro\\data_NextBasket.txt", "wb")
    pickle.dump(basketdic, f1)
    f1.close()
    #read
    # f1 = open("F:\\pythonPro\\RNN_lizk.txt", "rb")
    # d = pickle.load(f1)
    # f1.close()






except mdb.Error, e:

    print "Error %d: %s" % (e.args[0],e.args[1])
    sys.exit(1)

finally:

    if con:
        con.close()
