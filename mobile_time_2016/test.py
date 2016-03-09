import time
for i in range(1000):
    file1 = open('result.txt','w')
    file1.write(str(i))
    file1.close()    
    time.sleep(1)
	
