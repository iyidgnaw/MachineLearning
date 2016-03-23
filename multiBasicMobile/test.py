import multiprocessing
import time

import multiprocessing
import time

def worker_1(interval, count):
    print "worker_1"
    time.sleep(interval)
    print "end worker_1"
    count += 1
    print "count"+str(count)

def worker_2(interval, count):
    print "worker_2"
    time.sleep(interval)
    print "end worker_2"
    count += 1

def worker_3(interval, count):
    print "worker_3"
    time.sleep(interval)
    print "end worker_3"
    count += 1

if __name__ == "__main__":
    count=0
    p1 = multiprocessing.Process(target = worker_1, args = (1,count))
    p2 = multiprocessing.Process(target = worker_2, args = (1,count))
    p3 = multiprocessing.Process(target = worker_3, args = (1,count))

    p1.start()
    p1.join()
    p2.start()
    p2.join()
    p3.start()
    p3.join()
    print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print("child   p.name:" + p.name + "\tp.id" + str(p.pid))
    print "END!!!!!!!!!!!!!!!!!"

    print count