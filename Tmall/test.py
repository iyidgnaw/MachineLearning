import sys
f_handler=open('result.txt', 'w')
sys.stdout=f_handler
print "Hello"
