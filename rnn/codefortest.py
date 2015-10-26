import time


ISOTIMEFORMAT='%Y-%m-%d %X'
print "Start time is :"
print time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) ) 