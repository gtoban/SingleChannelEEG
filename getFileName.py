import sys
import time


if (len(sys.argv) < 4):
    print("Need patientId Night percentVar")
    exit(-1)
pid = sys.argv[1]
day = sys.argv[2]
percentVar = sys.argv[3]

print(time.strftime("%Y%m%d%H%M%S") +"."+ pid+day+percentVar)
