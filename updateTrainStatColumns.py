import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
destPath = dir_path + "/Results/"
sessions = []
for dirTup in os.walk(destPath):
    if (dirTup[0].split("/")[-1].strip()):
        sessions.append(dirTup[0].split("/")[-1])
osessions = sorted(sessions)
trainParamLines = []
for session in sessions:
    with open(destPath + session + "/trainParams.txt") as trainFile:
        headerB      = trainFile.readline()
        headerBsplit = headerB.split(",")
        headerAsplit = []
        trainB       = trainFile.readline()
        trainBsplit  = trainB.split(",")
        trainAsplit  = []
        for i in range(7):
            headerAsplit.append(headerBsplit[i])
            trainAsplit.append(trainBsplit[i])

        headerAsplit.append("     normalize")
        trainAsplit.append ("          none")

        for i in range(7,len(headerBsplit)):
            headerAsplit.append(headerBsplit[i])

        for i in range(7,len(trainBsplit)):
            trainAsplit.append(trainBsplit[i])
    #print(",".join(headerAsplit))
    #break
    with open(destPath + session + "/trainParams.txt", "w") as trainFile:
        trainFile.write(",".join(headerAsplit))
        trainFile.write(",".join(trainAsplit))
    
