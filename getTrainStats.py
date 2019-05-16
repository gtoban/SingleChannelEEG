import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
destPath = dir_path + "/Results/"
sessions = []
for dirTup in os.walk(destPath):
    if (dirTup[0].split("/")[-1].strip()):
        sessions.append(dirTup[0].split("/")[-1])

sessions = sorted(sessions)
trainParamLines = []
for session in sessions:
    if (session != "backup"):
        with open(destPath + session + "/trainParams.txt") as trainFile:
            header=trainFile.readline()
            trainParamLines.append(trainFile.readline())

with open(destPath + "sessionParams.csv", "w") as sessFile:
    sessFile.write("%23s," % ("trainId") + header)
    i = 0
    for trainParamLine in trainParamLines:
        sessFile.write("%23s," % (sessions[i]) + trainParamLine.rstrip() + "\n")
        i += 1
