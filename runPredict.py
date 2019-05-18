import tensorflow as tf
import numpy as np
import warnings #for numpy overflow
#import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import signal
import time
from eegData import eegData
from annFile import tf_ann
from paramFile import paramHolder
##################################################
#
#                DATA PREP
#
##################################################

gparam = paramHolder()

def main():
    #Predict args: softAnn.py fname dest [decision]
    #             decision format example: dec=8
    #             fname file example: 002PCA700
    #             fname flag example: all or all=700 (PCA LEVEL)
    if(len(sys.argv) >= 3):
        argv1 = sys.argv[1]
        argv2 = sys.argv[2]
	
        if (len(argv1.split("=")) > 1):
            gparam.trainId = argv1.split("=")[0]
            gparam.decomp = argv1.split("=")[1]
        else:
            gparam.trainId = argv1
            gparam.decomp = "700"
        fname = gparam.dir_path+"/PCAData/input" + gparam.trainId + ".csv"
        gparam.destPath += "/Results/" + sys.argv[2] + "/"
        if (gparam.trainId == "all"):
            predictAll(argv2,gparam.decomp)
        else:
            predict(fname,gparam.trainId)
    else:
        print("Not Enough Args")
        return   
    
def predict(fname,fileId):
    #global decision, normalize
    eegObj = eegData(gparam.destPath)
    
    X,Y = eegObj.getPredictData(fname,1)
    YT = np.argmax(Y,axis=1)
    annObj = tf_ann(gparam.dir_path, gparam.destPath)
    annObj.predict_init(gparam.decision, gparam.layerSizeMultiplier, gparam.predictOverfit, gparam.layers)
    P = annObj.predict(X,Y)
        
    acc,sens,spec=annObj.allStats(YT,P)
    print("Decision:", gparam.decision)
    print("Accuracy:",acc)
    print("Sensitivity:",sens)
    print("Specificity:",spec)
    print("Classification Rate",annObj.classification_rate(YT,P))
    tfile = open(gparam.destPath + "predictStats.txt", "a")
    tfile.write(fileId+ ", " + str(acc) + "," + str(sens) + "," +str(spec) + "," + str(annObj.classification_rate(YT,P)) + "\n")
    tfile.close()

def predictAll(trainDest, decomp):
    open(gparam.destPath + "predictStats.txt","w").close()
    with open(gparam.dir_path + "/sourceData/validPatientNights.csv") as nightsFile:
        validNights = nightsFile.readlines()
    for validNight in validNights:
        if (validNight.strip() != ""):
            fileId = validNight.split(",")[0].strip()
            fileId += validNight.split(",")[1].strip()
            fileId += "PCA" + decomp
            fname = gparam.dir_path+"/PCAData/input" + fileId + ".csv"
            predict(fname,fileId)


    
    tfile = open(gparam.destPath + "predictStats.txt")
    pstats = tfile.read()
    tfile.close()
    predictId = time.strftime("%Y%m%d%H%M%S")
    tfile = open(gparam.destPath + predictId + "predictStats.txt","w")
    tfile.write(pstats)
    tfile.close()
    avgPredictStats(trainDest,predictId)
    

def avgPredictStats(trainDest,predictId):
    global decision
    
    #dest = gparam.dir_path + "/Results/" + sys.argv[1] + "/"
    acc = []
    sens = []
    spec = []
    with open(gparam.destPath + "predictStats.txt") as pfile:
        line = pfile.readline()
        line = pfile.readline()

        while(line.strip()):
            lineSplit = line.split(",")
            acc.append(float(lineSplit[1]))
            sens.append(float(lineSplit[2]))
            spec.append(float(lineSplit[3]))
            line = pfile.readline()
    decform = "%14s" % ("%3.2f" % gparam.decision)
    accAvg = "%14s" % ("%3.2f" % (sum(acc)/len(acc)))
    sensAvg = "%14s" % ("%3.2f" % (sum(sens)/len(sens)))
    specAvg = "%14s" % ("%3.2f" % (sum(spec)/len(spec)))
    
    with open(gparam.dir_path + "/Results/" + "sessionStats.csv", "a") as sessFile:
        sessFile.write("%23s" % str(trainDest))
        sessFile.write(",%14s" % str(decform))
        sessFile.write(",%14s" % str(accAvg))
        sessFile.write(",%14s" % str(sensAvg))
        sessFile.write(",%14s" % str(specAvg))
        if (gparam.predictOverfit):
            sessFile.write(",%14s" % "overfit")
        else:
            sessFile.write(",%14s" % "train")
        sessFile.write(",%23s" % predictId)
        sessFile.write("\n")

if __name__ == "__main__":
    main()
