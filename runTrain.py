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
##################################################
#
#                DATA PREP
#
##################################################

##IMPORTANT VARIABLES
learningRate = 10e-3 #learning rate
trainIterations = 100000 #100000 #10000
trainPrintStep = 100
overfitSteps = 10
#overfitSteps:
# the number of trainPrintSteps
# without an increase in accuracy on the overfit data
# until its considered overfitting.
trainPrint = 0
trainRandomStart = 1
writeStartingGuess = 0
normalize = 1 ## used to specify in normalizing is used (x - mean)/std
inputnormalizeType="std Dev"
activationType="relu"
layers = 1
layerSizeMultiplier = 0.5 ## LAYER sizes are a multiple of input dimensions
decision = 50 # if REM ([REM NotREM]) >= decision then REM is choice
stopReason = "iterations"
predictOverfit = 0 #use weights from overfit stop

dir_path = os.path.dirname(os.path.realpath(__file__))
destPath = dir_path
trainId = ""
trainDestination = dir_path

#class GracefulKiller:
#  kill_now = False
#  def __init__(self):
#    signal.signal(signal.SIGINT, self.exit_gracefully)
#    signal.signal(signal.SIGTERM, self.exit_gracefully)
#
#  def exit_gracefully(self,signum, frame):
#    self.kill_now = True


def main():
    global destPath, trainId, layers, decision, inputnormalizeType, activationType
    #Train args  : softAnn.py fname oname dest
    #Predict args: softAnn.py fname dest [decision]
    #             decision format example: dec=8
    #             fname file example: 002PCA700
    #             fname flag example: all or all=700 (PCA LEVEL)
    trainChoice = 0
    predictChoice = 0
    argv1=argv2=argv3=""
    if (len(sys.argv) == 4):
        argv1 = sys.argv[1]
        argv2 = sys.argv[2]
        argv3 = sys.argv[3]
        if (len(argv3.split("=")) > 1):
            predictChoice = 1
            decision = float(argv3.split("=")[1])
        else:
            trainChoice = 1
    elif(len(sys.argv) == 3):
        predictChoice = 1
        argv1 = sys.argv[1]
        argv2 = sys.argv[2]
        
    if(trainChoice):
        fname = dir_path+"/PCAData/input" + sys.argv[1] + ".csv"
        oname = dir_path+"/PCAData/input" + sys.argv[2] + ".csv"
        destPath += "/Results/" + sys.argv[3] + "/"
        paramsFile=open(destPath+"trainParams.txt","w")
        paramsFile.write("    train Data,  Overfit Data, learning Rate,")
        paramsFile.write("max Iterations,   print Steps, Overfit Steps,")
        paramsFile.write("    activation,     normalize,    ANN layers,     layerSize,")
        paramsFile.write("    Decision %,        stopIt,")
        paramsFile.write("    stopReason,          Cost,     ClassRate,")
        paramsFile.write("  OvrftClassRt,")
        paramsFile.write("      OFStopIt,")
        paramsFile.write("        OFCost,   OFClassRate,OFOvrftClassRt\n")
        #paramsFile.write("%14s" %) #14 digits
        paramsFile.write("%14s," % sys.argv[1])
        paramsFile.write("%14s," % sys.argv[2])
        paramsFile.write("%14s," % str(learningRate))
        paramsFile.write("%14s," % str(trainIterations))
        paramsFile.write("%14s," % str(trainPrintStep))
        paramsFile.write("%14s," % str(overfitSteps))
        paramsFile.write("%14s," % activationType)
        paramsFile.write("%14s," % inputnormalizeType)
        paramsFile.write("%14s," % str(layers))
        

        #paramsFile.write("    train Data: " + sys.argv[1] + "\n")
        #paramsFile.write("  Overfit Data: " + sys.argv[2] + "\n")
        #paramsFile.write(" learning Rate: " + str(learningRate) + "\n")
        #paramsFile.write("max Iterations: " + str(trainIterations) + "\n")
        #paramsFile.write("   print Steps: " + str(trainPrintStep) + "\n")
        #paramsFile.write(" Overfit Steps: " + str(overfitSteps) + "\n")
        #paramsFile.write("    ANN layers: " + str(layers) + "\n")
        paramsFile.close()
        
        tf_train(fname,oname)
    elif (predictChoice):
        argv1 = sys.argv[1]
        argv2 = sys.argv[2]
        if (len(argv1.split("=")) > 1):
            trainId = argv1.split("=")[0]
            decomp = argv1.split("=")[1]
        else:
            trainId = argv1
            decomp = "700"
        fname = dir_path+"/PCAData/input" + trainId + ".csv"
        destPath += "/Results/" + sys.argv[2] + "/"
        if (trainId == "all"):
            predictAll(argv2,decomp)
        else:
            predict(fname,trainId)
    else:
        print("Not Enough Args")
        return   
    
    
#NEEDS EDITS BECAUSE M IS AN ARRAY
def writeStartGuess(W1,b1,W2,b2,D,M): #USED
    w1f = open("w1f.csv","w")
    for row in W1:
        w1f.write(','.join(str(item) for item in row) + "\n")
    w1f.close()
    
    b1f = open("b1f.csv","w")
    b1f.write(','.join(str(item) for item in b1) + "\n")
    b1f.close()
    
    w2f = open("w2f.csv","w")
    for row in W2:
        w2f.write(','.join(str(item) for item in row) + "\n")
    w2f.close()
    
    b2f = open("b2f.csv","w")
    
    b2f.write(','.join(str(item) for item in b2) + "\n")
    b2f.close()

def readStartGuess(): #USED
    w1f = pd.read_csv("w1f.csv",header=None)
    W1 = np.array(w1f.values)
    
    b1f = pd.read_csv("b1f.csv",header=None)
    b1 = np.array(b1f.values)
    b1 = b1.T
    b1.shape = (len(b1),)

    w2f = pd.read_csv("w2f.csv",header=None)
    W2 = np.array(w2f.values)

    b2f = pd.read_csv("b2f.csv",header=None)
    b2 = np.array(b2f.values)
    b2 = b2.T
    b2.shape = (len(b2),)
    
    return W1,b1,W2,b2
    
        
def tf_train(fname,oname):
    global normalize
    eegObj = eegData(destPath)
    X,Y = eegObj.getTrainData(fname)
    OX,OY = eegObj.getTrainData(oname)
    if (normalize):
        X, OX = eegObj.normalizeData(X, OX)        
    open(dir_path + "/trainStatusFile.txt","w").close()
    
    annObj = tf_ann(dir_path, destPath)
    annObj.train_init(layerSizeMultiplier, layers, trainRandomStart, trainIterations, trainPrintStep, learningRate, overfitSteps )
    annObj.fit(X,OX,Y,OY)

            
    
def predict(fname,fileId):
    global decision, normalize
    eegObj = eegData(destPath)
    
    X,Y = eegObj.getPredictData(fname,1)
    
    annObj = tf_ann(dir_path, destPath)
    annObj.predict_init(decision, layerSizeMultiplier, predictOverfit, layers)
    annObj.predict(X,Y)
    
def predictAll(trainDest, decomp):
    open(destPath + "predictStats.txt","w").close()
    with open(dir_path + "/sourceData/validPatientNights.csv") as nightsFile:
        validNights = nightsFile.readlines()
    for validNight in validNights:
        if (validNight.strip() != ""):
            fileId = validNight.split(",")[0].strip()
            fileId += validNight.split(",")[1].strip()
            fileId += "PCA" + decomp
            fname = dir_path+"/PCAData/input" + fileId + ".csv"
            predict(fname,fileId)


    
    tfile = open(destPath + "predictStats.txt")
    pstats = tfile.read()
    tfile.close()
    predictId = time.strftime("%Y%m%d%H%M%S")
    tfile = open(destPath + predictId + "predictStats.txt","w")
    tfile.write(pstats)
    tfile.close()
    avgPredictStats(trainDest,predictId)
    

def avgPredictStats(trainDest,predictId):
    global decision
    
    #dest = dir_path + "/Results/" + sys.argv[1] + "/"
    acc = []
    sens = []
    spec = []
    with open(destPath + "predictStats.txt") as pfile:
        line = pfile.readline()
        line = pfile.readline()

        while(line.strip()):
            lineSplit = line.split(",")
            acc.append(float(lineSplit[1]))
            sens.append(float(lineSplit[2]))
            spec.append(float(lineSplit[3]))
            line = pfile.readline()
    decform = "%14s" % ("%3.2f" % decision)
    accAvg = "%14s" % ("%3.2f" % (sum(acc)/len(acc)))
    sensAvg = "%14s" % ("%3.2f" % (sum(sens)/len(sens)))
    specAvg = "%14s" % ("%3.2f" % (sum(spec)/len(spec)))
    
    with open(dir_path + "/Results/" + "sessionStats.csv", "a") as sessFile:
        sessFile.write("%23s" % str(trainDest))
        sessFile.write(",%14s" % str(decform))
        sessFile.write(",%14s" % str(accAvg))
        sessFile.write(",%14s" % str(sensAvg))
        sessFile.write(",%14s" % str(specAvg))
        if (predictOverfit):
            sessFile.write(",%14s" % "overfit")
        else:
            sessFile.write(",%14s" % "train")
        sessFile.write(",%23s" % predictId)
        sessFile.write("\n")
if __name__ == "__main__":
    main()
