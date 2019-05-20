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
newStart = False
gparam = paramHolder(newStart) #dont overwrite old params

def main():
    #Train args  : softAnn.py fname oname dest

    trainChoice = 0
    predictChoice = 0
    argv1=argv2=argv3=""
    if (len(sys.argv) >= 4):
        argv1 = sys.argv[1]
        argv2 = sys.argv[2]
        argv3 = sys.argv[3]
        
        fname = gparam.dir_path+"/PCAData/input" + sys.argv[1] + ".csv"
        oname = gparam.dir_path+"/PCAData/input" + sys.argv[2] + ".csv"
        gparam.destPath += "/Results/" + sys.argv[3] + "/"
        if (newStart):
            paramsFile=open(gparam.destPath+"trainParams.txt","w")
            paramsFile.write("    train Data,  Overfit Data, learning Rate,")
            paramsFile.write("max Iterations,   print Steps, Overfit Steps,")
            paramsFile.write("    activation,     normalize,    ANN layers,     layerSize,")
            paramsFile.write("        stopIt,")
            paramsFile.write("    stopReason,regularization,    optimizer,       sample,")
            paramsFile.write("          Cost,     ClassRate,")
            paramsFile.write("  OvrftClassRt,")
            paramsFile.write("      OFStopIt,")
            paramsFile.write("        OFCost,   OFClassRate,OFOvrftClassRt\n")        
            paramsFile.close()
        
        if (len(sys.argv) > 4):
            trainingSets = int(sys.argv[4]) #the number of different sets of random variables to train with
            gparam.printWeights = False
            tf_trainMult(fname,oname, trainingSets)
        else:
            tf_train(fname,oname)
    else:
        print("Not Enough Args")
        return   
    
        
def tf_train(fname,oname):
    #global normalize
    eegObj = eegData(gparam.destPath)
    X,Y = eegObj.getTrainData(fname)
    OX,OY = eegObj.getTrainData(oname)
    if (gparam.normalize):
        X, OX = eegObj.normalizeData(X, OX)        
    open(gparam.dir_path + "/trainStatusFile.txt","w").close()

    #paramsFile.write("%14s" %) #14 digits
    with open(gparam.destPath+"trainParams.txt","a") as paramsFile:
        paramsFile.write("%14s," % sys.argv[1])
        paramsFile.write("%14s," % sys.argv[2])
        paramsFile.write("%14s," % str(gparam.learningRate))
        paramsFile.write("%14s," % str(gparam.trainIterations))
        paramsFile.write("%14s," % str(gparam.trainPrintStep))
        paramsFile.write("%14s," % str(gparam.overfitSteps))
        paramsFile.write("%14s," % gparam.activationType)
        paramsFile.write("%14s," % gparam.inputnormalizeType)
        paramsFile.write("%14s," % str(gparam.layers))
    
    annObj = tf_ann(gparam.dir_path, gparam.destPath)
    annObj.train_init(gparam.layerSizeMultiplier, gparam.layers, gparam.trainRandomStart, gparam.trainIterations, gparam.trainPrintStep, gparam.learningRate, gparam.overfitSteps, gparam.printWeights )
    annObj.setDropOutRegularization()
    #annObj.setRMSProp()
    annObj.setBatch()
    annObj.fit(X,OX,Y,OY)

def tf_trainMult(fname,oname, trainSets):
    #global normalize
    eegObj = eegData(gparam.destPath)
    X,Y = eegObj.getTrainData(fname)
    OX,OY = eegObj.getTrainData(oname)
    if (gparam.normalize):
        X, OX = eegObj.normalizeData(X, OX)        

    
    annObj = tf_ann(gparam.dir_path, gparam.destPath)
    gparam.getNewParamSet()
    for trainSet in range(trainSets):
        #paramsFile.write("%14s" %) #14 digits
        open(gparam.dir_path + "/trainStatusFile.txt","w").close()
        with open(gparam.destPath+"trainParams.txt","a") as paramsFile:
            paramsFile.write("%14s," % sys.argv[1])
            paramsFile.write("%14s," % sys.argv[2])
            paramsFile.write("%14s," % str(gparam.learningRate))
            paramsFile.write("%14s," % str(gparam.trainIterations))
            paramsFile.write("%14s," % str(gparam.trainPrintStep))
            paramsFile.write("%14s," % str(gparam.overfitSteps))
            paramsFile.write("%14s," % gparam.activationType)
            paramsFile.write("%14s," % gparam.inputnormalizeType)
            paramsFile.write("%14s," % str(gparam.layers))
        
        annObj.train_init(gparam.layerSizeMultiplier, gparam.layers, gparam.trainRandomStart, gparam.trainIterations, gparam.trainPrintStep, gparam.learningRate, gparam.overfitSteps, gparam.printWeights )
        
        
        if (gparam.regMethod ==  "L1"):
            annObj.setL1Regularization()
        elif (gparam.regMethod == "L2"):
            annObj.setL2Regularization()
        elif (gparam.regMethod == "dropout"): #IS DROPOUT
            annObj.setDropOutRegularization()
        elif (gparam.regMethod == "none"):
            annObj.setNoRegularization()
            
        if (gparam.optimizer == "gradientdescent"):
            annObj.setGradientDescent()
        elif (gparam.optimizer == "adam"):
            annObj.setAdamOptimizer()
        elif (gparam.optimizer == "momentum"):
            annObj.setMomentum()
        elif (gparam.optimizer == "rmsprop"):
            annObj.setRMSProp()
            
        if (gparam.batch == "batch"):
            annObj.setBatch()
        else:
            annObj.setNoBatch()
        annObj.fit(X,OX,Y,OY)

        gparam.getNewParamSet()

        
    
    
    
if __name__ == "__main__":
    main()
