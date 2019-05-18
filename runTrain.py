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
    #Train args  : softAnn.py fname oname dest

    trainChoice = 0
    predictChoice = 0
    argv1=argv2=argv3=""
    if (len(sys.argv) == 4):
        argv1 = sys.argv[1]
        argv2 = sys.argv[2]
        argv3 = sys.argv[3]
        
        fname = gparam.dir_path+"/PCAData/input" + sys.argv[1] + ".csv"
        oname = gparam.dir_path+"/PCAData/input" + sys.argv[2] + ".csv"
        gparam.destPath += "/Results/" + sys.argv[3] + "/"
        paramsFile=open(gparam.destPath+"trainParams.txt","w")
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
        paramsFile.write("%14s," % str(gparam.learningRate))
        paramsFile.write("%14s," % str(gparam.trainIterations))
        paramsFile.write("%14s," % str(gparam.trainPrintStep))
        paramsFile.write("%14s," % str(gparam.overfitSteps))
        paramsFile.write("%14s," % gparam.activationType)
        paramsFile.write("%14s," % gparam.inputnormalizeType)
        paramsFile.write("%14s," % str(gparam.layers))
        

        #paramsFile.write("    train Data: " + sys.argv[1] + "\n")
        #paramsFile.write("  Overfit Data: " + sys.argv[2] + "\n")
        #paramsFile.write(" learning Rate: " + str(learningRate) + "\n")
        #paramsFile.write("max Iterations: " + str(trainIterations) + "\n")
        #paramsFile.write("   print Steps: " + str(trainPrintStep) + "\n")
        #paramsFile.write(" Overfit Steps: " + str(overfitSteps) + "\n")
        #paramsFile.write("    ANN layers: " + str(layers) + "\n")
        paramsFile.close()
        
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
    
    annObj = tf_ann(gparam.dir_path, gparam.destPath)
    annObj.train_init(gparam.layerSizeMultiplier, gparam.layers, gparam.trainRandomStart, gparam.trainIterations, gparam.trainPrintStep, gparam.learningRate, gparam.overfitSteps )
    annObj.fit(X,OX,Y,OY)

            

if __name__ == "__main__":
    main()
