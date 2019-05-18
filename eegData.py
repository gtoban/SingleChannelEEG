import numpy as np
import warnings #for numpy overflow
#import matplotlib.pyplot as plt
import pandas as pd

class eegData(object):
    def __init__(self, destPath):
        self.normalize = 1
        self.destPath = destPath
    def getTrainData(self, fname):  #USED
        ###########
        #
        # Y = [REM, NOT REM]
        #
        ###########
        DATA = pd.read_csv(fname,header=None)
        data = np.array(DATA.values) #as_matrix())
        X = data[:,1:].astype(float) #np.float128) #/1000 #SHOULD I STANDARDIZE?? /1000

        YName = data[:,0]
        Y = []
        for i in range(0,len(data[:,0])): #lenXprep-1):
            Y.append([1,0] if YName[i] == "R" else [0,1])
        Y = np.array(Y).astype(float) #np.float128)
        
        return X,Y

    def getPredictData(self, fname, normalize):  #USED
        ###########
        #
        # Y = [REM, NOT REM]
        #
        ###########
        DATA = pd.read_csv(fname,header=None)
        data = np.array(DATA.values) #as_matrix())
        X = data[:,1:].astype(float) #float128) #/1000 #SHOULD I STANDARDIZE?? /1000
    
        YName = data[:,0]
        Y = []
        for i in range(0,len(data[:,0])): #lenXprep-1):
            Y.append([1,0] if YName[i] == "R" else [0,1])
        Y = np.array(Y).astype(float) #float128)

        X = self.normalizeData(X)
        return X,Y

    def normalizeData(self, X, OX = []):
        
        normSTD = np.std(X)
        normMean = np.mean(X)
        X = np.divide(np.subtract(X,normMean), normSTD)
        if (len(OX) != 0):
            OX = np.divide(np.subtract(OX,normMean), normSTD)
        with open(self.destPath + "normalMax.txt", "w") as normalFile:
            normalFile.write( str(normSTD) + "," +  str(normMean))
        
        #normMax = np.max(np.sum(X,0))
        #X = np.divide(X,normMax)
        #OX = np.divide(OX,normMax)
        #with open(destPath + "normalMax.txt", "w") as normalFile:
        #    normalFile.write(str(normMax))

        if (len(OX) == 0):
            return X
        else:
            return X, OX
