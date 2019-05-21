import os
import numpy as np

class paramHolder(object):
    def __init__(self, newStart = True):
        self.learningRate = 10e-3 #learning rate
        self.trainIterations = 100000 #100000 #10000
        self.trainPrintStep = 100
        self.overfitSteps = 10
        #overfitSteps:
        # the number of trainPrintSteps
        # without an increase in accuracy on the overfit data
        # until its considered overfitting.
        self.trainPrint = 0
        self.trainRandomStart = 1
        self.writeStartingGuess = 0
        self.normalize = 1 ## used to specify in normalizing is used (x - mean)/std
        self.inputnormalizeType="std Dev"
        self.activationType="relu"
        self.layers = 2
        self.layerSizeMultiplier = 0.5 ## LAYER sizes are a multiple of input dimensions
        self.decision = 50 # if REM ([REM NotREM]) >= decision then REM is choice
        self.stopReason = "iterations"
        self.predictOverfit = 0 #use weights from overfit stop

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.destPath = self.dir_path
        self.trainId = ""
        self.trainDestination = self.dir_path
        self.printWeights = True
        self.regMethod = "none"
        self.optimizer = "gradientdescent"
        self.batch = "none"
        self.getString()
        if (newStart):
            with open(self.dir_path + "/trainParamsFile.txt", "w") as pf:
                pf.write(self.paramString + "\n")
        

    def getNewParamSet(self):
        regMethods = ["none", "L2", "dropout"]
        optimizers = ["none", "adam", "gradientdescent", "momentum", "rmsprop"]
        batches = ["none", "batch"]
        duplicate = True

        while (duplicate): 
            self.layerSizeMultiplier = int(np.random.normal(.9,.175)*10)/10 # .2 - 1.5
            temp = 1 + abs(int(np.random.normal(0.0,4.0))) #want more 2
            self.layers = 2**temp #2 - 16
            if (self.layers > 16 or self.layers < 2):
                print("LAYERS OUT OF RANGE:", self.layers)
                exit()
              
        
            #10 ^ -2 - 10^-7 Want bigger steps for bigger layers
            # https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
            # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
            temp = int((((1/self.layers - 1/16) * (7 - 2)) / (1/2 - 1/16)) + 1) # get a new number
            temp += abs(int(np.random.normal(0.0,1))) # all it to change a little bit
            temp = -1*min(temp, 2 + abs(int(np.random.normal(0.0,2.5)))%5)
            self.learningRate = 10**temp
            if (self.learningRate > 10**-2 or self.learningRate < 10**-7):
                print("LEARNING RATE OUT OF RANGE:", self.learningRate)
                exit()
              
            self.regMethod = regMethods[np.random.randint(0,len(regMethods))]
            self.optimizer = optimizers[np.random.randint(0,len(optimizers))]
            self.batch = batches[np.random.randint(0,len(batches))]
            duplicate = self.checkDuplicate()

    def getString(self):
        self.paramString = (str(self.layerSizeMultiplier) + "|" +
                            str(self.learningRate) + "|" +
                            str(self.regMethod) + "|" +
                            str(self.optimizer) + "|" +
                            str(self.batch) + "|" +
                            str(self.layers))
    def checkDuplicate(self):
        self.getString()
        with open(self.dir_path + "/trainParamsFile.txt") as pf:
            line = pf.readline().strip()
            while (line != ""):
                if line == self.paramString:
                    return True
                line = pf.readline().strip()
        with open(self.dir_path + "/trainParamsFile.txt", "a") as pf:
            pf.write(self.paramString + "\n")
        return False
