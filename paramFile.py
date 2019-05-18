import os

class paramHolder(object):
    def __init__(self):
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
