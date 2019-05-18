import tensorflow as tf
import numpy as np
import warnings #for numpy overflow
#import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import signal
import time
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
normalizeType="std Dev"
activationType="relu"
layers = 4
layerSizeMultplier = 0.5 ## LAYER sizes are a multiple of input dimensions
decision = 50 # if REM ([REM NotREM]) >= decision then REM is choice
stopReason = "iterations"
predictOverfit = 0 #use weights from overfit stop

dir_path = os.path.dirname(os.path.realpath(__file__))
destPath = dir_path
trainId = ""
trainDestination = dir_path

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

def main():
    global destPath, trainId, layers, decision, normalizeType, activationType
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
        paramsFile.write("%14s," % normalizeType)
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

def writeTrainInfo(data,name):  #USED
   w1f = open(name,"w")
   for row in data:
       w1f.write(str(row) + "\n")
   w1f.close()
   
    
    
def getTrainData(fname):  #USED
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

def getPredictData(fname):  #USED
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
    return X,Y

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

def writeTrainedNodes(W,b,level): #USED
    w1f = open(destPath + "node" + str(level) + "w.csv","w")
    for row in W:
        w1f.write(','.join(str(item) for item in row) + "\n")
    w1f.close()
    
    b1f = open(destPath + "node" + str(level) + "b.csv","w")
    b1f.write(','.join(str(item) for item in b) + "\n")
    b1f.close()

def writeOverfitTrainedNodes(W,b,level): #USED
    w1f = open(destPath + "overfit" + str(level) + "w.csv","w")
    for row in W:
        w1f.write(','.join(str(item) for item in row) + "\n")
    w1f.close()
    
    b1f = open(destPath + "overfit" + str(level) + "b.csv","w")
    b1f.write(','.join(str(item) for item in b) + "\n")
    b1f.close()
    

def readOverfitTrainedNodes(Mmax):  #USED
    global layers
    for dirTup in os.walk(destPath):
        if dirTup[0] == destPath:
            allFiles = dirTup[2]
            break
    nodeFiles = []
    for fname in allFiles:
        if "~" not in fname and "#" not in fname and "node" in fname:
            nodeFiles.append(fname)
            
    layers = int(len(nodeFiles)/2)-1
    #W = np.empty((0,Mmax,Mmax), float)
    #b = np.empty((0,Mmax), float)
    Wb = []
    for level in range(int(len(nodeFiles)/2)):
        
        w1f = pd.read_csv(destPath + "overfit" + str(level) + "w.csv",header=None)
        Wb.append(np.array(w1f.values)) #as_matrix()))
        #W = np.append(W, np.array(w1f.as_matrix()), axis=0)
        
        b1f = pd.read_csv(destPath + "overfit" + str(level) + "b.csv",header=None)
        b1 = np.array(b1f.values) #as_matrix())
        b1 = b1.T
        b1.shape = (len(b1),)
        Wb.append(b1)
        #b = np.append(b, b1, axis=0)
        
    return Wb

def readTrainedNodes(Mmax):  #USED
    global layers
    for dirTup in os.walk(destPath):
        if dirTup[0] == destPath:
            allFiles = dirTup[2]
            break
    nodeFiles = []
    for fname in allFiles:
        if "~" not in fname and "#" not in fname and "node" in fname:
            nodeFiles.append(fname)
            
    layers = int(len(nodeFiles)/2)-1
    #W = np.empty((0,Mmax,Mmax), float)
    #b = np.empty((0,Mmax), float)
    Wb = []
    for level in range(int(len(nodeFiles)/2)):
        
        w1f = pd.read_csv(destPath + "node" + str(level) + "w.csv",header=None)
        Wb.append(np.array(w1f.values)) #as_matrix()))
        #W = np.append(W, np.array(w1f.as_matrix()), axis=0)
        
        b1f = pd.read_csv(destPath + "node" + str(level) + "b.csv",header=None)
        b1 = np.array(b1f.values) #as_matrix())
        b1 = b1.T
        b1.shape = (len(b1),)
        Wb.append(b1)
        #b = np.append(b, b1, axis=0)
        
    return Wb
##################################################
#
#                FUNCTIONS
#
##################################################

def classification_rate(Y,P):  #USED
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
            
    return float(n_correct) / n_total

def allStats(Y,P):
    TP = 0 #TRUE POSITIVE (CORRECTLY IDENTIFIED REM)
    FP = 0 #FALSE POSITIVE (INCORRECTLY IDENTIFIED REM)
    TN = 0 #TRUE NEGATIVE (CORRECTLY IDENTIFIED NOT REM)
    FN = 0 #FALSE NEGATIVE (INCORRECTLY IDENTIFIED NOT REM)
    for i in range(len(Y)):
        
        if Y[i] == P[i]:
            if P[i] == 0:
                TP += 1
            else:
                TN += 1
        else:
            if P[i] == 0:
                FP += 1
            else:
                FN += 1
    acc = (TP+TN)/(TP+FP+TN+FN) #accuracy
    sens = TP/(TP+FN) #sensitivity
    spec = TN/(TN+FP) #specificity

    return acc,sens,spec
            
            
    #return float(n_correct) / n_total

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))

def tf_forward(X, Wb):
    Z = tf.nn.relu(tf.matmul(X, Wb[0]) + Wb[1])
    for i in range(1,layers):
        index = i*2
        Z = tf.nn.relu(tf.matmul(Z, Wb[index]) + Wb[index+1])
    return tf.matmul(Z, Wb[-2]) + Wb[-1]

# Desc: gets hidden layer sizes based on layers (NUMBER OF HIDDEN LAYERS VAR)
# input: (D = number of dimensions of input)
# return: array M (an array of length layers with the size of each layer 
def getMVals(D):
    M = []
    for i in range(layers):
        M.append(D)
    return M

# Desc: Uses decision to convert percentage choice into choice
# input: prob = percentage choice from ANN [[REM NotREM]j ... [REM notREM]]
# return: p = absolute choice [ij ... ij]
#             i = 0 or 1 (index REM(0) or notREM(1)), 0 < j < N
def argDecision(prob):
    global decision
    perc = decision/100
    p = []
    for i in range(len(prob)):
        p.append(0 if prob[i][0] >= perc else 1)
    return p
        
def tf_train(fname,oname):
    global normalize
    killer = GracefulKiller()
    lr = learningRate
    X,Y = getTrainData(fname)
    OX,OY = getTrainData(oname)
    if (normalize):
        normSTD = np.std(X)
        normMean = np.mean(X)
        X = np.divide(np.subtract(X,normMean), normSTD)
        OX = np.divide(np.subtract(OX,normMean), normSTD)
        with open(destPath + "normalMax.txt", "w") as normalFile:
            normalFile.write( str(normSTD) + "," +  str(normMean))
        
        #normMax = np.max(np.sum(X,0))
        #X = np.divide(X,normMax)
        #OX = np.divide(OX,normMax)
        #with open(destPath + "normalMax.txt", "w") as normalFile:
        #    normalFile.write(str(normMax))
    open(dir_path + "/trainStatusFile.txt","w").close()
    #open(dir_path + "/safeStop.txt","w").close()
    #YT = np.argmax(Y,axis=1)
    YT = argDecision(Y)    
    #OYT = np.argmax(OY,axis=1)
    OYT = argDecision(OY)
    D = len(X[0])
    MD = int(len(X[0])*layerSizeMultplier) # number of input parameters
    with open(destPath+"trainParams.txt","a") as paramsFile:
        paramsFile.write("%14s," % str(MD))
        paramsFile.write("%14s," % str(decision))
    M = getMVals(MD) #hidden layer size
    K = int(Y.shape[1]) # of classes (number of output parameters)

    #==================================
    #
    # tensorflow X and Y
    #
    #=================================
    tfX = tf.placeholder(tf.float32, [None, D])
    tfY = tf.placeholder(tf.float32, [None, K])
    tfOX = tf.placeholder(tf.float32, [None, D])
    

    #==================================
    #
    # tensorflow Weights and Biases
    #
    #=================================
    
    if trainRandomStart == 1:
        #W1 = np.random.randn(D,M)
        #b1 = np.random.randn(M)
        #W2 = np.random.randn(M,K)
        #b2 = np.random.randn(K)
        Wb = []
        Wb.append(init_weights([D, M[0]])) # create symbolic variables
        Wb.append(init_weights([M[0]]))
        for i in range(len(M)-1):
            Wb.append(init_weights([M[i],M[i+1]]))
            Wb.append(init_weights([M[i+1]]))
        
        Wb.append(init_weights([M[-1], K]))
        Wb.append(init_weights([K]))
    else:
        W1,b1,W2,b2 = readStartGuess()
    
    cost=[]
    YTrate = []
    OYTrate = []

    stopReason="Iterations"
    stored_exception = 0
    rprev = 100
    rChangeCount = 0
    OFMax = 0
    #OFW1 = W1
    #OFb1 = b1
    #OFW1 = W1
    #OFb1 = b1
    OFiteration = 0
    OFstopC = 0
    OFstopr = 0
    OFstopro = 0
    OFstopiteration = 0
    #probPrev = Y
    #W1Prev = W1
    #b1Prev = b1
    #W2Prev = W2
    #b2Prev = b2

    #==================================
    #
    # tensorflow functions
    #
    #=================================

    logits = tf_forward(tfX, Wb)

    tf_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tfY,
            logits=logits
        )
    )

    train_op = tf.train.GradientDescentOptimizer(lr).minimize(tf_cost)
    
    predict_op = tf.argmax(logits, 1)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

   
    for iteration in range(trainIterations):
         warnings.filterwarnings("error")
         try:
            # Prob, Z = forward(X, W1, b1, W2, b2)#Predictions or Probability
            
            #==================================
            #
            # tensorflow forward
            #
            #=================================
            sess.run(train_op, feed_dict={tfX: X, tfY: Y})
            #Probo,Zo = forward(OX, W1, b1, W2, b2) #For OverFit Test
            if iteration%trainPrintStep == 0:
                #C = costFunc(Y,Prob)
                C = sess.run(tf_cost, feed_dict={tfX: X, tfY: Y}) 
                #P = np.argmax(Prob,axis=1)
                
                #expA = np.exp(tf_forward(X, Wb).eval())
                
                expA = np.exp(sess.run(logits, feed_dict={tfX: X, tfY: Y}))
                Prob = expA / expA.sum(axis=1, keepdims=True)
                P = argDecision(Prob)
                #P = sess.run(predict_op, feed_dict={tfX: X, tfY: Y})
                
                expAo = np.exp(sess.run(logits, feed_dict={tfX: OX, tfY: OY}))
                Probo = expAo / expAo.sum(axis=1, keepdims=True)
                Po = argDecision(Probo)            
                #Po = np.argmax(Probo,axis=1)
                #Po = sess.run(predict_op, feed_dict={tfX: OX, tfY: OY})            
                r = classification_rate(YT,P)
                ro = classification_rate(OYT,Po)
                #=================================
                # Stopping Conditions
                #================================
                #if stored_exception:
                #    stopReason = "User Stop"
                #    break
                if abs(rprev-r) < 0.001:
                    #if np.abs(C) < 10e-1:
                    rprev = r
                    rChangeCount += 1
                    if (rChangeCount > 10):
                        stopReason = "ClassRateChange"
                        break
                else:
                    rChangeCount = 0
                        
                if (ro>OFMax):
                    OFMax = ro
                    OFiteration=1
                    #OWb = Wb[:]
                    OWb = []
                    for i in range(int(len(Wb)/2)):
                        index = i*2
                        OWb.append(Wb[index].eval(session=sess))
                        OWb.append(Wb[index+1].eval(session=sess))
                        OFstopC = C
                        OFstopr = r
                        OFstopro = ro
                        OFstopiteration = iteration
                else:
                    if (ro < .3):
                            OFiteration = 1
                    else:
                        #if overfit is 10% < OFMax
                        if (r > ro and (OFMax - 0.10) > ro ):
                            OFiteration += 1
                        else:
                            OFiteration = 1
                                
                    #OFiteration+=1
                    if (OFiteration>overfitSteps):
                        stopReason = "Overfit"
                        break
                #print("\n----------\niteration:",iteration,"\ncost:",C,"\nclassifcation:",r,"\nOverfitClass:",ro,"\n")
                with open(dir_path + "/trainStatusFile.txt","a") as TSFile:
                    TSFile.write("\n----------\niteration:"+str(iteration)+"\ncost:"+str(C)+"\nclassifcation:"+str(r)+"\nOverfitClass:"+str(ro)+"\n")
                #with open(dir_path + "/safeStop.txt") as ssfile:
                #    line = ssfile.readline().strip()
                #    if (line != ""):
                #        killer.kill_now = True
                cost.append(C)
                YTrate.append(r)
                OYTrate.append(ro)
         except KeyboardInterrupt as e:
             stored_exception=sys.exc_info()
             stopReason = "User Stop"
             print("User Interrupt")
             print(str(e), file=sys.stderr)
             break
         except Warning as e:
             print("Caught Warning")
             print(str(e), file=sys.stderr)
             stopReason = "Compute Error"
             break

         if killer.kill_now:
             #stored_exception=sys.exc_info()
             stopReason = "User Stop"
             with open(dir_path + "/trainStatusFile.txt","a") as TSFile:
                TSFile.write("\nUser Interrupt\n")
             #print("User Interrupt")
             #print(str(e), file=sys.stderr)
             break

    #=================
    # OUTSIDE LOOP
    #================
    #C = costFunc(Y,Prob)
    C = sess.run(tf_cost, feed_dict={tfX: X, tfY: Y})

    #P = np.argmax(Prob,axis=1)
    expA = np.exp(sess.run(logits, feed_dict={tfX: X, tfY: Y}))
    Prob = expA / expA.sum(axis=1, keepdims=True)
    P = argDecision(Prob)
    #P2 = sess.run(predict_op, feed_dict={tfX: X, tfY: Y})

    #Po = np.argmax(Probo,axis=1)
    expAo = np.exp(sess.run(logits, feed_dict={tfX: OX, tfY: OY}))
    Probo = expAo / expAo.sum(axis=1, keepdims=True)
    Po = argDecision(Probo)
    #Po = sess.run(predict_op, feed_dict={tfX: OX, tfY: OY})                    

    r = classification_rate(YT,P)
    ro = classification_rate(OYT,Po)
    cost.append(C)
    YTrate.append(r)
    OYTrate.append(ro)

    #print("\n----------\niteration:",iteration,"\ncost:",C,"\nclassifcation:",r,"\nOverfitClass:",ro,"\n")
    with open(dir_path + "/trainStatusFile.txt","a") as TSFile:
        TSFile.write("\n----------\niteration:"+str(iteration)+"\ncost:"+str(C)+"\nclassifcation:"+str(r)+"\nOverfitClass:"+str(ro)+"\n")
    file = open(destPath + "trainStats.txt", "w")
    file.write("\n----------\niteration:" + str(iteration)+"\ncost:"+str(C)+"\nclassifcation:"+str(r)+"\noverfit classification:"+str(ro)+"\n")

    if(OFiteration > overfitSteps):
        
        file.write("\n----------\nOverFit Stop At:\n")
        file.write("\n----------\niteration:" + str(OFstopiteration)+"\ncost:"+str(OFstopC)+"\nclassifcation:"+str(OFstopr)+"\noverfit classification:"+str(OFstopro)+"\n")
    file.close()

    with open(destPath+"trainParams.txt","a") as trainFile:
        trainFile.write("%14s," % str(iteration))
        trainFile.write("%14s," % stopReason)
        trainFile.write("%14s," % (str("%4.2f" % C)))
        trainFile.write("%14s," % (str("%4.2f" % r)))
        trainFile.write("%14s," % (str("%4.2f" % ro)))
                        
        trainFile.write("%14s," % str(OFstopiteration))
        trainFile.write("%14s," % (str("%4.2f" % OFstopC)))
        trainFile.write("%14s," % (str("%4.2f" % OFstopr)))
        trainFile.write("%14s" % (str("%4.2f" % OFstopro)))
        
                         
    for i in range(int(len(Wb)/2)):
        index = i*2
        writeTrainedNodes(Wb[index].eval(session=sess),
                          Wb[index+1].eval(session=sess),
                          i)
        writeOverfitTrainedNodes(OWb[index],
                                 OWb[index+1],
                                 i)
    
    sess.close()
     
    writeTrainInfo(cost, destPath + "cost.dat")
    writeTrainInfo(YTrate, destPath + "YTrate.dat")
    writeTrainInfo(OYTrate, destPath + "OYTrate.dat")

    #if (SHOWPLOTS):
    #    plt.figure()
    #    plt.plot(cost)
    #    plt.title("Cost")
    #    plt.savefig(destPath + "cost.png")
    #    #plt.show()
    #    
    #    plt.figure()
    #    plt.plot(YTrate)
    #    plt.title("Classification Rate")
    #    plt.savefig(destPath + "realClassRate.png")
    #    #plt.show()
    #    
    #    plt.figure()
    #    plt.plot(OYTrate)
    #    plt.title("Overfit Classification Rate")
    #    plt.savefig(destPath + "overClassRate.png")
    #    #plt.show()
    #    #
    #    plt.figure()
    #    plt.plot(YTrate)
    #    plt.plot(OYTrate)
    #    plt.title("Classification Rate vs Overfit Classification Rate")
    #    plt.savefig("classRate.png")
    #    plt.show()
        
    
def predict(fname,fileId):
    global decision, normalize
    X,Y = getPredictData(fname)
    
    #print(np.shape(X))
    if (normalize == 900):
        with open(destPath + "normalMax.txt") as normalFile:
            line = normalFile.readline()
            normSTD = float(line.split(",")[0])
            normMean = float(line.split(",")[1])
        X = np.divide(np.subtract(X,normMean),normSTD)

        #with open(destPath + "normalMax.txt") as normalFile:
        #    normMax = float(normalFile.readline())
        #X = np.divide(X,normMax)
    #print(np.shape(X))
        
    YT = np.argmax(Y,axis=1)
    D = len(X[0])
    print("Input Params: %5d" % D)
    MD = int(len(X[0])*layerSizeMultplier) # number of input parameters
    M = getMVals(MD) #hidden layer size
    if (predictOverfit):
        print("Overfit Data: YES")
        Wb = readOverfitTrainedNodes(max(M))    
    else:
        print("Overfit Data: NO")
        Wb = readTrainedNodes(max(M))
    K = int(Y.shape[1]) 
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    #Prob = forward(X, W1, b1, W2, b2)
    with sess.as_default():
        expA = np.exp(tf_forward(X, Wb).eval())
    Prob = expA / expA.sum(axis=1, keepdims=True)
    #P = np.argmax(Prob,axis=1)
    P = argDecision(Prob)            
    acc,sens,spec=allStats(YT,P)
    print("Decision:",decision)
    print("Accuracy:",acc)
    print("Sensitivity:",sens)
    print("Specificity:",spec)
    print("Classification Rate",classification_rate(YT,P))
    file = open(destPath + "predictStats.txt", "a")
    file.write(fileId+ ", " + str(acc) + "," + str(sens) + "," +str(spec) + "," + str(classification_rate(YT,P)) + "\n")
    file.close()
    
    sess.close()

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
