import tensorflow as tf
import numpy as np
import warnings #for numpy overflow
#import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import signal
import time

#https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True


class tf_ann(object):
    def __init__(self, dir_path, destPath):
        self.dir_path = dir_path
        self.destPath = destPath
        self.decision = 50
    #check tf options

    def predict_init(self, decision, layerSizeMultiplier, predictOverfit, layers):
        self.decision = decision
        self.layerSizeMultiplier = layerSizeMultiplier
        self.predictOverfit = predictOverfit
        self.layers = layers
    
    def train_init(self, layerSizeMultiplier, layers, trainStart, trainIterations, trainPrintStep, learningRate, overfitSteps ):
        self.layerSizeMultiplier = layerSizeMultiplier
        self.layers = layers
        self.trainStart = trainStart
        self.trainIterations = trainIterations
        self.trainPrintStep = trainPrintStep
        self.learningRate = learningRate
        self.overfitSteps = overfitSteps
        self.regMethod = 0
        self.optimizer = "gradientdescent"

    def setL1Regularization(self,scale=0.005):
        self.regMethod = 1
        self.l1Scale = scale

    def setL2Regularization(self, scale=10e-3):
        self.regMethod = 2
        self.l2Scale = scale

    def setNoRegularization(self):
        self.regMethod = 0

    def setAdamOptimizer(self,beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.optimizer = "adam"
        self.adamBeta1 = beta1
        self.adamBeta2 = beta2
        self.adamEpsilon = epsilon

    def setGradientDescent(self):
        self.optimizer = "gradientdescent"

    def setMomentum(self, momentum=0.99, nesterov=True):
        self.optimizer = "momentum"
        self.momentum = momentum
        self.nesterov = nesterov

    def setRMSProp(self,decay=0.99, momentum=0.999):
        self.optimizer = "rmsprop"
        self.decay = decay
        self.momentum = momentum
        
    def predict(self,X,Y):
        
        D = len(X[0])
        print("Input Params: %5d" % D)
        MD = int(len(X[0])*self.layerSizeMultiplier) # number of input parameters
        M = self.getMVals(MD) #hidden layer size
        if (self.predictOverfit):
            print("Overfit Data: YES")
            Wb = self.readOverfitTrainedNodes(max(M))    
        else:
            print("Overfit Data: NO")
            Wb = self.readTrainedNodes(max(M))
        K = int(Y.shape[1]) 
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        #Prob = forward(X, W1, b1, W2, b2)
        with sess.as_default():
            expA = np.exp(self.tf_forward(X, Wb).eval())
        Prob = expA / expA.sum(axis=1, keepdims=True)
        #P = np.argmax(Prob,axis=1)
        P = self.argDecision(Prob)            
            
        sess.close()
        return P

    def fit(self, X,OX,Y,OY):
        killer = GracefulKiller()
        
        YT = self.argDecision(Y)    
        OYT = self.argDecision(OY)
        
        D = len(X[0])
        MD = int(len(X[0])*self.layerSizeMultiplier) # number of input parameters
        with open(self.destPath+"trainParams.txt","a") as paramsFile:
            paramsFile.write("%14s," % str(MD))
        M = self.getMVals(MD) #hidden layer size
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
    
        if self.trainStart == 1:
            Wb = []
            Wb.append(self.init_weights([D, M[0]])) # create symbolic variables
            Wb.append(self.init_weights([M[0]]))
            for i in range(len(M)-1):
                Wb.append(self.init_weights([M[i],M[i+1]]))
                Wb.append(self.init_weights([M[i+1]]))
            
            Wb.append(self.init_weights([M[-1], K]))
            Wb.append(self.init_weights([K]))
        #else:
        #W1,b1,W2,b2 = readStartGuess()
    
            
        cost=[]
        YTrate = []
        OYTrate = []
        
        stopReason="Iterations"
        stored_exception = 0
        rprev = 100
        rChangeCount = 0
        OFMax = 0
        OFiteration = 0
        OFstopC = 0
        OFstopr = 0
        OFstopro = 0
        OFstopiteration = 0
        
        
        #==================================
        #
        # tensorflow functions
        #
        #=================================

        logits = self.tf_forward(tfX, Wb)

        #==================================
        #
        # COST: L1 or L2 Regularization
        #
        #=================================


        tf_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tfY,
                logits=logits
            )
        )
        
        if (self.regMethod ==  1):
            #https://stackoverflow.com/questions/36706379/how-to-exactly-add-l1-regularisation-to-tensorflow-error-function
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=self.l1Scale, scope=None
                )
            tf_cost += tf.contrib.layers.apply_regularization(l1_regularizer, Wb)
        elif (self.regMethod == 2):
            l2_regulizer = self.l2Scale*sum([tf.nn.l2_loss(aweight) for aweight in Wb])
            tf_cost += l2_regulizer

        #==================================
        #
        # Optimizer: gradientdescent, adam, momentum, rmsprop
        #
        #=================================
            
        if (self.optimizer == "gradientdescent"):
            train_op = tf.train.GradientDescentOptimizer(self.learningRate).minimize(tf_cost)
        elif (self.optimizer == "adam"):
            train_op = tf.train.AdamOptimizer(self.learningRate, beta1=self.adamBeta1, beta2=self.adamBeta2, epsilon=self.adamEpsilon).minimize(tf_cost)
        elif (self.optimizer == "momentum"):
            train_op = tf.train.MomentumOptimizer(self.learningRate,momentum=self.momentum,use_nesterov=self.nesterov).minimize(tf_cost)
        elif (self.optimizer == "rmsprop"):
            train_op = tf.train.RMSPropOptimizer(self.learningRate,decay=self.decay,momentum=self.momentum).minimize(tf_cost)
        else:
            print("NO OPTIMIZER SELECTED")
            exit()
            
        predict_op = tf.argmax(logits, 1)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for iteration in range(self.trainIterations):
            warnings.filterwarnings("error")
            try:
                
                #==================================
                #
                # tensorflow forward
                #
                #=================================
                sess.run(train_op, feed_dict={tfX: X, tfY: Y})
                if iteration%self.trainPrintStep == 0:
                    #C = costFunc(Y,Prob)
                    C = sess.run(tf_cost, feed_dict={tfX: X, tfY: Y}) 
                    expA = np.exp(sess.run(logits, feed_dict={tfX: X, tfY: Y}))
                    Prob = expA / expA.sum(axis=1, keepdims=True)
                    P = self.argDecision(Prob)
                    expAo = np.exp(sess.run(logits, feed_dict={tfX: OX, tfY: OY}))
                    Probo = expAo / expAo.sum(axis=1, keepdims=True)
                    Po = self.argDecision(Probo)            
                    r = self.classification_rate(YT,P)
                    ro = self.classification_rate(OYT,Po)
                    #=================================
                    # Stopping Conditions
                    #================================
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
                    if (OFiteration>self.overfitSteps):
                        stopReason = "Overfit"
                        break
                    with open(self.dir_path + "/trainStatusFile.txt","a") as TSFile:
                        TSFile.write("\n----------\niteration:"+str(iteration)+"\ncost:"+str(C)+"\nclassifcation:"+str(r)+"\nOverfitClass:"+str(ro)+"\n")
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
                with open(self.dir_path + "/trainStatusFile.txt","a") as TSFile:
                    TSFile.write("\nUser Interrupt\n")
                #print("User Interrupt")
                #print(str(e), file=sys.stderr)
                break

        #=================
        # OUTSIDE LOOP
        #================
        C = sess.run(tf_cost, feed_dict={tfX: X, tfY: Y})
        expA = np.exp(sess.run(logits, feed_dict={tfX: X, tfY: Y}))
        Prob = expA / expA.sum(axis=1, keepdims=True)
        P = self.argDecision(Prob)
        
        expAo = np.exp(sess.run(logits, feed_dict={tfX: OX, tfY: OY}))
        Probo = expAo / expAo.sum(axis=1, keepdims=True)
        Po = self.argDecision(Probo)
        
        r = self.classification_rate(YT,P)
        ro = self.classification_rate(OYT,Po)
        cost.append(C)
        YTrate.append(r)
        OYTrate.append(ro)

        with open(self.dir_path + "/trainStatusFile.txt","a") as TSFile:
            TSFile.write("\n----------\niteration:"+str(iteration)+"\ncost:"+str(C)+"\nclassifcation:"+str(r)+"\nOverfitClass:"+str(ro)+"\n")
        tfile = open(self.destPath + "trainStats.txt", "w")
        tfile.write("\n----------\niteration:" + str(iteration)+"\ncost:"+str(C)+"\nclassifcation:"+str(r)+"\noverfit classification:"+str(ro)+"\n")

        if(OFiteration > self.overfitSteps):
        
            tfile.write("\n----------\nOverFit Stop At:\n")
            tfile.write("\n----------\niteration:" + str(OFstopiteration)+"\ncost:"+str(OFstopC)+"\nclassifcation:"+str(OFstopr)+"\noverfit classification:"+str(OFstopro)+"\n")
        tfile.close()

        with open(self.destPath+"trainParams.txt","a") as trainFile:
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
            self.writeTrainedNodes(Wb[index].eval(session=sess),
                            Wb[index+1].eval(session=sess),
                            i)
            self.writeOverfitTrainedNodes(OWb[index],
                                    OWb[index+1],
                                    i)
    
        sess.close()
        
        self.writeTrainInfo(cost, self.destPath + "cost.dat")
        self.writeTrainInfo(YTrate, self.destPath + "YTrate.dat")
        self.writeTrainInfo(OYTrate, self.destPath + "OYTrate.dat")


    

    def init_weights(self,shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))

    def tf_forward(self,X, Wb):
        Z = tf.nn.relu(tf.matmul(X, Wb[0]) + Wb[1])
        for i in range(1,self.layers):
            index = i*2
            Z = tf.nn.relu(tf.matmul(Z, Wb[index]) + Wb[index+1])
        return tf.matmul(Z, Wb[-2]) + Wb[-1]

    
    # Desc: gets hidden layer sizes based on layers (NUMBER OF HIDDEN LAYERS VAR)
    # input: (D = number of dimensions of input)
    # return: array M (an array of length layers with the size of each layer 
    def getMVals(self,D):
        M = []
        for i in range(self.layers):
            M.append(D)
        return M

    # Desc: Uses decision to convert percentage choice into choice
    # input: prob = percentage choice from ANN [[REM NotREM]j ... [REM notREM]]
    # return: p = absolute choice [ij ... ij]
    #             i = 0 or 1 (index REM(0) or notREM(1)), 0 < j < N

    def argDecision(self, prob):
        perc = self.decision/100
        p = []
        for i in range(len(prob)):
            p.append(0 if prob[i][0] >= perc else 1)
        return p

    def classification_rate(self, Y,P):  #USED
        n_correct = 0
        n_total = 0
        for i in range(len(Y)):
            n_total += 1
            if Y[i] == P[i]:
                n_correct += 1
            
        return float(n_correct) / n_total


    
    def allStats(self,Y,P):
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

    
    def writeTrainInfo(self,data,name):  #USED
        w1f = open(name,"w")
        for row in data:
            w1f.write(str(row) + "\n")
        w1f.close()


    def writeTrainedNodes(self, W,b,level): #USED
        w1f = open(self.destPath + "node" + str(level) + "w.csv","w")
        for row in W:
            w1f.write(','.join(str(item) for item in row) + "\n")
        w1f.close()
    
        b1f = open(self.destPath + "node" + str(level) + "b.csv","w")
        b1f.write(','.join(str(item) for item in b) + "\n")
        b1f.close()

    def writeOverfitTrainedNodes(self, W,b,level): #USED
        w1f = open(self.destPath + "overfit" + str(level) + "w.csv","w")
        for row in W:
            w1f.write(','.join(str(item) for item in row) + "\n")
        w1f.close()
    
        b1f = open(self.destPath + "overfit" + str(level) + "b.csv","w")
        b1f.write(','.join(str(item) for item in b) + "\n")
        b1f.close()


    
    def readOverfitTrainedNodes(self, Mmax):  #USED
        for dirTup in os.walk(self.destPath):
            if dirTup[0] == self.destPath:
                allFiles = dirTup[2]
                break
        nodeFiles = []
        for fname in allFiles:
            if "~" not in fname and "#" not in fname and "node" in fname:
                nodeFiles.append(fname)
            
        self.layers = int(len(nodeFiles)/2)-1
        #W = np.empty((0,Mmax,Mmax), float)
        #b = np.empty((0,Mmax), float)
        Wb = []
        for level in range(int(len(nodeFiles)/2)):
        
            w1f = pd.read_csv(self.destPath + "overfit" + str(level) + "w.csv",header=None)
            Wb.append(np.array(w1f.values)) #as_matrix()))
            #W = np.append(W, np.array(w1f.as_matrix()), axis=0)
            
            b1f = pd.read_csv(self.destPath + "overfit" + str(level) + "b.csv",header=None)
            b1 = np.array(b1f.values) #as_matrix())
            b1 = b1.T
            b1.shape = (len(b1),)
            Wb.append(b1)
            #b = np.append(b, b1, axis=0)
        
        return Wb


    def readTrainedNodes(self, Mmax):  #USED
        for dirTup in os.walk(self.destPath):
            if dirTup[0] == self.destPath:
                allFiles = dirTup[2]
                break
        nodeFiles = []
        for fname in allFiles:
            if "~" not in fname and "#" not in fname and "node" in fname:
                nodeFiles.append(fname)
            
        self.layers = int(len(nodeFiles)/2)-1
        #W = np.empty((0,Mmax,Mmax), float)
        #b = np.empty((0,Mmax), float)
        Wb = []
        for level in range(int(len(nodeFiles)/2)):
        
                w1f = pd.read_csv(self.destPath + "node" + str(level) + "w.csv",header=None)
                Wb.append(np.array(w1f.values)) #as_matrix()))
                #W = np.append(W, np.array(w1f.as_matrix()), axis=0)
        
                b1f = pd.read_csv(self.destPath + "node" + str(level) + "b.csv",header=None)
                b1 = np.array(b1f.values) #as_matrix())
                b1 = b1.T
                b1.shape = (len(b1),)
                Wb.append(b1)
                #b = np.append(b, b1, axis=0)
                
        return Wb
