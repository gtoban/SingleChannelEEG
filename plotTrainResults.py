import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
destPath = dir_path
def readTrainInfo(name):
    with open(name) as dfile:
        data = [float(row) for row in dfile.readlines()]
    return data
            
    
def main():
    global destPath
    if (len(sys.argv) != 2):
        print("Need trainId", file=sys.stderr)
        exit(1)
        
    destPath += "/Results/" + sys.argv[1] + "/"
    cost = readTrainInfo(destPath + "cost.dat")
    YTrate = readTrainInfo(destPath + "YTrate.dat")
    OYTrate = readTrainInfo(destPath + "OYTrate.dat")
    
    plt.figure()
    plt.plot(cost)
    plt.title("Cost")
    plt.savefig(destPath + "cost.png")
    #plt.show()
    
    plt.figure()
    plt.plot(YTrate)
    plt.title("Classification Rate")
    plt.savefig(destPath + "realClassRate.png")
    #plt.show()
    
    plt.figure()
    plt.plot(OYTrate)
    plt.title("Overfit Classification Rate")
    plt.savefig(destPath + "overClassRate.png")
    #plt.show()
    #
    plt.figure()
    plt.plot(YTrate)
    plt.plot(OYTrate)
    plt.title("Classification Rate vs Overfit Classification Rate")
    plt.savefig("classRate.png")
    plt.show()
    
if __name__ == "__main__":
    main()
