# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:57:00 2015

@author: Liu_Longpo
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

trainSet = []
w = []
b = 0
lens = 0
alpha = 0  # learn rate , default 1
    

def updateParm(sample):
    global w,b,lens,alpha
    for i in range(lens):
        w[i] = w[i] + alpha*sample[1]*sample[0][i]
    b = b + alpha*sample[1]
        
def calDistance(sample):
    global w,b
    res = 0
    for i in range(len(sample[0])):
        res += sample[0][i] * w[i]
    res += b
    res *= int(sample[1])
    return res

def trainMLP(Iter):
    print "training MLP..."
    print "-"*40
    epoch = 0
    for i in range(Iter):
        train_loss = 0
        update = False
        print "epoch",epoch, "  w: ",w,"b:",b,
        for sample in trainSet:
            res = calDistance(sample)
            if res <= 0:
                train_loss += -res
                update = True
                updateParm(sample)
        print 'train loss:',train_loss
        if update:
            epoch = epoch+1
        else:
            print "The training have convergenced,stop trianing "
            print "Optimum W:",w," Optimum b:",b
            #os._exit(0)
            break
        update = False
        
        
if __name__=="__main__":
    
    if len(sys.argv)!=4:
        print "Usage: python MLP.py trainFile modelFile"
        exit(0)
    alpha = float(sys.argv[1])
    trainFile = open(sys.argv[2])
    modelPath = sys.argv[3]
    lens = 0
    for line in trainFile:
        data = line.strip().split(' ')
        lens = len(data) - 1
        sample_all = []
        sample_data = []
        for i in range(0,lens):
            sample_data.append(float(data[i]))
        sample_all.append(sample_data) # add data
        sample_all.append(int(data[lens])) # add label
        trainSet.append(sample_all)
    trainFile.close()
    # initialize w by 0 
    for i in range(lens):
        w.append(0)
    # train model for max 100 Iteration
    trainMLP(100)
    '''
    x = np.linspace(-1,5,10)
    print x
    plt.figure()
    for i in range(len(trainSet)):
        plt.scatter(trainSet[i][0][0],trainSet[i][0][1],c=u'b')
    plt.plot(x,-(w[0]*x-b)/w[1],c=u'r')
    plt.show()
    '''
    

        
