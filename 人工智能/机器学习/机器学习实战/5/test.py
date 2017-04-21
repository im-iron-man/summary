# -*- coding: utf-8 -*-
import math
import copy

####################
# softmax function #
####################
def softmax(W, w, x):
    # calculate numerator
    numerator = sum([w[k]*x[k] for k in range(len(x))])
    numerator = math.pow(math.e, numerator)
    
    # calculate denominator
    denominator = []
    for v in W:
        t = sum([v[k]*x[k] for k in range(len(x))])
        denominator.append(t)
    denominator = sum([math.pow(math.e, d) for d in denominator])
    
    # return
    return numerator/denominator
    
####################
# gradient descent #
####################
def batch(X, Y, step=0.001, cycle=5000):
    # initialize train data
    X = copy.deepcopy(X)
    m = len(X); n = len(X[0])
    for i in range(m):
        X[i] = [1] + X[i]
    
    # calculate class number
    num = len(set(Y))
    
    # calculate coefficient
    W = [[1]*(n+1)]*num
    for _ in range(cycle):
        tmpW = copy.deepcopy(W)
        for i in range(m):
            for j in range(num):
                for l in range(n+1):
                    W[j][l] += step*((1 if Y[i] == j else 0)-softmax(tmpW, tmpW[j], X[i]))*X[i][l]
    return W
    
def stochastic(X, Y, step=0.001, cycle=5000):
    # initialize train data
    X = copy.deepcopy(X)
    m = len(X); n = len(X[0])
    for i in range(m):
        X[i] = [1] + X[i]
    
    # calculate class number
    num = len(set(Y))
    
    # calculate coefficient
    W = [[1]*(n+1)]*num
    for _ in range(cycle):
        for i in range(m):
            tmpW = W[:]
            for j in range(num):
                for l in range(n+1):
                    W[j][l] += step*((1 if Y[i] == j else 0)-softmax(tmpW, tmpW[j], X[i]))*X[i][l]
    return W

############
# SoftmaxR #
############
class SoftmaxR(object):
    
    def __init__(self, gd=batch, step=0.001, cycle=5000):
        self.gd    = gd
        self.step  = step
        self.cycle = cycle
        
    def fit(self, X, Y):
        setY = list(set(Y))
        self.map = {k: setY[k] for k in range(len(setY))}
        self.rev = {setY[k]: k for k in range(len(setY))}
        tmpY = [self.rev[y] for y in Y]
        self.W = self.gd(X, tmpY, self.step, self.cycle)
        
    def predict(self, X):
        Y = []
        for x in X:
            x = [1] + x
            k = 0
            v = 0
            for l in range(len(self.map)):
                t = softmax(self.W, self.W[l], x)
                if t > v:
                    k = l
                    v = t
            Y.append(self.map[l])
        return Y
        
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
    
X, Y = loadDataSet()
clf = SoftmaxR(step=1)
clf.fit(X, Y)
print clf.W