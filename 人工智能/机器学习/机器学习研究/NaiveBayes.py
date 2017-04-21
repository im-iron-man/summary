# -*- coding: utf-8 -*-
import math

##############
# NaiveBayes #
##############
class NaiveBayes(object):
    
    def fit(self, X, Y):
        trainNum = len(X)
        everyNum = len(X[0])
        
        # calculate class probility
        self.classProb = {}
        for y in Y:
            self.classProb[y] = self.classProb.get(y, 0.0) + 1.0
        for k, v in self.classProb.iteritems():
            self.classProb[k] = math.log(v/trainNum)
            
        # calculate conditional probility
        self.condProb = {}
        count = {}
        for i in range(trainNum):
            if Y[i] not in self.condProb:
                self.condProb[Y[i]] = [1.0] * everyNum
                count[Y[i]] = 2.0
            for j in range(everyNum):
                self.condProb[Y[i]][j] += X[i][j]
                count[Y[i]] += X[i][j]
        for k in count:
            for i in range(everyNum):
                self.condProb[k][i] = math.log(self.condProb[k][i]/count[k])
                
    def predict(self, X):
        Y = []
        for x in X:
            t = {}
            for k in self.classProb:
                t[k] = self.classProb[k] + sum([x[i]*self.condProb[k][i] for i in range(len(x))])
            index = t.keys()[0]
            for k, v in t.iteritems():
                if v > t[index]: index = k
            Y.append(index)
        return Y
        
########
# test #
########
def loadDataSet():
    X = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],   
    ]
    Y = [0, 1, 0, 1, 0, 1]
    return X, Y

class Standard(object):
    
    def __init__(self, X):
        setX = set([])
        for x in X:
            setX |= set(x)
        self.standard = list(setX)
        
    def word2vec(self, x):
        vector = [0] * len(self.standard)
        for _ in x:
            if _ in self.standard:
                vector[self.standard.index(_)] += 1
        return vector

if __name__ == '__main__':
    X, Y = loadDataSet()
    standard = Standard(X)
    newX = []
    for x in X:
        newX.append(standard.word2vec(x))
    clf = NaiveBayes()
    clf.fit(newX, Y)
    X1 = [['love', 'my', 'dalmation']]
    X2 = [['stupid', 'garbage']]
    print clf.predict([standard.word2vec(X1[0])])[0]
    print clf.predict([standard.word2vec(X2[0])])[0]