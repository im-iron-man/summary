# -*- coding: utf-8 -*-
import math
import operator

######################
# auxiliary function #
######################
def _splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def _calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt
 
def _chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = _calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = _splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*_calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def _majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(
        classCount.iteritems(),
        key     = operator.itemgetter(1),
        reverse = True,
    )
    return sortedClassCount[0][0]

def _createTree(dataSet, labels):
    labels = labels[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return _majorityCnt(classList)
    bestFeat = _chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = _createTree(
            _splitDataSet(dataSet, bestFeat, value),
            labels,
        )
    return myTree

def _classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict:
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = _classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#################
# DecisionTreeC #
#################    
class DecisionTreeC(object):
    
    def fit(self, X, Y):
        lenX = len(X[0])
        lenY = len(Y)
        self.myDat = [X[i] + [Y[i]] for i in range(lenY)]
        self.labels = range(lenX)
        self.myTree = _createTree(self.myDat, self.labels)
        
    def predict(self, X):
        Y = []
        for x in X:
            y = _classify(self.myTree, self.labels, x)
            Y.append(y)
        return Y

########
# test #
########     
def loadDataSet():
    X = [
        [1, 1],
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 1],
    ]
    Y = ['yes', 'yes', 'no', 'no', 'no']
    return X, Y

if __name__ == '__main__':
    X, Y = loadDataSet()
    clf = DecisionTreeC()
    clf.fit(X, Y)
    print clf.predict([[1, 0]])[0]
    print clf.predict([[1, 1]])[0]