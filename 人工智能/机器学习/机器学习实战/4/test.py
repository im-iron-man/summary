import math

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

def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec
    
def createVocabList(dataSet):
    vocabSet = set([])
    for _ in dataSet:
        vocabSet = vocabSet | set(_)
    return list(vocabSet)
    
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for _ in inputSet:
        if _ in vocabList:
            returnVec[vocabList.index(_)] = 1
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for _ in inputSet:
        if _ in vocabList:
            returnVec[vocabList.index(_)] += 1
    return returnVec
    
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for _ in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, _))
    clf = NaiveBayes()
    clf.fit(trainMat, listClasses)
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = [setOfWords2Vec(myVocabList, testEntry)]
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]
    
    testEntry = ['stupid', 'garbage']
    thisDoc = [setOfWords2Vec(myVocabList, testEntry)]
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
    
def spamTest():
    import numpy as np
    docList = []
    classList = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    
    trainingSet = range(50)
    
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
        
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    clf = NaiveBayes()
    clf.fit(trainMat, trainClasses)
    
    errorCount = 0
    for docIndx in testSet:
        wordVector = [setOfWords2Vec(vocabList, docList[docIndex])]
        if clf.predict(wordVector)[0] != classList[docIndex]:
            errorCount += 1
    
    print 'the error rate is:', float(errorCount)/len(testSet)
    
def testingNB2():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for _ in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, _))
    clf = NaiveBayes()
    clf.fit(trainMat, listClasses)
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = [bagOfWords2Vec(myVocabList, testEntry)]
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]
    
    testEntry = ['stupid', 'garbage']
    thisDoc = [bagOfWords2Vec(myVocabList, testEntry)]
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]
    
def spamTest2():
    import numpy as np
    docList = []
    classList = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    
    trainingSet = range(50)
    
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
        
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    clf = NaiveBayes()
    clf.fit(trainMat, trainClasses)
    
    errorCount = 0
    for docIndx in testSet:
        wordVector = [bagOfWords2Vec(vocabList, docList[docIndex])]
        if clf.predict(wordVector)[0] != classList[docIndex]:
            errorCount += 1
    
    print 'the error rate is:', float(errorCount)/len(testSet)
    
if __name__ == '__main__':
    testingNB()
    spamTest()
    
    testingNB2()
    spamTest2()