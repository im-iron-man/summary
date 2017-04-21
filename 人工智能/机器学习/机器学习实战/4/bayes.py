from numpy import *
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

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
    
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for _ in range(numTrainDocs):
        if trainCategory[_] == 1:
            p1Num += trainMatrix[_]
            p1Denom += sum(trainMatrix[_])
        else:
            p0Num += trainMatrix[_]
            p0Denom += sum(trainMatrix[_])
    p1Vect = log(p1Num) - log(p1Denom)
    p0Vect = log(p0Num) - log(p0Denom)
    
    return p0Vect, p1Vect, pAbusive 

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    return 1 if p1 > p0 else 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for _ in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, _))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb)
    
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb)
    
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
    
def spamTest():
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
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
        
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    
    errorCount = 0
    for docIndx in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    
    print 'the error rate is:', float(errorCount)/len(testSet)

def testingGaussianNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for _ in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, _))
        
    clf = GaussianNB()
    clf.fit(array(trainMat), array(listClasses))
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array([bagOfWords2Vec(myVocabList, testEntry)])
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]
    
    testEntry = ['stupid', 'garbage']
    thisDoc = array([bagOfWords2Vec(myVocabList, testEntry)])
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]
    
def spamGaussianNBTest():
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
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
        
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    clf = GaussianNB()
    clf.fit(array(trainMat), array(trainClasses))
    
    errorCount = 0
    for docIndx in testSet:
        wordVector = array([bagOfWords2Vec(vocabList, docList[docIndex])])
        if clf.predict(wordVector)[0] != classList[docIndex]:
            errorCount += 1
    
    print 'the error rate is:', float(errorCount)/len(testSet)
    
def testingMultinomialNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for _ in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, _))
        
    clf = MultinomialNB()
    clf.fit(array(trainMat), array(listClasses))
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array([bagOfWords2Vec(myVocabList, testEntry)])
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]
    
    testEntry = ['stupid', 'garbage']
    thisDoc = array([bagOfWords2Vec(myVocabList, testEntry)])
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]
    
def spamMultinomialNBTest():
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
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
        
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    clf = MultinomialNB()
    clf.fit(array(trainMat), array(trainClasses))
    
    errorCount = 0
    for docIndx in testSet:
        wordVector = array([bagOfWords2Vec(vocabList, docList[docIndex])])
        if clf.predict(wordVector)[0] != classList[docIndex]:
            errorCount += 1
    
    print 'the error rate is:', float(errorCount)/len(testSet)
    
def testingBernoulliNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for _ in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, _))
        
    clf = BernoulliNB()
    clf.fit(array(trainMat), array(listClasses))
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array([bagOfWords2Vec(myVocabList, testEntry)])
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]
    
    testEntry = ['stupid', 'garbage']
    thisDoc = array([bagOfWords2Vec(myVocabList, testEntry)])
    print testEntry, 'classified as:', clf.predict(thisDoc)[0]
    
def spamBernoulliNBTest():
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
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
        
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    clf = BernoulliNB()
    clf.fit(array(trainMat), array(trainClasses))
    
    errorCount = 0
    for docIndx in testSet:
        wordVector = array([bagOfWords2Vec(vocabList, docList[docIndex])])
        if clf.predict(wordVector)[0] != classList[docIndex]:
            errorCount += 1
    
    print 'the error rate is:', float(errorCount)/len(testSet)

if __name__ == '__main__':
    print 'test classifyNB'
    testingNB()
    spamTest()
    
    print '\ntest GaussianNB'
    testingGaussianNB()
    spamGaussianNBTest()
    
    print '\ntest MultinomialNB'
    testingMultinomialNB()
    spamMultinomialNBTest()
    
    print '\ntest BernoulliNB'
    testingBernoulliNB()
    spamBernoulliNBTest()