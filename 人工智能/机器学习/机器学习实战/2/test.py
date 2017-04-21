import operator
import numpy as np

def euclidean(x, y):
    length = len(x)
    s = sum([(x[i]-y[i])**2 for i in range(length)])
    return s**0.5

class NearestNeighbor(object):
    
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance
        
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
    
    def predict(self, X):
        Y = []
        for x in X:
            distances = []
            for y in self.X:
                distances.append(self.distance(x, y))
            nns = sorted(
                zip(distances, self.Y),
                key = operator.itemgetter(0),
            )
            nnd = {}
            for i in range(self.k):
                nnd[nns[i][1]] = nnd.get(nns[i][1], 0) + 1
            t = sorted(
                nnd.iteritems(),
                key     = operator.itemgetter(1),
                reverse = True,
            )
            Y.append(t[0][0])
        return Y
  
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect.tolist()[0]

def handwritingClassTest():
    import os
    
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = []
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat.append(img2vector('trainingDigits/%s' % fileNameStr))
    clf = NearestNeighbor(3, euclidean)
    clf.fit(trainingMat, hwLabels)
    
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = clf.predict([vectorUnderTest])[0]
        print 'the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr)
        if classifierResult != classNumStr: errorCount += 1.0
    
    print 'the total error rate is: %f' % (errorCount/float(mTest))

handwritingClassTest()