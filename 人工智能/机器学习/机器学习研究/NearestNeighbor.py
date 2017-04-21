# -*- coding: utf-8 -*-
import operator

############
# distance #
############
def euclidean(x, y):
    length = len(x)
    s = sum([(x[i]-y[i])**2 for i in range(length)])
    return s**0.5

def pearson(x, y):
    length = len(x)
    sumX = sum(x); sumY = sum(y)
    sumXSq = sum([i**2 for i in x])
    sumYSq = sum([j**2 for j in y])
    sumXY = sum([i*j for i, j in zip(x, y)])
    numerator = sumXY - sumX*sumY/length
    denominator = ((sumXSq-sumX**2/length)*(sumYSq-sumY**2/length))**0.5
    if denominator == 0: return 0
    return numerator/denominator

###################
# NearestNeighbor #
################### 
class NearestNeighbor(object):
    
    def __init__(self, k=3, distance=euclidean):
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

########
# test #
########
def loadDataSet():
    X = [
        [1.0, 1.1],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.1],
    ]
    Y = ['A', 'A', 'B', 'B']
    return X, Y

if __name__ == '__main__':
    X, Y = loadDataSet()
    clf = NearestNeighbor()
    clf.fit(X, Y)
    print clf.predict([[0.0, 0.0]])[0]