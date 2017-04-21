# -*- coding: utf-8 -*-

####################
# gradient descent #
####################
def batch(X, Y, f, step=0.001, cycle=500):
    X = X[:]
    m = len(X); n = len(X[0])
    for i in range(m):
        X[i] = [1] + X[i]
    W = [1]*(n+1)
    for _ in range(cycle):
        tmpW = W[:]
        for i in range(m):
            for j in range(n+1):
                t = sum([tmpW[k]*X[i][k] for k in range(n+1)])
                W[j] += step*(Y[i]-f(t))*X[i][j]
    return W
    
def stochastic(X, Y, f, step=0.001, cycle=500):
    X = X[:]
    m = len(X); n = len(X[0])
    for i in range(m):
        X[i] = [1] + X[i]
    W = [1]*(n+1)
    for _ in range(cycle):
        for i in range(m):
            tmpW = W[:]
            for j in range(n+1):
                t = sum([tmpW[k]*X[i][k] for k in range(n+1)])
                W[j] += step*(Y[i]-f(t))*X[i][j]
    return W