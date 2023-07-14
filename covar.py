import numpy as np

def covar(x,y):
    xbar , ybar =  x.mean() ,y.mean()
    return np.sum((x-xbar)*(y-ybar))/len(x)-1

def cov_mat(X):
    return np.array([[covar(X[0] , X[0]) , covar(X[0] , X[1])],
                    [covar(X[0] , X[1]) , covar(X[1] , X[1])]])
    
    