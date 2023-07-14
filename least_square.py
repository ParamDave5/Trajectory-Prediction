import numpy as np 

def standard_least_square(cx , cy):
    X = []
    
    for i in cx:
        X.append([i**2 , i , 1])
        
    X = np.array(X)
    Y = np.array(cy)
    
    XtXinv = np.linalg.inv(np.dot(X.T , X))
    XtY = np.dot(X.T , Y)
    result = np.dot(XtXinv , XtY)
    
    return result

def standard_least_square_line(cx , cy):
    X = []
    
    for i in cx:
        X.append([i , 1])
        # X.append(i)
        
    X = np.array(X)
    Y = np.array(cy)
    
    XtXinv = np.linalg.inv(np.dot(X.T , X))
    XtY = np.dot(X.T , Y)
    result = np.dot(XtXinv , XtY)
    
    return result