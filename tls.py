import numpy as np
from svd import *

def total_least_square_parabola(cX ,cY):
    X = []
    for i  in cX:
        X.append([i**2 , i])
    
    X.np.array(X)
    y = np.array(cY)
    
    if X.ndim is 1:
        n = 1
        X= X.reshape(len(X) , 1)
        
    else:
        n = np.array(X).shape[1]
        
    z = np.vstack((X.T, y)).T
    u , s , vt = svd(z)
    
    V = vt.T
    Vxy = V[:n,n:]
    Vyy = V[n:,n:]
    
    a_tls = -Vxy/Vyy
    
    return a_tls.flatten()

def total_least_square_line(cX ,cY):
    X = []
    for i  in cX:
        X.append(i)
    
    X = np.array(X)
    y = np.array(cY)
    
    if X.ndim is 1:
        n = 1
        X= X.reshape(len(X) , 1)
        
    else:
        n = np.array(X).shape[1]
        
    z = np.vstack((X.T, y)).T
    u , s , vt = svd(z)
    
    V = vt.T
    Vxy = V[:n,n:]
    Vyy = V[n:,n:]
    
    a_tls = -Vxy/Vyy
    
    return a_tls
    
    
    
    