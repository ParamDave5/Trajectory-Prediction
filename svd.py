import numpy as np


def svd(z):
    #calculating Vt
    
    r_values , r_vectors = np.linalg.eig(np.dot(z.T,z))
    idx = r_values.argsort() [::-1]
    r_values = r_values[idx]
    r_vectors = r_vectors[:,idx]
    Vt = r_vectors.T
    V = Vt.T
    
    #Computing sigma 
    
    index = []
    
    for i in range(len(r_values)):
        if r_values[i] <= 0.001:
            index.append(i)
    
    r_values = np.delete(r_values , index)
    
    s = np.zeros(shape = (z.shape))
    
    for i in range(len(r_values)):
        s[i,i] = r_values[i]**0.5
    
    #Calculating U
    l_values , l_vectors = np.linalg.eig(np.dot(z.T,z))
    idx = l_values.argsort()[::-1]
    l_values = l_values[idx]
    l_vectors = l_vectors[:,idx]
    U = l_vectors.real
    
    return U , s , Vt        
