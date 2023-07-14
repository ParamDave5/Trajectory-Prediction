import numpy as np
from least_square import *
import math
import random as rd
from scipy.optimize import fmin_cobyla

def ransac(x, y ):
    """
    RANSAC implemetation
    Parameters: 
    1) p : Desired probability that its a good sample
    2) e : Probability that a point is an outlier (# outliers / # datapoints)
    3) s : Minimum no. of points to fit the model
    4) N : No. of iterations
    5) t : threshold for inliners
    """
    max_couter = 0
    data = list(zip(x,y))
    samples = len(data)
    
    t = 13
    outliers = 4
    
    p = 0.999
    e = outliers/samples
    s = 2
    N = int(math.log(1-p)/math.log(1-(1-e)**s))
    
    best_model = np.array([])
    max_inlier_count = 0
    
    for i in range(N):
        print(f'\n..... iteration {i} .....\n')
        inlier_count = 0
        sample_points = rd.sample(data , s)
        print("Sample points", sample_points)
        list_x, list_y = zip(*sample_points)
        result = standard_least_square_line(list_x , list_y)
        
        print(result)
        
        for point in data:
            def f(x):
                return result[0]*x + result[1]
            
            def objective(X):
                x,y=X
                distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                return distance 
            
            def c1(X):
                x,y=X
                return y - f(x)
            
            def c2(X):
                x , y  = X
                return f(x) - y
            
            X = fmin_cobyla(objective, x0=[point[0], point[1]], cons=[c1, c2])
            distance = round(objective(X), 2)
        
            if distance <= t:
                inlier_count +=1
        
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = result    
    
    print("Max inliner count for ransac: ", max_inlier_count, "\n")
    # print("RANSAC solution \n", best_model)
    
    return best_model

