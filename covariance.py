import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def covar():
    df = pd.read_csv('hw1.csv')
    age = df[['age']].to_numpy()
    charges = df[['charges']].to_numpy()
    age = age.flatten()
    charges = charges.flatten()
    age_mean = age.mean()
    cherges_mean = charges.mean()
    x = []
    y = []
    for i in age:
        x.append(i)
    for i in charges:
        y.append(i)
        
    X = (x,y)
    x = np.array(x)
    y = np.array(y)
    xmean = x.mean()
    ymean = y.mean()

    xhat = x - xmean
    yhat = y - ymean

    covarxy = np.sum(xhat*yhat)/len(x)-1
    covarxx = np.sum(xhat*xhat)/len(x)-1
    covaryx = np.sum(yhat*xhat)/len(y)-1
    covaryy = np.sum(yhat*yhat)/len(y)-1
    
    cov = [[covarxx , covarxy] , [covaryx , covaryy]]
    w, v = np.linalg.eig(cov)
    
    plt.scatter(x,y)
    plt.plot(v[0][0] , v[0][1] , c = 'r')
    eig_vec1 = v[:,0]
    eig_vec2 = v[:,1]
    origin = [age_mean,cherges_mean]
    plt.xlabel("Age")
    plt.ylabel("Charges")
    
    arrow = plt.quiver(*origin, *eig_vec1, color=['y'], scale=10)
    arrow2 = plt.quiver(*origin, *eig_vec2, color=['r'], scale=10)
    legend1 = plt.legend((arrow ,arrow2) , ['eigenvector1 ' , 'eigenvector2' ] )
    plt.gca().add_artist(legend1)
    # plt.savefig('/home/sheriarty/ENMP 673/perceptionhw1/Outputs/covariance.png')
    plt.show()
    return x , y  
    
print(covar())