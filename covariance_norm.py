import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def covar_norm():
    df = pd.read_csv('hw1.csv')
    print(df)
    
    age = df[['age']].to_numpy()
    charges = df[['charges']].to_numpy()

    age = age.flatten()
    charges = charges.flatten()

    age_mean = age.mean()
    charges_mean = charges.mean()


    age_normal = (age - age.min())/(age.max() - age.min()) 
    charges_normal = (charges - charges.min())/(charges.max() - charges.min())


    # age_norm = np.linalg.norm(age)
    # age_normal = age/age_norm

    # charge_norm = np.linalg.norm(charges)
    # charge_normal = charges/charge_norm

    x = []
    y = []
    x_norm = []
    y_norm = []
    for i in age:
        x.append(i)
    for i in charges:
        y.append(i)
        
    for i in age_normal:
        x_norm.append(i)
    for i in charges_normal:
        y_norm.append(i)

    
    X_norm = (x_norm,y_norm)
    x_norm = np.array(x_norm)
    y_norm = np.array(y_norm)
    xmean = x_norm.mean()
    ymean = y_norm.mean()

    xhat = x_norm - xmean
    yhat = y_norm - ymean

    covarxy = np.sum(xhat*yhat)/(len(x_norm)-1)
    covarxx = np.sum(xhat*xhat)/(len(x_norm)-1)
    covaryx = np.sum(yhat*xhat)/(len(y_norm)-1)
    covaryy = np.sum(yhat*yhat)/(len(y_norm)-1)

    cov = [[covarxx , covarxy] , [covaryx , covaryy]]
    print(cov)

    w, v = np.linalg.eig(cov)
    print(v)
    print(w)
    a = v * w 
    print("final eigen vectors = " ,a )

    plt.scatter(x,y)
    plt.plot(v[0][0] , v[0][1] , c = 'r')
    eig_vec1 = a[:,0]
    eig_vec2 = a[:,1]
    origin = [age_mean,charges_mean]
    plt.xlabel("Age")
    plt.ylabel("Charges")

    arrow = plt.quiver(*origin, *eig_vec1, color=['y'], scale=0.5)
    arrow2 = plt.quiver(*origin, *eig_vec2, color=['r'], scale=0.5)
    legend1 = plt.legend((arrow ,arrow2) , ['eigenvector1 ' , 'eigenvector2' ] )
    plt.gca().add_artist(legend1)
    # plt.savefig('/home/sheriarty/ENMP 673/perceptionhw1/Outputs/covariance_norm.png')
    plt.show()

    return x , y
    
print(covar_norm())