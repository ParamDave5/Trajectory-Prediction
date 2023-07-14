import numpy as np
from covariance import *
from svd import *
from least_square import *
from tls import *
from ransac import *
from covariance_norm import *

#covariance
age , charges = covar()

#covariance with normalization
age_norm , charge_norm = covar_norm()

#SLS
sls = np.array(standard_least_square_line(age,charges),dtype = np.float32)

x_sls = np.linspace(18,65 , 100)
y_sls = sls[0]*x_sls + sls[1]

plt.title("Age vs Charges ")
plt.plot(x_sls,y_sls)
plt.xlabel('Age')
plt.ylabel("Charges")
plt.scatter(age, charges ,c = 'r')
plt.suptitle("Fiting a line using Standard Least Square", fontsize=16, fontweight = 'bold')
# plt.savefig('/home/sheriarty/ENMP 673/perceptionhw1/Outputs/sls.png')
plt.show()


#TLS

tls = np.array(total_least_square_line(age,charges),dtype = np.float32)
x_tls = np.linspace(18,65 , 100)
y_tls = tls[0]*x_tls 
plt.title("Age vs Charges ")
plt.plot(x_tls,y_tls , c = 'r')
plt.xlabel('Age')
plt.ylabel("Charges")
plt.scatter(age, charges ,c = 'b')
plt.suptitle("Fiting a line using Total Least Square", fontsize=16, fontweight = 'bold')
# plt.savefig('/home/sheriarty/ENMP 673/perceptionhw1/Outputs/tls.png')
plt.show()



#Ransac
ransac = ransac(age , charges)
print("best model : " , ransac)
x_ransac = np.linspace(18,65 , 100)
y_ransac = ransac[0]*x_ransac + ransac[1]
plt.title("Age vs Charges ")
plt.plot(x_ransac,y_ransac , c = 'r')
plt.xlabel('Age')
plt.ylabel("Charges")
plt.scatter(age, charges ,c = 'b')
plt.suptitle("Fiting a line using Ransac", fontsize=16, fontweight = 'bold')
# plt.savefig('/home/sheriarty/ENMP 673/perceptionhw1/Outputs/ransac.png')
plt.show()
