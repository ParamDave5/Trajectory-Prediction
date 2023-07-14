from video import *
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from least_square import *


path = 'ball_video1.mp4'    
center_x1 , center_y1 = video(path)

path1 = 'ball_video2.mp4'
center_x2 , center_y2 = video(path1)

sls1 = np.array(standard_least_square(center_x1 , center_y1) , dtype = np.float32)
sls2 = np.array(standard_least_square(center_x2 , center_y2) , dtype = np.float32)
center_x1 = np.array(center_x1 , dtype = np.float32)
center_x2 = np.array(center_x2 , dtype = np.float32)


#for video 1
x = np.linspace(0,720 ,28)
y_sls1  = sls1[0]*(center_x1**2) + sls1[1]*center_x1 + sls1[2]
y  = sls1[0]*(x**2) + sls1[1]*x + sls1[2]
plt.title("Standard Least Sqaure without noise")
plt.plot(x,y)
# plt.plot(x,y_sls1)
plt.scatter(center_x1, center_y1 ,c = 'r')
plt.suptitle("Parabola w/o noise", fontsize=20, fontweight = 'bold')
# plt.savefig('/home/sheriarty/ENMP 673/perceptionhw1/Outputs/para.png')
plt.show()

#for video2

x = np.linspace(0,1050,2000)
y_sls2 = sls2[0]*(center_x2**2) + sls2[1]*center_x2 + sls2[2]
y  = sls2[0]*(x**2) + sls2[1]*x + sls2[2]
plt.title("Standard Least Sqaure with noise")
plt.plot(x,y )
plt.scatter(center_x2 , center_y2 , c = 'g')
plt.suptitle("Parabola with noise", fontsize=20, fontweight = 'bold')
# plt.savefig('/home/sheriarty/ENMP 673/perceptionhw1/Outputs/para2.png')

plt.show()