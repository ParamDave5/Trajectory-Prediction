import numpy as np
import matplotlib.pyplot as plt
import math as m
import cv2 as cv

centers  = []
def loc(dst):
    min = []
    max = []
    
    loc = np.where(dst == 0 )

    min_x = np.min(loc[0])
    min.append(min_x)
    f = np.where(loc[0] == min_x)
    min.append(loc[1][f[0][0]])


    max_x = np.amax(loc[0])
    max.append(max_x)
    
    g = np.where(loc[0] == max_x)
    max.append(loc[1][g[0][0]])  
      
    center_x = (max[0] + min[0])//2 
    center_y = (max[1] + min[1])//2
    center = [center_x , center_y]

    centers.append(center)
    return centers
    
    
cap = cv.VideoCapture('ball_video1.mp4')
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

count = 0
# if (cap.isOpened() == False):
#     print("Error Opening video file")
while(count < length):
    ret , frame = cap.read()
    
    
    frame = np.array(frame , dtype = np.float32) 
    ret1, black_white = cv.threshold(frame, 120, 255, cv.THRESH_BINARY)
    th, dst = cv.threshold(black_white, 100, 255, cv.THRESH_TOZERO)
    dst = dst/255
    
    centers = loc(dst)
    count +=1    
# print(centers)

xs = [x[0] for x in centers]
ys = [x[1] for x in centers]
plt.plot(xs, ys)
plt.show()