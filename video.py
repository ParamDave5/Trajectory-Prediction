import cv2 
import numpy as np
import matplotlib.pyplot as plt

def video(path):
    
    center_x = []
    center_y = []
    
    cap = cv2.VideoCapture(path)
    try: 
        while(1):
            _ ,frame = cap.read()
            frame = np.array(frame)
            width = int(frame.shape[1]*0.3)
            height = int(frame.shape[0]*0.3)
                
            frame = cv2.resize(frame , (width , height) , interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            _ , thresh = cv2.threshold(gray , 127 , 255 ,cv2.THRESH_BINARY_INV)
            # cv2.imshow('frame' , thresh)
            conts , _ = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
            for c in conts:
                m = cv2.moments(c)
                cX = int(m["m10"] / m["m00"])
                cY = int(m["m01"] / m["m00"])
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cX, cY), 4, (255, 0, 0), -1)
            
                center_x.append(cX)
                center_y.append(height - cY)
            
        print(center_x)
        
        return center_x ,center_y

    except:
        return center_x , center_y



