import grabscreen
import re
from time import sleep
import cv2
import matplotlib.pyplot as plt
import numpy as np


hwnd = grabscreen.FindWindow_bySearch("envs")


print(hwnd)

def normalization_fram(x,maxpix):    
    # y = np.array((x**m)/(255**m)*255,dtype=np.uint8)
    y= np.array((x/maxpix)*255)
    y[y>255]=255
    y = np.array(y,dtype=np.uint8)
    return y


plt.show()

while True:
    
    frame = grabscreen.getWindow_Img(hwnd)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    frame  =  normalization_fram(frame,150)
    frame = frame[28:,:1600]
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    
    # fig = plt.figure()
    plt.ion()
    plt.plot(hist)
    plt.ylim((0, 120000))
    plt.show()
    print(frame.shape)
    plt.pause(0.1)
    cv2.imshow("screen box", frame)
    plt.cla()
    k = cv2.waitKey(30)&0xFF #64bits! need a mask
    if k ==27:
        cv2.destroyAllWindows()
        break