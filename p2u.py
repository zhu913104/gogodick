import grabscreen
import re
from time import sleep
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import pyautogui

hwnd = grabscreen.FindWindow_bySearch("envs")

reword = 0

def normalization_fram(x,m):    
    y = np.array((x**m)/(255**m)*255,dtype=np.uint8)
    return y


red = (0,0,255)

one = True

while True:
    #  Wait for next request from client
    frame = grabscreen.getWindow_Img(hwnd)
    frame = frame[28:,:1600]
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    frameNL  =  normalization_fram(frame,0.6)

    rewordpix = frame[7,-7]
    if rewordpix == 255:
        reword = 1
        cv2.imwrite("p.png",frame)
        cv2.imwrite("pNL.png",frameNL)
    elif rewordpix ==0:
        reword = -1
        cv2.imwrite("n.png",frame)
        cv2.imwrite("nNL.png",frameNL)
        one = False
    else:
        reword = 0
        cv2.imwrite("f.png",frame)
        cv2.imwrite("fNL.png",frameNL)
        
    print(reword)
    

    cv2.imshow("screen box", frame)

    k = cv2.waitKey(30)&0xFF #64bits! need a mask
    if k ==27:
        cv2.destroyAllWindows()
        break