import grabscreen
import re
from time import sleep
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import pyautogui

hwnd = grabscreen.FindWindow_bySearch("envs")

log = np.array([0,0,0])

reword = 0
t = time.time()
act = "forword"
for i in range(4,0,-1):
    print(i)
    time.sleep(1)

def normalization_fram(x,m):    
    y = np.array((x**m)/(255**m)*255,dtype=np.uint8)
    return y

def forword():
    pyautogui.keyUp('d')
    pyautogui.keyUp('a')
    pyautogui.keyDown('w')
    return "forword"


def left():
    pyautogui.keyUp('d')
    pyautogui.keyDown('a')
    return "left" 

def right():
    pyautogui.keyUp('a')
    pyautogui.keyDown('d')
    return "right"
  

def _stop():
    pyautogui.keyUp('d')
    pyautogui.keyUp('a')
    pyautogui.keyUp('w')
    time.sleep(1)
    pyautogui.keyDown('w')
    



act =forword()


while True:
    #  Wait for next request from client
    frame = grabscreen.getWindow_Img(hwnd)
    frame = frame[28:,:1600]
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    frameNL  =  normalization_fram(frame,0.6)
    frameNLL = cv2.resize(frameNL,(160,160))

    # if int(time.time()-t ) %4==0 and act!="forword":
    #     act =  forword()
    # elif int(time.time()-t ) %4==1 and act!="left":
    #     t1 = time.time()
    #     act =left()
    # elif int(time.time()-t ) %4==2 and act!="forword":
    #     act =forword()
    # elif int(time.time()-t ) %4==3 and act!="right":
    #     t1 = time.time()
    #     act =right()



    # rewordpix = frame[7,-7]
    
    # if rewordpix == 255:
    #     reword = 1
    # elif rewordpix ==0:
    #     reword = -1
    #     _stop()
    # else:
    #     reword = 0
    
    # log = np.vstack((log,[act,reword,time.time()-t]))


    # print([act,reword,time.time()-t])

    

    cv2.imshow("frameNLL", frameNLL)

    k = cv2.waitKey(30)&0xFF #64bits! need a mask
    if k ==27:
        cv2.destroyAllWindows()
        break