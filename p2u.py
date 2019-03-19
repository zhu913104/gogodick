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

# act = "forword"
# for i in range(4,0,-1):
#     print(i)
#     time.sleep(1)

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
    pyautogui.keyDown('w')
    
def get_state(hwnd,zoom):
    frame = grabscreen.getWindow_Img(hwnd)
    frame = frame[28:,:1600]
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    frame  =  normalization_fram(frame,0.6)
    frame = cv2.resize(frame,(zoom,zoom))

    return frame

def get_reword(frame):
    rewordpix = frame[1,-1]
    
    if rewordpix == 255:
        return 1
    elif rewordpix ==0:
        # _stop()
        return  -1
    else:
        return  0


def stack_state(frame,stack,num=3):
    if stack.any()==0:
        stack = np.stack((frame,frame))
        if num>2:
            for i in range(num-2):
                stack = np.vstack((stack,[frame]))
            return stack
        else:
            return stack
    else:
        stack = np.vstack(([frame],stack[:(num-1),:,:]))
        return stack
nums = 4
stack = np.array([0])
while True:
    #  Wait for next request from client
    
    frame = get_state(hwnd,500)
    x = get_reword(frame)

    stack = stack_state(frame,stack,nums)

    s = np.sum(stack,axis=0 )
    s =s/(nums)
    s= np.array(s,dtype = np.uint8)


    # log = np.vstack((log,[act,reword,time.time()-t]))


    # print([act,reword,time.time()-t])

    cv2.imshow("frameNLL", frame)
    cv2.imshow("s", s)

    k = cv2.waitKey(30)&0xFF #64bits! need a mask
    if k ==27:
        cv2.destroyAllWindows()
        break