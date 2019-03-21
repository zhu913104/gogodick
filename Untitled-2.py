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

class env(object):
    def __init__(self, hwnd, zoom,num):
        self.hwnd = hwnd
        self.zoom = zoom
        self.num = num
        self.stack = np.array([0])
        self.frame = 0


    def normalization_fram(self,x,m):    
        y = np.array((x**m)/(255**m)*255,dtype=np.uint8)
        return y

    def forword(self):
        pyautogui.keyUp('d')
        pyautogui.keyUp('a')
        pyautogui.keyDown('w')

    def left(self):
        pyautogui.keyUp('d')
        pyautogui.keyDown('a')


    def right(self):
        pyautogui.keyUp('a')
        pyautogui.keyDown('d')

    
    def _stop(self):
        pyautogui.keyUp('d')
        pyautogui.keyUp('a')
        pyautogui.keyUp('w')
        pyautogui.keyDown('w')
        
    def _state(self):
        self.frame = grabscreen.getWindow_Img(self.hwnd)
        self.frame = self.frame[28:,:1600]
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_RGB2GRAY)
        self.frame  =  self.normalization_fram(self.frame,0.8)
        self.frame = cv2.resize(self.frame,(self.zoom,self.zoom))
        return self.frame

    def get_reword(self):
        rewordpix = self.frame[1,-1]
        if rewordpix == 255:
            self.reword = 1
            return self.reword 
        elif rewordpix ==0:
            # _stop()
            self.reword =  -1
            return self.reword 
        else:
            self.reword = 0
            return self.reword 

    def stack_state(self):
        self._state()
        if self.num ==1:
            pass
        else :
            if self.stack.any()==0:
                self.stack = np.stack((self.frame,self.frame))
                if self.num>2:
                    for i in range(self.num-2):
                        self.stack = np.vstack((self.stack,[self.frame]))
            else:
                self.stack = np.vstack(([self.frame],self.stack[:(self.num-1),:,:]))

    def get_state(self):
        self.stack_state()
        if self.num==1:
            return self.frame
        else:
            self.stack_m = np.sum(self.stack,axis=0 )
            self.stack_m =self.stack_m/(self.num)
            self.stack_m= np.array(self.stack_m,dtype = np.uint8)
            return self.stack_m



    def action(self,act):
        if act ==0:
            self.forword()
        elif act ==1:
            self.left()
        elif act ==2:
            self.right() 

zoom = 160
nums =1
act = 0
eee = env(hwnd, zoom,nums)
t= 0

s = eee.get_state()
while True:
    #  Wait for next request from client
    if t%5==0:
        act = np.random.randint(3)


    # eee.action(act)
    s_ ,r= eee.get_state(),eee.get_reword()

    s=s_
    cv2.imshow("xx",s_)


    print("ACTION",act,"REWORD",r,"TIME",t)

    t+=1
    k = cv2.waitKey(30)&0xFF #64bits! need a mask
    if k ==27:
        cv2.destroyAllWindows()
        break
 