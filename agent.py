from PIL import ImageGrab
import numpy as np
import cv2
import time
import pyautogui

for i in range(4,0,-1):
    print(i)
    time.sleep(1)

print ('go')

pyautogui.keyDown('w')
time.sleep(2)

for i in range (100):
    a= 'a'
    b= 'd'
    if i%2==0:
        a , b = b ,a
    pyautogui.keyDown(a)
    time.sleep(.3)
    pyautogui.keyUp(b)
    pyautogui.keyDown(a)




# def process_image(image):
#     processed_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     processed_image = cv2.Canny(processed_image,threshold1=100,threshold2=100)
#     return processed_image

# last_time = time.time()

# while(True):
#     screen = np.array(ImageGrab.grab(bbox=(0,0,1920,1080)))
#     new_screen = process_image(screen)
#     print("Loop took {} seconds".format(time.time()-last_time))
#     last_time = time.time()
#     cv2.imshow("canny",new_screen)
#     # cv2.imshow('widow',cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY))
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows
#         break