import numpy as np
import matplotlib.pyplot as plt



mode5 = np.load("log/mode_5_2019_03_30_1444.npy")
mode6 = np.load("log/mode_6_2019_03_31_0309.npy")
mode4 = np.load("log/mode_4_2019_03_29_1202.npy")
mode1 = np.load("log/mode1/one_frame_no_stack_2019_03_22_1548.npy")
mode2 = np.load("log/mode2/one_frame_no_stack_2019_03_27_1910.npy")
mode3 = np.load("log/mode3/one_frame_no_stack_2019_03_28_2050.npy")
mode7 = np.load("log/mode_7_2019_04_01_1313.npy")
mode8 = np.load("log/mode_8_2019_04_02_0137.npy")




plt.plot(mode1[1:1000,0],mode1[1:1000,1],label = "mode1")
plt.plot(mode2[1:1000,0],mode2[1:1000,1],label = "mode2")
plt.plot(mode3[1:1000,0],mode3[1:1000,1],label = "mode3")
plt.plot(mode4[1:1000,0],mode4[1:1000,1],label = "mode4")
plt.plot(mode5[1:1000,0],mode5[1:1000,1],label = "mode5")
plt.plot(mode6[1:1000,0],mode6[1:1000,1],label = "mode6")
plt.plot(mode7[1:1000,0],mode7[1:1000,1],label = "mode7")
plt.plot(mode8[1:1000,0],mode8[1:1000,1],label = "mode8")


plt.legend(loc="lower right")


plt.show()