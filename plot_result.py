import numpy as np
import matplotlib.pyplot as plt



# mode5 = np.load("log/dqn/loss/loss2019_05_19_0017.npy")
mode6 = np.load("log/dqn/reword/reword2019_05_20_2108.npy")
# mode4 = np.load("log/mode_4_2019_03_29_1202.npy")
# mode1 = np.load("log/mode1/one_frame_no_stack_2019_03_22_1548.npy")
# mode2 = np.load("log/mode2/one_frame_no_stack_2019_03_27_1910.npy")
# mode3 = np.load("log/mode3/one_frame_no_stack_2019_03_28_2050.npy")
# mode7 = np.load("log/mode_7_2019_04_01_1313.npy")
# mode8 = np.load("log/mode_8_2019_04_02_0137.npy")
# mode9 = np.load("log/mode_9_2019_04_02_1408.npy")


# for i, element in enumerate(mode8[1:44,1]):
#     mode8[i,1] = element-58*((43-i)/42)

# for i, element in enumerate(mode6[:30,1]):
#     mode6[i,1] = element-70*((29-i)/29)
#     print(i)


# print("mode1 max:",mode1[500:1000,1].max(),"mode1 mean:",mode1[500:1000,1].mean(),"mean error: ",mode1[500:1000,1].std())
# print("mode2 max:",mode2[500:1000,1].max(),"mode2 mean:",mode2[500:1000,1].mean(),"mean error: ",mode2[500:1000,1].std())
# print("mode3 max:",mode3[500:1000,1].max(),"mode3 mean:",mode3[500:1000,1].mean(),"mean error: ",mode3[500:1000,1].std())
# print("mode4 max:",mode4[500:1000,1].max(),"mode4 mean:",mode4[500:1000,1].mean(),"mean error: ",mode4[500:1000,1].std())
# print("mode5 max:",mode5[500:1000,1].max(),"mode5 mean:",mode5[500:1000,1].mean(),"mean error: ",mode5[500:1000,1].std())
# print("mode6 max:",mode6[500:1000,1].max(),"mode6 mean:",mode6[500:1000,1].mean(),"mean error: ",mode6[500:1000,1].std())
# print("mode7 max:",mode7[500:1000,1].max(),"mode7 mean:",mode7[500:1000,1].mean(),"mean error: ",mode7[500:1000,1].std())
# print("mode8 max:",mode8[500:1000,1].max(),"mode8 mean:",mode8[500:1000,1].mean(),"mean error: ",mode8[500:1000,1].std())
# print("mode9 max:",mode9[500:1000,1].max(),"mode9 mean:",mode9[500:1000,1].mean(),"mean error: ",mode9[500:1000,1].std())



# plt.plot(mode1[1:1000,0],mode1[1:1000,1],"#FF4D00",label = "mode1")
# plt.plot(mode2[1:1000,0],mode2[1:1000,1],"#FFA500",label = "mode2")
# plt.plot(mode3[1:1000,0],mode3[1:1000,1],"#CC5500",label = "mode3")
# plt.plot(mode4[1:1000,0],mode4[1:1000,1],"#008080",label = "mode4")
# plt.plot(mode5[1:,0],mode5[1:,1],label = "mode5")
plt.plot(mode6[1:,0],mode6[1:,1], label = "mode6")
# plt.plot(mode7[1:1000,0],mode7[1:1000,1],"#5E86C1",label = "mode7")
# plt.plot(mode8[1:1000,0],mode8[1:1000,1],"#30D5C8",label = "mode8")
# plt.plot(mode9[1:1000,0],mode9[1:1000,1],"b",label = "mode9")



plt.xlabel("step")
plt.ylabel("reword")
plt.legend(loc="lower right",prop={'size': 20})


plt.show()