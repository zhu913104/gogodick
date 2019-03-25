import numpy as np
import matplotlib.pyplot as plt



x = np.load("log/real/one_frame_no_stack_2019_03_22_1548.npy")
y = np.load("log/real/one_frame_no_stack_2019_03_23_1833.npy")

print(x)

plt.plot(x[2:,0],x[2:,1])
plt.plot(y[2:,0],y[2:,1])


plt.show()