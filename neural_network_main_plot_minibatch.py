import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import copy


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
prop_cycle = (cycler('color', ['b', 'y', 'k', 'r', 'c', 'm', 'dimgrey', 'pink']))
ax1.set_prop_cycle(prop_cycle)

path = "outputs/neural_network_minibatch/train"
files = os.listdir(path)
for f in files:
    file_path = path+"/"+f
    data = np.genfromtxt(file_path, dtype='float')
    if len(data) != 0:
        if "ABGDcs" in f:
            data1 =  copy.deepcopy(data)
            ll = len(data[:, 1])
            for i in range(ll):
                if (i % 2) == 0:
                    data1[i,1] = data[i//2,1]
                else:
                    data1[i,1] = (data[i//2,1] + data[i//2-1,1] )/2.0
            ax1.plot(data[:, 0], data1[:, 1], '.-', label=f)
        else:
            ax1.plot(data[:, 0], data[:, 1], '.-', label=f)

plt.title("Neural Network Convergence - train")
plt.xlabel("minibatches")
plt.ylabel("loss")
legend = plt.legend()
# ax1.set_xlim(-1, 40)
# ax1.set_ylim(0, 2000000)



#
# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111)
# prop_cycle = (cycler('color', ['b', 'y', 'k', 'r', 'c', 'm']))
# ax2.set_prop_cycle(prop_cycle)
#
# path = "outputs/neural_network/test"
# files = os.listdir(path)
# for f in files:
#     file_path = path+"/"+f
#     data = np.genfromtxt(file_path, dtype='float')
#
#     if len(data) != 0:
#         if "ABGDcs" in f:
#             pass
#         else:
#             ax2.plot(data[:,0], data[:,1], '.-', label=f)
#
# plt.title("Neural Network Convergence - test")
# plt.xlabel("steps")
# plt.ylabel("loss")
# legend = plt.legend()


plt.show()
