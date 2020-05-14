import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
prop_cycle = (cycler('color', ['b', 'y', 'k', 'r', 'c', 'm']))
ax1.set_prop_cycle(prop_cycle)

path = "outputs/neural_network/train"
files = os.listdir(path)
for f in files:
    file_path = path+"/"+f
    data = np.genfromtxt(file_path, dtype='float')
    if len(data) != 0:
        ax1.plot(data[:,0], data[:,1], '.-', label=f)

plt.title("Neural Network Convergence - train")
plt.xlabel("steps")
plt.ylabel("loss")
legend = plt.legend()



fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
prop_cycle = (cycler('color', ['b', 'y', 'k', 'r', 'c', 'm']))
ax2.set_prop_cycle(prop_cycle)

path = "outputs/neural_network/test"
files = os.listdir(path)
for f in files:
    file_path = path+"/"+f
    data = np.genfromtxt(file_path, dtype='float')
    if len(data) != 0:
        ax2.plot(data[:,0], data[:,1], '.-', label=f)

plt.title("Neural Network Convergence - test")
plt.xlabel("steps")
plt.ylabel("loss")
legend = plt.legend()


plt.show()

