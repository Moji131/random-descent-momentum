import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

fig1 = plt.figure(1)

path = "outputs/neural_network/train"
files = os.listdir(path)
for f in files:
    file_path = path+"/"+f
    data = np.genfromtxt(file_path, dtype='float')
    if len(data) != 0:
        plt.plot(data[:,0], data[:,1], '.-', label=f)

plt.title("Neural Network Convergence - train")
plt.xlabel("steps")
plt.ylabel("loss")
legend = plt.legend()



fig2 = plt.figure(2)

path = "outputs/neural_network/test"
files = os.listdir(path)
for f in files:
    file_path = path+"/"+f
    data = np.genfromtxt(file_path, dtype='float')
    if len(data) != 0:
        plt.plot(data[:,0], data[:,1], '.-', label=f)

plt.title("Neural Network Convergence - test")
plt.xlabel("steps")
plt.ylabel("loss")
legend = plt.legend()


plt.show()

