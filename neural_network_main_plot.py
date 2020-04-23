import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


path = "outputs/neural_network"
files = os.listdir(path)
for f in files:
    file_path = path+"/"+f
    data = np.genfromtxt(file_path, dtype='float')
    if len(data) != 0:
        plt.plot(data[:,0], data[:,1], '.-', label=f)

plt.title("Neural Network Convergence")
plt.xlabel("steps")
plt.ylabel("loss")
legend = plt.legend()
plt.show()