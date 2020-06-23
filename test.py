d = 784
total_C = 10
n = 784
import numpy.random as rand
import numpy as np

rand.seed(5)
w = rand.randn(d*(total_C-1), 1)
print(w.shape)
w = np.ones(w.shape)

print(w)