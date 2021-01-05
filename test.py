import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x + y

x = np.linspace(0, 1, 2)
y = np.linspace(0, 2, 3)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

print(X)
print(Y)
print(Z)

plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.show()


