import math
import numpy as np

theta = 0 # to rotate the function around teh z axis
pi = 3.14159265359

def easom_main(x):

    # rotate function by theta
    x1 = x[0] * math.cos(theta) + x[1] * math.sin(theta)
    x2 = - x[0] * math.sin(theta) + x[1] * math.cos(theta)

    fact1 = -np.cos(x1) * np.cos(x2)
    fact2 = np.exp(-(x1 - pi) ** 2 - (x2 - pi) ** 2)
    f = fact1 * fact2
    return f


def easom_grad(x):

    # rotate function by theta
    x1 = x[0] * math.cos(theta) + x[1] * math.sin(theta)
    x2 = - x[0] * math.sin(theta) + x[1] * math.cos(theta)

    g = np.empty(x.shape)
    g[0] = -(2*pi - 2*x1)*np.exp(-(-pi + x1)**2 - (-pi + x2)**2)*np.cos(x1) * \
        np.cos(x2) + np.exp(-(-pi + x1)**2 -
                            (-pi + x2)**2)*np.sin(x1)*np.cos(x2)
    g[1] = -(2*pi - 2*x2)*np.exp(-(-pi + x1)**2 - (-pi + x2)**2)*np.cos(x1) * \
        np.cos(x2) + np.exp(-(-pi + x1)**2 -
                            (-pi + x2)**2)*np.sin(x2)*np.cos(x1)
    return g