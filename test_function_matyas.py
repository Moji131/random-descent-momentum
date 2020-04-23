import math
import numpy as np

pi = 3.14159265359
theta = 0.1 * pi / 6.00 # to rotate the function around teh z axis

def matyas_main(x):

    # rotate function by theta
    x1 = x[0] * math.cos(theta) - x[1] * math.sin(theta)
    x2 = x[0] * math.sin(theta) + x[1] * math.cos(theta)


    term1 = 0.26 * (x1 ** 2 + x2 ** 2)
    term2 = -0.48 * x1 * x2
    f = term1 + term2
    return f


def matyas_grad(x):

    # rotate function by theta
    x1 = x[0] * math.cos(theta) - x[1] * math.sin(theta)
    x2 = x[0] * math.sin(theta) + x[1] * math.cos(theta)

    g = np.empty(x.shape)
    g[0] = 0.52 * x1 - 0.48 * x2
    g[1] = -0.48*x1 + 0.52*x2
    return g