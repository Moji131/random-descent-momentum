import math
import numpy as np
import random as rn

pi = 3.14159265359
theta = pi/4.0 # 0.1 * pi / 6.00 # to rotate the function around teh z axis
a = 0.5

def quadratic_main(x):

    f = 100
    for i in range(len(x)):
        c1 = 10 * (i % 5)**2 + 1
        # c1 = 1
        f = f + c1 * x[i]**2

    # f = f + 1e1 * rn.gauss(0, 1)
    # f = f + 1e3 * rn.gauss(0, 1)


    # f = f + 1e0 * rn.gauss(0, 1)
    # f = f + 1e3 * rn.gauss(0, 1)





    #

    return f


def quadratic_grad(x):
    #
    # # rotate function by theta
    # # x1 = x[0] * math.cos(theta) - x[1] * math.sin(theta)
    # # x2 = x[0] * math.sin(theta) + x[1] * math.cos(theta)
    #
    # x1 = x[0]
    # x2 = x[1]
    #
    # g = np.empty(x.shape)
    # # g[0] = 0.52 * x1 - 0.48 * x2
    # # g[1] = -0.48*x1 + 0.52*x2
    # g[0] = 2*x1
    # g[1] = 2*a**2*x2

    g = np.zeros(len(x))
    for i in range(len(x)):
        # c1 = 5 * (i % 5) + 1
        c1 = 1
        g[i] = 2 * c1 * x[i]



    return g