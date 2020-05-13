import numpy as np
import math




def optimiser_GDM(x_start, gradient_func, max_iterations, tot_num_save):
# Nesterov accelerated gradient descent function

    lr = 1
    momentum = 0
    d = len(x_start)

    x = x_start
    g = gradient_func(x_start)


    t = 1
    x_out = np.array([x_start])

    v = np.zeros(d)

    while t <= max_iterations:

        g = gradient_func(x)
        v = momentum * v +  lr * g
        x = x - v

        t = t + 1

        # prints progress and saves the point
        c = max_iterations // tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x], axis=0)
            # print("Adam %: ", int((t-1)/max_iterations*100))

    return x_out, t - 1