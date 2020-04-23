import numpy as np
import math




def optimiser_NAGD(x_start, gradient_func, max_iterations, tot_num_save):
# Nesterov accelerated gradient descent function

    alpha = 0.001

    x_prev = x_start
    g = gradient_func(x_prev)


    t = 1
    x_out = np.array([x_start])
    x = x_prev

    while t <= max_iterations:

        gamma = 1 - 3 / (5 + t)

        y = x +  (x - x_prev) * (t-1) / (t+2)
        g = gradient_func(y)
        x_prev = x
        x = y - alpha * g

        t = t + 1

        # prints progress and saves the point
        c = max_iterations // tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x], axis=0)
            # print("Adam %: ", int((t-1)/max_iterations*100))

    return x_out, t - 1