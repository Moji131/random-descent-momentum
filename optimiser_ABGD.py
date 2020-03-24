import numpy as np


def optimiser_ABGD(x_start, gradient_func, max_iterations, tot_num_save):

    # bisection gradient descent optimization function
    d = len(x_start)
    step = np.ones(d) * 0.1  # inital step (can be anything)

    # put the initial point as the saved point

    x_0 = x_start  # set x_0 to x_start
    g_0 = gradient_func(x_0)  # gradient at new point
    g_0_unit = np.sign(g_0)

    g_0_m1 = np.ones(len(x_start))

    t = 1  # set the counter
    x_out = np.array([x_0])  # add to x_out array for graph

    while t <= max_iterations:  # steps
        # print(x_0, g_0_unit, step)
        x_1 = x_0 - g_0_unit * step  # advance x one step

        g_1 = gradient_func(x_1)  # gradient at new point
        g_1_unit = np.sign(g_1)

        g_1_0 = g_1_unit * g_0_unit


        step_mult = np.ones(d) * 2.0
        step_mult = np.where(g_0_m1 == -1.0, 1.0, step_mult)
        step_mult = np.where(g_1_0 == -1.0, 0.5, step_mult)
        step = step * step_mult

        x_0 = x_1
        g_0_unit = g_1_unit
        g_0_m1 = g_1_0

        t = t + 1

        # prints progress and saves the point
        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x_1], axis=0)
            # print("Adam %: ", int((t-1)/max_iterations*100))



    return x_out, t

