import numpy as np


def optimiser_ABGD(x_start, gradient_func, max_iterations, tot_num_save):

    # bisection gradient descent optimization function
    d = len(x_start)
    step = np.ones(d) * 0.01  # inital step (can be anything)


    # initialize
    x_0 = x_start  # set x_0 to x_start
    g_0 = gradient_func(x_0)  # gradient at x_0
    g_0_sign = np.sign(g_0)   # get the sign of the components
    g_0_m1 = np.ones(len(x_start))  # initialising product of sign of gradient of step 0 and -1


    t = 1  # iteration counter
    x_out = np.array([x_0])  # add to x_out array for graph

    while t <= max_iterations:  # steps

        x_1 = x_0 - g_0_sign * step  # advance x one step
        g_1 = gradient_func(x_1)  # gradient at the new point
        g_1_sign = np.sign(g_1)  # sign of the components
        g_1_0 = g_1_sign * g_0_sign  # product of sign of gradient of step 1 and 0


        step_mult = np.ones(d) * 2.0  #  setting all step multipliers to 2
        step_mult = np.where(g_0_m1 == -1.0, 1.0, step_mult) #  if g_0_m1 is -1 change step_mult component to 1
        step_mult = np.where(g_1_0 == -1.0, 0.5, step_mult)  #  if g_1_0 is -1 change step_mult component to 0.5
        step = step * step_mult  # use step_mult to update current step sizes

        #  preparation for  the next step
        x_0 = x_1
        g_0_sign = g_1_sign
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

