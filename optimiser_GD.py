import numpy as np

def optimiser_GD(x_start, gradient_func, max_iterations, tot_num_save):
    # gradient descent optimization function

    l_rate = 0.01
    x = x_start
    t = 1
    x_out = np.array([x])
    while t <= max_iterations:
        g = gradient_func(x)

        x = x - l_rate * g
        # x_out = np.append(x_out, np.array([x]), axis=0)
        t += 1

        # prints progress and saves the point
        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x], axis=0)
            print("GD %: ", int((t-1)/max_iterations*100))

    return x_out, t-1