import numpy as np


def optimiser_ADAM(x_start, gradient_func, max_iterations, tot_num_save):
    # ADAM optimization function
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1.0e-08
    l_rate = 10

    x = x_start
    m = 0
    v = 0
    t = 1

    x_out = np.array([x])

    #g = gradient_func(x)
    # while np.sum(g) > 0.001:
    while t <= max_iterations:
        g = gradient_func(x)

        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        m_hat = m / (1 - beta_1**t)
        v_hat = v / (1 - beta_2**t)
        x = x - (l_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
        # x_out = np.append(x_out, [x], axis=0)
        t = t+1


        # prints progress and saves the point
        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x], axis=0)
            # print("Adam %: ", int((t-1)/max_iterations*100))

    return x_out, t-1


