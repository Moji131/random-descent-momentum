import numpy as np
from matplotlib.pyplot import ion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from softmax import softmax
from numpy.linalg import norm
import numpy.random as rand
from regConvex import regConvex


########### Optimiser functions ###########################
###########################################################


def gd_opt(x_start, gradient_func, l_rate, max_iterations, min_g, tot_num_save):
    # gradient descent optimization function
    x = x_start
    t = 1
    x_out = np.array([x])
    while t <= max_iterations:
        g = gradient_func(x)
        if np.linalg.norm(g) < min_g:
            break
        x = x - l_rate * g
        # x_out = np.append(x_out, np.array([x]), axis=0)
        t += 1


        c = max_iterations//tot_num_save

        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x], axis=0)
            print("GD %: ", int((t-1)/max_iterations*100))

    return x_out, t-1


def adam_opt(x_start, gradient_func, l_rate, epsilon,  max_iterations, beta_1, beta_2, min_g, tot_num_save):
    # ADAM optimization function
    x = x_start
    m = 0
    v = 0
    t = 1

    x_out = np.array([x])

    #g = gradient_func(x)
    # while np.sum(g) > 0.001:
    while t <= max_iterations:
        g = gradient_func(x)

        if np.linalg.norm(g) < min_g:
            break
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        m_hat = m / (1 - beta_1**t)
        v_hat = v / (1 - beta_2**t)
        x = x - (l_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
        # x_out = np.append(x_out, [x], axis=0)
        t = t+1

        c = max_iterations//tot_num_save

        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x], axis=0)
            print("Adam %: ", int((t-1)/max_iterations*100))
    g_0_norm = np.linalg.norm(g)  # norm of the gradient
    return x_out, t-1




def bisec_gd_opt_s0(x_start, gradient_func, l_rate, step_mult_max, max_iterations, g_norm_min, tot_num_save):
    # bisection gradient descent optimization function
    t = 0  # step counter to zero
    step = l_rate  # inital step (can be anything)

    # put the initial point as the saved point
    t = t + 1  # increase step counter
    x_s = x_start  # set x_s to x_start
    x_out = np.array([x_s])  # add to x_out array for graph
    g_s = gradient_func(x_s)  # gradient at new point
    g_s_norm = np.linalg.norm(g_s)  # norm of the gradient
    g_s_normed = g_s / g_s_norm  # normalized gradient

    # advance x one step
    t = t + 1  # increase step counter
    x_0 = x_s - g_s_normed * step  # advance x
    # x_out = np.append(x_out, [x_0], axis=0) #add to x_out array for graph
    g_0 = gradient_func(x_0)  # gradient at new point
    g_0_norm = np.linalg.norm(g_0)  # norm of the gradient
    g_0_normed = g_0 / g_0_norm  # normalized gradient
    g_eff_normed = g_0_normed


    while t <= max_iterations:  # steps
        # advance x one step
        t = t + 1  # increase step counter
        # g = g_0_normed
        # g = g_0_normed
        # g_norm = np.linalg.norm(g)
        # g_normed = g / g_norm

        x_1 = x_0 - g_eff_normed * step  # advance x
        # x_out = np.append(x_out, [x_1], axis=0)  # add to x_out array for graph
        g_1 = gradient_func(x_1)  # gradient at new point
        g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
        g_1_normed = g_1 / g_1_norm  # normalized gradient

        # caculating step for the next iteration
        # dot product of gradient_1 and gradient_0
        g_1_0_dot = np.dot(g_1_normed, g_0_normed)
        # dot product of gradient_1 and gradient_s
        g_1_s_dot = np.dot(g_1_normed, g_s_normed)

        if g_1_0_dot < g_1_s_dot:
            g_1_s_dot = g_1_0_dot
            g_s_normed = g_0_normed

        print(t, g_1_normed)
        print(t, g_s_normed)
        g_sum = g_1_normed + g_s_normed
        print(t, g_sum)
        g_sum_norm = np.linalg.norm(g_sum)  # norm of the gradient
        g_sum_normed = g_sum / g_sum_norm  # normalized gradient
        print(t, g_sum_normed)


        # g_eff = g_1_normed + g_sum_normed
        g_eff = g_1_normed
        g_eff_norm = np.linalg.norm(g_eff)  # norm of the gradient
        g_eff_normed = g_eff / g_eff_norm  # normalized gradient
        print(t, g_eff_normed)
        if g_1_s_dot < 0:
            step = step * 0.5
            # None
        else:
            step = step * 2.0

            # step = l_rate

        # swap values for the next step
        x_0 = x_1
        g_0_normed = g_1_normed

        # break if
        if g_1_norm < g_norm_min:
            break

        # save the points for graph and print progress
        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x_0], axis=0)
            print("BGD_3 %: ", int((t-1)/max_iterations*100))


    return x_out, t-1






def bisec_gd_opt_s1(x_start, gradient_func, l_rate, step_mult_max, max_iterations, g_norm_min, tot_num_save):
    # bisection gradient descent optimization function
    t = 0  # step counter to zero
    step_g = l_rate  # inital step (can be anything)
    step_sum = l_rate

    # put the initial point as the saved point
    t = t + 1  # increase step counter
    x_s = x_start  # set x_s to x_start
    x_out = np.array([x_s])  # add to x_out array for graph
    g_s = gradient_func(x_s)  # gradient at new point
    g_s_norm = np.linalg.norm(g_s)  # norm of the gradient
    g_s_normed = g_s / g_s_norm  # normalized gradient

    # advance x one step
    t = t + 1  # increase step counter
    x_0 = x_s - g_s_normed * step_g  # advance x
    # x_out = np.append(x_out, [x_0], axis=0) #add to x_out array for graph
    g_0 = gradient_func(x_0)  # gradient at new point
    g_0_norm = np.linalg.norm(g_0)  # norm of the gradient
    g_0_normed = g_0 / g_0_norm  # normalized gradient

    g_sum_0 = g_0_normed + g_s_normed
    g_sum_0_norm = np.linalg.norm(g_sum_0)  # norm of the gradient
    g_sum_0_normed = g_sum_0 / g_sum_0_norm  # normalized gradient

    g_sum_s_normed = g_sum_0_normed

    while t <= max_iterations:  # steps
        # advance x one step
        t = t + 1  # increase step counter
        # x_1 = x_0 - g_0_normed * step_g + g_sum_0_normed * step_sum # advance x
        x_1 = x_0 - g_0_normed * step_g - g_sum_0_normed * step_sum
        g_1 = gradient_func(x_1)  # gradient at new point
        g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
        g_1_normed = g_1 / g_1_norm  # normalized gradient

        g_sum_1 = g_1_normed + g_s_normed
        g_sum_1_norm = np.linalg.norm(g_sum_1)  # norm of the gradient
        g_sum_1_normed = g_sum_1 / g_sum_1_norm  # normalized gradient


        # caculating step_g for the next iteration
        # dot product of gradient_1 and gradient_0
        g_1_0_dot = np.dot(g_1_normed, g_0_normed)
        # dot product of gradient_1 and gradient_s
        g_1_s_dot = np.dot(g_1_normed, g_s_normed)

        if g_1_0_dot < g_1_s_dot:
            g_1_s_dot = g_1_0_dot
            g_s_normed = g_0_normed

        if g_1_s_dot < 0:
            step_g = step_g * 0.5
        else:
            step_g = step_g * 2.0

        # swap values for the next step
        x_0 = x_1
        g_0_normed = g_1_normed



        # caculating step_sum for the next iteration
        # dot product of g_sum_1 and g_sum_0
        g_sum_1_0_dot = np.dot(g_sum_1_normed, g_sum_0_normed)
        # dot product of g_sum_1 and g_sum_s
        g_sum_1_s_dot = np.dot(g_sum_1_normed, g_sum_s_normed)

        if g_sum_1_0_dot < g_sum_1_s_dot:
            g_sum_1_s_dot = g_sum_1_0_dot
            g_sum_s_normed = g_sum_0_normed

        if g_sum_1_s_dot < 0:
            step_sum = step_sum * 0.5
        else:
            step_sum = l_rate

        # swap values for the next step
        x_0 = x_1
        g_0_normed = g_1_normed
        g_sum_0_normed = g_sum_1_normed



        # break for small gradient
        if g_1_norm < g_norm_min:
            break

        print(t, step_g, step_sum)

        # save the points for graph and print progress
        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x_0], axis=0)
            print("BGD_3 %: ", int((t-1)/max_iterations*100))


    return x_out, t-1







def bisec_gd_opt_s1(x_start, gradient_func, l_rate, step_mult_max, max_iterations, g_norm_min, tot_num_save):


    # This one does not have saved. just compares both g and g_sum with previous and halves the step if they are opposing
    # it also resets step_sum every time that we do not have a pair

    # bisection gradient descent optimization function
    t = 0  # step counter to zero
    step_g = l_rate  # inital step (can be anything)
    step_sum = l_rate
    step_sum_save = step_sum


    # put the initial point as the saved point
    t = t + 1  # increase step counter
    x_s = x_start  # set x_s to x_start
    x_out = np.array([x_s])  # add to x_out array for graph
    g_s = gradient_func(x_s)  # gradient at new point
    g_s_norm = np.linalg.norm(g_s)  # norm of the gradient
    g_s_normed = g_s / g_s_norm  # normalized gradient

    # advance x one step
    t = t + 1  # increase step counter
    x_0 = x_s - g_s_normed * step_g  # advance x
    # x_out = np.append(x_out, [x_0], axis=0) #add to x_out array for graph
    g_0 = gradient_func(x_0)  # gradient at new point
    g_0_norm = np.linalg.norm(g_0)  # norm of the gradient
    g_0_normed = g_0 / g_0_norm  # normalized gradient

    g_sum_0 = g_0_normed + g_s_normed
    g_sum_0_norm = np.linalg.norm(g_sum_0)  # norm of the gradient
    g_sum_0_normed = g_sum_0 / g_sum_0_norm  # normalized gradient

    c_sum = 1

    while t <= max_iterations:  # steps
        # advance x one step
        t = t + 1  # increase step counter
        # x_1 = x_0 - g_0_normed * step_g - g_sum_0_normed * step_sum # advance x
        x_1 = x_0 - g_0_normed * step_g - c_sum * g_sum_0_normed * step_sum # advance x
        g_1 = gradient_func(x_1)  # gradient at new point
        g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
        g_1_normed = g_1 / g_1_norm  # normalized gradient

        g_sum_1 = g_1_normed + g_s_normed
        g_sum_1_norm = np.linalg.norm(g_sum_1)  # norm of the gradient
        g_sum_1_normed = g_sum_1 / g_sum_1_norm  # normalized gradient






        # caculating step_sum for the next iteration
        # dot product of g_sum_1 and g_sum_0
        g_sum_1_0_dot = np.dot(g_sum_1_normed, g_sum_0_normed)

        if g_sum_1_0_dot < 0:
            step_sum = step_sum * 0.5
        else:
            step_sum = step_sum * 1.5



        # caculating step_g for the next iteration
        # dot product of gradient_1 and gradient_0
        g_1_0_dot = np.dot(g_1_normed, g_0_normed)

        if g_1_0_dot < 0:
            step_g = step_g * 0.5
            # c_sum = 1
        else:
            step_g = step_g * 1.5
            step_sum = l_rate
            # c_sum = 0




        # swap values for the next step
        x_0 = x_1
        g_0_normed = g_1_normed
        g_sum_0_normed = g_sum_1_normed



        # break for small gradient
        if g_1_norm < g_norm_min:
            break

        print(t, step_g, step_sum)

        # save the points for graph and print progress
        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x_0], axis=0)
            print("BGD_3 %: ", int((t-1)/max_iterations*100))


    return x_out, t-1














def bisec_gd_opt_s3(x_start, gradient_func, l_rate, step_mult_max, max_iterations, g_norm_min, tot_num_save):

    # this function doubles the step until it finds a pair
    # after finding a pair it returns half a step in the direction of the current gradient
    # From this point it goes in the direction of the sum of the previous step and the one before that
    # if the the direction of gradient at this point is against the prevoius sum direction it halves the step_sum
    # otherwise it does step_sum * 1.01
    # then it resets the saved gradient to the curent one and starts over

    # bisection gradient descent optimization function
    t = 0  # step counter to zero
    step_g = l_rate  # inital step (can be anything)
    step_sum = l_rate
    step_sum_save = step_sum


    # put the initial point as the saved point
    t = t + 1  # increase step counter
    x_s = x_start  # set x_s to x_start
    x_out = np.array([x_s])  # add to x_out array for graph
    g_s = gradient_func(x_s)  # gradient at new point
    g_s_norm = np.linalg.norm(g_s)  # norm of the gradient
    g_s_normed = g_s / g_s_norm  # normalized gradient

    # advance x one step
    t = t + 1  # increase step counter
    x_0 = x_s - g_s_normed * step_g  # advance x
    # x_out = np.append(x_out, [x_0], axis=0) #add to x_out array for graph
    g_0 = gradient_func(x_0)  # gradient at new point
    g_0_norm = np.linalg.norm(g_0)  # norm of the gradient
    g_0_normed = g_0 / g_0_norm  # normalized gradient

    g_sum_0 = g_0_normed + g_s_normed
    g_sum_0_norm = np.linalg.norm(g_sum_0)  # norm of the gradient
    g_sum_0_normed = g_sum_0 / g_sum_0_norm  # normalized gradient

    g_sum_s_normed = g_sum_0_normed

    c_sum = 1

    while t <= max_iterations:  # steps
        # advance x one step
        t = t + 1  # increase step counter
        # x_1 = x_0 - g_0_normed * step_g - g_sum_0_normed * step_sum # advance x
        x_1 = x_0 - g_0_normed * step_g
        g_1 = gradient_func(x_1)  # gradient at new point
        g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
        g_1_normed = g_1 / g_1_norm  # normalized gradient


        # # caculating step_sum for the next iteration
        # # dot product of g_sum_1 and g_sum_0
        # g_sum_1_0_dot = np.dot(g_sum_1_normed, g_sum_0_normed)
        #
        # if g_sum_1_0_dot < 0:
        #     step_sum = step_sum * 0.5
        # else:
        #     step_sum = step_sum * 1.5



        # caculating step for the next iteration
        # dot product of gradient_1 and gradient_0
        g_1_0_dot = np.dot(g_1_normed, g_0_normed)
        # dot product of gradient_1 and gradient_s
        g_1_s_dot = np.dot(g_1_normed, g_s_normed)

        if g_1_0_dot < g_1_s_dot:
            x_s = x_0
            g_1_s_dot = g_1_0_dot
            g_s_normed = g_0_normed

        x_0 = x_1
        g_0_normed = g_1_normed
        x_out = np.append(x_out, [x_0], axis=0)

        if g_1_s_dot < 0:

            step_g = step_g * 0.5

            t = t + 1  # increase step counter
            x_1 = x_0 - g_0_normed * step_g
            g_1 = gradient_func(x_1)  # gradient at new point
            g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
            g_1_normed = g_1 / g_1_norm  # normalized gradient

            g_sum_1 = g_0_normed + g_s_normed
            g_sum_1_norm = np.linalg.norm(g_sum_1)  # norm of the gradient
            g_sum_1_normed = g_sum_1 / g_sum_1_norm  # normalized gradient

            x_0 = x_1
            g_0_normed = g_1_normed
            g_sum_0_normed = g_sum_1_normed
            x_out = np.append(x_out, [x_0], axis=0)

            t = t + 1  # increase step counter
            x_1 = x_0 - g_sum_0_normed * step_sum
            g_1 = gradient_func(x_1)  # gradient at new point
            g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
            g_1_normed = g_1 / g_1_norm  # normalized gradient

            g_1_sum0_dot = np.dot(g_1_normed, g_sum_0_normed)
            g_sums_sum0_dot = np.dot(g_sum_s_normed, g_sum_0_normed)
            if g_1_sum0_dot < -1/np.sqrt(2) or g_sums_sum0_dot < 0 :
                print(t, g_1_sum0_dot, g_sums_sum0_dot)
                step_sum = step_sum * 0.5
            else:
                step_sum = step_sum * 1.1

            x_0 = x_1
            g_0_normed = g_1_normed
            g_sum_s_normed = g_sum_0_normed
            x_out = np.append(x_out, [x_0], axis=0)

            x_s = x_0
            g_s_normed = g_0_normed




        else:
            step_g = step_g * 1.5
            # c_sum = 0



        print(t, step_g, step_sum)




        # break for small gradient
        if g_1_norm < g_norm_min:
            break


        # save the points for graph and print progress
        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            # x_out = np.append(x_out, [x_0], axis=0)
            print("BGD_3 %: ", int((t-1)/max_iterations*100))


    return x_out, t-1





def bisec_gd_opt_s4(x_start, gradient_func, l_rate, step_mult_max, max_iterations, g_norm_min, tot_num_save):
    # bisection gradient descent optimization function
    t = 0  # step counter to zero
    step_g = l_rate  # inital step (can be anything)
    step_sum = l_rate
    m =0
    v = 0

    # put the initial point as the saved point
    t = t + 1  # increase step counter
    x_s = x_start  # set x_s to x_start
    x_out = np.array([x_s])  # add to x_out array for graph
    g_s = gradient_func(x_s)  # gradient at new point
    g_s_norm = np.linalg.norm(g_s)  # norm of the gradient
    g_s_normed = g_s / g_s_norm  # normalized gradient

    # advance x one step
    t = t + 1  # increase step counter
    x_0 = x_s - g_s_normed * step_g  # advance x
    # x_out = np.append(x_out, [x_0], axis=0) #add to x_out array for graph
    g_0 = gradient_func(x_0)  # gradient at new point
    g_0_norm = np.linalg.norm(g_0)  # norm of the gradient
    g_0_normed = g_0 / g_0_norm  # normalized gradient

    g_sum_0 = g_0_normed + g_s_normed
    g_sum_0_norm = np.linalg.norm(g_sum_0)  # norm of the gradient
    g_sum_0_normed = g_sum_0 / g_sum_0_norm  # normalized gradient

    g_sum_s_normed = g_sum_0_normed

    while t <= max_iterations:  # steps
        # # advance x one step
        t = t + 1  # increase step counter
        # x_1 = x_0 - g_0_normed * step_g + g_sum_0_normed * step_sum # advance x
        x_1 = x_0 - g_0_normed * step_g
        x_out = np.append(x_out, [x_1], axis=0)
        g_1 = gradient_func(x_1)  # gradient at new point
        g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
        g_1_normed = g_1 / g_1_norm  # normalized gradient







        # g_sum_1 = g_1_normed + g_s_normed
        # g_sum_1_norm = np.linalg.norm(g_sum_1)  # norm of the gradient
        # g_sum_1_normed = g_sum_1 / g_sum_1_norm  # normalized gradient


        # caculating step_g for the next iteration
        # dot product of gradient_1 and gradient_0
        g_1_0_dot = np.dot(g_1_normed, g_0_normed)
        # dot product of gradient_1 and gradient_s
        g_1_s_dot = np.dot(g_1_normed, g_s_normed)

        if g_1_0_dot < g_1_s_dot:
            g_1_s_dot = g_1_0_dot
            g_s_normed = g_0_normed

        x_0 = x_1
        g_0_normed = g_1_normed

        if g_1_s_dot < 0:
            step_g = step_g * 0.5

            t = t + 1  # increase step counter
            # x_1 = x_0 - g_0_normed * step_g + g_sum_0_normed * step_sum # advance x
            x_1 = x_0 - g_0_normed * step_g
            x_out = np.append(x_out, [x_1], axis=0)
            g_1 = gradient_func(x_1)  # gradient at new point
            g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
            g_1_normed = g_1 / g_1_norm  # normalized gradient

            x_0 = x_1
            g_0_normed = g_1_normed

            m = beta_1 * m + (1 - beta_1) * g_0
            v = beta_2 * v + (1 - beta_2) * np.power(g_0, 2)
            m_hat = m / (1 - beta_1 ** t)
            v_hat = v / (1 - beta_2 ** t)
            x_1 = x_0 - (l_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
            x_out = np.append(x_out, [x_1], axis=0)
            t = t + 1

            g_1 = gradient_func(x_1)  # gradient at new point
            g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
            g_1_normed = g_1 / g_1_norm  # normalized gradient

        else:
            step_g = step_g * 2.0
            v = 0
            m = 0

        # swap values for the next step
        x_0 = x_1
        g_0_normed = g_1_normed



        # # caculating step_sum for the next iteration
        # # dot product of g_sum_1 and g_sum_0
        # g_sum_1_0_dot = np.dot(g_sum_1_normed, g_sum_0_normed)
        # # dot product of g_sum_1 and g_sum_s
        # g_sum_1_s_dot = np.dot(g_sum_1_normed, g_sum_s_normed)
        #
        # if g_sum_1_0_dot < g_sum_1_s_dot:
        #     g_sum_1_s_dot = g_sum_1_0_dot
        #     g_sum_s_normed = g_sum_0_normed
        #
        # if g_sum_1_s_dot < 0:
        #     step_sum = step_sum * 0.5
        # else:
        #     step_sum = l_rate

        # swap values for the next step
        x_0 = x_1
        g_0_normed = g_1_normed
        # g_sum_0_normed = g_sum_1_normed



        # break for small gradient
        if g_1_norm < g_norm_min:
            break

        print(t, step_g, step_sum)

        # save the points for graph and print progress
        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            # x_out = np.append(x_out, [x_0], axis=0)
            print("BGD_3 %: ", int((t-1)/max_iterations*100))


    return x_out, t-1










def bisec_gd_opt(x_start, gradient_func, l_rate, step_mult_max, max_iterations, g_norm_min, tot_num_save):

    # this function doubles the step until it finds a pair
    # after finding a pair it returns half a step in the direction of the current gradient
    # From this point it goes in the direction of the sum of the previous step and the one before that
    # if the the direction of gradient at this point is against the prevoius sum direction it halves the step_sum
    # otherwise it does step_sum * 1.01
    # then it resets the saved gradient to the curent one and starts over

    # bisection gradient descent optimization function
    t = 0  # step counter to zero
    step_g = l_rate  # inital step (can be anything)
    step_sum = l_rate
    step_sum_save = step_sum


    # put the initial point as the saved point
    t = t + 1  # increase step counter
    x_s = x_start  # set x_s to x_start
    x_out = np.array([x_s])  # add to x_out array for graph
    g_s = gradient_func(x_s)  # gradient at new point
    g_s_norm = np.linalg.norm(g_s)  # norm of the gradient
    g_s_normed = g_s / g_s_norm  # normalized gradient

    # advance x one step
    t = t + 1  # increase step counter
    x_0 = x_s - g_s_normed * step_g  # advance x
    # x_out = np.append(x_out, [x_0], axis=0) #add to x_out array for graph
    g_0 = gradient_func(x_0)  # gradient at new point
    g_0_norm = np.linalg.norm(g_0)  # norm of the gradient
    g_0_normed = g_0 / g_0_norm  # normalized gradient

    g_sum_0 = g_0_normed + g_s_normed
    g_sum_0_norm = np.linalg.norm(g_sum_0)  # norm of the gradient
    g_sum_0_normed = g_sum_0 / g_sum_0_norm  # normalized gradient

    g_sum_s_normed = g_sum_0_normed

    c_sum = 1

    while t <= max_iterations:  # steps
        # advance x one step
        t = t + 1  # increase step counter
        # x_1 = x_0 - g_0_normed * step_g - g_sum_0_normed * step_sum # advance x
        x_1 = x_0 - g_0_normed * step_g
        g_1 = gradient_func(x_1)  # gradient at new point
        g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
        g_1_normed = g_1 / g_1_norm  # normalized gradient


        # caculating step for the next iteration
        # dot product of gradient_1 and gradient_0
        g_1_0_dot = np.dot(g_1_normed, g_0_normed)
        # dot product of gradient_1 and gradient_s
        g_1_s_dot = np.dot(g_1_normed, g_s_normed)

        if g_1_0_dot < g_1_s_dot:
            x_s = x_0
            g_1_s_dot = g_1_0_dot
            g_s_normed = g_0_normed

        x_0 = x_1
        g_0_normed = g_1_normed
        x_out = np.append(x_out, [x_0], axis=0)

        if g_1_s_dot < 0:

            step_g = step_g * 0.5

            t = t + 1  # increase step counter
            x_1 = x_0 - g_0_normed * step_g
            g_1 = gradient_func(x_1)  # gradient at new point
            g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
            g_1_normed = g_1 / g_1_norm  # normalized gradient

            g_sum_1 = g_0_normed + g_s_normed
            g_sum_1_norm = np.linalg.norm(g_sum_1)  # norm of the gradient
            g_sum_1_normed = g_sum_1 / g_sum_1_norm  # normalized gradient

            x_0 = x_1
            g_0_normed = g_1_normed
            g_sum_0_normed = g_sum_1_normed
            x_out = np.append(x_out, [x_0], axis=0)

            # t = t + 1  # increase step counter
            # x_1 = x_0 - g_sum_0_normed * step_sum
            # g_1 = gradient_func(x_1)  # gradient at new point
            # g_1_norm = np.linalg.norm(g_1)  # norm of the gradient
            # g_1_normed = g_1 / g_1_norm  # normalized gradient
            #
            # g_1_sum0_dot = np.dot(g_1_normed, g_sum_0_normed)
            # g_sums_sum0_dot = np.dot(g_sum_s_normed, g_sum_0_normed)
            # if g_1_sum0_dot < -1/np.sqrt(2) or g_sums_sum0_dot < 0 :
            #     print(t, g_1_sum0_dot, g_sums_sum0_dot)
            #     step_sum = step_sum * 0.5
            # else:
            #     step_sum = step_sum * 1.1

            m = beta_1 * m + (1 - beta_1) * g_sum_1_normed
            v = beta_2 * v + (1 - beta_2) * np.power(g_sum_1_normed, 2)
            m_hat = m / (1 - beta_1 ** t)
            v_hat = v / (1 - beta_2 ** t)
            x_1 = x_0 - (l_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
            # x_out = np.append(x_out, [x], axis=0)

            x_0 = x_1
            g_0_normed = g_1_normed
            g_sum_s_normed = g_sum_0_normed
            x_out = np.append(x_out, [x_0], axis=0)

            x_s = x_0
            g_s_normed = g_0_normed




        else:
            step_g = step_g * 1.5
            m = 0
            v = 0
            # c_sum = 0







        # break for small gradient
        if g_1_norm < g_norm_min:
            break


        # save the points for graph and print progress
        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            # x_out = np.append(x_out, [x_0], axis=0)
            print("BGD_3 %: ", int((t-1)/max_iterations*100))


    return x_out, t-1










########### Test functions ###########################
###########################################################


def rosenbrock_main(x):
    """The Rosenbrock function"""
    a = 0
    b = 100
    return (a-x[0])**2 + b*(x[1]-x[0]**2)**2.0


def rosenbrock_grad(x):
    """Gradient of Rosenbrock function"""
    a = 0
    b = 100
    g = np.empty(x.shape)
    g[0] = - 2.0*(a-x[0]) - 2.0*b*(x[1]-x[0]**2)*2.0*x[0]
    g[1] = 2.0*b*(x[1]-x[0]**2)
    return g


def EASOM_main(x):
    pi = 3.14159265359
    x1 = x[0]
    x2 = x[1]
    fact1 = -np.cos(x1) * np.cos(x2)
    fact2 = np.exp(-(x1 - pi) ** 2 - (x2 - pi) ** 2)
    f = fact1 * fact2
    return f


def EASOM_grad(x):
    pi = 3.14159265359
    x1 = x[0]
    x2 = x[1]
    g = np.empty(x.shape)
    g[0] = -(2*pi - 2*x1)*np.exp(-(-pi + x1)**2 - (-pi + x2)**2)*np.cos(x1) * \
        np.cos(x2) + np.exp(-(-pi + x1)**2 -
                            (-pi + x2)**2)*np.sin(x1)*np.cos(x2)
    g[1] = -(2*pi - 2*x2)*np.exp(-(-pi + x1)**2 - (-pi + x2)**2)*np.cos(x1) * \
        np.cos(x2) + np.exp(-(-pi + x1)**2 -
                            (-pi + x2)**2)*np.sin(x2)*np.cos(x1)
    return g


def MATYAS_main(x):
    x1 = x[0]
    x2 = x[1]
    term1 = 0.26 * (x1 ** 2 + x2 ** 2)
    term2 = -0.48 * x1 * x2
    f = term1 + term2
    return f


def MATYAS_grad(x):
    x1 = x[0]
    x2 = x[1]
    g = np.empty(x.shape)
    g[0] = 0.52 * x1 - 0.48 * x2
    g[1] = -0.48*x1 + 0.52*x2
    return g




############## Soft Max ###############
rand.seed(2)
n = 300
d = 2
total_C = 2

# X = rand.randn(n, d)   #Let X be a random matrix

A = rand.randn(n, d)
D = np.logspace(1, 8, d)
X = A*D  # set X as a ill conditioned Matrix


I = np.eye(total_C, total_C - 1)
ind = rand.randint(total_C, size=n)

Y = I[ind, :]



lamda = 1
reg = None


def reg(x): return regConvex(x, lamda)


# def reg(x): return regNonconvex(x, lamda)
w = rand.randn(d*(total_C-1), 1)


def fun(w): return softmax(X, Y, w, reg=reg)


f, g, Hv = fun(w)


def softMax_grad(x):
    f, g, Hv = fun(x)
    return g.T[0]


def softMax_main(x):
    f, g, Hv = fun(x)
    return f[0][0]




########### Main part ###########################
###########################################################



# converging parameters
max_iterations = 1000# maximum number of iterations
tot_num_save = max_iterations # max(int(max_iterations/100), max_iterations)
min_g = -1  # 0.0001   #minimum gradient


# parameters for ADAM optimization
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1.0e-08
l_rate = 0.01

# parameters for bisection gradient descent
step_mult_max = 0.9


func_main = softMax_main
func_grad = softMax_grad
# x_start = np.random.randint(2, size=d)
# x_start = np.zeros(d)
# x_start = np.ones(d)
x_start = rand.randn(d)
x_min = -0.5
x_max = 0.1
y_min = -0.05
y_max = 0.05
x_opt = 0
y_opt = 0

#
# # for this one enable the log plot below
# func_main = rosenbrock_main
# func_grad = rosenbrock_grad
# x_start = np.array([1, 2.5])
# x_min = -2.0
# x_max = 2.0
# y_min = -1.0
# y_max = 3.0
# x_opt = 0
# y_opt = 0


#
# # for this one enable the normal (not log) plot below
# func_main = EASOM_main
# func_grad = EASOM_grad
# x_start = np.array([2, 2])
# x_min = 1.5
# x_max = 4.75
# y_min = 1.5
# y_max = 4.75
#
# x_opt = 3.1416080159444633
# y_opt = 3.1416080159444633



# func_main = MATYAS_main
# func_grad = MATYAS_grad
# x_start = np.array([5, 0])
# x_min = -10.0
# x_max = 10.0
# y_min = -10.0
# y_max = 10.0




################# applying optimisers #####################
###########################################################


# applying gradient dhescent
x_out_1, t_1 = gd_opt(x_start, func_grad, l_rate, max_iterations, min_g, tot_num_save)
# applying ADAM
x_out_2, t_2 = adam_opt(x_start, func_grad, l_rate, epsilon, max_iterations, beta_1, beta_2, min_g, tot_num_save)
# applying Bisection Gradient Descent
x_out_3, t_3 = bisec_gd_opt(x_start, func_grad, l_rate, step_mult_max, max_iterations, min_g, tot_num_save)








########### Plots for two dimensional tests ###############
###########################################################

num_var = d


if d  == 2:
    # genrating vectors for countour plot
    kx = np.linspace(x_min, x_max, 50)
    ky = np.linspace(y_min, y_max, 50)
    mx, my = np.meshgrid(kx, ky)

    f = np.empty((ky.shape[0], kx.shape[0]))
    gg = np.empty((ky.shape[0], kx.shape[0],num_var))
    gg_norm = np.empty((ky.shape[0], kx.shape[0],num_var))



    for ix in range(f.shape[0]):
        for iy in range(f.shape[1]):
            f[ix, iy ] = func_main(np.array([mx[ix, iy], my[ix, iy]]))

    for ix in range(gg.shape[0]):
        for iy in range(gg.shape[1]):
            gg[ix, iy , :] = func_grad(np.array([mx[ix, iy], my[ix, iy]]))
            #gg_norm[ix, iy, :] = gg[ix, iy , :] / np.linalg.norm(gg[ix, iy , :])


    # drawing contour plot of the function
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    cmap = plt.get_cmap('seismic')
    cs = ax1.contourf(mx, my, np.log10(f), 100, cmap=cmap) #enable this for Rosenbrock function
    # cs = ax1.contourf(mx, my, f, 100, cmap=cmap)  #enable this for EASOM function
    fig1.colorbar(cs, ax=ax1, shrink=0.9)

    # fig3 = plt.figure(3)
    # ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
    # surf = ax3.plot_surface(mx, my, f, rstride=1, cstride=1,linewidth=0, antialiased=False)

    # u = gg_norm[:, :, 0]
    # v = gg_norm[:, :, 1]
    # q = ax1.quiver(mx, my, u, v)

    ### adding trajectories to two dimensional test plots #####

    # plotting the trajectory of gradient descent
    # ax1.plot(x_out_1[:,0],x_out_1[:,1],'b.-')

    # plotting the trajectory of ADAM
    ax1.plot(x_out_2[:,0],x_out_2[:,1],'b.-')

    # plotting the trajectory of bisection gradient descent
    ax1.plot(x_out_3[:, 0], x_out_3[:, 1], 'y.-')

###########################################################
###########################################################



################# final plots #########################
###########################################################

# creating step number arrays for diffrent outputs
t_1_arr = np.array([i*max_iterations//100 for i in range(0, x_out_1.shape[0])])
t_2_arr = np.array([i*max_iterations//100 for i in range(0, x_out_2.shape[0])])
t_3_arr = np.array([i*max_iterations//100 for i in range(0, x_out_3.shape[0])])

# ploting the convergence graph
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)


f3 = np.array([func_main(i) for i in x_out_3])
f2 = np.array([func_main(i) for i in x_out_2])
f1 = np.array([func_main(i) for i in x_out_1])

# ax2.plot(t_1_arr,  f1, '.-', label='GD')
ax2.plot(t_2_arr, f2, 'b.-', label='ADAM')
ax2.plot(t_3_arr, f3, 'y.-', label='BGD')

plt.xlabel("steps")
plt.ylabel("function value")
legend = ax2.legend()
plt.show()
