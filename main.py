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
            print("GD %: ", t/c)

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
            print("Adam %: ", t/c)
    g_0_norm = np.linalg.norm(g)  # norm of the gradient
    print("adam", g_0_norm)
    return x_out, t-1




def bisec_gd_opt(x_start, gradient_func, step_mult_max, max_iterations, g_norm_min, tot_num_save):
    # bisection gradient descent optimization function
    t = 0  # step counter to zero
    step = 0.01  # inital step (can be anything)

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

    while t <= max_iterations:  # steps
        # advance x one step
        t = t + 1  # increase step counter
        x_1 = x_0 - g_0_normed * step  # advance x
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

        if g_1_s_dot < 0:
            # step = step * 0.5
            None
        else:
            step = step * 0.9999

        # swap values for the next step
        x_0 = x_1
        g_0_normed = g_1_normed

        # break if
        if g_1_norm < g_norm_min:
            break

        c = max_iterations//tot_num_save
        c_1 = t % c
        if c_1 == 0:
            # add to x_out array for graph
            x_out = np.append(x_out, [x_0], axis=0)
            print("BGD_3 %: ", t/c)

        # ax1.quiver(x_1[0], x_1[1], -g_1_normed[0], -g_1_normed[1], facecolor='b', scale=3, headlength=7)
        # print(-g_1_normed[0], -g_1_normed[1])
        # print(x_1, x_0 , g1_norm, step)
        # ax1.plot(x_out[:, 0], x_out[:, 1], 'ko-')
        # plt.show()
        # exit()
    print("BGD", g_1_norm)

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
n = 100
d = 50
total_C = 2

# X = rand.randn(n, d)   #Let X be a random matrix

A = rand.randn(n, d)
D = np.logspace(1, 8, d)
X = A*D  # set X as a ill conditioned Matrix

I = np.eye(total_C, total_C - 1)
ind = rand.randint(total_C, size=n)
Y = I[ind, :]
lamda = 0
reg = None


def reg(x): return regConvex(x, lamda)


# def reg(x): return regNonconvex(x, lamda)
w = rand.randn(d*(total_C-1), 1)


def fun(x): return softmax(X, Y, x, reg=reg)


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
max_iterations = 5000  # maximum number of iterations
tot_num_save = 100
min_g = -1  # 0.0001   #minimum gradient


# parameters for ADAM optimization
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1.0e-08
l_rate = 0.1

# parameters for bisection gradient descent
step_mult_max = 0.9


func_main = softMax_main
func_grad = softMax_grad
# x_start = np.random.randint(2, size=d)
# x_start = np.zeros(d)
# x_start = np.ones(d)
x_start = rand.randn(d)
x_min = -2.0
x_max = 2.0
y_min = -1.0
y_max = 3.0
x_opt = 0
y_opt = 0


# for this one enable the log plot below
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
#


# func_main = MATYAS_main
# func_grad = MATYAS_grad
# x_start = np.array([5, 0])
# x_min = -10.0
# x_max = 10.0
# y_min = -10.0
# y_max = 10.0


########### Plots for two dimensional tests ###############
###########################################################


# genrating vectors for countour plot
# kx = np.linspace(x_min, x_max, 50)
# ky = np.linspace(y_min, y_max, 50)
# mx, my = np.meshgrid(kx, ky)

# f = np.empty((ky.shape[0], kx.shape[0]))
# gg = np.empty((ky.shape[0], kx.shape[0],num_var))
# gg_norm = np.empty((ky.shape[0], kx.shape[0],num_var))
# f = func_main(np.array([mx, my]))
# for ix in range(gg.shape[0]):
#     for iy in range(gg.shape[1]):
#         gg[ix, iy , :] = func_grad(np.array([mx[ix, iy], my[ix, iy]]))
#         #gg_norm[ix, iy, :] = gg[ix, iy , :] / np.linalg.norm(gg[ix, iy , :])


# drawing contour plot of the function
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(111)
# cmap = plt.get_cmap('seismic')
# cs = ax1.contourf(mx, my, np.log10(f), 100, cmap=cmap) #enable this for Rosenbrock function
# # cs = ax1.contourf(mx, my, f, 100, cmap=cmap)  #enable this for EASOM function
# fig1.colorbar(cs, ax=ax1, shrink=0.9)

#fig3 = plt.figure(3)
#ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
#surf = ax3.plot_surface(mx, my, f, rstride=1, cstride=1,linewidth=0, antialiased=False)

#u = gg_norm[:, :, 0]
#v = gg_norm[:, :, 1]
#q = ax1.quiver(mx, my, u, v)


###########################################################
###########################################################


################# applying optimisers #####################
###########################################################


# applying gradient dhescent
# x_out_1, t_1 = gd_opt(x_start, func_grad, l_rate, max_iterations, min_g, tot_num_save)
# applying ADAM
x_out_2, t_2 = adam_opt(x_start, func_grad, l_rate, epsilon, max_iterations, beta_1, beta_2, min_g, tot_num_save)
# applying Bisection Gradient Descent
x_out_3, t_3 = bisec_gd_opt(x_start, func_grad,step_mult_max, max_iterations, min_g, tot_num_save)




### adding trajectories to two dimensional test plots #####
###########################################################

# plotting the trajectory of gradient descent
# ax1.plot(x_out_1[:,0],x_out_1[:,1],'k.-')

# plotting the trajectory of ADAM
# ax1.plot(x_out_2[:,0],x_out_2[:,1],'k.-')

# plotting the trajectory of bisection gradient descent
# ax1.plot(x_out_3[:, 0], x_out_3[:, 1], 'k.-')

###########################################################
###########################################################



################# final plots #########################
###########################################################

# creating step number arrays for diffrent outputs
# t_1_arr = np.array([i*max_iterations//100 for i in range(0, x_out_1.shape[0])])
t_2_arr = np.array([i*max_iterations//100 for i in range(0, x_out_2.shape[0])])
t_3_arr = np.array([i*max_iterations//100 for i in range(0, x_out_3.shape[0])])

# ploting the convergence graph
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)


f3 = np.array([func_main(i) for i in x_out_3])
f2 = np.array([func_main(i) for i in x_out_2])
# f1 = np.array([func_main(i) for i in x_out_1])

ax2.plot(t_3_arr, f3, '.-', label='BGD')
ax2.plot(t_2_arr, f2, '.-', label='ADAM')
# ax2.plot(t_1_arr,  f1, '.-', label='GD')
plt.xlabel("steps")
plt.ylabel("function value")
legend = ax2.legend()
plt.show()
