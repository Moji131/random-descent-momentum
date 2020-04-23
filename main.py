import numpy as np
from matplotlib.pyplot import ion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from numpy.linalg import norm
import numpy.random as rand



########### Optimiser functions ###########################
###########################################################

from optimiser_ABGD import optimiser_ABGD
from optimiser_ADAM import optimiser_ADAM
from optimiser_NAGD import optimiser_NAGD




########### Selecting the test fucntion ###################
###########################################################


############ Softmax test funcion ###############


# To change the input data of the softmax regression go to test_function_softmax.py
from test_function_softmax import softMax_main as func_main
from test_function_softmax import softMax_grad as func_grad
from test_function_softmax import d_func
from test_function_softmax import description_func
d = d_func()

x_start = np.random.randint(2, size=d)
x_start = np.zeros(d)
x_start = np.ones(d)

x_min = -1
x_max = 1.2
y_min = -1
y_max = 1.2
x_opt = 0
y_opt = 0

log_plot = False
description = description_func()
convergence_plt_title = "Convergence - " + description
trajectory_plt_title = "Trajectories - " + description


#########  Rosenbrock test function  #################

# from test_function_rosenbrock import rosenbrock_main as func_main
# from test_function_rosenbrock import rosenbrock_grad as func_grad
#
# x_start = np.array([1, 2.5])
# d = 2
#
# x_min = -2.0
# x_max = 2.0
# y_min = -1.0
# y_max = 3.0
# x_opt = 0
# y_opt = 0
# log_plot = True
# convergence_plt_title = "Convergence - Rosenbrock Function"
# trajectory_plt_title = "Trajectories - Rosenbrock Function"


#########  easom test function  #################

# from test_function_easom import easom_main as func_main
# from test_function_easom import easom_grad as func_grad
# x_start = np.array([4, 4])
# d = 2
# x_min = 1.7
# x_max = 4.5
# y_min = 1.7
# y_max = 4.5
#
# x_opt = 3.1416080159444633
# y_opt = 3.1416080159444633
# log_plot = False
# convergence_plt_title = "Convergence - EASOM Function"
# trajectory_plt_title = "Trajectories - EASOM Function"



#########  MATYAS test function  #################

# from test_function_matyas import matyas_main as func_main
# from test_function_matyas import matyas_grad as func_grad
# d = 2
# x_start = np.array([2, 3])
# x_start = np.array([5, 0])
# x_min = -10.0
# x_max = 10.0
# y_min = -10.0
# y_max = 10.0
# log_plot = True
# convergence_plt_title = "Convergence - MATYAS Function"
# trajectory_plt_title = "Trajectories - MATYAS Function"




################# applying optimisers #####################
###########################################################

# converging parameters
max_iterations = 200 # maximum number of iterations
tot_num_save = max_iterations # max(int(max_iterations/100), max_iterations)


# applying gradient descent
x_out_1, t_1 = optimiser_NAGD(x_start, func_grad, max_iterations, tot_num_save)
# applying ADAM
x_out_2, t_2 = optimiser_ADAM(x_start, func_grad, max_iterations, tot_num_save)
# applying Bisection Gradient Descent
x_out_3, t_3 = optimiser_ABGD(x_start, func_grad, max_iterations, tot_num_save)





########### Trajectory Plot for two dimensional tests #####
###########################################################

if d  == 2:
    # genrating vectors for countour plot
    kx = np.linspace(x_min, x_max, 50)
    ky = np.linspace(y_min, y_max, 50)
    mx, my = np.meshgrid(kx, ky)

    f = np.empty((ky.shape[0], kx.shape[0]))
    gg = np.empty((ky.shape[0], kx.shape[0],d))
    gg_norm = np.empty((ky.shape[0], kx.shape[0],d))



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

    if log_plot:
        cs = ax1.contourf(mx, my, np.log10(f), 100, cmap=cmap) #enable this for Rosenbrock function
    else:
        cs = ax1.contourf(mx, my, f, 100, cmap=cmap)  #enable this for EASOM function

    fig1.colorbar(cs, ax=ax1, shrink=0.9)

    # fig3 = plt.figure(3)
    # ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
    # surf = ax3.plot_surface(mx, my, f, rstride=1, cstride=1,linewidth=0, antialiased=False)

    # u = gg_norm[:, :, 0]
    # v = gg_norm[:, :, 1]
    # q = ax1.quiver(mx, my, u, v)

    ### adding trajectories to two dimensional test plots #####

    # plotting the trajectory of gradient descent
    ax1.plot(x_out_1[:,0],x_out_1[:,1],'k.-', label='NAGD')

    # plotting the trajectory of ADAM
    ax1.plot(x_out_2[:,0],x_out_2[:,1],'b.-', label='ADAM')

    # plotting the trajectory of bisection gradient descent
    ax1.plot(x_out_3[:, 0], x_out_3[:, 1], 'y.-', label='ABGD')

    plt.title(trajectory_plt_title)
    legend = ax1.legend()

###########################################################
###########################################################



################# Convergence plot #########################
###########################################################

# ploting the convergence graph
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)


t_1_arr = np.array([i for i in range(0, x_out_1.shape[0])])
f1 = np.array([func_main(i) for i in x_out_1])
ax2.plot(t_1_arr,  f1, 'k.-', label='NAGD')

t_2_arr = np.array([i for i in range(0, x_out_2.shape[0])])
f2 = np.array([func_main(i) for i in x_out_2])
ax2.plot(t_2_arr, f2, 'b.-', label='ADAM')

t_3_arr = np.array([i for i in range(0, x_out_3.shape[0])])
f3 = np.array([func_main(i) for i in x_out_3])
ax2.plot(t_3_arr, f3, 'y.-', label='ABGD')


plt.title(convergence_plt_title)
plt.xlabel("steps")
plt.ylabel("function value")
legend = ax2.legend()
plt.show()
