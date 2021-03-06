import numpy as np
from matplotlib.pyplot import ion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from numpy.linalg import norm
import numpy.random as rand

from cycler import cycler
import os



max_iterations = 10 # maximum number of iterations



# ########### Selecting the test fucntion ###################
# ###########################################################


#########  Quadratic test function  #################

from test_function_quadratic import quadratic_main as func_main
from test_function_quadratic import quadratic_grad as func_grad
d = 2
list1 = []
for i in range(d):
    list1 = list1 + [1+(6-i%5)**5 /100 ]
x_start = np.array(list1)
max1 = max(list1)
x_min = -max1 *1.1
x_max = max1 *1.1
y_min = -max1 *1.1
y_max = max1 *1.1
log_plot = True
convergence_plt_title = "Convergence - Quadratic Function"
trajectory_plt_title = "Trajectories - Quadratic Function"



############ Softmax test funcion ###############
# To change the input data of the softmax regression go to test_function_softmax.py


# from test_function_softmax import softMax_main as func_main
# from test_function_softmax import softMax_grad as func_grad
# from test_function_softmax import d_func
# from test_function_softmax import description_func
# d = d_func()
#
# # x_start = np.random.randint(2, size=d)
# # x_start = np.zeros(d)
# x_start = np.ones(d)*1.05
#
# x_min = -1.5
# x_max = 1.3
# y_min = -1.5
# y_max = 1.3
# x_opt = 0
# y_opt = 0
#
# log_plot = False
# description = description_func()
# convergence_plt_title = "Convergence - " + description
# trajectory_plt_title = "Trajectories - " + description


# #########  Rosenbrock test function  #################

# from test_function_rosenbrock import rosenbrock_main as func_main
# from test_function_rosenbrock import rosenbrock_grad as func_grad
#
# x_start = np.array([0.5, 1.5])
# d = 2
#
# x_min = -0.2
# x_max = 1.5
# y_min = -0.5
# y_max = 2.0
# x_opt = 0
# y_opt = 0
# log_plot = True
# convergence_plt_title = "Convergence - Rosenbrock Function"
# trajectory_plt_title = "Trajectories - Rosenbrock Function"


#########  easom test function  #################

# from test_function_easom import easom_main as func_main
# from test_function_easom import easom_grad as func_grad
# x_start = np.array([3.14 - 1.5, 3.14 - 0])
# d = 2
# x_min = 3.14 - 2
# x_max = 3.14 + 2
# y_min = 3.14 - 2
# y_max = 3.14 + 2
#
# x_opt = 3.1416080159444633
# y_opt = 3.1416080159444633
# log_plot = False
# convergence_plt_title = "Convergence - EASOM Function"
# trajectory_plt_title = "Trajectories - EASOM Function"





############ import optimizers ###############
##############################################

# opt_list = ['optimizer_ADAM', 'optimizer_ANGD', 'optimizer_ANGDc', 'optimizer_GDM']
# lr =  [ 1e-1,              1e-1,                 1e-1,           ,  1e-1            ]

opt_list = ['optimizer_GDM']
lr =  [ 1e-1             ]

opt_n = len(opt_list) # number of optimizers defined


optimizer = [ 0 for i in range(opt_n)]
label = [ 0 for i in range(opt_n)]
x_out = [ 0 for i in range(opt_n)]
t_out = [ 0 for i in range(opt_n)]
closure_list = [ 0 for i in range(opt_n)]




for i, opt in enumerate(opt_list):
    opt1 = __import__(opt)
    optimizer[i] = opt1.opt(x_start, lr[i])
    optimizer[i].x = x_start
    label[i] = opt_list[i] + " lr=" + str(lr[i])
    x_out[i] = [x_start]
    t_out[i] = [0]
    #defining the function to reevaluate function and gradient if needed
    def closure():
        optimizer[i].g = func_grad(optimizer[i].x)
        loss = func_main(optimizer[i].x)
        return loss
    closure_list[i] = closure


######  creating files to output data ########
##############################################
path = "outputs/main"
if not os.path.exists(path):
    os.makedirs(path)
files = os.listdir(path)
for f in files:
    file_path = path + "/" + f
    os.remove(file_path)
file = [ 0 for i in range(opt_n)]


############## Running optimizers #################
###################################################




for t in range(1,max_iterations):
    print(t)

    for opt_i, opt in enumerate(opt_list):
        fv = func_main(optimizer[opt_i].x)
        optimizer[opt_i].loss1 = fv
        optimizer[opt_i].g = func_grad(optimizer[opt_i].x)

        optimizer[opt_i].step(closure_list[opt_i] )
        x_out[opt_i] = np.append(x_out[opt_i], [optimizer[opt_i].x], axis=0)
        t_out[opt_i] = np.append(t_out[opt_i], [t], axis=0)


        lr1 =  str( optimizer[opt_i].lr )
        lr1 = "-lr=" + lr1
        file[opt_i] = open('outputs/main/' + opt_list[opt_i] + lr1 , 'a')
        str_to_file = str(t) + "\t" + str(fv) + "\n"
        file[opt_i].write(str_to_file)
        file[opt_i].close()

        # print(opt)







########### Trajectory Plot for two dimensional tests #####
###########################################################

if d  == 2:
    # genrating vectors for countour plot
    kx = np.linspace(x_min, x_max, 50)
    ky = np.linspace(y_min, y_max, 50)
    mx, my = np.meshgrid(kx, ky)
    f = np.empty((ky.shape[0], kx.shape[0]))
    for ix in range(f.shape[0]):
        for iy in range(f.shape[1]):
            f[ix, iy ] = func_main(np.array([mx[ix, iy], my[ix, iy]]))



    # drawing contour plot of the function
    fig1 = plt.figure(2)
    ax1 = fig1.add_subplot(111)
    prop_cycle = (cycler('color', ['y', 'k', 'dimgrey', 'c', 'r', 'b', 'm', 'pink']))
    ax1.set_prop_cycle(prop_cycle)
    cmap = plt.get_cmap('seismic')
    if log_plot:
        cs = ax1.contourf(mx, my, np.log10(f), 100, cmap=cmap) #enable this for Rosenbrock function
    else:
        cs = ax1.contourf(mx, my, f, 100, cmap=cmap)  #enable this for EASOM function
    fig1.colorbar(cs, ax=ax1, shrink=0.9)



    ### adding trajectories to two dimensional test plots #####

    # plotting the trajectory of bisection gradient descent
    for i_opt, opt in enumerate(opt_list):
        ax1.plot(x_out[i_opt][:,0], x_out[i_opt][:,1], '-o', label=opt_list[i_opt])

    plt.title(trajectory_plt_title)
    legend = ax1.legend()






################# Convergence plot #########################
###########################################################

# # ploting the convergence graph
# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111)
#
# prop_cycle=(cycler('color', ['b', 'y', 'k', 'c', 'm' ]))
# ax2.set_prop_cycle(prop_cycle)
#
# for i_opt in opt_list:
#     f1 = np.array([func_main(x_i) for x_i in x_out[i_opt]])
#     ax2.plot(t_out[i_opt], f1, '.-', label=label[i_opt])
#
#
# plt.title(convergence_plt_title)
# plt.xlabel("steps")
# plt.ylabel("function value")
# # ax2.set_ylim(-130, 530)
# legend = ax2.legend()
# plt.show()

import main_plot

