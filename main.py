import numpy as np
from matplotlib.pyplot import ion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from numpy.linalg import norm
import numpy.random as rand

from cycler import cycler




########### Optimiser functions ###########################
###########################################################

from optimiser_ABGDc import abgd_c
from optimiser_ABGDv import abgd_v
from optimiser_ABGDcs import abgd_cs
from optimiser_ABGDvmd import abgd_vmd
from optimiser_ADAM import adam
from optimiser_GDM import gdm
from optimiser_ABGDcsd import abgd_csd


### parameters
opt_n = 7 # number of optimizers defined
# opt_list = [0,1,2,3,4,5] # list optimizers to be applied
# opt_list = [0,2,4,6]
opt_list = [2,6]

max_iterations = 130 #maximum number of iterations




# ########### Selecting the test fucntion ###################
# ###########################################################
#
#
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


#########  Rosenbrock test function  #################

from test_function_rosenbrock import rosenbrock_main as func_main
from test_function_rosenbrock import rosenbrock_grad as func_grad

x_start = np.array([1, 1.5])
d = 2

x_min = -0.2
x_max = 1.5
y_min = -0.5
y_max = 2.0
x_opt = 0
y_opt = 0
log_plot = True
convergence_plt_title = "Convergence - Rosenbrock Function"
trajectory_plt_title = "Trajectories - Rosenbrock Function"


#########  easom test function  #################

# from test_function_easom import easom_main as func_main
# from test_function_easom import easom_grad as func_grad
# x_start = np.array([3.14 - 2, 3.14 - 2])
# d = 2
# x_min = 3.14 - 5
# x_max = 3.14 + 5
# y_min = 3.14 - 5
# y_max = 3.14 + 5
#
# x_opt = 3.1416080159444633
# y_opt = 3.1416080159444633
# log_plot = False
# convergence_plt_title = "Convergence - EASOM Function"
# trajectory_plt_title = "Trajectories - EASOM Function"



#########  Quadratic test function  #################

# from test_function_matyas import matyas_main as func_main
# from test_function_matyas import matyas_grad as func_grad
# d = 2
# x_start = np.array([50, 1000])
# x_min = -200.0
# x_max = 220.0
# y_min = -100.0
# y_max = 1200.0
# log_plot = True
# convergence_plt_title = "Convergence - Quadratic Function"
# trajectory_plt_title = "Trajectories - Quadratic Function"
#



################# preperaing optimiser objects #####################
####################################################################


### defining lists to hold parameters for diffrent optmizers
lr = [ 0 for i in range(opt_n)]
name = [ 0 for i in range(opt_n)]
optimizer = [ 0 for i in range(opt_n)]
label = [ 0 for i in range(opt_n)]
x_out = [ 0 for i in range(opt_n)]
t_out = [ 0 for i in range(opt_n)]
closure_list = [ 0 for i in range(opt_n)]




################# creating optimiser objects #####################
####################################################################

##### Creating ABGDc object
lr[0] = 0.01
name[0] = "ABGDc"
optimizer[0] = abgd_c(x_start, lr=lr[0])
optimizer[0].x = x_start
label[0] = name[0] + " lr=" + str(lr[0])
x_out[0] = [x_start]
t_out[0] = [0]
#defining the function to reevalute function and gradient if needed
closure_list[0] = None



### Creating ABGDv object
lr[1] = 0.001
name[1] = "ABGDv"
optimizer[1] = abgd_v(x_start, lr=lr[1])
optimizer[1].x =  x_start
label[1] = name[1] + " lr=" + str(lr[1])
x_out[1] = [x_start]
t_out[1] = [0]
closure_list[1] = None


### Creating ABGDcs object
lr[2] = 0.01
name[2] = "ABGDcs"
optimizer[2] = abgd_cs(x_start, lr=lr[2])
optimizer[2].x = x_start
label[2] = name[2] + " lr=" + str(lr[2])
x_out[2] = [x_start]
t_out[2] = [0]
#defining the function to reevalute function and gradient if needed
closure_list[2] = None
# defining the function to reevalute function and gradient if needed
def closure():
    optimizer[2].g = func_grad(optimizer[2].x)
    loss = func_main(optimizer[2].x)
    return loss
closure_list[2] = closure



### Creating ABGDvmd object
lr[3] = 0.01
name[3] = "ABGDvmd"
momentum = 0.7
drift = True
optimizer[3] = abgd_vmd(x_start, lr=lr[3], momentum=momentum, drift=drift)
optimizer[3].x = x_start
label[3] = name[3] + " lr=" + str(lr[3])
x_out[3] = [x_start]
t_out[3] = [0]
# defining the function to reevalute function and gradient if needed
def closure():
    optimizer[3].g = func_grad(optimizer[3].x)
    loss = func_main(optimizer[3].x)
    return loss
closure_list[3] = closure


### Creating ADAM object
lr[4] = 0.1
name[4] = "ADAM"
optimizer[4] = adam(x_start, lr=lr[4])
optimizer[4].x = x_start
label[4] = name[4] + " lr=" + str(lr[4])
x_out[4] = [x_start]
t_out[4] = [0]
#defining the function to reevalute function and gradient if needed
closure_list[4] = None


### Creating GDM object
lr[5] = 0.001
name[5] = "GDM"
optimizer[5] = gdm(x_start, lr=lr[5])
optimizer[5].x = x_start
label[5] = name[5] + " lr=" + str(lr[5])
x_out[5] = [x_start]
t_out[5] = [0]
#defining the function to reevalute function and gradient if needed
closure_list[5] = None


### Creating ABGDcsd object
lr[6] = 0.01
name[6] = "ABGDcsd"
optimizer[6] = abgd_csd(x_start, lr=lr[6])
optimizer[6].x = x_start
label[6] = name[6] + " lr=" + str(lr[6])
x_out[6] = [x_start]
t_out[6] = [0]
#defining the function to reevalute function and gradient if needed
closure_list[6] = None
# defining the function to reevalute function and gradient if needed
def closure():
    optimizer[6].g = func_grad(optimizer[6].x)
    loss = func_main(optimizer[6].x)
    return loss
closure_list[6] = closure


############## Running optimizers #################
###################################################



for t in range(1,max_iterations):
    print(t)

    for opt_i in opt_list:

        optimizer[opt_i].g = func_grad(optimizer[opt_i].x)
        optimizer[opt_i].t = t
        optimizer[opt_i].step(closure_list[opt_i])
        x_out[opt_i] = np.append(x_out[opt_i], [optimizer[opt_i].x], axis=0)
        t_out[opt_i] = np.append(t_out[opt_i], [t], axis=0)







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
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    prop_cycle = (cycler('color', ['b', 'y', 'k', 'r', 'c', 'm']))
    ax1.set_prop_cycle(prop_cycle)
    cmap = plt.get_cmap('seismic')
    if log_plot:
        cs = ax1.contourf(mx, my, np.log10(f), 100, cmap=cmap) #enable this for Rosenbrock function
    else:
        cs = ax1.contourf(mx, my, f, 100, cmap=cmap)  #enable this for EASOM function
    fig1.colorbar(cs, ax=ax1, shrink=0.9)



    ### adding trajectories to two dimensional test plots #####

    # plotting the trajectory of bisection gradient descent
    for i_opt in opt_list:
        ax1.plot(x_out[i_opt][:,0], x_out[i_opt][:,1], '-o', label=label[i_opt])

    plt.title(trajectory_plt_title)
    legend = ax1.legend()





################# Convergence plot #########################
###########################################################

# ploting the convergence graph
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)

prop_cycle=(cycler('color', ['b', 'y', 'k', 'r', 'c', 'm' ]))
ax2.set_prop_cycle(prop_cycle)

for i_opt in opt_list:
    f1 = np.array([func_main(x_i) for x_i in x_out[i_opt]])
    ax2.plot(t_out[i_opt], f1, '.-', label=label[i_opt])


plt.title(convergence_plt_title)
plt.xlabel("steps")
plt.ylabel("function value")
# ax2.set_ylim(-130, 530)
legend = ax2.legend()
plt.show()
