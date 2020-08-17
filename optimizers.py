
################# preperaing optimiser objects #####################
####################################################################


### defining lists to hold parameters for diffrent optmizers
opt_n = 9 # number of optimizers defined
name = [ 0 for i in range(opt_n)]
optimizer = [ 0 for i in range(opt_n)]
label = [ 0 for i in range(opt_n)]
x_out = [ 0 for i in range(opt_n)]
t_out = [ 0 for i in range(opt_n)]
closure_list = [ 0 for i in range(opt_n)]

#    [0    , 1    , 2   , 3  , 4      , 5      , 6     , 7           , 8          ]
#    [ABGDc, ABGDc, ADAM, GDM, ABGDvmd, ABGDcm2, ABGDvm, ABGDcm2_copy, ABGDvm_copy]


################# creating optimiser objects #####################
####################################################################
##### Creating ABGDc object
n = 0
name[0] = "ABGDc"
from optimiser_ABGDc import abgd_c
optimizer[0] = abgd_c(x_start, lr=lr[0])
optimizer[0].x = x_start
label[0] = name[0] + " lr=" + str(lr[0])
x_out[0] = [x_start]
t_out[0] = [0]
#defining the function to reevalute function and gradient if needed
def closure():
    optimizer[0].g = func_grad(optimizer[0].x)
    loss = func_main(optimizer[0].x)
    return loss
closure_list[0] = closure


### Creating ABGDv object
n = 1
name[1] = "ABGDv"
from optimiser_ABGDv import abgd_v
optimizer[1] = abgd_v(x_start, lr=lr[1])
optimizer[1].x = x_start
label[1] = name[1] + " lr=" + str(lr[1])
x_out[1] = [x_start]
t_out[1] = [0]
#defining the function to reevalute function and gradient if needed
def closure():
    optimizer[1].g = func_grad(optimizer[1].x)
    loss = func_main(optimizer[1].x)
    return loss
closure_list[1] = closure


### Creating ADAM object
n = 2
name[2] = "ADAM"
from optimiser_ADAM import adam
optimizer[2] = adam(x_start, lr=lr[2])
optimizer[2].x = x_start
label[2] = name[2] + " lr=" + str(lr[2])
x_out[2] = [x_start]
t_out[2] = [0]
# defining the function to reevalute function and gradient if needed
def closure():
    optimizer[2].g = func_grad(optimizer[2].x)
    loss = func_main(optimizer[2].x)
    return loss
closure_list[2] = closure


### Creating GDM object
n = 3
name[3] = "GDM"
from optimiser_GDM import gdm
optimizer[3] = gdm(x_start, lr=lr[3])
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


### Creating ABGDvmd object
n = 4
name[4] = "ABGDvmd"
from optimiser_ABGDvmd import abgd_vmd
optimizer[4] = abgd_vmd(x_start, lr=lr[4])
optimizer[4].x = x_start
label[4] = name[4] + " lr=" + str(lr[4])
x_out[4] = [x_start]
t_out[4] = [0]
# defining the function to reevalute function and gradient if needed
def closure():
    optimizer[4].g = func_grad(optimizer[4].x)
    loss = func_main(optimizer[4].x)
    return loss
closure_list[4] = closure


####### ABGDcm2 ##################
n = 5
name[5] = "ABGDcm2"
from optimiser_ABGDcm2 import abgd_cm2
optimizer[5] = abgd_cm2(x_start, lr=lr[5])
optimizer[5].x = x_start
label[5] = name[5] + " lr=" + str(lr[5])
x_out[5] = [x_start]
t_out[5] = [0]
# defining the function to reevalute function and gradient if needed
def closure():
    optimizer[5].g = func_grad(optimizer[5].x)
    loss = func_main(optimizer[5].x)
    return loss
closure_list[5] = closure


####### ABGDvm ##################
n = 6
name[6] = "ABGDvm"
from optimiser_ABGDvm import abgd_vm
optimizer[6] = abgd_vm(x_start, lr=lr[6])
optimizer[6].x = x_start
label[6] = name[6] + " lr=" + str(lr[6])
x_out[6] = [x_start]
t_out[6] = [0]
# defining the function to reevalute function and gradient if needed
def closure():
    optimizer[6].g = func_grad(optimizer[6].x)
    loss = func_main(optimizer[6].x)
    return loss
closure_list[6] = closure


####### ABGDcm2_copy ##################
n = 7
name[7] = "ABGDcm2_copy"
from optimiser_ABGDcm2_copy import abgd_cm2_copy
optimizer[7] = abgd_cm2_copy(x_start, lr=lr[7])
optimizer[7].x = x_start
label[7] = name[7] + " lr=" + str(lr[7])
x_out[7] = [x_start]
t_out[7] = [0]
# defining the function to reevalute function and gradient if needed
def closure():
    optimizer[7].g = func_grad(optimizer[7].x)
    loss = func_main(optimizer[7].x)
    return loss
closure_list[7] = closure


####### ABGDvm ##################
n = 8
name[8] = "ABGDvm_copy"
from optimiser_ABGDvm_copy import abgd_vm_copy
optimizer[8] = abgd_vm_copy(x_start, lr=lr[8])
optimizer[8].x = x_start
label[8] = name[8] + " lr=" + str(lr[8])
x_out[8] = [x_start]
t_out[8] = [0]
# defining the function to reevalute function and gradient if needed
def closure():
    optimizer[8].g = func_grad(optimizer[8].x)
    loss = func_main(optimizer[8].x)
    return loss
closure_list[8] = closure