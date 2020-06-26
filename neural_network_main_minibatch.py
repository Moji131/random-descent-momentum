import torch  ### go to https://pytorch.org/ for installation command 
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

from neural_network_ABGDc import abgd_c
from neural_network_ABGDv import abgd_v
from neural_network_ABGDcs import abgd_cs
from neural_network_ABGDcsd import abgd_csd
from neural_network_ABGDvmd import abgd_vmd



############ settings #######################
opt_n = 7 # number of optimizers
model = [0 for i in range(opt_n)] # list of models to be used with optimizers
opt_list = [0,1,2,3,4,5,6] # list of active optimizers
# opt_list = [0,2,4,6]
save_count = 5
print_count = 10
epochs = 1000




############ Data load #######################
##############################################


###### random data #########

# N = 500
# D_in, H1, D_out = 8, 36, 1
#
# # Create random Tensors to hold inputs and outputs
# torch.manual_seed(2)
# x_train = torch.randn(N, D_in)
# y_train = torch.randn(N, D_out)
#
# test_con = False



########## MNIST load #########


# transform = transforms.ToTensor()
# train_set = datasets.MNIST("data/mnist/trainset", transform=transform, download=True)
# test_set = datasets.MNIST("data/mnist/testset", transform=transform, train=False, download=True)
# train_loader = DataLoader(train_set, batch_size=len(train_set))
# test_loader = DataLoader(test_set, batch_size=len(test_set))
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# train_inputs, train_targets = iter(train_loader).next()
# train_inputs = train_inputs.reshape(60000, 784).to(device)
# train_targets = train_targets.reshape(60000,-1).to(device)
#
# test_inputs, test_targets = iter(test_loader).next()
# test_inputs = test_inputs.reshape(10000, 784).to(device)
# test_targets = test_targets.reshape(10000,-1).to(device)
#
# n_train =60000
# x_train = train_inputs[0:n_train,:]
# y_train = train_targets[0:n_train].float()
#
# n_test =10000
# x_test = test_inputs[0:n_test,:]
# y_test = test_targets[0:n_test].float()
#
# # D_in is input dimension; Hs are hidden dimensions; D_out is output dimension.
# D_in, H1, D_out = 784, 20, 1
#
# test_con = False



########## CIFAR10 load #########

transform = transforms.ToTensor()
train_set = datasets.CIFAR10("data/CIFAR10/trainset", transform=transform, download=True, batch_size=100, shuffle=True)
test_set = datasets.CIFAR10("data/CIFAR10/testset", transform=transform, train=False, download=True)
train_loader = DataLoader(train_set, batch_size=len(train_set))
test_loader = DataLoader(test_set, batch_size=len(test_set))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_inputs, train_targets = iter(train_loader).next()
# train_inputs = train_inputs.reshape(50000,3072).to(device)
# train_targets = train_targets.reshape(50000,-1).to(device)
#
test_inputs, test_targets = iter(test_loader).next()
test_inputs = test_inputs.reshape(10000,3072).to(device)
test_targets = test_targets.reshape(10000,-1).to(device)

# n_train =50000
# x_train = train_inputs[0:n_train,:]
# y_train = train_targets[0:n_train].float()
#
n_test =1000
x_test = test_inputs[0:n_test,:]
y_test = test_targets[0:n_test].float()

# D_in is input dimension; Hs are hidden dimensions; D_out is output dimension.
D_in, H1, D_out = 3072, 20, 1

test_con = True






#################  Defining neural netwrok model ###########
############################################################


### Custom made netwrok

model_o = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, D_out),
)


### Resnet34
# model_o = torchvision.models.resnet34()



#########
loss_fn = torch.nn.MSELoss(reduction='sum') # selecting loss function
for opt_i in range(opt_n):
    # Use the nn package to define a model for each optimizer
    model[opt_i] = copy.deepcopy(model_o)



######## optimizers ##################3
########################################

lr = [ 0 for i in range(opt_n)]
name = [ 0 for i in range(opt_n)]
optimizer = [ 0 for i in range(opt_n)]

# loss_train = [ 0 for i in range(opt_n)]
# y_pred_train = [ 0 for i in range(opt_n)]

# loss_test = [ 0 for i in range(opt_n)]
# y_pred_test = [ 0 for i in range(opt_n)]

closure_list = [ 0 for i in range(opt_n)]


###########  ABGDc #############
lr[0] = 1e-3
name[0] = "ABGDc lr=" + str(lr[0])
drift=True
optimizer[0] = abgd_c(model[0].parameters(), lr=lr[0], min_step_r=2**5, max_step_r=2**5 )

#### defining the function to reevalute function and gradient if needed
closure_list[0] = None


###########  ABGDv #############
lr[1] = 1e-2
name[1] = "ABGDv lr=" + str(lr[1])
drift=True
optimizer[1] = abgd_v(model[1].parameters(), lr=lr[1])

#### defining the function to reevalute function and gradient if needed
closure_list[1] = None


###########  ABGDcs #############
lr[2] = 1e-3
name[2] = "ABGDcs lr=" + str(lr[2])
drift=True
optimizer[2] = abgd_cs(model[2].parameters(), lr=lr[2])

#### defining the function to reevalute function and gradient if needed
def closure():
    optimizer[2].np_to_params()
    optimizer[2].zero_grad()
    y_pred = model[2](x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer[2].params_to_np()
    return loss
closure = torch.enable_grad()(closure)
closure_list[2] = closure




###########  ABGDvmd #############
lr[3] = 1e-2
name[3] = "ABGDvmd lr=" + str(lr[3])
momentum = 0.7
drift = True
optimizer[3] = abgd_vmd(model[3].parameters(), lr=lr[3], momentum=momentum, drift=drift)

#### defining the function to reevalute function and gradient if needed
def closure():
    optimizer[3].np_to_params()
    optimizer[3].zero_grad()
    y_pred = model[3](x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer[3].params_to_np()
    return loss
closure = torch.enable_grad()(closure)
closure_list[3] = closure



# ###### ADAM ##################
lr[4] = 1e-3
name[4] = "ADAM lr=" + str(lr[4])
optimizer[4] = torch.optim.Adam(model[4].parameters(), lr=lr[4])
closure_list[4] = None


# ###### GD ##################
lr[5] = 1e-8
name[5] = "GD lr=" + str(lr[5])
momentum = 0.9
optimizer[5] = torch.optim.SGD(model[5].parameters(), lr=lr[5], momentum=momentum)

closure_list[5] = None


###########  ABGDcsd #############
lr[6] = 1e-3
name[6] = "ABGDcsd lr=" + str(lr[6])
drift=True
optimizer[6] = abgd_csd(model[6].parameters(), lr=lr[6])

#### defining the function to reevalute function and gradient if needed
def closure():
    optimizer[6].np_to_params()
    optimizer[6].zero_grad()
    y_pred = model[6](x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer[6].params_to_np()
    return loss
closure = torch.enable_grad()(closure)
closure_list[6] = closure



######  creating files to output data ########
##############################################
path = "outputs/neural_network/train"
if not os.path.exists(path):
    os.makedirs(path)
files = os.listdir(path)
for f in files:
    file_path = path + "/" + f
    os.remove(file_path)
path = "outputs/neural_network/test"
if not os.path.exists(path):
    os.makedirs(path)
files = os.listdir(path)
for f in files:
    file_path = path + "/" + f
    os.remove(file_path)

file = [ 0 for i in range(opt_n)]





######### main loop ############
################################



for t in range(epochs):
    for i, (x_train, y_train) in enumerate(train_loader):

        for opt_i in opt_list:

            # Forward pass: compute predicted y by passing x to the model.
            y_pred_train = model[opt_i](x_train)

            # Compute and print loss.
            loss_train = loss_fn(y_pred_train, y_train)

            if test_con:
                y_pred_test = model[opt_i](x_test)
                loss_test = loss_fn(y_pred_test, y_test)

            if t % save_count == 0:
                file[opt_i] = open('outputs/neural_network/train/' + name[opt_i], 'a')
                str_to_file = str(t) + "\t" + str(loss_train.data) + "\n"
                file[opt_i].write(str_to_file)
                file[opt_i].close()

                if test_con:
                    file[opt_i] = open('outputs/neural_network/test/' + name[opt_i], 'a')
                    str_to_file = str(t) + "\t" + str(loss_test.data) + "\n"
                    file[opt_i].write(str_to_file)
                    file[opt_i].close()


            # Before the backward pass, use the optimizer object to zero all of the
            optimizer[opt_i].zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss_train.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer[opt_i].step(closure=closure_list[opt_i])

        if t % print_count == 0:
            print("progress:", t , " of ", epochs )

import neural_network_main_plot
