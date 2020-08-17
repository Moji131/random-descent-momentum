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




############ settings #######################
# opt_n = 8 # number of optimizers
# model = [0 for i in range(opt_n)] # list of models to be used with optimizers
# opt_list = [ 1, 4, 7] # list of active optimizers
# # opt_list = [0,2,4,6]

### parameters
opt_n = 9
#    [0    , 1    , 2    , 3   , 4      , 5      , 6     , 7           , 8          ]
#    [ABGDc, ABGDc, ADAM , GDM , ABGDvmd, ABGDcm2, ABGDvm, ABGDcm2_copy, ABGDvm_copy]
lr = [0.1  , 0.1  , 1e-2 , 1e-5, 0.1    , 1e-1   , 1e-2  , 1e-2         , 1e-4        ]

# lr = [0.1  , 0.1  , 1e-2 , 1e-5, 0.1    , 1e-1   , 1e-1  , 1e-2         , 1e-4        ]
# opt_list = [2,6,7] # list of optimizers to be applied
opt_list = [2,6,7] # list of optimizers to be applied

save_count = 1
print_count = 1
epochs = 50000




############ Data load #######################
##############################################


###### random data #########

N = 8000
D_in, H1, D_out = 3,200, 1

# Create random Tensors to hold inputs and outputs
torch.manual_seed(43)
x_train = torch.randn(N, D_in)
y_train = torch.randn(N, D_out)

test_con = False



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
#
# transform = transforms.ToTensor()
# train_set = datasets.CIFAR10("data/CIFAR10/trainset", transform=transform, download=True)
# test_set = datasets.CIFAR10("data/CIFAR10/testset", transform=transform, train=False, download=True)
# train_loader = DataLoader(train_set, batch_size=len(train_set))
# test_loader = DataLoader(test_set, batch_size=len(test_set))
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# train_inputs, train_targets = iter(train_loader).next()
# train_inputs = train_inputs.reshape(50000,3072).to(device)
# train_targets = train_targets.reshape(50000,-1).to(device)
#
# test_inputs, test_targets = iter(test_loader).next()
# test_inputs = test_inputs.reshape(10000,3072).to(device)
# test_targets = test_targets.reshape(10000,-1).to(device)
#
# n_train =50000
# x_train = train_inputs[0:n_train,:]
# y_train = train_targets[0:n_train].float()
#
# n_test =10000
# x_test = test_inputs[0:n_test,:]
# y_test = test_targets[0:n_test].float()
#
# # D_in is input dimension; Hs are hidden dimensions; D_out is output dimension.
# D_in, H1, D_out = 3072, 20, 1
#
# test_con = True




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




############ import optimizers ###############
##############################################

exec(open("./neural_netwrok_optimizers.py").read())



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




for t in range(1,epochs+1):

    for opt_i in opt_list:

        # Forward pass: compute predicted y by passing x to the model.
        y_pred_train = model[opt_i](x_train)

        # Compute and print loss.
        loss_train = loss_fn(y_pred_train, y_train)

        if test_con:
            y_pred_test = model[opt_i](x_test)
            loss_test = loss_fn(y_pred_test, y_test)




        # Before the backward pass, use the optimizer object to zero all of the
        optimizer[opt_i].zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_train.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer[opt_i].step(closure=closure_list[opt_i])

        if t % save_count == 0:
            str1 = 'outputs/neural_network/train/' + name[opt_i] + "-lr=" + str(optimizer[opt_i].lr)
            file[opt_i] = open(str1, 'a')
            str_to_file = str(t) + "\t" + str(loss_train.data.item()) + "\n"
            file[opt_i].write(str_to_file)
            file[opt_i].close()

            if test_con:
                file[opt_i] = open('outputs/neural_network/test/' + name[opt_i], 'a')
                str_to_file = str(t) + "\t" + str(loss_test.data.item()) + "\n"
                file[opt_i].write(str_to_file)
                file[opt_i].close()


    if t % print_count == 0:
        print("progress:", t , " of ", epochs )

import neural_network_main_plot
