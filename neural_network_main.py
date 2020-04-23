import torch
import torch.nn as nn
import torchvision
from neural_network_abgd import abgd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt
import pickle
import os





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
# x = test_inputs
# y = test_targets.float()

# D_in is input dimension; Hs are hidden dimensions; D_out is output dimension.
D_in, H1, D_out = 784, 20, 1




###### random data #########
N = 53
D_in, H1, D_out = 5, 5, 1

# Create random Tensors to hold inputs and outputs
torch.manual_seed(0)
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)



#################  Defining neural netwrok model ###########


# Use the nn package to define our model and loss function.
model_1 = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

model_2 = copy.deepcopy(model_1)
model_3 = copy.deepcopy(model_1)


######## ptimizers ###########
########################################



###########  accelerated bisection gradient descent #############
learning_rate = 1e-4
t_list_1 = []
loss_list_1 = []
optimizer_1 = abgd(model_1.parameters(), lr=learning_rate)

###### ADAM ##################
learning_rate = 1e-2
t_list_2 = []
loss_list_2 = []
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate)

####### SGDM ############
learning_rate = 1e-4
momentum = 0
t_list_3 = []
loss_list_3 = []
optimizer_3 = torch.optim.SGD(model_3.parameters(), lr=learning_rate, momentum=momentum)




######  files to output data ########
path = "outputs/neural_network"
files = os.listdir(path)
for f in files:
    file_path = path + "/" + f
    os.remove(file_path)

file_1 = open('outputs/neural_network/ABGD', 'a')

file_2 = open('outputs/neural_network/ADAM', 'a')

file_3 = open('outputs/neural_network/SGDM', 'a')




######### main loop ############

save_count = 1
print_count = 100
epochs = 1000

for t in range(epochs):

    #######  model_1 ABGD ###########

    # Forward pass: compute predicted y by passing x to the model.
    y_pred_1 = model_1(x)

    # Compute and print loss.
    loss_1 = loss_fn(y_pred_1, y)
    if t % print_count == 0:
        print("ABGD", t,loss_1.item())

    if t % save_count == 0:
        t_list_1.append(t)
        loss_list_1.append(loss_1.item())

        str_to_file = str(t) + "\t" + str(loss_1.item()) + "\n"
        file_1.write(str_to_file)


    # Before the backward pass, use the optimizer object to zero all of the
    optimizer_1.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss_1.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer_1.step()

    if t % 100 == 0:
        for p in optimizer_1._params:
            pass
            # print("ABGD p", t, p)
            # print("ABGD g", t, p.grad.data)
            # print("ABGD step", t,optimizer_1._params_step)
            # print("ABGD sign", t, optimizer_1._params_g_1_0neural_network_abgd.py)





    #######  model_2 ADAM ###########

    # Forward pass: compute predicted y by passing x to the model.
    y_pred_2 = model_2(x)

    # Compute and print loss.
    loss_2 = loss_fn(y_pred_2, y)
    if t % print_count == 0:
        print("ADAM", t,loss_2.item())
    if t % save_count == 0:
        t_list_2.append(t)
        loss_list_2.append(loss_2.item())

        str_to_file = str(t) + "\t" + str(loss_2.item()) + "\n"
        file_2.write(str_to_file)

    # Before the backward pass, use the optimizer object to zero all of the
    optimizer_2.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss_2.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer_2.step()


    #
    #
    # #######  model_3 SGDM ########
    # # Forward pass: compute predicted y by passing x to the model.
    # y_pred_3 = model_3(x)
    #
    # # Compute and print loss.
    # loss_3 = loss_fn(y_pred_3, y)
    # if t % print_count == 0:
    #     print("SGDM", t, loss_3.item())
    #
    # if t % save_count == 0:
    #     t_list_3.append(t)
    #     loss_list_3.append(loss_3.item())
    #
    #     str_to_file = str(t) + "\t" + str(loss_3.item()) + "\n"
    #     file_3.write(str_to_file)
    #
    # # Before the backward pass, use the optimizer object to zero all of the
    # optimizer_3.zero_grad()
    #
    # # Backward pass: compute gradient of the loss with respect to model parameters
    # loss_3.backward()
    #
    # # Calling the step function on an Optimizer makes an update to its parameters
    # optimizer_3.step()





file_1.close()
file_2.close()
file_3.close()

import neural_network_main_plot