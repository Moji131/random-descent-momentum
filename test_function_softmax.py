# -*- coding: utf-8 -*-
"""
##output as a column vector
Created on Wed Nov  8 10:48:59 2017

@author: Liu Yang
"""
import numpy as np
from scipy.sparse import spdiags, identity
import numpy.random as rand
# from derivativetest import derivativetest
from scipy.linalg import block_diag
from scipy import sparse
from test_function_softmax_regConvex import regConvex
# from regNonconvex import regNonconvex
from sklearn import preprocessing
# from loaddata import loaddata
from numpy.linalg import norm


def softmax(X, Y, w, arg=None, reg=None, HProp=None, gProp=None):
    if reg == None:
        reg_f = 0
        reg_g = 0
        def reg_Hv(v): return 0
    else:
        reg_f, reg_g, reg_Hv = reg(w)
    # X [n x d]
    # Y [n x C]
    global d, C
    n, d = X.shape
    C = int(len(w)/d)


    w = w.reshape(d*C, 1)  # [d*C x 1]

    W = w.reshape(C, d).T  # [d x C]



    XW = np.dot(X, W)  # [n x C]
    large_vals = np.amax(XW, axis=1).reshape(n, 1)  # [n,1 ]
    large_vals = np.maximum(0, large_vals)  # M(x), [n, 1]
    # XW - M(x)/<Xi,Wc> - M(x), [n x C]
    XW_trick = XW - np.tile(large_vals, (1, C))
    # sum over b to calc alphax, [n x total_C]
    XW_1_trick = np.append(-large_vals, XW_trick, axis=1)
    #alphax, [n, ]
    sum_exp_trick = np.sum(np.exp(XW_1_trick), axis=1).reshape(n, 1)
    log_sum_exp_trick = large_vals + np.log(sum_exp_trick)  # [n, 1]

    f = np.sum(log_sum_exp_trick)/n - np.sum(np.sum(XW*Y, axis=1))/n + reg_f
    if arg == 'f':
        return f

    inv_sum_exp = 1./sum_exp_trick
    inv_sum_exp = np.tile(inv_sum_exp, (1, np.size(W, axis=1)))
    S = inv_sum_exp*np.exp(XW_trick)  # h(x,w), [n x C]
    g = np.dot(X.T, S-Y)/n  # [d x C]
    g = g.T.flatten().reshape(d*C, 1) + reg_g  # [d*C, ]
    if gProp != None:
        if gProp == 'HProp':
            gProp = HProp
        n_g = np.int(np.floor(n*gProp))
        np.random.seed(n_g)  # fixed random seed for same subsamping para
#        np.random.seed(4) #fixed random seed for same subsamping para
        idx_g = np.random.choice(n, n_g, replace=False)
        inv_sum_exp_g = 1./(sum_exp_trick[idx_g, :])
        inv_sum_exp_g = np.tile(inv_sum_exp_g, (1, np.size(W, axis=1)))
        S_g = inv_sum_exp_g*np.exp(XW_trick[idx_g, :])  # h(x,w), [n x C]
        subg = np.dot(X[idx_g, :].T, S_g-Y[idx_g, :])/n_g  # [d x C]
        subg = subg.T.flatten().reshape(d*C, 1) + reg_g  # [d*C, ]

    if arg == 'g':
        return g

    if arg == 'subg_g':
        return subg

    if arg == 'fg':
        return f, g

    if arg == None:
        # write in one function to ensure no array inputs
        def Hv(v): return hessvec(X, S, n, v) + reg_Hv(v)
        return f, g, Hv
    else:
        if gProp == 'HProp':
            n_H = n_g
            idx_H = idx_g
        else:
            n_H = np.int(np.floor(n*HProp))
#            np.random.seed(n_g) #fixed random seed for same subsamping para
#            np.random.seed(4) #fixed random seed for same subsamping para
            idx_H = np.random.choice(n, n_H, replace=False)
        inv_sum_exp_H = 1./(sum_exp_trick[idx_H, :])
        inv_sum_exp_H = np.tile(inv_sum_exp_H, (1, np.size(W, axis=1)))
        S_H = inv_sum_exp_H*np.exp(XW_trick[idx_H, :])  # h(x,w), [S x C]
        if arg == 'subH':
            def Hv(v): return hessvec(X[idx_H, :], S_H, n_H, v) + reg_Hv(v)
            # write in one function to ensure no array inputs
            def fullHv(v): return hessvec(X, S, n, v) + reg_Hv(v)
            return f, g, Hv, fullHv
        if arg == 'subg':
            def Hv(v): return hessvec(X[idx_H, :], S_H, n_H, v) + reg_Hv(v)
            def subgHv(v): return hessvec(X[idx_g, :], S_g, n_g, v) + reg_Hv(v)
            return f, g, Hv, subg, subgHv




############## selection of data source ###############
#######################################################

# ############## This section for random data generation ###############
rand.seed(3)
n = 30
d = 2
total_C = 2
# X = rand.randn(n, d)   #Let X be a random matrix
A = rand.randn(n, d)
cond_number = 6
D = np.logspace(1, cond_number, d)
X = A*D  # set X as a ill conditioned Matrix
I = np.eye(total_C, total_C - 1)
ind = rand.randint(total_C, size=n)
Y = I[ind, :]
description = "Softmax - Random Data, d=" + str(d) + ", n=" + str(n) + ", \n condition number = " + str(cond_number)




########### This section for loading MNIST ############


# import tensorflow as tf
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
transform = transforms.ToTensor()

# d = 784
# total_C = 10
# n = 784
#
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
#
# X = np.array(x_train)
# Y = np.array(y_train)
#
# n_test =10000
# x_test = test_inputs[0:n_test,:]
# y_test = test_targets[0:n_test].float()
#
# description = "Softmax - MNIST, d=" + str(d) + ", n=" + str(n)



############ This section for loading CIFAR10 ############

# import tensorflow as tf
# cifar10 = tf.keras.datasets.cifar10
#
# d = 3072
# total_C = 10
# n = 3072 * 2
# (train_X, train_Y), (test_X, test_Y) = cifar10.load_data()
#
# train_X = train_X[0:n,:]
# X = train_X.reshape(-1, d).astype(np.float64)
#
# train_Y = train_Y[0:n]
# Y = train_Y.reshape(-1, 1).astype(np.float64)
# description = "Softmax - CIFAR10, d=" + str(d) + ", n=" + str(n)
#




########## Defining functions ##############
############################################

def d_func():
    return d

def description_func():
    return description


lamda = 1
# reg = None
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
