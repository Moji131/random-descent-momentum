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
from regConvex import regConvex
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

#    if arg == 'explicit':
#        f = np.sum(log_sum_exp_trick) - np.sum(np.sum(XW*Y,axis=1)) + reg_f
#        g = np.dot(X.T, S-Y) #[d x C]
#        g = g.T.flatten().reshape(d*C,1) + reg_g #[d*C, ]
#        Hv = lambda v: hessvec(X, S, v, reg)
#        #S is divided into C parts {1:b}U{c}, [n, ] * C
#        S_cell = np.split(S.T,C)
#        SX_cell = np.array([]).reshape(n,0) #empty [n x 0] array
#        SX_self_cell = np.array([]).reshape(0,0)
#        for column in S_cell:
#            c = spdiags(column,0,n,n) #value of the b/c class
#            SX_1_cell = np.dot(c.A,X) #WX = W x X,half of W, [n x d]
#            #fill results from columns, [n x d*C]
#            SX_cell = np.c_[SX_cell, SX_1_cell]
#            SX_cross = np.dot(SX_cell.T,SX_cell) #take square, [d*C x d*C]
#            #X.T x WX        half of W, [d x d]
#            SX_1self_cell = np.dot(X.T,SX_1_cell)
#            #put [d x d] in diag, W_cc, [d*C x d*C]
#            SX_self_cell = block_diag(SX_self_cell,SX_1self_cell)
#            H = SX_self_cell - SX_cross #compute W_cc, [d*C x d*C]
#        H = H + 2*reg*identity(d*C)
#        return f, g, Hv, H


def hessvec(X, S, n, v):
    v = v.reshape(len(v), 1)
    V = v.reshape(C, d).T  # [d x C]
    A = np.dot(X, V)  # [n x C]
    AS = np.sum(A*S, axis=1).reshape(n, 1)
    rep = np.matlib.repmat(AS, 1, C)  # A.dot(B)*e*e.T
    XVd1W = A*S - S*rep  # [n x C]
    Hv = np.dot(X.T, XVd1W)/n  # [d x C]
#    if norm(Hv)< 1E-10:
#        print('a')
    Hv = Hv.T.flatten().reshape(d*C, 1)  # [d*C, ] #[d*C, ]
    return Hv

# def SSHv(X, S, v):
#    v = v.reshape(len(v),1)
#    V = v.reshape(C, d).T #[d x C]
#    A = np.dot(X,V) #[n x C]
#    AS = np.sum(A*S, axis=1).reshape(n, 1)
#    rep = np.matlib.repmat(AS, 1, C)#A.dot(B)*e*e.T
#    XVd1W = A*S - S*rep #[n x C]
#    Hv = np.dot(X.T, XVd1W)/n #[d x C]
# if norm(Hv)< 1E-10:
# print('a')
#    Hv = Hv.T.flatten().reshape(d*C,1)#[d*C, ] #[d*C, ]
#    return Hv

# @profile


def main():
    rand.seed(1)
    n = 100
    d = 50
    total_C = 2
    X = rand.randn(n, d)
    I = np.eye(total_C, total_C - 1)
    ind = rand.randint(total_C, size=n)
    Y = I[ind, :]
    lamda = 0
#    reg = None
    reg = lambda x: regConvex(x, lamda)
#     def reg(x): return regNonconvex(x, lamda)
    w = rand.randn(d*(total_C-1), 1)
    def fun(x): return softmax(X, Y, x, reg=reg)
    # derivativetest(fun, w)

########################################################################
#    data = 'mnist'
##    data = 'cifar10'
#    standarlize = False
#    normalize = False
#
#    data_dir = '../Data'
#    train_X, train_Y, test_X, test_Y, idx = loaddata(data_dir, data)
#
#    print('Dataset_shape:', train_X.shape, end=' ')
#    train_X = train_X[0:100,:]
#    train_Y = train_Y[0:100]
#    print('Dataset_shape_in_use:', train_X.shape)
#
#    if standarlize:
#        train_X = preprocessing.scale(train_X)
#        test_X = preprocessing.scale(test_X)
#
#    if normalize:
#        train_X = preprocessing.normalize(train_X, norm='l2')
#        test_X = preprocessing.normalize(test_X, norm='l2')
#
#    n, d= train_X.shape
#    Classes = sorted(set(train_Y))
#    Total_C  = len(Classes)
#    l = d*(Total_C-1)
#    I = np.ones(n)
#
#    X_label = np.array([i for i in range(n)])
#    Y = sparse.coo_matrix((I,(X_label, train_Y)), shape=(n, Total_C)).tocsr().toarray()
#    Y = Y[:,:-1]
#    X = train_X.astype(np.float64)
#
#    np.random.seed(0)
##    x = np.zeros((l,1))
#    x = np.random.randn(l,1)
#    f,g,Hv,H = softmax(X, Y, x, arg='explicit', reg=0)
#    eig = np.linalg.eigvals(H)
#    tt = np.all(eig > 0)
#    print(tt)
#    sum(eig[eig<0])

#
# if __name__ == '__main__':
#     main()



############## Soft Max ###############
rand.seed(2)
n = 300
d = 50
def d_func():
    return d
total_C = 2

# X = rand.randn(n, d)   #Let X be a random matrix

A = rand.randn(n, d)
D = np.logspace(1, 8, d)
X = A*D  # set X as a ill conditioned Matrix


I = np.eye(total_C, total_C - 1)
ind = rand.randint(total_C, size=n)

Y = I[ind, :]



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
