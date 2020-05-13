# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 17:53:19 2018

@author: ly
"""
import numpy as np
from numpy.random import randn
# from derivativetest import derivativetest


def regConvex(x, lamda, arg=None):
    x = x.reshape(len(x), 1)
    f = lamda*np.dot(x.T, x)
    if arg == 'f':
        return f

    g = 2*lamda*x
    if arg == 'g':
        return g
    if arg == 'fg':
        return f, g

    def Hv(v): return 2*lamda*v

    if arg is None:
        return f, g, Hv


def main():
    #    rand.seed(1)
    d = 500
    lamda = 2
    w = randn(d, 1)
    def fun1(x): return regConvex(x, lamda)
    # derivativetest(fun1, w)


if __name__ == '__main__':
    main()
