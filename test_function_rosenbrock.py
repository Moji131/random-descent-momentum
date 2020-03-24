import numpy as np

def rosenbrock_main(x):
    """The Rosenbrock function"""
    a = 0
    b = 100
    return (a-x[0])**2 + b*(x[1]-x[0]**2)**2.0


def rosenbrock_grad(x):
    """Gradient of Rosenbrock function"""
    a = 0
    b = 100
    g = np.empty(x.shape)
    g[0] = - 2.0*(a-x[0]) - 2.0*b*(x[1]-x[0]**2)*2.0*x[0]
    g[1] = 2.0*b*(x[1]-x[0]**2)
    return g
