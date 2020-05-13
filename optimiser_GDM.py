


import numpy as np

class gdm():
    def __init__(self, params, lr=0.01, momentum = 0.9):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_GDM_init.py").read())



    def _update_params(self):
        self.v = self.m * self.v +  self.lr * self.g
        self.x = self.x - self.v



    def step(self, closure = None):
        self._update_params()





