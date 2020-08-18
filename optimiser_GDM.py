


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

    def step(self, closure):
        if self.t == 1:
            self._find_lr(closure)
            self.t = 1
        self._update_params(closure)

    def _find_lr(self, closure):
        self.step_g = self.lr / 1000
        xx = self.x[:]
        loss0 = closure()
        loss2 = loss0
        loss1 = loss0

        while not loss2 > loss1:
            loss1 = loss2
            self.step_g = 10 * self.step_g
            self._update_params(closure)
            loss2 = closure()
            self.x = xx

        loss0 = closure()
        self.step_g = float('{:0.1e}'.format(
            (self.step_g / 10 + (loss0 - loss1) * (self.step_g - self.step_g / 10) / (loss2 - loss1)) / 20))
        self.lr = self.step_g







