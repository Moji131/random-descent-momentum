import numpy as np


class abgd_v():
    def __init__(self, params, lr=0.01):

        self.lr = lr  # learning rate

        self.d = len(params)  # input dimension

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)


        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDv_init.py").read())



    def _update_params(self, closure):

        ### normalizing gradient
        g_0_norm = np.linalg.norm(self.g)  # norm of the gradient
        g_0_normed = self.g / g_0_norm  # normalized gradient

        ### caculating step_g_g for the next iteration
        g_0_m1_dot = np.dot(g_0_normed, self.g_m1_normed)
        if g_0_m1_dot < 0:
            self.step_g = self.step_g * 0.5
        elif self.g_0_m1_dot_m1 < 0:
            self.step_g = self.step_g * 1.0
        else:
            self.step_g = self.step_g * 2.0

        ### updating parameters
        self.x = self.x - g_0_normed * self.step_g

        ### save values for the next step
        self.g_m1_normed = g_0_normed
        self.g_0_m1_dot_m1 = g_0_m1_dot




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
        self.step_g = float('{:0.1e}'.format(  ( self.step_g/10 + (loss0-loss1)*(self.step_g-self.step_g/10)/(loss2-loss1) ) / 20 ))
        self.lr = self.step_g





