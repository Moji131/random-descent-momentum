


import numpy as np

class abgd_csd():
    def __init__(self, params, lr=0.01, min_step_r=2**4, max_step_r=2**20 ):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDcsd_init.py").read())



    def _update_params(self, closure):

        g_m1_sign = np.sign(self.g)  # sign of the components

        self.x = self.x - g_m1_sign * self.step_g  # advance x one step_g

        loss = closure() # reevaluate gradient

        g_0_sign = np.sign(self.g)  # sign of the components
        g_0_m1 = g_0_sign * g_m1_sign  # product of sign of gradient of step 1 and 0

        self.step_g_mult_count = np.where(g_0_m1 == -1.0, self.step_g_mult_count+1, 0)  # if g_1_0 is -1 change step_g_mult component to 0.5
        step_g_mult = np.where(g_0_m1 == -1.0, 1, 2.0)  # if g_1_0 is -1 change step_g_mult component to 0.5
        step_g_mult = np.where(self.step_g_mult_count >= self.delay, 0.5, step_g_mult)  # if g_1_0 is -1 change step_g_mult component to 0.5



        self.step_g = self.step_g * step_g_mult  # use step_g_mult to update current step_g sizes

        self.step_g = np.where(self.step_g < (self.lr/self.min_step_r), (self.lr/self.min_step_r), self.step_g)  # minimum step size check
        self.step_g = np.where(self.step_g > (self.lr*self.max_step_r), (self.lr*self.max_step_r), self.step_g)  # maximum step size check

        self.x = self.x - g_0_sign * self.step_g  # advance x one step_g



    def step(self, closure = None):
        self._update_params(closure)





