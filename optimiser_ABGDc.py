


import numpy as np

class abgd_c():
    def __init__(self, params, lr=0.01, min_step_r=2**20, max_step_r=2**20 ):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDc_init.py").read())



    def _update_params(self):

        g_0_sign = np.sign(self.g)  # sign of the components
        g_0_m1 = g_0_sign * self.g_m1_sign  # product of sign of gradient of step 1 and 0

        step_g_mult = np.ones(self.d) * 2.0  #  setting all step_g multipliers to 2
        step_g_mult = np.where(self.g_m1_m2 == -1.0, 1.0, step_g_mult) #  if g_0_m1 is -1 change step_g_mult component to 1
        step_g_mult = np.where(g_0_m1 == -1.0, 0.5, step_g_mult)  #  if g_1_0 is -1 change step_g_mult component to 0.5
        self.step_g = self.step_g * step_g_mult  # use step_g_mult to update current step_g sizes

        self.step_g = np.where(self.step_g < (self.lr/self.min_step_r), (self.lr/self.min_step_r), self.step_g)  # minimum step size check
        self.step_g = np.where(self.step_g > (self.lr*self.max_step_r), (self.lr*self.max_step_r), self.step_g)  # maximum step size check

        self.x = self.x - g_0_sign * self.step_g  # advance x one step_g

        #  preparation for  the next step
        self.g_m1_sign = g_0_sign  # get the sign of the components
        self.g_m1_m2 = g_0_m1



    def step(self, closure = None):
        self._update_params()





