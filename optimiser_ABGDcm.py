


import numpy as np


class abgd_cm():
    def __init__(self, params, lr=0.01, min_step_r=2 ** 20, max_step_r=2 ** 20, momentum_max = 5):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)


        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDcm_init.py").read())




    def _update_params(self):

        g_0_sign = np.sign(self.g)  # sign of the components
        g_0_m1 = g_0_sign * self.g_m1_sign  # product of sign of gradient of step 1 and 0

        # self.m = self.m + g_0_sign   #- 0.1 * self.step_g
        self.m = np.where( (( self.m == 0) & (self.g_m1_m2 == -1.0)) , 0, self.m + g_0_sign )
        self.m = np.where(self.m > self.m_max, self.m_max, self.m)
        self.m = np.where(self.m < -self.m_max, -self.m_max, self.m)

        m_sign = np.sign(self.m)
        g_0_m = g_0_sign * m_sign  # product of sign of gradient of step 1 and 0

        step_g_mult = np.ones(self.d) * 2.0  # setting all step_g multipliers to 2

        step_g_mult = np.where(self.g_m1_m2 == -1.0 , 1.0, step_g_mult)  # if g_1_0 is -1 change step_g_mult component to 0.5
        step_g_mult = np.where(self.g_0_m_m1 == -1.0, 1.0, step_g_mult)  # if g_1_0 is -1 change step_g_mult component to 0.5

        step_g_mult = np.where(g_0_m == -1.0, 0.5, step_g_mult)  # if g_1_0 is -1 change step_g_mult component to 0.5
        step_g_mult = np.where(g_0_m1 == -1.0 , 0.5, step_g_mult)  # if g_1_0 is -1 change step_g_mult component to 0.5

        self.step_g = self.step_g * step_g_mult  # use step_g_mult to update current step_g sizes

        self.x = np.where( m_sign == 0 ,self.x - g_0_sign * self.step_g, self.x - m_sign * self.step_g)   # advance x one step_g

        self.g_m1_sign = g_0_sign
        self.g_0_m_m1 = g_0_m
        self.g_m1_m2 = g_0_m1



    def step(self, closure=None):
        self._update_params()



