


import numpy as np

class abgd_cm():
    def __init__(self, params, lr=0.01, min_step_r=2**20, max_step_r=2**20, momentum = 0.7 ):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDcm_init.py").read())





    def _update_params(self):

        g_0_sign = np.sign(self.g)  # sign of the components of gradient
        g_0_m1 = g_0_sign * self.g_m1_sign  # product of sign of gradients of step 1 and 0 to detect change of sign.



        beta_1 = 0.9
        beta_2 = 0.999

        dm = (self.g + self.g_m1) * self.step_g * 0.5
        self.m = beta_1 * self.m + (1 - beta_1) * dm
        self.v = beta_2 * self.v + (1 - beta_2) * np.power(dm, 2)
        self.v = np.ones(self.d)
        dd = self.m / self.v
        dd_sign = np.sign(dd)
        self.x = self.x - dd_sign * self.step_g  # advance x one step_g

        self.g_m1 = self.g



        # # self.v = self.vcoef * self.v + np.power(self.g, 2) # calculate momentum
        # m_0_sign = np.sign(self.m) # get sign of the components of momentum
        # m_g_0 = m_0_sign * g_0_sign # check if momentum and gradient have the same sign.
        # self.m = self.momentum * self.m + self.g # calculate momentum
        # # self.m = np.where( m_g_0 == -1, self.m + 0.05*self.g, self.m) #state 1. going against the gradient because of momentum. half the step.
        #
        #
        # # u = self.m / np.sqrt(self.v)
        # # u_sign = np.sign(u)
        #
        # m_0_sign = np.sign(self.m) # get sign of the components of momentum
        # m_0_m1 = m_0_sign * self.m_m1_sign # # product of sign of momentum of step 1 and 0 to detect change of sign.
        #
        # m_g_0 = m_0_sign * g_0_sign # check if momentum and gradient have the same sign.
        #
        # state = np.zeros(self.d) # state 0. doubling step for explore
        # state = np.where( m_g_0 == -1, 1, state) #state 1. going against the gradient because of momentum. half the step.
        # state = np.where((m_0_m1 == -1) & (g_0_m1 == 1), 2, state) # state2. when momentum changes sign and trajectory returns (no change of sign for gradient). do not change step size.
        # state = np.where(self.state_m1 == 4  , 3, state)  # state 3. convergence and halving step-size in the previous step. do not update momentum. Do not change step size.
        # state = np.where((m_0_m1 == -1) & (g_0_m1 == -1) , 4, state)  #state 4.  both g and m changed sign. halving step for convergence.
        #
        # step_g_mult = np.ones(self.d) * 2.0  # setting all step_g multipliers to 2 (default for state 0, exploring)
        # step_g_mult = np.where(  state == 1 , 0.5, step_g_mult)  # for state 1, change the step_g_mult component to 1
        # step_g_mult = np.where(  state == 2 , 1, step_g_mult)  # for state 2, change the step_g_mult component to 1
        # step_g_mult = np.where(  state == 3 ,1, step_g_mult)  # for state 3, change the step_g_mult component to 1
        # step_g_mult = np.where(  state == 4 , 0.5, step_g_mult)  # for state 1, change the step_g_mult component to 1
        # self.step_g = self.step_g * step_g_mult
        #
        #
        # self.step_g = np.where(self.step_g < (self.lr/self.min_step_r), (self.lr/self.min_step_r), self.step_g)  # minimum step size check
        # self.step_g = np.where(self.step_g > (self.lr*self.max_step_r), (self.lr*self.max_step_r), self.step_g)  # maximum step size check
        #
        # self.x = self.x - m_0_sign * self.step_g  # advance x one step_g
        #
        # # self.m = np.where(state == 2, 0, self.m)  # reset momentum to zero for state 2
        # # self.m = np.where(state == 3, 0, self.m)  # reset momentum to zero for state 3
        # # self.m = np.where(state == 4, 0, self.m)  # reset momentum to zero for state 4
        #
        #
        #
        # #  preparation for  the next step
        # self.g_m1_sign = g_0_sign
        # self.g_m1_m2 = g_0_m1
        # self.m_m1_sign = m_0_sign
        # self.state_m1 = state
        #
        #


    def step(self, closure = None):
        self._update_params()





