#
#
#
# import numpy as np
#
# class adam():
#     def __init__(self, params, lr=0.01):
#
#         self.lr = lr
#
#         self.d = len(params)
#
#         self.x = np.zeros(self.d)
#         self.g = np.zeros(self.d)
#
#         ##### initialising parameters specific to the algorithm #######
#         exec(open("./optimiser_ADAM_init.py").read())
#
#
#
#     def _update_params(self):
#         beta_1 = 0.9
#         beta_2 = 0.999
#         epsilon = 1.0e-08
#
#         self.m = beta_1 * self.m + (1 - beta_1) * self.g
#         self.v = beta_2 * self.v + (1 - beta_2) * np.power(self.g, 2)
#         m_hat = self.m / (1 - beta_1**self.t)
#         v_hat = self.v / (1 - beta_2**self.t)
#
#         g_0_sign = np.sign(self.g)  # sign of the components of gradient
#         a_sign = np.sign((m_hat) / (np.sqrt(v_hat) + epsilon))
#         m_g_0 = a_sign * g_0_sign # check if momentum and gradient have the same sign.
#         self.lr = np.where( m_g_0 == -1, self.lr*0.5, self.lr*2) #state 1. going against the gradient because of momentum. half the step.
#
#
#         self.x = self.x - self.lr * a_sign
#
#         # self.x = self.x - (self.lr * m_hat) / (np.sqrt(v_hat) + epsilon)
#
#
#
#     def step(self, closure = None):
#         self._update_params()
#
#
#
#







import numpy as np

class adam():
    def __init__(self, params, lr=0.01):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ADAM_init.py").read())



    def _update_params(self, closure):
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1.0e-08

        self.m = beta_1 * self.m + (1 - beta_1) * self.g
        self.v = beta_2 * self.v + (1 - beta_2) * np.power(self.g, 2)
        m_hat = self.m / (1 - beta_1**self.t)
        v_hat = self.v / (1 - beta_2**self.t)


        self.x = self.x - (self.step_g * m_hat) / (np.sqrt(v_hat) + epsilon)


        self.t = self.t + 1




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
        self.step_g = float('{:0.1e}'.format(  ( self.step_g/10 + (loss0-loss1)*(self.step_g-self.step_g/10)/(loss2-loss1) ) / 5 ))
        self.lr = self.step_g










