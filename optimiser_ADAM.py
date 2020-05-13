


import numpy as np

class adam():
    def __init__(self, params, lr=0.01):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ADAM_init.py").read())



    def _update_params(self):
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1.0e-08

        self.m = beta_1 * self.m + (1 - beta_1) * self.g
        self.v = beta_2 * self.v + (1 - beta_2) * np.power(self.g, 2)
        m_hat = self.m / (1 - beta_1**self.t)
        v_hat = self.v / (1 - beta_2**self.t)
        self.x = self.x - (self.lr * m_hat) / (np.sqrt(v_hat) + epsilon)



    def step(self, closure = None):
        self._update_params()





