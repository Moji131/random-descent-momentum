import numpy as np

class opt():
    def __init__(self, params, lr=0.01):

        self.lr = lr

        self.d = len(params)
        print(self.d)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initializing parameters specific to the algorithm #######

        self.m = np.zeros(self.d)
        self.v = np.zeros(self.d)
        self.t = 1

        self.step_m = self.lr



    def _update_params(self, closure):
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1.0e-08

        self.m = beta_1 * self.m + (1 - beta_1) * self.g
        self.v = beta_2 * self.v + (1 - beta_2) * np.power(self.g, 2)
        m_hat = self.m / (1 - beta_1**self.t)
        v_hat = self.v / (1 - beta_2**self.t)
        p = m_hat / (np.sqrt(v_hat) + epsilon)

        if self.t == 1:
            self.lr = self.step_m

        self.x = self.x - self.step_m * p

        self.t = self.t + 1





    def step(self, closure):
        self._update_params(closure)


