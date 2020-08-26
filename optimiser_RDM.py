


import numpy as np

class rdm():
    def __init__(self, params, lr=0.01, momentum = 0.95):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_RDM_init.py").read())



    def _update_params(self, closure):

        if self.t == 1:
            l = closure()
            p = np.random.random(self.d)
            p_normed = p / np.linalg.norm(p)
        else:
            l = closure()
            g = (l-self.l_m1) * (self.x - self.x_m1) / self.step_g /self.step_g
            self.m = self.momentum * self.m +  g
            m_normed = self.m / np.linalg.norm(self.m)

            s = np.random.random(self.d)
            s_normed = s / np.linalg.norm(s)
            s_m_dot = np.dot(s_normed, m_normed)
            s_perp = s_normed - m_normed * s_m_dot

            p =  m_normed + s_perp
            p_normed = p / np.linalg.norm(p)

        self.x_m1 = self.x
        self.x = self.x - p_normed * self.step_g

        self.l_m1 = l




    def step(self, closure):
        # if self.t == 1:
        #     self._find_lr(closure)
        #     self.t = 1
        self.step_g = self.lr
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







