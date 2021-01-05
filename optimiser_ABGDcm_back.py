


import numpy as np


class abgd_cm():
    def __init__(self, params, lr=0.01, min_step_r=2 ** 20, max_step_r=2 ** 20, momentum_max = 5):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)


        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ALR_ADAM_init.py").read())




    def _update_params(self):

        for i in range(self.d):
            self.g_0_sign[i] = np.sign(self.g[i])  # sign of the components
            print("self.x[i]", i, self.x[i])

            print("self.converge_count[i]", i, self.converge_count[i])


            if self.converge_count[i] == 0:
                self.m[i] = self.m[i] + self.g_0_sign[i]
                if self.m[i] > self.m_max:
                    self.m[i] = self.m_max
                if self.m[i] == 0:
                    self.converge_count[i] = self.m_max + 1
                    self.m[i] = 0
            else:
                self.m[i] = 0




            if self.converge_count[i] == 0:
                m_sign = np.sign(self.m[i])
                g_0_m = self.g_0_sign[i] * m_sign  # product of sign of gradient of step 1 and 0
                if g_0_m == 1.0:
                    self.step_g[i] = self.step_g[i] * 2.0
                    print("2.0 step", i)
                else:
                    print("2.0 step not", i)
                self.x[i] = self.x[i] - m_sign * self.step_g[i]  # advance x one step_g
            else:

                g_0_m = self.g_0_sign[i] * self.g_m1_sign[i]  # product of sign of gradient of step 1 and 0
                print(" g_0_m = self.g_0_sign[i] * self.g_m1_sign[i]",  i, g_0_m, self.g_0_sign[i], self.g_m1_sign[i])
                if g_0_m  == -1.0:
                    self.step_g[i] = self.step_g[i] * 0.5
                    self.converge_count[i] = 2
                    print("0.5 step", i)
                else:
                    self.converge_count[i] = self.converge_count[i] - 1
                    print("0.5 step not", i)

                self.x[i] = self.x[i] - self.g_0_sign[i] * self.step_g[i]  # advance x one step_g




            self.g_m1_sign[i] = self.g_0_sign[i]








    def step(self, closure=None):
        self._update_params()



