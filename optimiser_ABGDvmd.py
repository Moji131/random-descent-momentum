import numpy as np

class abgd_vmd():
    def __init__(self, params, lr=0.01, min_step_r=2**10, max_step_r=2**10, momentum = 0.9, drift= True):

        self.lr = lr # learning rate

        self.d = len(params) # input dimension

        self.x = np.zeros(self.d) # value of parameter
        self.g = np.zeros(self.d) # gradient


        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDvmd_init.py").read())



    def _update_params(self, closure):

        g_0_norm = np.linalg.norm(self.g)  # norm of the gradient
        if g_0_norm == 0:
            print("Zero gradient error!")
            exit()
        g_0_normed = self.g / g_0_norm  # normalized gradient
        g_0_mom_dot = np.dot(g_0_normed, self.mom_normed)


        if g_0_mom_dot < self.pair_con:
            # self.convrge_state = True

            self.step_g = self.step_g * 0.5
            self.mom = g_0_normed
            # self.convrge_state_m1 = self.convrge_state

        elif self.g_0_mom_dot_m1 < self.pair_con:
            # self.convrge_state = True

            self.mom = g_0_normed
            # self.convrge_state_m1 = self.convrge_state

        else:
            # self.convrge_state = False

            if not self.mom_con:
                self.mom = np.zeros(self.d)
            self.step_g = self.step_g * 2.0
            self.mom = self.beta * self.mom + (1-self.beta) * g_0_normed

            # self.convrge_state_m1 = self.convrge_state


        ###### Drift section ##########

        drift_update = False # used to detect end of the drift period.
        g_0_m1_dot = np.dot(g_0_normed, self.g_m1_normed) # dot product of gradient of 0 and -1 step to be used in drift


        # runs drift step if drift is True, we have a pair of gradients and dot products of th last two gardeints is smaller than drift condition, drift_con.
        if self.drift and g_0_mom_dot < self.pair_con and g_0_m1_dot < self.drift_con:
            g_drift_0 = ( g_0_normed + self.g_m1_normed ) # when there is a


            g_drift_0_norm = np.linalg.norm(g_drift_0)  # norm of the gradient
            if g_drift_0_norm > 1e-30:
                g_drift_0_normed = g_drift_0 / g_drift_0_norm  # normalized gradient
                g_drift_0_m1_dot = np.dot(g_drift_0_normed, self.g_drift_m1_normed)
                if g_drift_0_m1_dot < 0:
                    self.step_drift = self.step_drift * 0.5
                elif self.g_drift_0_m1_dot_m1 < 0:
                    self.step_drift = self.step_drift * 1.0
                else:
                    self.step_drift = self.step_drift * 2.0

                x_save = self.x
                self.x = self.x - g_drift_0_normed * self.step_drift

                loss = closure()

                g_norm = np.linalg.norm(self.g)
                g_normed = self.g / g_norm
                g_drift_dot = np.dot(g_drift_0_normed, g_normed)
                if g_drift_dot < self.drift_reject_con or g_drift_dot > self.drift_reject_con:
                    self.g_drift_m1_normed = g_drift_0_normed
                    self.g_drift_0_m1_dot_m1 = g_drift_0_m1_dot
                    drift_update = True
                else:
                    self.x = x_save


        elif self.drift and self.g_0_mom_dot_m1 < self.pair_con and g_0_m1_dot > -self.drift_con:
            drift_update = True

        if not drift_update:
            self.step_drift = self.step_g * 0.5
            self.g_drift_m1_normed = np.zeros(self.d)
            self.g_drift_0_m1_dot_m1 = 1



        mom_norm = np.linalg.norm(self.mom)  # norm of the gradient
        self.mom_normed = self.mom / mom_norm  # normalized gradient

        self.x = self.x - self.mom_normed * self.step_g


        self.g_0_mom_dot_m1 = g_0_mom_dot
        self.g_m1_normed = g_0_normed



    def step(self, closure = None):
        self._update_params(closure)



