import numpy as np

class abgd_vmd():
    def __init__(self, params, lr=0.01, momentum = 0.7, drift= True):

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
        g_0_m1_dot = np.dot(g_0_normed, self.g_m1_normed) # dot product of gradient of 0 and -1 step to be used in drift
        g_0_mom_dot = np.dot(g_0_normed, self.mom_normed) # dot product of current gradient and momentum



        if g_0_mom_dot < self.pair_con:
            self.step_g = self.step_g * 0.5
            self.mom = g_0_normed

            ### drift move
            if self.drift and g_0_m1_dot < self.drift_con:
                g_drift_0 = (g_0_normed + self.g_m1_normed)  # when there is a
                g_drift_0_norm = np.linalg.norm(g_drift_0)  # norm of the gradient
                if g_drift_0_norm > 1e-40: # avoiding division by zero
                    g_drift_0_normed = g_drift_0 / g_drift_0_norm  # normalized gradient
                    g_drift_0_m1_dot = np.dot(g_drift_0_normed, self.g_drift_m1_normed) # dot product of the last two drift vectors
                    if g_drift_0_m1_dot < 0: # if the last two drift vectors pointing at each other
                        self.step_drift = self.step_drift * 0.5
                    elif self.g_drift_0_m1_dot_m1 < 0: # if the last two drift vectors for the previous step pointing at each other
                        self.step_drift = self.step_drift * 1.0
                    else:
                        self.step_drift = self.step_drift * 2.0

                    x_save = self.x # save x in case of rejected drift move
                    self.x = self.x - g_drift_0_normed * self.step_drift # advance x

                    loss = closure() # reevaluate gradient

                    g_norm = np.linalg.norm(self.g)
                    g_normed = self.g / g_norm # normalised gradient
                    g_drift_dot = np.dot(g_drift_0_normed, g_normed) # dot product of new gradient and the drift vector
                    if g_drift_dot > self.drift_reject_con: # if move is not rejected
                        # preparation for the next step
                        self.g_drift_m1_normed = g_drift_0_normed
                        self.g_drift_0_m1_dot_m1 = g_drift_0_m1_dot
                    else: # if move is rejected
                        self.x = x_save # return to previous x
                        # preparation for the next step
                        self.g_drift_m1_normed = np.zeros(self.d)
                        self.g_drift_0_m1_dot_m1 = 1


        elif self.g_0_mom_dot_m1 < self.pair_con:
            self.mom = g_0_normed
        else:
            if not self.mom_con:
                self.mom = np.zeros(self.d)
            self.step_g = self.step_g * 2.0
            self.mom = self.beta * self.mom + (1-self.beta) * g_0_normed

            # resetting values of the drift move
            if self.drift:
                self.step_drift = self.step_g * 0.5
                self.g_drift_m1_normed = np.zeros(self.d)
                self.g_drift_0_m1_dot_m1 = 1




        mom_norm = np.linalg.norm(self.mom)  # norm of the gradient
        self.mom_normed = self.mom / mom_norm  # normalized gradient

        self.x = self.x - self.mom_normed * self.step_g # advanced x one step

        #prepreation for the next step
        self.g_0_mom_dot_m1 = g_0_mom_dot
        self.g_m1_normed = g_0_normed



    def step(self, closure = None):
        self._update_params(closure)






# import numpy as np
#
# class abgd_vmd():
#     def __init__(self, params, lr=0.01, min_step_r=2**10, max_step_r=2**10, momentum = 0.7, drift= True):
#
#         self.lr = lr # learning rate
#
#         self.d = len(params) # input dimension
#
#         self.x = np.zeros(self.d) # value of parameter
#         self.g = np.zeros(self.d) # gradient
#
#
#         ##### initialising parameters specific to the algorithm #######
#         exec(open("./optimiser_ABGDvmd_init.py").read())
#
#
#
#
#
#     def _update_params(self, closure):
#
#         g_0_norm = np.linalg.norm(self.g)  # norm of the gradient
#         if g_0_norm == 0:
#             print("Zero gradient error!")
#             exit()
#         g_0_normed = self.g / g_0_norm  # normalized gradient
#         g_0_mom_dot = np.dot(g_0_normed, self.mom_normed)
#
#
#
#         if g_0_mom_dot < self.pair_con:
#             # self.convrge_state = True
#
#             self.step_g = self.step_g * 0.5
#             self.mom = g_0_normed
#             # self.convrge_state_m1 = self.convrge_state
#
#         elif self.g_0_mom_dot_m1 < self.pair_con:
#             # self.convrge_state = True
#
#             self.mom = g_0_normed
#             # self.convrge_state_m1 = self.convrge_state
#
#         else:
#             # self.convrge_state = False
#
#             if not self.mom_con:
#                 self.mom = np.zeros(self.d)
#             self.step_g = self.step_g * 2.0
#             self.mom = self.beta * self.mom + (1-self.beta) * g_0_normed
#
#             # self.convrge_state_m1 = self.convrge_state
#
#         # if self.step_g > self.step_g_r * self.lr:
#         #     self.step_g = self.step_g_r * self.lr
#
#
#         ###### Drift section ##########
#
#         drift_state = False # used to detect end of the drift period.
#         g_0_m1_dot = np.dot(g_0_normed, self.g_m1_normed) # dot product of gradient of 0 and -1 step to be used in drift
#
#         # runs drift step if drift is True, we have a pair of gradients and dot products of th last two gardeints is smaller than drift condition, drift_con.
#         if self.drift and g_0_m1_dot < self.drift_con:
#             g_drift_0 = ( g_0_normed + self.g_m1_normed ) # when there is a
#
#
#             g_drift_0_norm = np.linalg.norm(g_drift_0)  # norm of the gradient
#             if g_drift_0_norm > 1e-30:
#                 g_drift_0_normed = g_drift_0 / g_drift_0_norm  # normalized gradient
#                 g_drift_0_m1_dot = np.dot(g_drift_0_normed, self.g_drift_m1_normed)
#                 if g_drift_0_m1_dot < 0:
#                     self.step_drift = self.step_drift * 0.5
#                 elif self.g_drift_0_m1_dot_m1 < 0:
#                     self.step_drift = self.step_drift * 1.0
#                 else:
#                     self.step_drift = self.step_drift * 2.0
#
#
#                 # if self.step_drift > self.step_drift_r * self.lr:
#                 #     self.step_drift = self.step_drift_r * self.lr
#
#
#                 x_save = self.x
#                 self.x = self.x - g_drift_0_normed * self.step_drift
#
#                 loss = closure()
#
#                 g_norm = np.linalg.norm(self.g)
#                 g_normed = self.g / g_norm
#                 g_drift_dot = np.dot(g_drift_0_normed, g_normed)
#                 if g_drift_dot > self.drift_reject_con:
#                     self.g_drift_m1_normed = g_drift_0_normed
#                     self.g_drift_0_m1_dot_m1 = g_drift_0_m1_dot
#                     drift_state = True
#                 else:
#                     self.x = x_save
#
#
#         elif self.drift and self.g_0_mom_dot_m1 < self.pair_con and g_0_m1_dot > -self.drift_con:
#             drift_state = True
#
#         if not drift_state:
#             self.step_drift = self.step_g * 0.5
#             self.g_drift_m1_normed = np.zeros(self.d)
#             self.g_drift_0_m1_dot_m1 = 1
#
#
#
#         mom_norm = np.linalg.norm(self.mom)  # norm of the gradient
#         self.mom_normed = self.mom / mom_norm  # normalized gradient
#
#         self.x = self.x - self.mom_normed * self.step_g
#
#
#         self.g_0_mom_dot_m1 = g_0_mom_dot
#         self.g_m1_normed = g_0_normed
#
#
#
#     def step(self, closure = None):
#         self._update_params(closure)
#
#
#
