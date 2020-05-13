##### initialising parameters specific to algorithm #######
self.min_step_r = min_step_r
self.max_step_r = max_step_r

self.step_g = np.ones(self.d) * lr  # inital step_g (can be anything)
self.g_m1_sign = np.ones(self.d)   # get the sign of the components
self.g_m1_m2 = np.ones(self.d)  # initialising product of sign of gradient of step_g 0 and -1



