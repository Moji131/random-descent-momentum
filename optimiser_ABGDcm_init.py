##### initialising parameters specific to the algorithm #######

self.min_step_r = min_step_r
self.max_step_r = max_step_r

self.m_max = momentum_max

self.step_g = np.ones(self.d) * lr  # inital step_g (can be anything)
self.g_m1_sign = np.ones(self.d)  # get the sign of the components
self.g_0_m_m1 = np.ones(self.d)
self.g_m1_m2 = np.ones(self.d)

self.m = np.zeros(self.d)