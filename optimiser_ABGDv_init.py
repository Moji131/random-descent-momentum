##### initialising parameters specific to the algorithm #######

self.step_g = np.ones(self.d) * self.lr
self.g_m1_normed = np.zeros(self.d)
self.g_0_m1_dot_m1 = 1.0

