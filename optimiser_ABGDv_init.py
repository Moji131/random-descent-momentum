##### initialising parameters specific to the algorithm #######

self.step_g = self.lr * np.ones(self.d)
self.g_0_m1_dot_m1 = 0
self.g_m1_normed = np.zeros(self.d)