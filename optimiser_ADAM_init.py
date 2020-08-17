##### initialising parameters specific to the algorithm #######
self.m = np.zeros(self.d)
self.v = np.zeros(self.d)
self.t = 1

self.g_m1= np.zeros(self.d)
self.x_s = np.zeros(self.d)
self.g_m1_normed = np.zeros(self.d)
self.x = np.zeros(self.d)

self.step_g = self.lr

