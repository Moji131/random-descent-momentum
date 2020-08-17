##### initialising parameters specific to the algorithm #######

self.step_g = self.lr
self.step_d = self.lr

self.l_m1 = 999999999999999.0
self.l_m2 = 9999999999999999.0
self.t = 1

self.converge = 2

self.g_list = [np.zeros(self.d)] * 1

self.m = np.zeros(self.d)
self.v = np.zeros(self.d)

self.m10 = np.zeros(self.d)
self.m100 = np.zeros(self.d)
self.mx10 = np.zeros(self.d)




self.m_m1_normed = np.zeros(self.d)
self.v_m1_normed = np.zeros(self.d)

self.momentum = 0.5
self.converge_m1 = True

self.g_m1_normed = np.zeros(self.d)
self.g_s_normed = np.zeros(self.d)

self.d_m1_normed = np.zeros(self.d)
self.d_s_normed = np.zeros(self.d)


self.m0_mm1_dot_m1 = 0
self.decrease_m1 = True

self.g_m1= np.zeros(self.d)
self.x_m1 = self.x
self.x_s = self.x


self.m_m1 = np.zeros(self.d)
self.v_m1 = np.zeros(self.d)

self.g_sum_normed_m1 = np.zeros(self.d)