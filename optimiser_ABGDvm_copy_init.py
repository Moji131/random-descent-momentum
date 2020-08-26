##### initialising parameters specific to the algorithm #######

self.step_g = self.lr
self.step_m = self.lr
self.g_delay = 0

self.beta_list = np.array([0,0.8,0.9,0.97])
self.beta_size = np.size(self.beta_list)
self.m_i = self.beta_size - 1

sh = (self.beta_size, self.d)
self.m_list = np.zeros(sh)
self.m_list_m1 = np.zeros(sh)
self.md_list = np.zeros(sh)
sh = (self.beta_size, 1)
self.vd_list = np.zeros(sh)
self.ind_d_list = np.zeros(sh)

self.g0_gm1_dot_m1 = 1



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

self.x_m1 = np.zeros(self.d)
self.md10 = np.zeros(self.d)
self.vd10 = 0

