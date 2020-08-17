# #### initialising parameters specific to the algorithm #######

self.step_g = self.lr
# self.step_g = self.lr
# self.step_g_mean = self.lr



self.beta_list = np.array([0,0.88,0.92])
self.beta2_list = np.array([0.999,0.999, 0.999])

# self.beta_list = np.array([0,0.75,0.9])
# self.beta2_list = np.array([0.999,0.999,0.999])

# self.beta_list = np.array([0, 0.9])
# self.beta2_list = np.array([0.999,0.999])

# self.beta_list = np.array([0.9])
# self.beta2_list = np.array([0.999])


self.beta_size = np.size(self.beta_list)
self.beta_i = self.beta_size - 1

sh = (self.beta_size, self.d)
self.m_list = np.zeros(sh)
self.v_list = np.zeros(sh)

self.md_list = np.zeros(sh)
sh = (self.beta_size, 1)
self.vd_list = np.zeros(sh)
self.ind_d_list = np.zeros(sh)

self.g0_gm1_dot_m1 = 1

self.m = np.zeros(self.d)
self.v = np.zeros(self.d)
self.p_m1_normed = np.ones(self.d) / np.sqrt(self.d)

self.v10 = np.zeros(self.d)
self.mx10 = np.zeros(self.d)


self.m10 = np.zeros(self.d)
self.v100 = np.zeros(self.d)

self.m100 = np.zeros(self.d)
self.v1000 = np.zeros(self.d)

self.m1000 = np.zeros(self.d)
self.v10000 = np.zeros(self.d)

self.m_dis = np.zeros(self.d)
self.v_dis = np.zeros(self.d)


self.md = np.zeros(self.d)
self.g_s = np.zeros(self.d)


self.t = 1

self.d_m1 = np.zeros(self.d)

self.g_m1= np.zeros(self.d)
self.x_s = np.zeros(self.d)
self.g_m1_normed = np.zeros(self.d)
self.d_m1_normed = np.zeros(self.d)
self.d0_dm1_dot_m1 = 0

self.converge = 1

self.l_m1 = 999999999999999.0
self.l_m2 = 9999999999999999.0
self.x_m1 = self.x

self.m_m1 = np.zeros(self.d)
self.v_m1 = np.zeros(self.d)

self.dis = np.zeros(self.d)

self.m_d0_g0_dot = 0

self.d_sum_normed_m1 = np.zeros(self.d)

self.md = 0

self.m250 = np.zeros(self.d)
self.m100 = np.zeros(self.d)
self.m040 = np.zeros(self.d)
self.v001 = 0

self.mr250 = np.zeros(self.d)
self.mr100 = np.zeros(self.d)
self.mr040 = np.zeros(self.d)
self.vr001 = 0

self.beta_1 = 0.9 * np.ones(self.d)
self.p_sign_m1 = np.ones(self.d)
self.g_sign_m1 = np.ones(self.d)
self.turn_count = np.zeros(self.d)
self.step_count = np.ones(self.d) * self.d * 2

self.d_sign = np.zeros(self.d)
self.p = np.zeros(self.d)



