# #### initialising parameters specific to the algorithm #######

self.t = 1

self.step_m = self.lr
self.find_lr = find_lr
self.reset_min = reset_min


self.beta_list = np.array(beta_list)
self.alpha_list = 1 - self.beta_list
self.alpha2_list = self.alpha_list / 100.0
self.beta2_list = 1 - self.alpha2_list
self.beta_size = np.size(self.beta_list)
self.m_i = self.beta_size - 1

sh = (self.beta_size, self.d)
self.m_list = np.zeros(sh)
self.v_list = np.zeros(sh)
self.md_list = np.zeros(sh)
self.ms_list = np.zeros(sh)
sh = (self.beta_size, 1)
self.ind_d_list = np.ones(sh)
self.ind_d_list_m1 = np.ones(sh)

sh = (self.beta_size, 1)
self.vd_list = np.zeros(sh)

self.delay_step_up = 1
self.delay_step = int(2/self.alpha_list[self.m_i])

self.loss0 = 0
self.loss1 = 0
self.loss_min = 1e100
self.x_min =  np.zeros(self.d)

