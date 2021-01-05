# #### initialising parameters specific to the algorithm #######

self.t = 1

self.step_m = self.lr
self.find_lr = find_lr
self.reset_min = reset_min


self.beta_list = np.array(beta_list)
self.alpha_list = 1 - self.beta_list
# self.alpha2_list = self.alpha_list / 100.0
# self.beta2_list = 1 - self.alpha2_list
self.beta_size = np.size(self.beta_list)
self.m_i = self.beta_size - 1

sh = (self.beta_size, self.d)
self.m_list = np.zeros(sh)
self.m2_list = np.zeros(sh)
# self.v_list = np.zeros(sh)
self.mm_list = np.zeros(sh)
self.mm2_list = np.zeros(sh)
self.ms_list = np.zeros(sh)

sh = (self.beta_size, 1)
self.vm_list = np.zeros(sh)





# self.delay_step = 1
# # self.delay_step_max = np.log(0.25)/np.log(1 - self.alpha_list / 2.0 )
# self.delay_step_max = np.log(0.25)/np.log(1 - self.alpha_list)
# self.delay_step_max = self.delay_step_max.astype(int)
# self.delay_step = self.delay_step_max[self.m_i]

self.loss0 = 0
self.loss1 = 0
self.loss_min = 1e100
self.x_min =  np.zeros(self.d)
self.loss1_m1 = 1e40


self.ind_m_m1 = 0
self.ind_m = 0

self.ind_mm_m1 = 1
self.ind_mm  = 1
