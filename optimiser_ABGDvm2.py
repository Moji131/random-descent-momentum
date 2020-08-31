import numpy as np


class abgd_vm():
    def __init__(self, params, lr=0.01):

        self.lr = lr  # learning rate

        self.d = len(params)  # input dimension

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)


        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDvm_init.py").read())

    def _update_params(self, closure):

        ### normalizing gradient


        g_0_normed = self.g / np.linalg.norm(self.g)  # normalized gradient
        g0_gm1_dot = np.dot(g_0_normed, self.g_m1_normed)

        # gg = self.g
        gg = g_0_normed

        if self.t == 1:
            # self.mx10[:] = self.x[:]
            self.x_m1 = self.x + g_0_normed * self.step_g
            for i in range(self.beta_size):
                # self.m_list[i] = gg[:]
                self.md_list[i] = - g_0_normed * self.step_g
                self.vd_list[i] = np.linalg.norm(g_0_normed * self.step_g)




        # self.m10 = 0.9 * self.m10  + 0.1 * gg
        for i in range(self.beta_size):
            self.m_list[i] = self.beta_list[i] * self.m_list[i] + (1- self.beta_list[i]) * gg


        d = self.x - self.x_m1
        for i in range(self.beta_size):
            alpha_d = min((1 - self.beta_list[i]), 0.5) # to work for momentum 0
            self.md_list[i] = (1-alpha_d) * self.md_list[i]  + alpha_d * d
            self.vd_list[i] = (1-alpha_d) * self.vd_list[i]  + alpha_d * np.linalg.norm(d)
            self.ind_d_list[i] = np.linalg.norm(self.md_list[i]) / self.vd_list[i]


        if  self.ind_d_list[self.beta_i,0] < 0.35 and self.beta_i > 0:
            self.beta_i = self.beta_i - 1
            self.step_g_save = self.step_g
        if self.beta_i < self.beta_size-1:
            if self.ind_d_list[self.beta_i+1, 0] > 0.4:
                if self.beta_i == 0:
                    # print("recovery", self.step_g, self.step_g_save)
                    self.step_g = self.step_g_save
                self.beta_i = self.beta_i + 1
                # self.step_g = self.step_g * 2.0

        # if self.beta_i == self.beta_size-1 and self.ind_d_list[self.beta_i, 0] > 0.95 :
        #     self.step_g = self.step_g * 1.1


        # scale_main = 50000
        # ff = open('outputs/main/0-ind_d-vm', 'a')
        # str_to_file = str(self.t) + "\t" + str( self.ind_d_list[self.beta_i,0]* scale_main) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        # ff = open('outputs/main/0-beta_i-vm', 'a')
        # str_to_file = str(self.t) + "\t" + str( self.beta_i * scale_main / (self.beta_size)) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        #
        # scale_nn = 140
        # ff = open('outputs/neural_network/train/0-ind_d-vm', 'a')
        # str_to_file = str(self.t) + "\t" + str( self.ind_d_list[self.beta_i,0]* scale_nn) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        # ff = open('outputs/neural_network/train/0-beta_i-vm', 'a')
        # str_to_file = str(self.t) + "\t" + str( self.beta_i * scale_nn / (self.beta_size)) + "\n"
        # ff.write(str_to_file)
        # ff.close()




        # step_mult_max = 1.0
        # if self.beta_i == 0 and g0_gm1_dot < -1/2/step_mult_max:
        #     step_mult = -1/2/g0_gm1_dot
        # else:
        #     step_mult = step_mult_max
        # self.step_g = step_mult * self.step_g

        m_normed = self.m_list[self.beta_i] / np.linalg.norm(self.m_list[self.beta_i])
        g_m_dot = np.dot(g_0_normed , m_normed)

        if self.beta_list[self.beta_i] == 0:
            if g0_gm1_dot < 0:
                self.step_g = 0.5 * self.step_g
            elif self.g0_gm1_dot_m1  < 0:
                self.step_g = 1.0 * self.step_g
            else:
                self.step_g = 2.0 * self.step_g


        if self.beta_i == self.beta_size - 1 and self.beta_list[self.beta_i] != 0:
            if g_m_dot > 0:
                self.step_g = self.step_g * 1.01
            else:
                self.step_g = self.step_g / 1.0


        self.x_m1[:] = self.x[:]
        self.x = self.x - m_normed * self.step_g
        # self.x = self.x - self.m_list[self.beta_i] * self.step_g


        # ff = open('outputs/neural_network/train/step_ABGDM.txt', 'a')
        # str_to_file = str(self.t) + "\t" + str(800 + self.step_g * 1000/5) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        #


        self.g0_gm1_dot_m1 = g0_gm1_dot
        self.g_m1_normed = g_0_normed
        self.t = self.t + 1





    def step(self, closure):
        self._update_params(closure)






