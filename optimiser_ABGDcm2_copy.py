
import numpy as np

class abgd_cm2_copy():
    def __init__(self, params, lr=0.01):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDcm2_copy_init.py").read())



    def _update_params(self, closure):

        beta_2 = 0.999
        epsilon = 1.0e-08

        g_0_normed = self.g / np.linalg.norm(self.g)  # normalized gradient
        g0_gm1_dot = np.dot(g_0_normed, self.g_m1_normed)
        g_sign = np.sign(self.g)
        g_gm1_sign = g_sign * self.g_sign_m1

        gg = g_0_normed
        # gg = self.g

        if self.t == 1:
            # self.mx10[:] = self.x[:]
            self.x_m1 = self.x + g_0_normed * self.step_g
            for i in range(self.beta_size):
                # self.m_list[i] = gg[:]
                # self.v_list[i] = np.power(gg,2)
                self.md_list[i] = - g_0_normed * self.step_g
                self.vd_list[i] = np.linalg.norm(g_0_normed * self.step_g)


        for i in range(self.beta_size):
            self.m_list[i] = self.beta_list[i] * self.m_list[i] + (1- self.beta_list[i]) * gg
            self.v_list[i] = self.beta2_list[i] * self.v_list[i] + (1 - self.beta2_list[i]) * np.power(gg,2)

        m_hat = self.m_list[self.beta_i] / (1 - self.beta_list[self.beta_i]**self.t)
        v_hat = self.v_list[self.beta_i] / (1 - self.beta2_list[self.beta_i] ** self.t)
        p =  m_hat / (np.sqrt(v_hat) + epsilon)
        # p_sign = np.sign(p)
        p_normed = p / np.linalg.norm(p)
        g_p_dot = np.dot(g_0_normed , p_normed)

        d = self.x - self.x_m1
        for i in range(self.beta_size):
            alpha_d = min((1 - self.beta_list[i]), 0.5)
            self.md_list[i] = (1-alpha_d) * self.md_list[i]  + alpha_d * d
            self.vd_list[i] = (1-alpha_d) * self.vd_list[i]  + alpha_d * np.linalg.norm(d)
            self.ind_d_list[i] = np.linalg.norm(self.md_list[i]) / self.vd_list[i]



        if  self.ind_d_list[self.beta_i,0] < 0.35 and self.beta_i > 0:
            self.beta_i = self.beta_i - 1
            self.step_g_save = self.step_g
        if self.beta_i < self.beta_size-1:
            if self.ind_d_list[self.beta_i+1, 0] > 0.4:
                if self.beta_i == 0:
                    self.step_g = self.step_g_save
                self.beta_i = self.beta_i + 1
                # self.step_g = self.step_g * 2.0


        # if self.beta_i == self.beta_size-1 and self.ind_d_list[self.beta_i, 0] > 0.85 :
        #     self.step_g = self.step_g * 1.1


        scale_main = 50000


        ff = open('outputs/main/0-ind_d-cm2', 'a')
        str_to_file = str(self.t) + "\t" + str( self.ind_d_list[self.beta_i,0]* scale_main) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/main/0-beta_i-cm2', 'a')
        str_to_file = str(self.t) + "\t" + str( self.beta_i * scale_main / (self.beta_size)) + "\n"
        ff.write(str_to_file)
        ff.close()

        scale_nn = 140
        ff = open('outputs/neural_network/train/0-ind_d-cm2', 'a')
        str_to_file = str(self.t) + "\t" + str( self.ind_d_list[self.beta_i,0]* scale_nn) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/neural_network/train/0-beta_i-cm2', 'a')
        str_to_file = str(self.t) + "\t" + str( self.beta_i * scale_nn / (self.beta_size)) + "\n"
        ff.write(str_to_file)
        ff.close()




        if self.beta_list[self.beta_i] == 0:
            if g0_gm1_dot < 0:
                self.step_g = 0.5 * self.step_g
            elif self.g0_gm1_dot_m1  < 0:
                self.step_g = 1.0 * self.step_g
            else:
                self.step_g = 2.0 * self.step_g

        # if self.beta_i == self.beta_size-1 and self.beta_list[self.beta_i] != 0:
        #     if g_p_dot > 0:
        #         self.step_g = self.step_g  * 1.0001
        #     else:
        #         self.step_g = self.step_g / 1



        self.x_m1[:] = self.x[:]
        # self.x = self.x - self.step_g * p
        self.x = self.x - self.step_g * p_normed
        # self.x = self.x - self.step_g * p_sign

        #
        # ff = open('outputs/neural_network/train/step_ADAM2.txt', 'a')
        # str_to_file = str(self.t) + "\t" + str(800+ self.step_g * 1000) + "\n"
        # ff.write(str_to_file)
        # ff.close()



        self.t = self.t + 1

        self.g_m1_normed = g_0_normed
        # self.p_sign_m1 = p_sign
        self.g_sign_m1 = g_sign
        self.g0_gm1_dot_m1 = g0_gm1_dot





    def step(self, closure):
        if self.t == 1:
            self._find_lr(closure)
        self._update_params(closure)




    def _find_lr(self, closure):
        self.step_g = self.lr / 1000
        xx = self.x[:]
        loss0 = closure()
        loss2 = loss0
        loss1 = loss0

        while not loss2 > loss1:
            loss1 = loss2
            self.step_g = 10 * self.step_g
            self._update_params(closure)
            loss2 = closure()
            self.x = xx


        loss0 = closure()
        self.step_g = float('{:0.1e}'.format(  ( self.step_g/10 + (loss0-loss1)*(self.step_g-self.step_g/10)/(loss2-loss1) ) / 10  ))



