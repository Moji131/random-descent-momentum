
import numpy as np

class abgd_cm2():
    def __init__(self, params, lr=0.01):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDcm2_init.py").read())



    def _update_params(self, closure):

        # if self.t == 1:
        #     self.m_i_list = np.array([0]*self.d)
        #
        # if self.t == 10:
        #     self.m_i_list = np.array([1]*self.d)
        #
        # if self.t == 20:
        #     self.m_i_list = np.array([0]*self.d)

        beta_2 = 0.999
        epsilon = 1.0e-08

        g_0_normed = self.g / np.linalg.norm(self.g)  # normalized gradient
        g0_gm1_dot = np.dot(g_0_normed, self.g_m1_normed)
        g_sign = np.sign(self.g)
        g_gm1_sign = g_sign * self.g_sign_m1

        # gg = g_0_normed # use normalised gardient
        gg = self.g # use normal gradient

        self.v = beta_2 * self.v + (1 - beta_2) * np.power(gg, 2)
        v_hat = self.v / (1 - beta_2 ** self.t)

        for i in range(self.beta_list_size):
            self.m_list[i] = self.beta_list[i] * self.m_list[i] + gg * (1 - self.beta_list[i])
            self.m_hat_list[i] = self.m_list[i] / (1 - self.beta_list[i]**self.t)
            self.d_list[i] =  self.m_hat_list[i] / (np.sqrt(v_hat) + epsilon)

        for n in range(self.d):
            self.p[n] = self.d_list[self.m_i_list[n],n]
        p_sign = np.sign(self.p)

        if self.t == 1:
            self.turn_count = np.ones(self.d) * 0
        else:
            p_pm1_sign = p_sign * self.p_sign_m1
            self.turn_count = np.where(p_pm1_sign == -1, self.turn_count + 1, self.turn_count)
            self.m_i_list = np.where( ((self.turn_count == self.beta_turn_max) & ~ (self.m_i_list == 0 )) , self.m_i_list - 1, self.m_i_list)
            self.turn_count = np.where(self.turn_count >= self.beta_turn_max, 0, self.turn_count)


            self.step_count = np.where(p_pm1_sign == -1, 0, self.step_count + 1 )
            self.m_i_list = np.where( (self.step_count > self.beta_step_max[self.m_i_list]) & ~ (self.m_i_list == self.beta_list_size - 1), self.m_i_list + 1, self.m_i_list )

            # self.step_g = np.where(  (self.m_i_list == self.beta_list_size - 1) & (self.step_count > self.beta_step_max[self.m_i_list]), self.step_g * 1.001,  self.step_g)
            # self.step_g = np.where(  (self.beta_list[self.m_i_list] == 0) & (p_pm1_sign == -1), self.step_g * 0.5,  self.step_g)  # halve if momentum is zero and p turns
            self.step_g = np.where((g_gm1_sign == -1) & (p_pm1_sign == -1), self.step_g * 0.5, self.step_g) # halve if both p and g turn


        self.x = self.x - self.step_g * self.p
        # self.x = self.x - self.step_g * p_sign


        #
        # ff = open('outputs/neural_network/train/step_ADAM2.txt', 'a')
        # str_to_file = str(self.t) + "\t" + str(800+ self.step_g * 1000) + "\n"
        # ff.write(str_to_file)
        # ff.close()



        self.t = self.t + 1

        self.g_m1_normed = g_0_normed
        self.p_sign_m1 = p_sign
        self.g_sign_m1 = g_sign





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





