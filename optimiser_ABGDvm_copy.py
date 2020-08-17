import numpy as np


class abgd_vm_copy():
    def __init__(self, params, lr=0.01):

        self.lr = lr  # learning rate

        self.d = len(params)  # input dimension

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)


        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDvm_copy_init.py").read())

    def _update_params(self, closure):

        ### normalizing gradient


        g_0_normed = self.g / np.linalg.norm(self.g)  # normalized gradient
        g0_gm1_dot = np.dot(g_0_normed, self.g_m1_normed)

        # gg = self.g
        gg = g_0_normed

        if self.t == 1:
            self.mx10 = self.x
            self.m10 = gg

        self.m10 = 0.95 * self.m10  + 0.05 * gg
        # self.m04 = 0.80 * self.m04  + 0.4 * self.g

        # m10_sign = np.sign(self.m10)
        m10_normed = self.m10 / np.linalg.norm(self.m10)


        # self.mx10 = 0.9 * self.mx10 + 0.1 * self.x
        # r_m = self.x - self.mx10
        # away = np.sign(-self.m10 * r_m)
        #
        # step_d = np.where( away < 0, np.abs(r_m), self.step_g )
        # self.step_g = np.where( away < 0, self.step_g/2, self.step_g  )



        self.x = self.x - m10_normed * self.step_g
        # self.x = self.x - self.m10 * self.step_g

        # self.x = np.where( away < 0, self.mx10, self.x )




        # print("ABGDv step: ", self.step_g)

        # ff = open('outputs/neural_network/train/step_ABGDM.txt', 'a')
        # str_to_file = str(self.t) + "\t" + str(800 + self.step_g * 1000/5) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        #
        # ff = open('outputs/neural_network_minibatch/train/step_ABGDM.txt', 'a')
        # str_to_file = str(self.t) + "\t" + str(self.step_g * 10000/5) + "\n"
        # ff.write(str_to_file)
        # ff.close()

        self.g_m1_normed = g_0_normed
        self.t = self.t + 1




    def step(self, closure):
        if self.t == 1:
            self._find_lr(closure)
            self.t = 1
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
        self.step_g = float('{:0.1e}'.format(  ( self.step_g/10 + (loss0-loss1)*(self.step_g-self.step_g/10)/(loss2-loss1) ) / 10 ))
        self.lr = self.step_g






