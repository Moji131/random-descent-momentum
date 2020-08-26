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
            m_normed = gg *(1- self.beta_list[self.m_i]) / np.linalg.norm(gg *(1- self.beta_list[self.m_i]))
            self.step_m = self._find_step_m(closure, self.step_m,  m_normed )
            self.step_g = self.step_m

            self.lr = self.step_g
            self.x_m1 = self.x + g_0_normed * self.step_g


        # update all momentum
        for i in range(self.beta_size):
            self.m_list_m1[i][:] = self.m_list[i][:]
            self.m_list[i] = self.beta_list[i] * self.m_list[i] + (1- self.beta_list[i]) * gg


        # indicators for oscillation
        for i in range(self.beta_size):
            # d = self.m_list[i]
            d = self.m_list[i] / np.linalg.norm(self.m_list[i])
            self.md_list[i] = self.beta_list[i] * self.md_list[i] + (1- self.beta_list[i] ) * d
            self.vd_list[i] = self.beta_list[i] * self.vd_list[i] + (1- self.beta_list[i] ) * np.linalg.norm(d)
            self.ind_d_list[i] = np.linalg.norm(self.md_list[i]) / self.vd_list[i]

        # condition to push momentum down
        if  self.ind_d_list[self.m_i,0] < 0.5 and self.m_i > 0:
            self.m_i = self.m_i - 1
            print(">>>>>  push down, momentum", self.beta_list[self.m_i])
            if self.m_i == 0:
                self.step_g = self._find_step_g(closure, self.step_m, self.m_list[self.m_i])
                g0_gm1_dot = 1
                self.g_delay = 2
            else:
                m_normed = self.m_list[self.m_i] / np.linalg.norm(self.m_list[self.m_i])
                # self.step_m = self._find_step_m(closure, self.step_m, m_normed)

        # condition to push momentum up
        elif self.m_i < self.beta_size-1:
            if self.ind_d_list[self.m_i+1, 0] > 0.7:
                self.m_i = self.m_i + 1
                print(">>>>>  push up, momentum", self.beta_list[self.m_i])
                # self.step_m = self._find_step_g(closure, self.step_m, self.m_list[self.m_i])


        scale_main = 50000
        ff = open('outputs/main/0-ind_d-vm', 'a')
        str_to_file = str(self.t) + "\t" + str( self.ind_d_list[self.m_i,0]* scale_main) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/main/0-beta_i-vm', 'a')
        str_to_file = str(self.t) + "\t" + str( self.m_i * scale_main / (self.beta_size)) + "\n"
        ff.write(str_to_file)
        ff.close()

        scale_nn = 100
        ff = open('outputs/neural_network/train/0-ind_d-vm', 'a')
        str_to_file = str(self.t) + "\t" + str( self.ind_d_list[self.m_i,0]* scale_nn) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/neural_network/train/0-beta_i-vm', 'a')
        str_to_file = str(self.t) + "\t" + str( self.m_i * scale_nn / (self.beta_size)) + "\n"
        ff.write(str_to_file)
        ff.close()



        # step_mult_max = 1.0
        # if self.m_i == 0 and g0_gm1_dot < -1/2/step_mult_max:
        #     step_mult = -1/2/g0_gm1_dot
        # else:
        #     step_mult = step_mult_max
        # self.step_g = step_mult * self.step_g

        m_normed = self.m_list[self.m_i] / np.linalg.norm(self.m_list[self.m_i])
        g_m_dot = np.dot(g_0_normed , m_normed)

        if self.beta_list[self.m_i] == 0:
            if g0_gm1_dot < 0:
                self.step_g = 0.5 * self.step_g
                self.g_delay = 1
            elif self.g_delay  > 0:
                self.step_g = 1.0 * self.step_g
                self.g_delay = self.g_delay - 1
            else:
                self.step_g = 2.0 * self.step_g
                self.g_delay = self.g_delay - 1
        else:
            if np.linalg.norm(self.m_list[self.m_i]) > np.linalg.norm(self.m_list_m1[self.m_i]):
                self.step_m = self.step_m * 1.1
            else:
                self.step_m = self.step_m / 1.1


        # if self.m_i == self.beta_size - 1 and self.beta_list[self.m_i] != 0:
        #     if g_m_dot > 0:
        #         self.step_g = self.step_g * 1.01
        #     else:
        #         self.step_g = self.step_g / 1.0


        # input()
        self.x_m1[:] = self.x[:]
        # self.x = self.x - m_normed * self.step_g
        if self.m_i == 0:
            self.x = self.x - m_normed * self.step_g
        else:
            # self.x = self.x - self.m_list[self.m_i] * self.step_m
            self.x = self.x - m_normed * self.step_m


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



    #
    # def _find_lr(self, closure):
    #     self.step_g = self.lr / 1000
    #     xx = self.x[:]
    #     loss0 = closure()
    #     loss2 = loss0
    #     loss1 = loss0
    #
    #     while not loss2 > loss1:
    #         loss1 = loss2
    #         self.step_g = 10 * self.step_g
    #         self._update_params(closure)
    #         loss2 = closure()
    #         self.x = xx
    #
    #
    #     loss0 = closure()
    #     self.step_g = float('{:0.1e}'.format(  ( self.step_g/10 + (loss0-loss1)*(self.step_g-self.step_g/10)/(loss2-loss1) ) / 2 ))
    #     # self.step_g = ( self.step_g/10 + (loss0-loss1)*(self.step_g-self.step_g/10)/(loss2-loss1) ) / 10
    #     self.lr = self.step_g



    def _find_step_g(self, closure, step, o):
        print("_find_step_g")

        stage0_step = np.array([])
        stage0_loss = np.array([])
        stage1_step = np.array([])
        stage1_loss = np.array([])
        stage2_step = np.array([])
        stage2_loss = np.array([])
        stage3_step = np.array([])
        stage3_loss = np.array([])

        x_s = self.x[:]
        loss0 = closure()
        stage0_step = np.append(stage0_step, np.array([0]))
        stage0_loss = np.append(stage0_loss, np.array([loss0]))

        self.x = x_s - o * step
        loss2 = closure()
        self.x = x_s
        stage0_step = np.append(stage0_step, np.array([step]))
        stage0_loss = np.append(stage0_loss, np.array([loss2]))

        print("stage0_step", stage0_step)
        print("stage0_loss", stage0_loss)





        if loss2 <= loss0:
            n1 = 0
            while loss2 <= loss0 and n1 < 10:
                step = step * 10
                self.x = x_s - o * step
                loss2 = closure()
                stage1_step = np.append(stage1_step, np.array([step]))
                stage1_loss = np.append(stage1_loss, np.array([loss2]))
                self.x = x_s
                n1 = n1 + 1

            print("loss2 <= loss0. stage1_step", stage1_step)
            print("loss2 <= loss0. stage1_loss", stage1_loss)

            if n1 == 10:
                print("loss2 <= loss0, smaller not found.")
            else:
                f = 0.9
                n2 = 0
                while loss2 > loss0 and f > 0.1:
                    ss = step * f
                    self.x = x_s - o * ss
                    loss2 = closure()
                    stage2_step = np.append(stage2_step, np.array([ss]))
                    stage2_loss = np.append(stage2_loss, np.array([loss2]))
                    self.x = x_s
                    f = f - 0.1
                    n2 = n2 + 1
                print("loss2 <= loss0. stage2_step", stage2_step)
                print("loss2 <= loss0. stage2_loss", stage2_loss)

                for i in range(9):
                    ss2 = ss * (i + 1) / 10.0
                    self.x = x_s - o * ss2
                    loss2 = closure()
                    stage3_step = np.append(stage3_step, np.array([ss2]))
                    stage3_loss = np.append(stage3_loss, np.array([loss2]))
                    self.x = x_s
                print("loss2 <= loss0. stage3_step", stage3_step)
                print("loss2 <= loss0. stage3_loss", stage3_loss)


        else:
            n1 = 0
            while loss2 > loss0 and n1 < 10:
                step = step / 10
                self.x = x_s - o * step
                loss2 = closure()
                stage1_step = np.append(stage1_step, np.array([step]))
                stage1_loss = np.append(stage1_loss, np.array([loss2]))
                self.x = x_s
            print("loss2 > loss0. stage1_step", stage1_step)
            print("loss2 > loss0. stage1_loss", stage1_loss)


            if n1 == 10:
                print("loss2 > loss0, smaller not found.")
            else:
                f = 2
                n2 = 0
                while loss2 <= loss0 and f < 10:

                    ss = step * f
                    self.x = x_s - o * ss
                    loss2 = closure()
                    stage2_step = np.append(stage2_step, np.array([ss]))
                    stage2_loss = np.append(stage2_loss, np.array([loss2]))
                    self.x = x_s
                    f = f + 1
                    n2 = n2 + 1
                print("loss2 > loss0. stage2_step", stage2_step)
                print("loss2 > loss0. stage2_loss", stage2_loss)

                for i in range(9):
                    ss2 = ss * (i + 1) / 10
                    self.x = x_s - o * ss2
                    loss2 = closure()
                    stage3_step = np.append(stage3_step, np.array([ss2]))
                    stage3_loss = np.append(stage3_loss, np.array([loss2]))
                    self.x = x_s
                print("loss2 > loss0. stage3_step", stage3_step)
                print("loss2 > loss0. stage3_loss", stage3_loss)

        loss0 = closure()

        stageall_step = np.concatenate((stage0_step, stage1_step, stage2_step, stage3_step))
        stageall_loss = np.concatenate((stage0_loss, stage1_loss, stage2_loss, stage3_loss))
        loss_min = np.amin(stageall_loss)
        min_i = np.where(stageall_loss == loss_min)
        step_min_arr = stageall_step[min_i]
        step_min = np.min(step_min_arr)

        print("step_min", step_min)


        p_g = 4
        sout = float('{:0.1e}'.format(step_min / p_g))
        print("step out", sout)
        # input()
        return sout






    def _find_step_m(self, closure, step, o):
        print("_find_step_m")
        g_0_normed = self.g / np.linalg.norm(self.g)
        dot = np.dot(o,g_0_normed)
        if dot < 0:
            o = -o
        print("m dot g: ",dot)

        stage0_step = np.array([])
        stage0_loss = np.array([])
        stage1_step = np.array([])
        stage1_loss = np.array([])
        stage2_step = np.array([])
        stage2_loss = np.array([])
        stage3_step = np.array([])
        stage3_loss = np.array([])

        x_s = self.x[:]
        loss0 = closure()
        stage0_step = np.append(stage0_step, np.array([0]))
        stage0_loss = np.append(stage0_loss, np.array([loss0]))

        self.x = x_s - o * step
        loss2 = closure()
        self.x = x_s
        stage0_step = np.append(stage0_step, np.array([step]))
        stage0_loss = np.append(stage0_loss, np.array([loss2]))

        print("stage0_step", stage0_step)
        print("stage0_loss", stage0_loss)





        if loss2 <= loss0:
            n1 = 0
            while loss2 <= loss0 and n1 < 10:
                step = step * 10
                self.x = x_s - o * step
                loss2 = closure()
                stage1_step = np.append(stage1_step, np.array([step]))
                stage1_loss = np.append(stage1_loss, np.array([loss2]))
                self.x = x_s
                n1 = n1 + 1

            print("loss2 <= loss0. stage1_step", stage1_step)
            print("loss2 <= loss0. stage1_loss", stage1_loss)

            if n1 == 10:
                print("loss2 <= loss0, smaller not found.")
            else:
                f = 0.9
                n2 = 0
                while loss2 > loss0 and f > 0.1:
                    ss = step * f
                    self.x = x_s - o * ss
                    loss2 = closure()
                    stage2_step = np.append(stage2_step, np.array([ss]))
                    stage2_loss = np.append(stage2_loss, np.array([loss2]))
                    self.x = x_s
                    f = f - 0.1
                    n2 = n2 + 1
                print("loss2 <= loss0. stage2_step", stage2_step)
                print("loss2 <= loss0. stage2_loss", stage2_loss)

                for i in range(9):
                    ss2 = ss * (i + 1) / 10.0
                    self.x = x_s - o * ss2
                    loss2 = closure()
                    stage3_step = np.append(stage3_step, np.array([ss2]))
                    stage3_loss = np.append(stage3_loss, np.array([loss2]))
                    self.x = x_s
                print("loss2 <= loss0. stage3_step", stage3_step)
                print("loss2 <= loss0. stage3_loss", stage3_loss)


        else:
            n1 = 0
            while loss2 > loss0 and n1 < 10:
                step = step / 10
                self.x = x_s - o * step
                loss2 = closure()
                stage1_step = np.append(stage1_step, np.array([step]))
                stage1_loss = np.append(stage1_loss, np.array([loss2]))
                self.x = x_s
            print("loss2 > loss0. stage1_step", stage1_step)
            print("loss2 > loss0. stage1_loss", stage1_loss)


            if n1 == 10:
                print("loss2 > loss0, smaller not found.")
            else:
                f = 2
                n2 = 0
                while loss2 <= loss0 and f < 10:

                    ss = step * f
                    self.x = x_s - o * ss
                    loss2 = closure()
                    stage2_step = np.append(stage2_step, np.array([ss]))
                    stage2_loss = np.append(stage2_loss, np.array([loss2]))
                    self.x = x_s
                    f = f + 1
                    n2 = n2 + 1
                print("loss2 > loss0. stage2_step", stage2_step)
                print("loss2 > loss0. stage2_loss", stage2_loss)

                for i in range(9):
                    ss2 = ss * (i + 1) / 10
                    self.x = x_s - o * ss2
                    loss2 = closure()
                    stage3_step = np.append(stage3_step, np.array([ss2]))
                    stage3_loss = np.append(stage3_loss, np.array([loss2]))
                    self.x = x_s
                print("loss2 > loss0. stage3_step", stage3_step)
                print("loss2 > loss0. stage3_loss", stage3_loss)

        loss0 = closure()

        stageall_step = np.concatenate((stage0_step, stage1_step, stage2_step, stage3_step))
        stageall_loss = np.concatenate((stage0_loss, stage1_loss, stage2_loss, stage3_loss))
        loss_min = np.amin(stageall_loss)
        min_i = np.where(stageall_loss == loss_min)
        step_min_arr = stageall_step[min_i]
        step_min = np.min(step_min_arr)

        print("step_min", step_min)


        p_m = 20
        sout = float('{:0.1e}'.format(step_min / p_m))
        print("step out", sout)
        # input()
        return sout

    #
    # def _find_step_m(self, closure, step, o):
    #     x_s = self.x[:]
    #     loss0 = closure()
    #
    #     self.x = x_s - o * step
    #     loss2 = closure()
    #     self.x = x_s
    #
    #     if loss2 <= loss0:
    #         # add max iteration check
    #         while loss2 <= loss0:
    #             step = step * 10
    #             self.x = x_s - o * step
    #             loss2 = closure()
    #             self.x = x_s
    #         step = step / 10
    #         f = 0.9
    #         # add min f check?
    #         while loss2 > loss0:
    #             ss = step * f
    #             self.x = x_s - o * ss
    #             loss2 = closure()
    #             self.x = x_s
    #             f = f - 0.1
    #     else:
    #         while loss2 > loss0:
    #             step = step / 10
    #             self.x = x_s - o * step
    #             loss2 = closure()
    #             self.x = x_s
    #         f = 2
    #         while loss2 <= loss0:
    #             ss = step * f
    #             self.x = x_s - o * ss
    #             loss2 = closure()
    #             self.x = x_s
    #             f = f + 1
    #
    #     loss0 = closure()
    #     print("ss/2", ss / 2)
    #
    #     return ss / 2
    #
    #     # loss0 = closure()
    #     # self.step_g = float('{:0.1e}'.format(  ( self.step_g/10 + (loss0-loss1)*(self.step_g-self.step_g/10)/(loss2-loss1) ) / 2 ))
    #     # self.step_g = ( self.step_g/10 + (loss0-loss1)*(self.step_g-self.step_g/10)/(loss2-loss1) ) / 10
    #     self.step_g =step




