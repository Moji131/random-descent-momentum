import numpy as np


class ALR_GDM():
    def __init__(self, params, lr=0.01,  beta_list = [0.9], find_lr = True, reset_min = True):

        self.lr = lr  # learning rate

        self.d = len(params)  # input dimension

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)


        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ALR_GDM_init.py").read())

    def _update_params(self, closure):

        # save minimum loss and its x if reset_min is true
        if self.reset_min:
            if self.loss1 <= self.loss_min:
                self.loss_min = self.loss1
                self.x_min = self.x

        ### normalizing gradient and setting input
        g_normed = self.g / np.linalg.norm(self.g)
        # self.input = self.g
        self.input = g_normed

        # update all momentum
        for i_m in range(self.beta_size):
            self.m_list[i_m] = (1 - self.alpha_list[i_m]) * self.m_list[i_m] + self.alpha_list[i_m] * self.input  # updating momentum
            self.m2_list[i_m] = (1 - self.alpha_list[i_m]/2) * self.m2_list[i_m] + self.alpha_list[i_m]/2 * self.input  # updating momentum
            self.ind_m_m1 = self.ind_m
            self.ind_m = np.linalg.norm(self.m_list[self.m_i])


        # calculating output
        self.output = self.m_list[self.m_i] / np.linalg.norm(self.m_list[self.m_i])  # normalized momentum
        if self.t == 1 and self.find_lr:  # calculate step_m if this is step 1
            n_step_min = 1
            self.step_m = self._find_step_m(closure, self.step_m, self.output, n_step_min=n_step_min)  # finding initial step size
            self.lr = self.step_m

        # calculating momentum of momentum indicator
        for i_m in range(0, self.beta_size):
            m_normed = self.m_list[i_m] / np.linalg.norm(self.m_list[i_m])
            self.mm_list[i_m] = (1 -self.alpha_list[i_m]) * self.mm_list[i_m] + self.alpha_list[i_m] * m_normed
            self.ind_mm_m1 = self.ind_mm
            self.ind_mm = np.linalg.norm(self.mm_list[i_m])

        # adjusting step_m
        scale_step_m = 1 + self.alpha_list[self.m_i]
        if self.ind_mm > 0.7:
            if self.ind_mm > self.ind_mm_m1 and self.ind_m > self.ind_m_m1:
                self.step_m = self.step_m * scale_step_m
        elif self.ind_mm < 0.3:
            if self.ind_m < self.ind_m_m1:
                self.step_m = self.step_m * self.ind_mm
                # resetting this indicator
                m_normed = self.m_list[self.m_i] / np.linalg.norm(self.m_list[self.m_i])
                self.mm_list[self.m_i] = self.alpha_list[self.m_i] * m_normed
                self.ind_mm = np.linalg.norm(self.mm_list[self.m_i])



        # self.sub_m2 = True
        # if self.sub_m2:
        #     if self.ind_mm < 0.3 and self.ind_mm2 > 0.7:
        #         self.m_list[self.m_i] = self.m2_list[self.m_i]



        # # calculating momentum of momentum indicators
        # for i_m in range(0, self.beta_size):
        #     m_normed = self.m_list[i_m] / np.linalg.norm(self.m_list[i_m])
        #
        #     self.mm_list[i_m] = (1 -self.alpha_list[i_m]) * self.mm_list[i_m] + self.alpha_list[i_m] * m_normed
        #     # self.vm_list[i_m] = (1- self.alpha_list[i_m]) * self.vm_list[i_m] + self.alpha_list[i_m] * 1
        #
        #     self.ind_mm_m1 = self.ind_mm
        #     # self.ind_mm = np.linalg.norm(self.mm_list[i_m]) / self.vm_list[i_m,0]
        #     self.ind_mm = np.linalg.norm(self.mm_list[i_m])
        #


        # # adjusting step_m
        # scale_step_m = 1 + self.alpha_list[self.m_i]
        # alpha = self.alpha_list[self.m_i]
        #
        # if self.ind_mm > 0.6:
        #         if self.ind_mm > self.ind_mm_m1 and self.ind_m > self.ind_m_m1:
        #             self.step_m = self.step_m * scale_step_m
        # elif self.ind_mm < 0.4:
        #     if (self.ind_m < (3 * alpha) ) and (self.ind_m < self.ind_m_m1) :
        #         self.step_m = self.step_m / 4
        #         # resetting this indicator
        #         m_normed = self.m_list[self.m_i] / np.linalg.norm(self.m_list[self.m_i])
        #         self.mm_list[self.m_i] = self.alpha_list[self.m_i] * m_normed
        #         self.vm_list[self.m_i] = self.alpha_list[self.m_i] * 1
        #         self.ind_mm = 1





        self.x_m1 = self.x
        # update x
        print("ALR-GDM update. momentum, step_m:", self.beta_list[self.m_i], self.step_m)
        self.x = self.x - self.output * self.step_m

        # calculating step for going to higher momentum
        for i_m in range(0,self.beta_size):
            self.ms_list[i_m] = (1- self.alpha_list[i_m]) * self.ms_list[i_m]  + self.alpha_list[i_m] * self.output * self.step_m

        # change momentum
        if self.m_i > 0 and self.ind_o_list[self.m_i, 0] < 0.25:
            self.m_i = self.m_i - 1
            self.delay_step = int(2 / self.alpha_list[self.m_i]) - 1
            # resetting this indicator
            self.mo_list[self.m_i] = 0
            self.vo_list[self.m_i] = 0
            for i_m in range(self.beta_size):  # reset all momentum to zero
                self.m_list[i_m] = 0
                self.ms_list[i_m] = 0
            if self.reset_min: # reset to minimum found till now
                self.x = self.x_min  # reset position to min found till now
        elif self.m_i < self.beta_size-1:
            if self.ind_o_list[self.m_i+1, 0] > self.ind_o_list[self.m_i, 0]:
                self.m_i = self.m_i + 1
                self.step_m = np.linalg.norm(self.ms_list[self.m_i])
                self.delay_step = int(2 / self.alpha_list[self.m_i]) - 1
                # resetting this indicator
                self.mo_list[self.m_i] = 0
                self.vo_list[self.m_i] = 0




        # ######## ouput indicators to file
        # if self.t == 1:
        #     self.loss0 = self.loss1
        #
        # scale = self.loss0
        #
        # ff = open('outputs/main/0-ind_m-ALR-GDM', 'a')
        # str_to_file = str(self.t) + "\t" + str(self.ind_m * scale) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        # ff = open('outputs/main/0-ind_mm-ALR-GDM', 'a')
        # str_to_file = str(self.t) + "\t" + str(self.ind_mm * scale) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        # ff = open('outputs/main/0-step_m-ALR-GDM', 'a')
        # str_to_file = str(self.t) + "\t" + str( scale * self.step_m / self.lr / 2 ) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        # # ff = open('outputs/main/0-beta_i-ALR-GDM', 'a')
        # # str_to_file = str(self.t) + "\t" + str(self.m_i * scale / (self.beta_size)) + "\n"
        # # ff.write(str_to_file)
        # # ff.close()
        #
        # ff = open('outputs/neural_network/train/0-ind_m-ALR-GDM', 'a')
        # str_to_file = str(self.t) + "\t" + str(self.ind_m * scale) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        # ff = open('outputs/neural_network/train/0-ind_mm-ALR-GDM', 'a')
        # str_to_file = str(self.t) + "\t" + str(self.ind_mm * scale) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        # ff = open('outputs/neural_network/train/0-step_m-ALR-GDM', 'a')
        # str_to_file = str(self.t) + "\t" + str( scale * self.step_m / self.lr / 2 ) + "\n"
        # ff.write(str_to_file)
        # # ff.close()
        # # ff = open('outputs/neural_network/train/0-beta_i-ALR-GDM', 'a')
        # # str_to_file = str(self.t) + "\t" + str(self.m_i * scale / (self.beta_size)) + "\n"
        # # ff.write(str_to_file)
        # # ff.close()
        #
        #
        # ff = open('outputs/neural_network_minibatch/train/0-ind_m-ALR-GDM', 'a')
        # str_to_file = str(self.t) + "\t" + str(self.ind_m * scale) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        # ff = open('outputs/neural_network_minibatch/train/0-ind_mm-ALR-GDM', 'a')
        # str_to_file = str(self.t) + "\t" + str(self.ind_mm * scale) + "\n"
        # ff.write(str_to_file)
        # ff.close()
        # ff = open('outputs/neural_network_minibatch/train/0-step_m-ALR-GDM', 'a')
        # str_to_file = str(self.t) + "\t" + str( scale * self.step_m / self.lr / 2 ) + "\n"
        # ff.write(str_to_file)
        # # ff.close()
        # # ff = open('outputs/neural_network_minibatch/train/0-beta_i-ALR-GDM', 'a')
        # # str_to_file = str(self.t) + "\t" + str(self.m_i * scale / (self.beta_size)) + "\n"
        # # ff.write(str_to_file)
        # # ff.close()

        # saving values for next step
        self.t = self.t + 1
        self.loss_m1 = self.loss1







    def step(self, closure):
        self._update_params(closure)



    def _find_step_g(self, closure, step, o):

        x_s = self.x[:]
        loss0 = self.loss1
        self.x = x_s - o * step
        loss2 = closure()

        n1 = 0
        while loss2 > loss0 and n1 < 10:
            step = step /2
            self.x = x_s - o * step
            loss2 = closure()
            self.x = x_s
            n1 = n1 + 1

        loss0 = closure()
        print("ALR-GDM _find_step_g: step_m is", self.step_m)
        print("ALR-GDM _find_step_g: step_g is set to", step)

        return step






    def _find_step_m(self, closure, step, o, n_step_min = 10):
        print("ALR-GDM, _find_step_m:")

        s = np.zeros(5)
        # stage1_step = np.array([])
        # stage1_loss = np.array([])
        # stage2_step = np.array([])
        # stage2_loss = np.array([])
        # stage3_step = np.array([])
        # stage3_loss = np.array([])

        x_s = self.x[:]
        loss0 = closure()
        stage0_step = np.array([0])
        stage0_loss = np.array([loss0])
        stage0_g = np.array(self.g)

        self.x = x_s - o * step
        loss2 = closure()
        self.x = x_s
        stage0_step = np.append(stage0_step, np.array([step]))
        stage0_loss = np.append(stage0_loss, np.array([loss2]))
        stage0_g = np.vstack((stage0_g, self.g))


        if loss2 <= loss0:
            n1 = 0
            while loss2 <= loss0 and n1 < 10:
                step = step * 10
                self.x = x_s - o * step
                loss2 = closure()
                stage0_step = np.append(stage0_step, np.array([step]))
                stage0_loss = np.append(stage0_loss, np.array([loss2]))
                stage0_g = np.vstack( (stage0_g,self.g) )
                self.x = x_s
                n1 = n1 + 1

            # print("loss2 <= loss0. stage0_step", stage0_step)
            # print("loss2 <= loss0. stage0_loss", stage0_loss)
        else:  # loss2 > loss0
            n1 = 0
            while loss2 > loss0 and n1 < 10:
                step = step / 10
                self.x = x_s - o * step
                loss2 = closure()
                stage0_step = np.append(stage0_step, np.array([step]))
                stage0_loss = np.append(stage0_loss, np.array([loss2]))
                stage0_g = np.vstack( (stage0_g,self.g) )
                self.x = x_s
                n1 = n1 + 1
            # print("loss2 > loss0. stage0_step", stage0_step)
            # print("loss2 > loss0. stage0_loss", stage0_loss)

        if n1 == 10:
            loss_min = np.amin(stage0_loss)
            min_i = np.where(stage0_loss == loss_min)
            ind_min = min_i[0][0]
            step_min = stage0_step[ind_min]
            sout = float('{:0.1e}'.format(step_min / n_step_min))
            ds = sout
            print("Step loop reached max iteration.")


        else: # loop did not reach max
            loss_min = np.amin(stage0_loss)
            min_i = np.where(stage0_loss == loss_min)
            ind_min = min_i[0][0]
            s[2] = stage0_step[ind_min]
            s[0] = s[2]/10
            s[4] = s[2] * 10
            s[1] = (s[0] + s[2]) / 2
            s[3] = (s[2] + s[4]) / 2
            ds = s[3] - s[1]

            n2 = 0
            while n2 < 10 and ds > s[2]/3:
                self.x = x_s - o * s[1]
                loss2 = closure()
                stage0_step = np.append(stage0_step, np.array([s[1]]))
                stage0_loss = np.append(stage0_loss, np.array([loss2]))
                l1 = loss2
                stage0_g = np.vstack( (stage0_g,self.g) )
                self.x = x_s

                self.x = x_s - o * s[3]
                loss2 = closure()
                stage0_step = np.append(stage0_step, np.array([s[3]]))
                stage0_loss = np.append(stage0_loss, np.array([loss2]))
                stage0_g = np.vstack( (stage0_g,self.g) )
                l3 = loss2
                self.x = x_s

                if loss_min < min(l1,l3):
                    s[0] = s[1]
                    s[2] = s[2]
                    s[4] = s[3]
                    s[1] = (s[0] + s[2]) / 2
                    s[3] = (s[2] + s[4]) / 2
                    ds = s[3] - s[1]

                elif l1 < l3:
                    s[4] = s[2]
                    s[2] = s[1]
                    s[0] = s[0]
                    s[1] = (s[0] + s[2]) / 2
                    s[3] = (s[2] + s[4]) / 2
                    ds = s[3] - s[1]
                else:
                    s[0] = s[2]
                    s[2] = s[3]
                    s[4] = s[4]
                    s[1] = (s[0] + s[2]) / 2
                    s[3] = (s[2] + s[4]) / 2
                    ds = s[3] - s[1]
                n2 = n2 + 1

            step_min = s[2]
            sout = float('{:0.1e}'.format(step_min / n_step_min))

        print("step list:", stage0_step)
        print("loss list:", stage0_loss)


        # import matplotlib.pyplot as plt
        # plt.plot(stage0_step, stage0_loss, 'bo')
        # plt.axvline(x=sout)
        # plt.xlabel('step')
        # plt.ylabel('loss')
        # plt.title('ALR-GDM step finder')
        # plt.show()

        print("step_min, error, step_out :", step_min , ds , sout)
        print("")
        self.loss1 = closure()

        return sout








# import numpy as np
#
#
# class abgd_vm():
#     def __init__(self, params, lr=0.01,  beta_list = [0,0.9,0.99], find_lr = True, reset_min = True, cal_step_g = True):
#
#         self.lr = lr  # learning rate
#
#         self.d = len(params)  # input dimension
#
#         self.x = np.zeros(self.d)
#         self.g = np.zeros(self.d)
#
#
#         ##### initialising parameters specific to the algorithm #######
#         exec(open("./optimiser_ALR_GDM_init.py").read())
#
#     def _update_params(self, closure):
#
#         ### normalizing gradient
#
#         # normalized g
#         g_normed = self.g / np.linalg.norm(self.g)  # normalized gradient
#         g0_gm1_dot = np.dot(g_normed, self.g_m1_normed)
#
#         # update all momentums
#         self.input = g_normed
#         # self.input = self.g
#         for i in range(self.beta_size):
#             self.m_list[i] = self.beta_list[i] * self.m_list[i] + (1- self.beta_list[i]) * self.input # updating momentum
#
#         # save minimum loss and its x if reset_min is true
#         if self.reset_min:
#             if self.loss1 <= self.loss_min:
#                 self.loss_min = self.loss1
#                 self.x_min = self.x
#
#         # caculate indicators for oscillation
#         if self.t == 1:
#             self.x_m1 = self.x + g_normed * 1e-12
#         d = self.x - self.x_m1
#         for i in range(0,self.beta_size):
#             if self.beta_list[i] == 0: # for mometum zero
#                 self.ind_o_list[i] = np.linalg.norm(g_normed + self.g_m1_normed)/2
#             else:  # for momentum nonzero
#                 self.md_list[i] = self.beta_list[i] * self.md_list[i]  + (1- self.beta_list[i]) * d
#                 self.vd_list[i] = self.beta_list[i] * self.vd_list[i]  + (1- self.beta_list[i]) * np.linalg.norm(d)
#                 self.ind_o_list[i] = np.linalg.norm(self.md_list[i]) / self.vd_list[i]
#
#         # adjusting steps
#         if self.beta_list[self.m_i] == 0: # for mometum zero
#             if self.ind_o_list[self.m_i,0] < 0.5:
#                 self.step_g = 0.5 * self.step_g
#                 self.g_delay = 1
#             elif self.ind_o_list[self.m_i,0] > 0.6 and self.g_delay < 1:
#                 self.step_g = 2.0 * self.step_g
#             else:
#                 self.g_delay = self.g_delay - 1
#         else: # for momentum nonzero
#             scale_step_m = 1 + (1-self.beta_list[self.m_i]) * 0.1
#             if self.ind_o_list[self.m_i,0] < 0.5: # decrease step for low indicator
#                 self.step_m = self.step_m / scale_step_m
#             elif self.ind_o_list[self.m_i,0] > 0.6: # increase step for high indicator
#                 self.step_m = self.step_m * scale_step_m
#
#
#         # condition to push momentum down
#         if self.m_i > 0 and self.ind_o_list[self.m_i,0] < 0.2:
#             print(">>>>>  push down, momentum is now", self.beta_list[self.m_i])
#             self.m_i = self.m_i - 1
#             if self.beta_list[self.m_i] == 0 and self.cal_step_g: # if momentum is zero recalculate step_g
#                 self.step_g = self._find_step_g(closure, self.step_m, self.m_list[self.m_i])
#                 self.g_delay = 2
#             else:
#                 self.step_g = self.step_g
#             if self.reset_min: # reset x to x_min of minimum loss till now
#                 self.x = self.x_min
#                 self.loss1 = closure()
#                 # resetting all momentum
#                 for i in range(self.beta_size):
#                     self.m_list[i] =  (1 - self.beta_list[i]) * g_normed
#
#         # condition to push momentum up
#         if self.m_i < self.beta_size-1:
#             if self.ind_o_list[self.m_i+1, 0] > 0.4:
#                 print(">>>>>  push up, momentum is now", self.beta_list[self.m_i])
#                 self.m_i = self.m_i + 1
#
#
#
#         self.x_m1[:] = self.x[:] # saving previous x
#
#         if self.beta_list[self.m_i] == 0: # for momentum zero
#             print("ABGDvm gradient update. step_g:", self.step_g)
#             self.x = self.x - g_normed * self.step_g # advance one step
#         else:
#             m_normed = self.m_list[self.m_i] / np.linalg.norm(self.m_list[self.m_i])  # normalized momentum
#             if self.t == 1 and self.find_lr:  # calculate step_m if this is step 1
#                 n_step_min = 1
#                 self.step_m = self._find_step_m(closure, self.step_m, m_normed, n_step_min)  # finding initial step size
#                 self.lr = self.step_m
#             print("ABGDvm momentum update. step_m:", self.step_m)
#             # advance one step
#             self.x = self.x - m_normed * self.step_m
#
#
#
#
#         ######## ouput indicators to file
#         if self.t == 1:
#             self.loss0 = self.loss1
#
#         # scale_main = self.loss0
#         # ff = open('outputs/main/0-ind_o-vm', 'a')
#         # str_to_file = str(self.t) + "\t" + str(self.ind_o_list[self.m_i, 0] * scale_main) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#         # ff = open('outputs/main/0-beta_i-vm', 'a')
#         # str_to_file = str(self.t) + "\t" + str(self.m_i * scale_main / (self.beta_size)) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#
#         # scale_nn = self.loss0
#         # ff = open('outputs/neural_network/train/0-ind_o-vm', 'a')
#         # str_to_file = str(self.t) + "\t" + str(self.ind_o_list[self.m_i, 0] * scale_nn) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#         # ff = open('outputs/neural_network/train/0-beta_i-vm', 'a')
#         # str_to_file = str(self.t) + "\t" + str(self.m_i * scale_nn / (self.beta_size)) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#         # ff = open('outputs/neural_network/train/0-step_m-vm', 'a')
#         # str_to_file = str(self.t) + "\t" + str( scale_nn * self.step_m / self.lr / 2 ) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#
#         scale_nn = self.loss0
#         ff = open('outputs/neural_network_minibatch/train/0-ind_o-vm', 'a')
#         str_to_file = str(self.t) + "\t" + str(self.ind_o_list[self.m_i, 0] * scale_nn) + "\n"
#         ff.write(str_to_file)
#         ff.close()
#         ff = open('outputs/neural_network_minibatch/train/0-beta_i-vm', 'a')
#         str_to_file = str(self.t) + "\t" + str(self.m_i * scale_nn / (self.beta_size)) + "\n"
#         ff.write(str_to_file)
#         ff.close()
#         ff = open('outputs/neural_network_minibatch/train/0-step_m-vm', 'a')
#         str_to_file = str(self.t) + "\t" + str( scale_nn * self.step_m / self.lr / 2 ) + "\n"
#         ff.write(str_to_file)
#         ff.close()
#
#
#         # saving values for next step
#         self.g0_gm1_dot_m1 = g0_gm1_dot
#         self.g_m1_normed = g_normed
#         self.t = self.t + 1
#
#
#
#
#
#
#     def step(self, closure):
#         self._update_params(closure)
#
#
#
#     def _find_step_g(self, closure, step, o):
#
#         x_s = self.x[:]
#         loss0 = self.loss1
#         self.x = x_s - o * step
#         loss2 = closure()
#
#         n1 = 0
#         while loss2 > loss0 and n1 < 10:
#             step = step /2
#             self.x = x_s - o * step
#             loss2 = closure()
#             self.x = x_s
#             n1 = n1 + 1
#
#         loss0 = closure()
#         print("_find_step_g: step_m is", self.step_m)
#         print("_find_step_g: step_g is set to", step)
#
#         return step
#
#
#
#
#
#
#     def _find_step_m(self, closure, step, o, n_step_min = 10):
#         print("ABGDvm, _find_step_m:")
#
#         s = np.zeros(5)
#         # stage1_step = np.array([])
#         # stage1_loss = np.array([])
#         # stage2_step = np.array([])
#         # stage2_loss = np.array([])
#         # stage3_step = np.array([])
#         # stage3_loss = np.array([])
#
#         x_s = self.x[:]
#         loss0 = closure()
#         stage0_step = np.array([0])
#         stage0_loss = np.array([loss0])
#         stage0_g = np.array(self.g)
#
#         self.x = x_s - o * step
#         loss2 = closure()
#         self.x = x_s
#         stage0_step = np.append(stage0_step, np.array([step]))
#         stage0_loss = np.append(stage0_loss, np.array([loss2]))
#         stage0_g = np.vstack((stage0_g, self.g))
#
#
#         if loss2 <= loss0:
#             n1 = 0
#             while loss2 <= loss0 and n1 < 10:
#                 step = step * 10
#                 self.x = x_s - o * step
#                 loss2 = closure()
#                 stage0_step = np.append(stage0_step, np.array([step]))
#                 stage0_loss = np.append(stage0_loss, np.array([loss2]))
#                 stage0_g = np.vstack( (stage0_g,self.g) )
#                 self.x = x_s
#                 n1 = n1 + 1
#
#             # print("loss2 <= loss0. stage0_step", stage0_step)
#             # print("loss2 <= loss0. stage0_loss", stage0_loss)
#         else:  # loss2 > loss0
#             n1 = 0
#             while loss2 > loss0 and n1 < 10:
#                 step = step / 10
#                 self.x = x_s - o * step
#                 loss2 = closure()
#                 stage0_step = np.append(stage0_step, np.array([step]))
#                 stage0_loss = np.append(stage0_loss, np.array([loss2]))
#                 stage0_g = np.vstack( (stage0_g,self.g) )
#                 self.x = x_s
#                 n1 = n1 + 1
#             # print("loss2 > loss0. stage0_step", stage0_step)
#             # print("loss2 > loss0. stage0_loss", stage0_loss)
#
#         if n1 == 10:
#             loss_min = np.amin(stage0_loss)
#             min_i = np.where(stage0_loss == loss_min)
#             ind_min = min_i[0][0]
#             step_min = stage0_step[ind_min]
#             sout = float('{:0.1e}'.format(step_min / n_step_min))
#             ds = sout
#             print("Step loop reached max iteration.")
#
#
#         else: # loop did not reach max
#             loss_min = np.amin(stage0_loss)
#             min_i = np.where(stage0_loss == loss_min)
#             ind_min = min_i[0][0]
#             s[2] = stage0_step[ind_min]
#             s[0] = s[2]/10
#             s[4] = s[2] * 10
#             s[1] = (s[0] + s[2]) / 2
#             s[3] = (s[2] + s[4]) / 2
#             ds = s[3] - s[1]
#
#             n2 = 0
#             while n2 < 10 and ds > s[2]/3:
#                 self.x = x_s - o * s[1]
#                 loss2 = closure()
#                 stage0_step = np.append(stage0_step, np.array([s[1]]))
#                 stage0_loss = np.append(stage0_loss, np.array([loss2]))
#                 l1 = loss2
#                 stage0_g = np.vstack( (stage0_g,self.g) )
#                 self.x = x_s
#
#                 self.x = x_s - o * s[3]
#                 loss2 = closure()
#                 stage0_step = np.append(stage0_step, np.array([s[3]]))
#                 stage0_loss = np.append(stage0_loss, np.array([loss2]))
#                 stage0_g = np.vstack( (stage0_g,self.g) )
#                 l3 = loss2
#                 self.x = x_s
#
#                 if loss_min < min(l1,l3):
#                     s[0] = s[1]
#                     s[2] = s[2]
#                     s[4] = s[3]
#                     s[1] = (s[0] + s[2]) / 2
#                     s[3] = (s[2] + s[4]) / 2
#                     ds = s[3] - s[1]
#
#                 elif l1 < l3:
#                     s[4] = s[2]
#                     s[2] = s[1]
#                     s[0] = s[0]
#                     s[1] = (s[0] + s[2]) / 2
#                     s[3] = (s[2] + s[4]) / 2
#                     ds = s[3] - s[1]
#                 else:
#                     s[0] = s[2]
#                     s[2] = s[3]
#                     s[4] = s[4]
#                     s[1] = (s[0] + s[2]) / 2
#                     s[3] = (s[2] + s[4]) / 2
#                     ds = s[3] - s[1]
#                 n2 = n2 + 1
#
#             step_min = s[2]
#             sout = float('{:0.1e}'.format(step_min / n_step_min))
#
#         print("step list:", stage0_step)
#         print("loss list:", stage0_loss)
#
#
#         # import matplotlib.pyplot as plt
#         # plt.plot(stage0_step, stage0_loss, 'bo')
#         # plt.axvline(x=sout)
#         # plt.xlabel('step')
#         # plt.ylabel('loss')
#         # plt.title('ABGDvm step finder')
#         # plt.show()
#
#         print("step_min, error, step_out :", step_min , ds , sout)
#         print("")
#         loss0 = closure()
#
#         return sout
#
#
#
