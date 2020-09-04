import numpy as np


class abgd_cm():
    def __init__(self, params, lr=0.01, beta_list = [0,0.9,0.95], find_lr = True, reset_min = True):

        self.lr = lr  # learning rate

        self.d = len(params)  # input dimension

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)


        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDcm_init.py").read())


    def _update_params(self, closure):

        # save minimum loss and its x if reset_min is true
        if self.reset_min:
            if self.loss1 <= self.loss_min:
                self.loss_min = self.loss1
                self.x_min = self.x

        ### normalizing gradient and setting input
        g_normed = self.g / np.linalg.norm(self.g)  # normalized gradient
        self.input = g_normed

        # update all momentum
        for i_m in range(self.beta_size):
            self.m_list[i_m] = (1 - self.alpha_list[i_m]) * self.m_list[i_m] + self.alpha_list[i_m] * self.input  # updating momentum
            self.v_list[i_m] = (1 - self.alpha2_list[i_m]) * self.v_list[i_m] + self.alpha2_list[i_m] * np.power(self.input, 2)

        # calculating output
        m_hat = self.m_list[self.m_i] / (1 - self.beta_list[self.m_i] ** self.t)
        v_hat = self.v_list[self.m_i] / (1 - self.beta2_list[self.m_i] ** self.t)
        epsilon = 1.0e-08
        p = m_hat / (np.sqrt(v_hat) + epsilon)
        self.output = p / np.linalg.norm(p)
        if self.t == 1 and self.find_lr:  # calculate step_m if this is step 1
            n_step_min = 1
            self.step_m = self._find_step_m(closure, self.step_m, self.output, n_step_min)  # finding initial step size
            self.lr = self.step_m

        # calculating indicators
        for i_m in range(0, self.beta_size):
            self.ind_d_list_m1[i_m, :] = self.ind_d_list[i_m, :]
            self.md_list[i_m] = (1 - self.alpha_list[i_m] / 2.0) * self.md_list[i_m] + self.alpha_list[
                i_m] / 2.0 * self.output
            self.vd_list[i_m] = (1 - self.alpha_list[i_m] / 2.0) * self.vd_list[i_m] + self.alpha_list[
                i_m] / 2.0 * np.linalg.norm(self.output)
            self.ind_d_list[i_m] = np.linalg.norm(self.md_list[i_m]) / self.vd_list[i_m]

        # adjusting step_m
        scale_step_m = 1 + self.alpha_list[self.m_i]
        if self.ind_d_list[self.m_i, 0] < 0.3:
            self.step_m = self.step_m / 2
            self.delay = 1
        elif self.ind_d_list[self.m_i, 0] < self.ind_d_list_m1[self.m_i, 0]:  # decrease step for low indicator
            self.step_m = self.step_m / scale_step_m
            self.delay = 1
        elif self.delay > 0:  # increase step for high indicator
            self.delay = self.delay - 1
        else:
            self.step_m = self.step_m * scale_step_m

        # update x
        print("ABGDcm update. momentum, step_m:", self.beta_list[self.m_i], self.step_m)
        self.x = self.x - self.output * self.step_m

        # calculating step for going to higher momentum
        for i_m in range(0, self.beta_size):
            self.ms_list[i_m] = (1 - self.alpha_list[i_m]) * self.ms_list[i_m] + self.alpha_list[
                i_m] * self.output * self.step_m

        # decide if change momentum
        if self.m_i > 0 and self.ind_d_list[self.m_i, 0] < 0.3:
            self.m_i = self.m_i - 1
            if self.reset_min:  # reset to minimum found till now
                self.x = self.x_min  # reset position to min found till now
                for i_m in range(self.beta_size):  # reset all momentum to zero
                    self.ms_list[i_m] = 0
        elif self.m_i < self.beta_size - 1:
            if self.ind_d_list[self.m_i + 1, 0] > self.ind_d_list[self.m_i, 0]:
                self.m_i = self.m_i + 1
                self.step_m = np.linalg.norm(self.ms_list[self.m_i])


        ######## ouput indicators to file
        if self.t == 1:
            self.loss0 = self.loss1

        scale_main = self.loss0
        ff = open('outputs/main/0-ind_d-cm', 'a')
        str_to_file = str(self.t) + "\t" + str(self.ind_d_list[self.m_i, 0] * scale_main) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/main/0-beta_i-cm', 'a')
        str_to_file = str(self.t) + "\t" + str(self.m_i * scale_main / (self.beta_size)) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/main/0-step_m-cm', 'a')
        str_to_file = str(self.t) + "\t" + str( scale_main * self.step_m / self.lr / 2 ) + "\n"
        ff.write(str_to_file)
        ff.close()

        scale_nn = self.loss0
        ff = open('outputs/neural_network/train/0-ind_d-cm', 'a')
        str_to_file = str(self.t) + "\t" + str(self.ind_d_list[self.m_i, 0] * scale_nn) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/neural_network/train/0-beta_i-cm', 'a')
        str_to_file = str(self.t) + "\t" + str(self.m_i * scale_nn / (self.beta_size)) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/neural_network/train/0-step_m-cm', 'a')
        str_to_file = str(self.t) + "\t" + str( scale_nn * self.step_m / self.lr / 2 ) + "\n"
        ff.write(str_to_file)
        ff.close()

        scale_nn = self.loss0
        ff = open('outputs/neural_network_minibatch/train/0-ind_d-cm', 'a')
        str_to_file = str(self.t) + "\t" + str(self.ind_d_list[self.m_i, 0] * scale_nn) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/neural_network_minibatch/train/0-beta_i-cm', 'a')
        str_to_file = str(self.t) + "\t" + str(self.m_i * scale_nn / (self.beta_size)) + "\n"
        ff.write(str_to_file)
        ff.close()
        ff = open('outputs/neural_network_minibatch/train/0-step_m-cm', 'a')
        str_to_file = str(self.t) + "\t" + str( scale_nn * self.step_m / self.lr / 2 ) + "\n"
        ff.write(str_to_file)
        ff.close()


        # saving values for next step
        self.t = self.t + 1



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
        print("ABGDcm _find_step_g: step_m is", self.step_m)
        print("ABGDcm _find_step_g: step_g is set to", step)

        return step







    def _find_step_m(self, closure, step, o, n_step_min = 10):
        print("ABGDcm, _find_step_m:")

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
        # plt.title('ABGDcm step finder')
        # plt.show()

        print("step_min, error, step_out :", step_min , ds , sout)
        print("")
        loss0 = closure()

        return sout





# import numpy as np
#
#
# class abgd_cm():
#     def __init__(self, params, lr=0.01, beta_list = [0,0.9, 0.93], beta2_list = [0.999,0.999,0.999], find_lr = True, reset_min = True, cal_step_g = True):
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
#         exec(open("./optimiser_ABGDcm_init.py").read())
#
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
#         gg = g_normed
#         # gg = self.g
#         for i in range(self.beta_size):
#             self.m_list[i] = self.beta_list[i] * self.m_list[i] + (1- self.beta_list[i]) * gg # updating momentum
#             self.v_list[i] = self.beta2_list[i] * self.v_list[i] + (1 - self.beta2_list[i]) * np.power(gg, 2)
#
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
#                 self.ind_d_list[i] = np.linalg.norm(g_normed + self.g_m1_normed)/2
#             else:  # for momentum nonzero
#                 self.md_list[i] = self.beta_list[i] * self.md_list[i]  + (1- self.beta_list[i]) * d
#                 self.vd_list[i] = self.beta_list[i] * self.vd_list[i]  + (1- self.beta_list[i]) * np.linalg.norm(d)
#                 self.ind_d_list[i] = np.linalg.norm(self.md_list[i]) / self.vd_list[i]
#
#         # adjusting steps
#         if self.beta_list[self.m_i] == 0: # for mometum zero
#             print(self.ind_d_list[self.m_i,0])
#             if self.ind_d_list[self.m_i,0] < 0.5:
#                 self.step_g = 0.5 * self.step_g
#                 self.delay = 1
#             elif self.ind_d_list[self.m_i,0] > 0.6 and self.delay < 1:
#                 self.step_g = 2.0 * self.step_g
#             else:
#                 self.delay = self.delay - 1
#         else: # for momentum nonzero
#             scale_step_m = 1 + (1-self.beta_list[self.m_i]) * 0.1
#             # scale_step_m = 1
#             if self.ind_d_list[self.m_i,0] < 0.5: # decrease step for low indicator
#                 self.step_m = self.step_m / scale_step_m
#             elif self.ind_d_list[self.m_i,0] > 0.6: # increase step for high indicator
#                 self.step_m = self.step_m * scale_step_m
#
#
#         # condition to push momentum down
#         if self.m_i > 0 and self.ind_d_list[self.m_i,0] < 0.2:
#             print(">>>>>  push down, momentum is now", self.beta_list[self.m_i])
#             self.m_i = self.m_i - 1
#             if self.beta_list[self.m_i] == 0 and self.cal_step_g: # if momentum is zero recalculate step_g
#                 self.step_g = self._find_step_g(closure, self.step_m, self.m_list[self.m_i])
#                 self.delay = 1
#             else:
#                 self.step_g = self.step_g
#             if self.reset_min: # reset x to x_min of minimum loss till now
#                 self.x = self.x_min
#                 self.loss1 = closure()
#                 # resetting all momentum
#                 for i in range(self.beta_size):
#                     self.m_list[i] =  (1 - self.beta_list[i]) * g_normed
#                     self.v_list[i] = (1 - self.beta2_list[i]) * np.power(gg, 2)
#
#         # condition to push momentum up
#         if self.m_i < self.beta_size-1:
#             if self.ind_d_list[self.m_i+1, 0] > 0.4:
#                 print(">>>>>  push up, momentum is now", self.beta_list[self.m_i])
#                 self.m_i = self.m_i + 1
#
#
#
#         self.x_m1[:] = self.x[:] # saving previous x
#
#         if self.beta_list[self.m_i] == 0: # for momentum zero
#             print("ABGDcm gradient update. step_g:", self.step_g)
#             self.x = self.x - g_normed * self.step_g # advance one step
#         else:
#             m_hat = self.m_list[self.beta_i] / (1 - self.beta_list[self.beta_i] ** self.t)
#             v_hat = self.v_list[self.beta_i] / (1 - self.beta2_list[self.beta_i] ** self.t)
#             epsilon = 1.0e-08
#             p = m_hat / (np.sqrt(v_hat) + epsilon)
#             p_normed = p / np.linalg.norm(p)
#
#             if self.t == 1:
#                  n_step_min = 3
#                  self.step_m = self._find_step_m(closure, self.step_m, p_normed, n_step_min)  # finding initial step size
#                  self.lr = self.step_m
#             print("ABGDcm momentum update. step_m:", self.step_m)
#             self.x = self.x - p_normed * self.step_m  # advance one step
#
#
#
#
#         ######## ouput indicators to file
#         if self.t == 1:
#             self.loss0 = self.loss1
#
#         # scale_main = self.loss0
#         # ff = open('outputs/main/0-ind_d-cm', 'a')
#         # str_to_file = str(self.t) + "\t" + str(self.ind_d_list[self.m_i, 0] * scale_main) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#         # ff = open('outputs/main/0-beta_i-cm', 'a')
#         # str_to_file = str(self.t) + "\t" + str(self.m_i * scale_main / (self.beta_size)) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#
#         # scale_nn = self.loss0
#         # ff = open('outputs/neural_network/train/0-ind_d-cm', 'a')
#         # str_to_file = str(self.t) + "\t" + str(self.ind_d_list[self.m_i, 0] * scale_nn) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#         # ff = open('outputs/neural_network/train/0-beta_i-cm', 'a')
#         # str_to_file = str(self.t) + "\t" + str(self.m_i * scale_nn / (self.beta_size)) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#         #
#         # ff = open('outputs/neural_network/train/0-step_m-cm', 'a')
#         # str_to_file = str(self.t) + "\t" + str( scale_nn * self.step_m / self.lr / 2 ) + "\n"
#         # ff.write(str_to_file)
#         # ff.close()
#
#         scale_nn = self.loss0
#         ff = open('outputs/neural_network_minibatch/train/0-ind_d-cm', 'a')
#         str_to_file = str(self.t) + "\t" + str(self.ind_d_list[self.m_i, 0] * scale_nn) + "\n"
#         ff.write(str_to_file)
#         ff.close()
#         ff = open('outputs/neural_network_minibatch/train/0-beta_i-cm', 'a')
#         str_to_file = str(self.t) + "\t" + str(self.m_i * scale_nn / (self.beta_size)) + "\n"
#         ff.write(str_to_file)
#         ff.close()
#
#         ff = open('outputs/neural_network_minibatch/train/0-step_m-cm', 'a')
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
#
#     def _find_step_m(self, closure, step, o, n_step_min = 10):
#         print("ABGDcm, _find_step_m:")
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
#         # plt.title('ABGDcm step finder')
#         # plt.show()
#
#         print("step_min, error, step_out :", step_min , ds , sout)
#         print("")
#         loss0 = closure()
#
#         return sout
#
