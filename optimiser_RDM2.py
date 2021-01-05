

import random
import numpy as np
import random as rn

class rdm2():
    def __init__(self, params, lr=0.01, momentum = 0.95):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_RDM2_init.py").read())


    def _update_params(self, closure):

        n_sample = 1
        n_skip = n_sample * 3

        if self.t ==1:
            self.dx = self.lr * np.ones(self.d) * 0.01
            self.x = self.x - self.dx
            self.loss_m1 =  0
            for i in range(n_sample):
                self.loss_m1 = self.loss_m1 + closure()
            self.loss_m1 = self.loss_m1 / n_sample
            self.x = self.x + self.dx

            self.x = self.x + self.dx
            self.loss1 = 0
            for i in range(n_sample):
                self.loss1 = self.loss1 + closure()
            self.loss1 = self.loss1 / n_sample

            dx = self.dx
            dx_norm = np.linalg.norm(dx)
            dx_normed = dx / dx_norm
            ghat_norm = (self.loss1 - self.loss_m1) / dx_norm
            ghat1 = ghat_norm * dx_normed

            self.m = (1 - self.beta) * ghat1

        if self.t%n_skip == 0:


            self.loss1 = 0
            for i in range(n_sample):
                self.loss1 = self.loss1 + closure()
            self.loss1 = self.loss1 / n_sample

            dx = self.dx
            dx_norm = np.linalg.norm(dx)
            dx_normed = dx / dx_norm
            ghat_norm = (self.loss1 - self.loss_m1) / dx_norm
            ghat1 = ghat_norm * dx_normed

            # self.m = self.beta * self.m + (1 - self.beta) * ghat
            # self.m = ghat

            norm_m = np.linalg.norm(self.m)
            if norm_m * self.step_m < 0.0001:  # limit on how small sampling distance can be for experimental cases
                print("RDM2, norm_m too small", norm_m)
                norm_m = 0.0001 / self.step_m
            vec = np.array([rn.gauss(0, 1) for i in range(self.d)])
            dot = np.dot(vec, self.m)
            par = self.m * dot / norm_m / norm_m
            perpen = vec - par
            mag = np.sum(perpen ** 2) ** .5
            perpen = perpen / mag * norm_m
            delta = perpen / 1

            f1=0
            f2=0
            for i in range(n_sample):
                self.x = self.x - delta * self.step_m
                f1 = f1 + closure()
                self.x = self.x + 2 * delta * self.step_m
                f2 = f2 + closure()
                self.x = self.x - delta * self.step_m

            f1 = f1 / n_sample
            f2 = f2 / n_sample

            dx = 2 * delta * self.step_m
            dx_norm = np.linalg.norm(dx)
            dx_normed = dx / dx_norm
            ghat_norm = (f2 - f1) / dx_norm
            ghat2 = ghat_norm * dx_normed

            ghat = ghat1 + ghat2


            self.m = self.beta * self.m + (1 - self.beta) * ghat
            p = self.m

            self.x = self.x - p * self.step_m
            self.dx = - p * self.step_m

            self.t = self.t + 1
            self.loss_m1 = self.loss1
        else:
            self.t = self.t + 1





    # def _update_params(self, closure):
    #
    #     n_sample = 100
    #
    #     self.loss1 =  0
    #     for i in range(n_sample):
    #         self.loss1 = self.loss1 + closure()
    #     self.loss1 = self.loss1 / n_sample
    #
    #     dx = self.dx
    #     dx_norm = np.linalg.norm(dx)
    #     dx_normed = dx / dx_norm
    #     ghat_norm = (self.loss1 - self.loss_m1) / dx_norm
    #     ghat1 = ghat_norm * dx_normed
    #
    #     # self.m = self.beta * self.m + (1-self.beta) *  ghat
    #     # self.m = ghat
    #
    #     norm_m = np.linalg.norm(self.m)
    #     if norm_m * self.step_m < 0.0001: # limit on how small sampling distance can be for experimental cases
    #         print("norm_m too small", norm_m)
    #         norm_m = 0.0001 / self.step_m
    #     vec = np.array([rn.gauss(0, 1) for i in range(self.d)])
    #     dot = np.dot(vec,self.m)
    #     par = self.m * dot / norm_m / norm_m
    #     perpen = vec - par
    #     mag = np.sum(perpen ** 2) ** .5
    #     perpen = perpen / mag * norm_m
    #     delta = perpen / 10
    #
    #     f1 = 0
    #     f2 = 0
    #     for i in range(n_sample):
    #         self.x = self.x + delta
    #         f1 = f1 + closure()
    #         self.x = self.x - 2 * delta
    #         f2 = f1 + closure()
    #         self.x = self.x + delta
    #     f1 = f1 / n_sample
    #     f2 = f2 / n_sample
    #
    #     dx = 2 * delta
    #     dx_norm = np.linalg.norm(dx)
    #     dx_normed = dx / dx_norm
    #     ghat_norm = (f1 - f2) / dx_norm
    #     ghat2 = ghat_norm * dx_normed
    #
    #     # ghat = ghat1
    #     ghat = ghat1 + ghat2
    #     self.m = self.beta * self.m + (1 - self.beta) * ghat
    #     p = self.m
    #
    #     self.x = self.x - p * self.step_m
    #     self.dx = - p * self.step_m
    #
    #     self.t = self.t + 1
    #     self.loss_m1 = self.loss1


    #
    # def _update_params(self, closure):
    #
    #
    #     dx = self.dx
    #     dx_norm = np.linalg.norm(dx)
    #     dx_normed = dx / dx_norm
    #     ghat_norm = (self.loss_m1 - self.loss1) / dx_norm
    #     ghat = ghat_norm * dx_normed
    #
    #     norm_m = np.linalg.norm(self.m)
    #     if norm_m < 0.0001: # limit on how small sampling distance can be for experimental cases
    #         norm_m = 0.0001
    #
    #     vec = np.array([rn.gauss(0, 1) for i in range(self.d)])
    #     mag = np.sum(vec ** 2) ** .5
    #     vec = vec / mag * norm_m
    #
    #     if norm_m < 0.0001: # limit on how small sampling distance can be for experimental cases
    #         self.m = vec
    #
    #     self.x = self.x + vec * self.step_m
    #
    #     loss_r = closure()
    #     dx = vec * self.step_m
    #     dx_norm = np.linalg.norm(dx)
    #     dx_normed = dx / dx_norm
    #     ghat_norm = (loss_r - self.loss1) / dx_norm
    #     ghat = ghat + ghat_norm * dx_normed
    #
    #     self.m = self.beta * self.m + (1 - self.beta) * ghat
    #
    #     self.x = self.x - vec * self.step_m - self.m * self.step_m
    #     self.dx = - self.m * self.step_m
    #
    #     self.t = self.t + 1
    #     self.loss_m1 = self.loss1




    def step(self, closure):
        self._update_params(closure)





    def _find_step_m(self, closure, step, o, n_step_min = 10):
        print("GDM, _find_step_m:")

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
        # plt.title('ABGDvm step finder')
        # plt.show()

        print("step_min, error, step_out :", step_min , ds , sout)
        print("")
        loss0 = closure()

        return sout




















#
#
#
# import numpy as np
#
# class rdm():
#     def __init__(self, params, lr=0.01, momentum = 0.95):
#
#         self.lr = lr
#
#         self.d = len(params)
#
#         self.x = np.zeros(self.d)
#         self.g = np.zeros(self.d)
#
#         ##### initialising parameters specific to the algorithm #######
#         exec(open("./optimiser_RDM_init.py").read())
#
#
#
#     def _update_params(self, closure):
#
#         if self.t == 1:
#             l = closure()
#             p = np.random.random(self.d)
#             p_normed = p / np.linalg.norm(p)
#         else:
#             l = closure()
#             g = (l-self.l_m1) * (self.x - self.x_m1) / self.step_g /self.step_g
#             self.m = self.momentum * self.m +  g
#             m_normed = self.m / np.linalg.norm(self.m)
#
#             s = np.random.random(self.d)
#             s_normed = s / np.linalg.norm(s)
#             s_m_dot = np.dot(s_normed, m_normed)
#             s_perp = s_normed - m_normed * s_m_dot
#
#             p =  m_normed + s_perp
#             p_normed = p / np.linalg.norm(p)
#
#         self.x_m1 = self.x
#         self.x = self.x - p_normed * self.step_g
#
#         self.l_m1 = l
#
#
#
#
#     def step(self, closure):
#         # if self.t == 1:
#         #     self._find_lr(closure)
#         #     self.t = 1
#         self.step_g = self.lr
#         self._update_params(closure)
#
#     def _find_lr(self, closure):
#         self.step_g = self.lr / 1000
#         xx = self.x[:]
#         loss0 = closure()
#         loss2 = loss0
#         loss1 = loss0
#
#         while not loss2 > loss1:
#             loss1 = loss2
#             self.step_g = 10 * self.step_g
#             self._update_params(closure)
#             loss2 = closure()
#             self.x = xx
#
#         loss0 = closure()
#         self.step_g = float('{:0.1e}'.format(
#             (self.step_g / 10 + (loss0 - loss1) * (self.step_g - self.step_g / 10) / (loss2 - loss1)) / 20))
#         self.lr = self.step_g
#
#
#
#
#
#
#
