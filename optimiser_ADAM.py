import numpy as np

class adam():
    def __init__(self, params, lr=0.01):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ADAM_init.py").read())



    def _update_params(self, closure):
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1.0e-08

        self.m = beta_1 * self.m + (1 - beta_1) * self.g
        self.v = beta_2 * self.v + (1 - beta_2) * np.power(self.g, 2)
        m_hat = self.m / (1 - beta_1**self.t)
        v_hat = self.v / (1 - beta_2**self.t)
        p = m_hat / (np.sqrt(v_hat) + epsilon)

        if self.t == 1:
            n_step_min = 1
            self.step_m = self._find_step_m(closure, self.step_m, p, n_step_min)  # finding initial step size
            self.lr = self.step_m

        self.x = self.x - self.step_m * p

        self.t = self.t + 1





    def step(self, closure):
        self._update_params(closure)



    def _find_step_m(self, closure, step, o, n_step_min = 10):
        print("ADAM, _find_step_m:")

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
# class adam():
#     def __init__(self, params, lr=0.01):
#
#         self.lr = lr
#
#         self.d = len(params)
#
#         self.x = np.zeros(self.d)
#         self.g = np.zeros(self.d)
#
#         ##### initialising parameters specific to the algorithm #######
#         exec(open("./optimiser_ADAM_init.py").read())
#
#
#
#     def _update_params(self):
#         beta_1 = 0.9
#         beta_2 = 0.999
#         epsilon = 1.0e-08
#
#         self.m = beta_1 * self.m + (1 - beta_1) * self.g
#         self.v = beta_2 * self.v + (1 - beta_2) * np.power(self.g, 2)
#         m_hat = self.m / (1 - beta_1**self.t)
#         v_hat = self.v / (1 - beta_2**self.t)
#
#         g_0_sign = np.sign(self.g)  # sign of the components of gradient
#         a_sign = np.sign((m_hat) / (np.sqrt(v_hat) + epsilon))
#         m_g_0 = a_sign * g_0_sign # check if momentum and gradient have the same sign.
#         self.lr = np.where( m_g_0 == -1, self.lr*0.5, self.lr*2) #state 1. going against the gradient because of momentum. half the step.
#
#
#         self.x = self.x - self.lr * a_sign
#
#         # self.x = self.x - (self.lr * m_hat) / (np.sqrt(v_hat) + epsilon)
#
#
#
#     def step(self, closure = None):
#         self._update_params()
#
#
#
#






