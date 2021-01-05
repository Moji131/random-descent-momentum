import numpy as np


class ALR_GD():
    def __init__(self, params, lr=0.01):

        self.lr = lr  # learning rate

        self.d = len(params)  # input dimension



        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)


        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ALR_GD_init.py").read())



    def _update_params(self, closure):

        ### normalizing gradient
        g_0_norm = np.linalg.norm(self.g)  # norm of the gradient
        g_0_normed = self.g / g_0_norm  # normalized gradient

        if self.t == 1:  # calculate step_m if this is step 1
            n_step_min = 0.5
            self.output = g_0_normed
            self.step_g = self._find_step_m(closure, self.step_g, self.output, n_step_min)  # finding initial step size
            self.lr = self.step_g

        ### caculating step_g_g for the next iteration
        g_0_m1_dot = np.dot(g_0_normed, self.g_m1_normed)
        if g_0_m1_dot < 0:
            self.step_g = self.step_g * 0.5
        elif self.g_0_m1_dot_m1 < 0:
            self.step_g = self.step_g * 1.0
        else:
            self.step_g = self.step_g * 2.0

        ### updating parameters
        self.x = self.x - g_0_normed * self.step_g

        ### save values for the next step
        self.g_m1_normed = g_0_normed
        self.g_0_m1_dot_m1 = g_0_m1_dot

        self.t = self.t + 1




    def step(self, closure):
        self._update_params(closure)




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
    #     self.step_g = float('{:0.1e}'.format(  ( self.step_g/10 + (loss0-loss1)*(self.step_g-self.step_g/10)/(loss2-loss1) ) / 20 ))
    #     self.lr = self.step_g



    def _find_step_m(self, closure, step, o, n_step_min = 10):
        print("ALR-GD, _find_step_m:")

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


