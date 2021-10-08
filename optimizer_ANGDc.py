


import numpy as np

class opt():
    def __init__(self, params, lr=0.01, min_step_r=2**20, max_step_r=2**20 ):

        self.lr = lr

        self.d = len(params)

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)
        
        self.min_step_r = min_step_r
        self.max_step_r = max_step_r

        self.step_g = np.ones(self.d) * lr  # inital step_g (can be anything)
        self.g_m1_sign = np.ones(self.d)   # get the sign of the components
        self.g_m1_m2 = np.ones(self.d)  # initialising product of sign of gradient of step_g 0 and -1


        self.t = 1



    def _update_params(self, closure):

        g_0_sign = np.sign(self.g)  # sign of the components
        g_0_m1 = g_0_sign * self.g_m1_sign  # product of sign of gradient of step 1 and 0

        if self.t == 1:  # calculate step_m if this is step 1
            n_step_min = 0.5
            self.output = g_0_sign
            self.step_g = self.lr * np.ones(self.d)


        step_g_mult = np.ones(self.d) * 2.0  #  setting all step_g multipliers to 2
        step_g_mult = np.where(self.g_m1_m2 == -1.0, 1.0, step_g_mult) #  if g_0_m1 is -1 change step_g_mult component to 1
        step_g_mult = np.where(g_0_m1 == -1.0, 0.5, step_g_mult)  #  if g_1_0 is -1 change step_g_mult component to 0.5
        self.step_g = self.step_g * step_g_mult  # use step_g_mult to update current step_g sizes

        self.step_g = np.where(self.step_g < (self.lr/self.min_step_r), (self.lr/self.min_step_r), self.step_g)  # minimum step size check
        self.step_g = np.where(self.step_g > (self.lr*self.max_step_r), (self.lr*self.max_step_r), self.step_g)  # maximum step size check

        self.x = self.x - g_0_sign * self.step_g  # advance x one step_g

        #  preparation for  the next step
        self.g_m1_sign = g_0_sign  # get the sign of the components
        self.g_m1_m2 = g_0_m1

        self.t = self.t + 1




    def step(self, closure):
        self._update_params(closure)






    def _find_step_m(self, closure, step, o, n_step_min = 10):
        print("ALR-GDc, _find_step_m:")

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


