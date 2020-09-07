import torch
import numpy as np
import copy
from torch.autograd import Variable
import copy

import optimiser_ABGDvm



class abgd_vm(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, beta_list = [0,0.9,0.95], find_lr = True, reset_min = True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}."
                             " It must be non-negative.".format(lr))
        defaults = dict(lr=lr)
        super(abgd_vm, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']

        self.lr = lr

        self.d = 0
        self.find_d()

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDvm_init.py").read())

    _update_params = optimiser_ABGDvm.abgd_vm._update_params
    _find_step_g = optimiser_ABGDvm.abgd_vm._find_step_g
    _find_step_m = optimiser_ABGDvm.abgd_vm._find_step_m



    @torch.no_grad()
    def step(self, closure=None):

        self.params_to_np()
        self._update_params(closure)
        self.np_to_params()

        return






    def find_d(self):
        for p in self._params:
            d_tensor = len(p.size())
            if d_tensor == 1:
                for i in range(p.size()[0]):
                    self.d = self.d + 1
            elif d_tensor == 2:
                for i in range(p.size()[0]):
                    for j in range(p.size()[1]):
                        self.d = self.d + 1
            elif d_tensor == 3:
                for i in range(p.size()[0]):
                    for j in range(p.size()[1]):
                        for k in range(p.size()[2]):
                            self.d = self.d + 1
            else:
                for i in range(p.size()[0]):
                    for j in range(p.size()[1]):
                        for k in range(p.size()[2]):
                            for l in range(p.size()[3]):
                                self.d = self.d + 1

    def params_to_np(self):
        n = 0
        for p in self._params:
            d = len(p.size())
            if d == 1:
                for i in p:
                    self.x[n] = i.data.item()
                    n = n + 1
            elif d == 2:
                for i in p:
                    for j in i:
                        self.x[n] = j.data.item()
                        n = n + 1
            elif d ==3:
                for i in p:
                    for j in i:
                        for k in j:
                            self.x[n] = k.data.item()
                            n = n + 1
            elif d == 4:
                for i in p:
                    for j in i:
                        for k in j:
                            for l in k:
                                self.x[n] = l.data.item()
                                n = n + 1

        n = 0
        for p in self._params:
            d = len(p.size())
            if d == 1:
                for i in p.grad:
                    self.g[n] = i.data.item()
                    n = n + 1
            elif d ==2:
                for i in p.grad:
                    for j in i:
                        self.g[n] = j.data.item()
                        n = n + 1
            elif d == 3:
                for i in p.grad:
                    for j in i:
                        for k in j:
                            self.g[n] = k.data.item()
                            n = n + 1
            else:
                for i in p.grad:
                    for j in i:
                        for k in j:
                            for l in k:
                                self.g[n] = l.data.item()
                                n = n + 1



    def np_to_params(self):
        n = 0
        for q in self._params:
            p = copy.copy(q)
            d = len(p.size())
            if d == 1:
                for i in range(p.size()[0]):
                    p.data[i] = self.x[n]
                    n = n + 1
            elif d == 2:
                for i in range(p.size()[0]):
                    for j in range(p.size()[1]):
                        p[i,j] = self.x[n]
                        n = n + 1
            elif d == 3:
                for i in range(p.size()[0]):
                    for j in range(p.size()[1]):
                        for k in range(p.size()[2]):
                            p[i,j,k] = self.x[n]
                            n = n + 1
            else:
                for i in range(p.size()[0]):
                    for j in range(p.size()[1]):
                        for k in range(p.size()[2]):
                            for l in range(p.size()[3]):
                                p[i,j,k, l] = self.x[n]
                                n = n + 1
            with torch.enable_grad():
                q = p








