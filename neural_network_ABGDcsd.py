import torch
import numpy as np
import copy
import optimiser_ABGDcsd
from torch.autograd import Variable
import copy

class abgd_csd(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, min_step_r=2**4, max_step_r=2**20):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}."
                             " It must be non-negative.".format(lr))
        defaults = dict(lr=lr)
        super(abgd_csd, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']

        self.lr = lr

        self.d = 0
        self.find_d()

        self.x = np.zeros(self.d)
        self.g = np.zeros(self.d)

        ##### initialising parameters specific to the algorithm #######
        exec(open("./optimiser_ABGDcsd_init.py").read())




    _update_params = optimiser_ABGDcsd.abgd_csd._update_params


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
            else:
                for i in range(p.size()[0]):
                    for i in range(p.size()[1]):
                        self.d = self.d + 1

    def params_to_np(self):
        n = 0
        for p in self._params:
            d = len(p.size())
            if d == 1:
                for i in p:
                    self.x[n] = i.data.item()
                    n = n + 1
            else:
                for i in p:
                    for j in i:
                        self.x[n] = j.data.item()
                        n = n + 1

        n = 0
        for p in self._params:
            d = len(p.size())
            if d == 1:
                for i in p.grad:
                    self.g[n] = i.data.item()
                    n = n + 1
            else:
                for i in p.grad:
                    for j in i:
                        self.g[n] = j.data.item()
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
            else:
                for i in range(p.size()[0]):
                    for j in range(p.size()[1]):
                        p[i,j] = self.x[n]
                        n = n + 1
            with torch.enable_grad():
                q = p








