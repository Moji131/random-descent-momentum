import torch
import copy

class abgd(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}."
                             " It must be non-negative.".format(lr))
        defaults = dict(lr=lr)
        super(abgd, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']

        self.min_step = lr / 10**10
        self.step_init = lr
        self.max_step = lr * 10**10

        self._params_step = copy.deepcopy(self._params)
        for p in self._params_step:
            p.data = torch.ones(p.data.size()) * self.step_init

        self._params_g_0_sign = copy.deepcopy(self._params)
        for p in self._params_g_0_sign:
            p.data = torch.zeros(p.data.size())

        self._params_g_1_sign = copy.deepcopy(self._params)
        for p in self._params_g_1_sign:
            p.data = torch.zeros(p.data.size())

        self._params_g_0_m1 = copy.deepcopy(self._params)
        for p in self._params_g_0_m1:
            p.data = torch.ones(p.data.size())

        self._params_g_1_0 = copy.deepcopy(self._params)
        for p in self._params_g_1_0:
            p.data = torch.ones(p.data.size())

        self._params_step_mult = copy.deepcopy(self._params)




    def _update_params(self):
        """Update parameters given an update direction and step-size. """

        for p, q in zip(self._params, self._params_g_1_sign):
            # print(p.grad.data)
            q.data = torch.sign(p.grad.data)
            # print(q.data)

        for p, q, r in zip(self._params_g_1_0, self._params_g_1_sign, self._params_g_0_sign):
            # print(q.data, r.data)
            p.data = q.data * r.data
            # print(p.data)

        for p, q, r, s in zip(self._params_step_mult , self._params_g_0_m1, self._params_g_1_0, self._params_step):
            p.data = torch.ones(p.data.size()) * 2.0
            # print(p.data)
            p.data = torch.where(q.data == -1.0, torch.ones(p.data.size()) * 1.0, p.data)
            # print(p.data)
            p.data = torch.where(r.data == -1.0, torch.ones(p.data.size()) * 0.5, p.data)
            # print(p.data)
            # print(s.data)
            s.data = s.data * p.data

            s.data = torch.where(s.data > self.max_step , torch.ones(p.data.size()) * self.max_step, s.data)
            s.data = torch.where(s.data < self.min_step , torch.ones(p.data.size()) * self.min_step, s.data)


            # print(s.data)


        # updating values
        for p, q, r in zip(self._params, self._params_step, self._params_g_1_sign):
            p.data = p.data - q.data * r.data
            # print("w",p.data)
            # print("step", q.data)
            # print("sign", r.data)


        self._params_g_0_sign = copy.deepcopy(self._params_g_1_sign)
        self._params_g_0_m1 = copy.deepcopy(self._params_g_1_0)

        # for p in self._params:
        #     print("w", p.data)
            

    @torch.no_grad()
    def step(self, closure = None):
        """ Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and returns
                the loss.
        """

        loss = None

        self._update_params()

        return loss