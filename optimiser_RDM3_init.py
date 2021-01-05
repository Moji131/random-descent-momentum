##### initialising parameters specific to the algorithm #######

self.beta = momentum
self.m = np.zeros(self.d)
self.step_m = self.lr
self.t = 1
self.loss_m1 = 0
self.dx = np.ones(self.d) * 100