##### initialising parameters specific to the algorithm #######

self.beta = momentum
self.m = np.zeros(self.d)
self.step_m = self.lr
self.t = 1