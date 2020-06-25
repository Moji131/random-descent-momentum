##### initialising parameters specific to algorithm #######
self.min_step_r = min_step_r
self.max_step_r = max_step_r

self.step_g = np.ones(self.d) * lr  # inital step_g (can be anything)
self.step_g_mult_count = np.ones(self.d)
self.delay = 4



