##### initialising parameters specific to the algorithm #######

self.step_g = lr  # inital step_g (can be anything)
self.step_drift = lr
# self.min_step_r = min_step_r # minimum step size ratio with respect to learning rate
# self.max_step_r = max_step_r # maximum step size ratio with respect to learning rate


self.mom = np.zeros(self.d) # momentum
self.mom_normed = np.zeros(self.d) # momentum normalised
self.g_0_mom_dot_m1 = 1 # dot product of momentum with gradient in the -1 step

self.g_drift__m1_normed = np.zeros(self.d) # drift in the previous drift step

self.g_m1_normed = np.zeros(self.d) # gradient normalised


self.pair_con = 0 # the value of dot product that under that is considered a a pair of gradients pointing at each other

self.drift_con = -0.95 # the value of the dot product of gradient that lower than that activates the drift move
self.drift = drift # Boolean for doing or not doing the drift move
self.drift_reject_con = -0.3 # value of dot product of drift and new gradient that lower than that rejects the move and undoes the drift move
self.g_drift_m1_normed = np.zeros(self.d) # drift vector normalised for the previous step
self.g_drift_0_m1_dot_m1 = 1  # dot product of the drift and the previous gradient for the -1 step



# momentum value. Turns momentum off if it is not above zero and under one
if momentum > 0 and momentum < 1:
    self.mom_con = True
    self.beta = momentum
else:
    self.mom_con = False
    self.beta = 0


#
# self.step_g_r = 2**5
# self.step_drift_r = 2**5
