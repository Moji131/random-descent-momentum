# An example of a recursive function to
# find the factorial of a number
import numpy as np

s = np.array([1, -0.10, +10.1, -3 ])
s_sign = np.sign(s)


g_0_m1 = np.array([-1, 1, 1])
g_1_0 = np.array([-1, -1, 1])
d = 3
step_mult = np.ones(d) * 2.0
step_mult = np.where(g_0_m1 == -1.0, 1.0, step_mult)
step_mult = np.where(g_1_0 == -1.0, 0.5, step_mult)

print(step_mult)




