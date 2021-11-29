from gps.cost.cost_sigmoid import CostSigmoid

import numpy as np

import matplotlib.pyplot as plt

sig = CostSigmoid(None, x_offset=0.025, steepness=-155.0)

l, lx, lxx = sig.func(np.arange(0.0, 0.1, 0.001))

plt.plot(np.arange(0.0, 0.1, 0.001), l)
plt.title("l")
plt.figure()
plt.plot(np.arange(0.0, 0.1, 0.001), lx)
plt.title("lx")
plt.figure()
plt.plot(np.arange(0.0, 0.1, 0.001), lxx)
plt.title("lxx")

plt.show()