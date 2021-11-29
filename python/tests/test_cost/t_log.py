from gps.algorithm.cost import CostTargetState, CostAction
import numpy as np

from gps.cost.cost_log import CostLog
from gps.sample.sample_collator import SampleCollator
from gps.utility.labeled_data_packer import LabeledDataPacker

cs_0 = CostTargetState(0, np.array([1, 2]), np.array([2, 1]))
ca = CostAction(np.array([20, 1]))

state_packer = LabeledDataPacker([(0, 2), (1, 1)])

# dx1 = 1, dx2 = 2, dx3 = 3
# u1 = 5, u2 = 3
x0 = {
    0: np.array([0, 0]),
    1: np.array([3])
}
s = SampleCollator(x0, 2, 2, state_packer)

for t in range(s.time_steps - 1):
    s.add(np.array([5, 3]),
          {
              0: np.array([0, 0]),
              1: np.array([3])
          })

c = CostLog(ca)

print(c)

for v, d in zip(c.eval(s.finalize()), ["l", "lx", "lu", "lxx", "luu", "lux"]):
    print()
    print("{0}:\n{1}".format(d, v))
print("Act")
for v, d in zip(ca.eval(s.finalize()), ["l", "lx", "lu", "lxx", "luu", "lux"]):
    print()
    print("{0}:\n{1}".format(d, v))
print("State")
for v, d in zip(cs_0.eval(s.finalize()), ["l", "lx", "lu", "lxx", "luu", "lux"]):
    print()
    print("{0}:\n{1}".format(d, v))
