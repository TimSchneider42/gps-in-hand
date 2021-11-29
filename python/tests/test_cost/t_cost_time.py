from collections import namedtuple

from gps.algorithm.cost import CostScalar, CostTimeDependant

cost_list = [(t, (CostScalar(c) + 3) * 5) for t, c in [(2, 2), (7, 3), (0, 1)]]

cost = CostTimeDependant(cost_list)

Sample = namedtuple("Sample", ("state_dimensions", "action_dimensions", "time_steps"))

print(cost)

for v, d in zip(cost.eval(Sample(1, 1, 10)), ["l", "lx", "lu", "lxx", "luu", "lux"]):
    print()
    print("{0}: {1}".format(d, v))
