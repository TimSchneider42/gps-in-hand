from typing import Tuple, List, Union
import numpy as np

from gps.cost.cost import Cost
from gps.sample import Sample


class CostTimeVariant(Cost):
    """
    Defines a time variant cost function. This lets you decide which cost function to use at each point in time.
    """

    def __init__(self, cost_list: List[Tuple[int, Union[Cost, float]]]):
        """

        :param cost_list:   Cost function and respective activation time step. At each time step only the cost function
                            most recently activated cost function is active. At least one activation time step must be
                            0.
        """
        self._cost_list = sorted([(t, Cost._to_cost(c)) for t, c in cost_list], key=lambda t: t[0])
        assert any(t == 0 for t, _ in self._cost_list), "At least one activation time step must be 0."
        assert all(t[0] >= 0 for t in self._cost_list)

    def eval(self, sample: Sample) -> Tuple[np.ndarray, ...]:
        """
        Evaluate cost function and derivatives
        :param sample: A single sample
        :return:  (l, lx, lu, lxx, luu, lux) where x denotes the state, u the action and l the cost function.
                  lx denotes the derivative of l with respect to x, lxx the second derivative of l with respect to x
                  and so on. Each entry of the returned tuple is a matrix with a first dimension of T where T is the
                  total number of time steps of the sample.
        """
        evaluations = [c.eval(sample) for t, c in self._cost_list if t < sample.time_steps]
        final: List[np.ndarray] = []
        for i in range(6):
            current = np.empty(evaluations[0][i].shape)
            start_ts = 0
            for j in range(len(self._cost_list) - 1):
                start_ts = self._cost_list[j][0]
                end_ts = self._cost_list[j + 1][0]
                if start_ts >= sample.time_steps:
                    break
                current[start_ts:end_ts + 1] = evaluations[j][i][start_ts:end_ts + 1]
            if start_ts < sample.time_steps:
                current[self._cost_list[-1][0]:] = evaluations[-1][i][self._cost_list[-1][0]:]
            final.append(current)
        return tuple(final)

    def __repr__(self) -> str:
        entries = ["{} <= t <= {}: {}".format(
            self._cost_list[i][0], self._cost_list[i + 1][0] - 1, self._cost_list[i][1])
                      for i in range(len(self._cost_list) - 1)] + \
                  ["{} <= t: {}".format(self._cost_list[-1][0], self._cost_list[-1][1])]
        return "{{{}}}".format("; ".join(entries))
