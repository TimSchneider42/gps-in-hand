import numpy as np
from typing import List, Tuple

from gps.cost.cost import Cost
from gps.sample import Sample


class CostMin(Cost):
    """
    Computes the minimum of the given cost functions.
    """

    def __init__(self, c: List[Cost]):
        assert len(c) > 0
        self._c = c

    def eval(self, sample: Sample) -> Tuple[np.ndarray, ...]:
        """
        Evaluate cost function and derivatives
        :param sample: A single sample
        :return:  (l, lx, lu, lxx, luu, lux) where x denotes the state, u the action and l the cost function.
                  lx denotes the derivative of l with respect to x, lxx the second derivative of l with respect to x
                  and so on. Each entry of the returned tuple is a matrix with a first dimension of T where T is the
                  total number of time steps of the sample.
        """
        results = [np.array(e) for e in zip(*[c.eval(sample) for c in self._c])]
        min_indices = np.argmin(results[0], axis=0)
        return tuple(np.swapaxes(r, 0, 1)[range(sample.time_steps), min_indices] for r in results)

    def __repr__(self) -> str:
        return f"min({', '.join(map(str, self._c))})"
