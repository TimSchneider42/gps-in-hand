""" This file defines a cost sum of arbitrary other costs. """
import numpy as np
from typing import List, Tuple

from gps.cost.cost import Cost
from gps.sample import Sample

class CostSum(Cost):
    """
    Adds any number of given costs.
    """

    def __init__(self, c: List[Cost]):
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
        l, lx, lu, lxx, luu, lux = zip(*[c.eval(sample) for c in self._c])
        final_l = sum(l)
        final_lx = sum(lx)
        final_lu = sum(lu)
        final_lxx = sum(lxx)
        final_luu = sum(luu)
        final_lux = sum(lux)
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux

    def __repr__(self) -> str:
        return "(" + " + ".join(map(str, self._c)) + ")"
