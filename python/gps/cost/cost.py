""" This file defines the base cost class. """
import abc
from abc import ABC
from typing import Tuple, Union
import numpy as np

from gps.sample import Sample


class Cost(ABC):
    @abc.abstractmethod
    def eval(self, sample: Sample) -> Tuple[np.ndarray, ...]:
        """
        Evaluate cost function and derivatives
        :param sample: A single sample
        :return:  (l, lx, lu, lxx, luu, lux) where x denotes the state, u the action and l the cost function.
                  lx denotes the derivative of l with respect to x, lxx the second derivative of l with respect to x
                  and so on. Each entry of the returned tuple is a matrix with a first dimension of T where T is the
                  total number of time steps of the sample.
        """
        pass

    @staticmethod
    def _to_cost(element: Union["Cost", float]):
        from gps.cost import CostScalar
        if isinstance(element, Cost):
            return element
        return CostScalar(element)

    def __add__(self, other: Union["Cost", float]):
        from gps.cost import CostSum
        c = Cost._to_cost(other)
        return CostSum([self, c])

    def __radd__(self, other: Union["Cost", float]):
        from gps.cost import CostSum
        c = Cost._to_cost(other)
        return CostSum([c, self])

    def __mul__(self, other: Union["Cost", float]):
        from gps.cost import CostProduct
        c = Cost._to_cost(other)
        return CostProduct([self, c])

    def __rmul__(self, other: Union["Cost", float]):
        from gps.cost import CostProduct
        c = Cost._to_cost(other)
        return CostProduct([c, self])

    def __neg__(self):
        return -1 * self

    def __sub__(self, other: Union["Cost", float]):
        c = Cost._to_cost(other)
        return self + (-c)

    def __rsub__(self, other: Union["Cost", float]):
        c = Cost._to_cost(other)
        return c + (-self)

    def __call__(self, sample: Sample):
        return self.eval(sample)
