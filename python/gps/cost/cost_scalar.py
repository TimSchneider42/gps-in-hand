""" This file defines a constant scalar cost value. """
import numpy as np
from typing import Tuple, Union

from gps.cost.cost import Cost
from gps.sample import Sample


class CostScalar(Cost):
    """
    Represents a constant scalar value.
    """

    def __init__(self, value: Union[float, np.ndarray]):
        """

        :param value:   Value of this scalar. If an array is given here, the i-th value of that array will be used for
                        the i-th time step. Should the array be shorter as the number of time steps in a sample, the
                        last array value will be repeated.
        """
        self._value = value
        assert not isinstance(self._value, np.ndarray) or self._value.shape[0] >= 1, "At least one value is required"

    def eval(self, sample: Sample) -> Tuple[np.ndarray, ...]:
        """
        Evaluate cost function and derivatives
        :param sample: A single sample
        :return:  (l, lx, lu, lxx, luu, lux) where x denotes the state, u the action and l the cost function.
                  lx denotes the derivative of l with respect to x, lxx the second derivative of l with respect to x
                  and so on. Each entry of the returned tuple is a matrix with a first dimension of T where T is the
                  total number of time steps of the sample.
        """
        ts = sample.time_steps
        du = sample.action_dimensions
        dx = sample.state_dimensions

        if isinstance(self._value, np.ndarray):
            l = np.ones(ts) * self._value[-1]
            min_len = min(len(l), len(self._value))
            l[:min_len] = self._value[:min_len]
        else:
            l = np.ones(ts) * self._value
        lx = np.zeros((ts, dx))
        lu = np.zeros((ts, du))
        lxx = np.zeros((ts, dx, dx))
        luu = np.zeros((ts, du, du))
        lux = np.zeros((ts, du, dx))
        return l, lx, lu, lxx, luu, lux

    def __repr__(self) -> str:
        return str(self._value)
