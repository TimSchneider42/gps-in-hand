from abc import abstractmethod

import numpy as np
from typing import Tuple

from gps.cost.cost import Cost
from gps.sample import Sample


class CostUnaryFunction(Cost):
    """
    Function with a scalar input and output range.
    """

    def __init__(self, input_cost: Cost):
        """

        :param input_cost:  Input cost for this function.
        """
        self.__input_cost = input_cost

    @abstractmethod
    def func(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        :param x: Array of scalars to evaluate function on
        :return: A three-tuple containing:
            - An array of function values for each point in x
            - An array of 1st order derivatives for each point in x
            - An array of 2nd order derivatives for each point in x
        """
        pass

    @property
    @abstractmethod
    def function_name(self) -> str:
        pass

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
        dx = sample.state_dimensions
        du = sample.action_dimensions
        l, lx, lu, lxx, luu, lux = self.__input_cost.eval(sample)
        f, fl, fll = self.func(l)
        final_l = f
        final_lx = (lx.T * fl).T
        final_lu = (lu.T * fl).T
        final_lxx = (lxx.T * fl + (lx.reshape((ts, dx, 1)) @ lx.reshape((ts, 1, dx))).T * fll).T
        final_luu = (luu.T * fl + (lu.reshape((ts, du, 1)) @ lu.reshape((ts, 1, du))).T * fll).T
        final_lux = (lux.T * fl + (lu.reshape((ts, du, 1)) @ lx.reshape((ts, 1, dx))).T * fll).T
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux

    def __repr__(self) -> str:
        return f"{self.function_name}({self.__input_cost})"
