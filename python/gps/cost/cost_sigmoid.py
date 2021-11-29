import numpy as np
from typing import Tuple

from gps.cost.cost import Cost
from gps.cost.cost_unary_function import CostUnaryFunction


class CostSigmoid(CostUnaryFunction):
    """
    Sigmoid function: 1 / (1 + exp(-k(x - x0)))
    """

    def __init__(self, input_cost: Cost, x_offset: float = 0.0, steepness: float = 1.0):
        """

        :param input_cost:  Cost function to take logistic function of.
        :param x_offset:    Midpoint x0 of the function.
        :param steepness:   Steepness k of the function.
        """
        super(CostSigmoid, self).__init__(input_cost)
        self.__x_offset = x_offset
        self.__steepness = steepness

    def func(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Avoiding overflows here
        f = 1.0 / (1.0 + np.exp(np.maximum(np.minimum(-self.__steepness * (x - self.__x_offset), 700.0), -700.0)))
        fx = self.__steepness * f * (1.0 - f)
        fxx = self.__steepness * fx * (1.0 - 2.0 * f)
        return f, fx, fxx

    @property
    def function_name(self) -> str:
        return "sigmoid"
