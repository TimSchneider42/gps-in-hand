import numpy as np
from typing import Tuple

from gps.cost.cost import Cost
from gps.cost.cost_unary_function import CostUnaryFunction


class CostArccos(CostUnaryFunction):
    """
    Arccosine of given cost function.
    """

    def __init__(self, input_cost: Cost):
        """

        :param input_cost: Cost function to take natural logarithm of
        """
        super(CostArccos, self).__init__(input_cost)

    def func(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ndx = 1.0 / np.sqrt(1 - x ** 2)  # type: np.ndarray
        return np.arccos(x), -ndx, x * ndx ** 3

    @property
    def function_name(self) -> str:
        return "log"
