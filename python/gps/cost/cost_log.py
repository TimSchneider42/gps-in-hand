import numpy as np
from typing import Tuple

from gps.cost.cost import Cost
from gps.cost.cost_unary_function import CostUnaryFunction


class CostLog(CostUnaryFunction):
    """
    Natural logarithm of given cost function.
    """

    def __init__(self, input_cost: Cost):
        """

        :param input_cost: Cost function to take natural logarithm of
        """
        super(CostLog, self).__init__(input_cost)

    def func(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.log(x), 1.0 / x, -1.0 / (x ** 2)

    @property
    def function_name(self) -> str:
        return "log"
