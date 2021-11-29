import numpy as np
from typing import Tuple

from gps.cost.cost import Cost
from gps.cost.cost_unary_function import CostUnaryFunction


class CostSqrt(CostUnaryFunction):
    """
    Natural logarithm of given cost function.
    """

    def __init__(self, input_cost: Cost):
        """

        :param input_cost: Cost function to take natural logarithm of
        """
        super(CostSqrt, self).__init__(input_cost)

    def func(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sqrt = np.sqrt(x)
        return sqrt, 1.0 / (2.0 * sqrt), -1.0 / (4 * sqrt ** 3)

    @property
    def function_name(self) -> str:
        return "sqrt"
