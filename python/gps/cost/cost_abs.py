import numpy as np
from typing import Tuple

from gps.cost import CostUnaryFunction
from gps.cost.cost import Cost


class CostAbs(CostUnaryFunction):
    """
    Computes the absolute value of the given cost functions.
    """

    def __init__(self, c: Cost):
        super(CostAbs, self).__init__(c)

    def func(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.abs(x), (x > 0) * 2 - 1, np.zeros(x.shape)

    @property
    def function_name(self) -> str:
        return "abs"
