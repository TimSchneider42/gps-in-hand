import operator
from typing import List, Iterable, Tuple
import numpy as np

from functools import reduce

from gps.cost.cost import Cost
from gps.sample import Sample


class CostProduct(Cost):
    """
    Multiplies any number of given cost functions.
    """

    def __init__(self, c: List[Cost]):
        self._c = c

    def eval(self, sample: Sample) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate cost function and derivatives
        :param sample: A single sample
        :return:  (l, lx, lu, lxx, luu, lux) where x denotes the state, u the action and l the cost function.
                  lx denotes the derivative of l with respect to x, lxx the second derivative of l with respect to x
                  and so on. Each entry of the returned tuple is a matrix with a first dimension of T where T is the
                  total number of time steps of the sample.
        """

        def mul(iterable: Iterable) -> float:
            """
            Multiplies all values in iterable.
            :param iterable: Iterable containing all values to be multiplied
            :return: The product of all values in iterable
            """
            return reduce(operator.mul, iterable, 1)

        l, lx, lu, lxx, luu, lux = zip(*[c.eval(sample) for c in self._c])

        ts = sample.time_steps
        du = sample.action_dimensions
        dx = sample.state_dimensions

        final_l = mul(l)
        final_lx = np.zeros((ts, dx))
        final_lu = np.zeros((ts, du))
        final_lxx = np.zeros((ts, dx, dx))
        final_luu = np.zeros((ts, du, du))
        final_lux = np.zeros((ts, du, dx))
        for i in range(len(self._c)):
            # sum over all factors except c_i
            l_product_without_i = mul([l[k] for k in range(len(l)) if k != i])  # type: np.ndarray
            expand_dims_i = np.expand_dims(l_product_without_i, axis=-1)
            repeat_state_i = np.repeat(expand_dims_i, dx, axis=-1)
            repeat_action_i = np.repeat(expand_dims_i, du, axis=-1)
            expand_dims_state_i = np.expand_dims(repeat_state_i, axis=-1)
            expand_dims_action_i = np.expand_dims(repeat_action_i, axis=-1)
            repeat_state_state_i = np.repeat(expand_dims_state_i, dx, axis=-1)
            repeat_action_action_i = np.repeat(expand_dims_action_i, du, axis=-1)
            repeat_action_state_i = np.repeat(expand_dims_action_i, dx, axis=-1)
            final_lx += lx[i] * repeat_state_i
            final_lu += lu[i] * repeat_action_i
            final_lxx += lxx[i] * repeat_state_state_i
            final_luu += luu[i] * repeat_action_action_i
            final_lux += lux[i] * repeat_action_state_i
            for j in range(i + 1, len(self._c)):
                # sum over all factors except c_i and c_j
                l_product_without_ij = mul([l[k] for k in range(len(l)) if k != i and k != j])  # type: np.ndarray
                expand_dims_ij = np.expand_dims(l_product_without_ij, axis=-1)
                repeat_state_ij = np.repeat(expand_dims_ij, dx, axis=-1)
                repeat_action_ij = np.repeat(expand_dims_ij, du, axis=-1)
                expand_dims_state_ij = np.expand_dims(repeat_state_ij, axis=-1)
                expand_dims_action_ij = np.expand_dims(repeat_action_ij, axis=-1)
                repeat_state_state_ij = np.repeat(expand_dims_state_ij, dx, axis=-1)
                repeat_action_action_ij = np.repeat(expand_dims_action_ij, du, axis=-1)
                repeat_action_state_ij = np.repeat(expand_dims_action_ij, dx, axis=-1)
                final_lxx += (np.expand_dims(lx[i], -1) * np.expand_dims(lx[j], -2) +
                              np.expand_dims(lx[j], -1) * np.expand_dims(lx[i], -2)) * repeat_state_state_ij
                final_luu += (np.expand_dims(lu[i], -1) * np.expand_dims(lu[j], -2) +
                              np.expand_dims(lu[j], -1) * np.expand_dims(lu[i], -2)) * repeat_action_action_ij
                final_lux += (np.expand_dims(lu[i], -1) * np.expand_dims(lx[j], -2) +
                              np.expand_dims(lu[j], -1) * np.expand_dims(lx[i], -2)) * repeat_action_state_ij
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux

    def __repr__(self) -> str:
        return " * ".join(map(str, self._c))
