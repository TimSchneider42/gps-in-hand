""" This file defines the torque (action) cost. """
from typing import Tuple, Optional

import numpy as np

from gps.cost.cost import Cost
from gps.sample import Sample


class CostAction(Cost):
    """
    Computes quadratic action penalties.
    costs(t) = 0.5 * U(t)^T * diag(w) * U(t)
    where U(t) is the action vector at time t and w is the specified weight vector.
    """

    def __init__(self, weights: Optional[np.ndarray] = None):
        """

        :param weights: Weights for each component of the action vector (Default: 1 for each component).
        """
        self._weights = weights

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

        weights = self._weights if self._weights is not None else np.ones(du)

        l = np.zeros(ts)
        # the last value will remain zero, because we don't have an action here
        l[0:-1] = 0.5 * np.sum(weights * (sample.actions ** 2), axis=1)
        lu = np.zeros((ts, du))
        lu[0:-1, :] = weights * sample.actions
        lx = np.zeros((ts, dx))
        luu = np.zeros((ts, du, du))
        luu[0:-1, :, :] = np.tile(np.diag(weights), [ts - 1, 1, 1])
        lxx = np.zeros((ts, dx, dx))
        lux = np.zeros((ts, du, dx))
        return l, lx, lu, lxx, luu, lux

    def __repr__(self):
        if self._weights is None:
            return "c_action(u)"
        return "c_action({}, u)".format(self._weights)
