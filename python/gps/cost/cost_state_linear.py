from typing import Tuple, Optional, Any

import numpy as np

from gps.cost.cost import Cost
from gps.sample import Sample


class CostStateLinear(Cost):
    """
    Computes linear state penalties.
    costs(t) = w^T * X(t)
    where X(t) is the state vector at time t and w is the specified weight vector.
    """

    def __init__(self, sensor: Any, weights: Optional[np.ndarray] = None):
        """

        :param sensor:  Sensor for which to compute l1/l2 distance.
        :param weights: Weights for each component of the selected sensor (Default: 1 for each component).
        """
        self._weights = weights
        self._sensor = sensor

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

        state_slice = sample.state_packer.label_slices[self._sensor]

        # Extract sensor state
        x = sample.full_state[self._sensor]
        dim_sensor = x.shape[1]

        weights = self._weights if self._weights is not None else np.ones(dim_sensor)

        assert weights.shape == (dim_sensor,)

        # the last value will remain zero, because we don't have an action here
        l = np.sum(weights * x, axis=1)
        lu = np.zeros((ts, du))
        lx = np.zeros((ts, dx))
        if self._sensor in sample.state_packer.labels:
            lx[:, state_slice] = weights * np.ones((ts, dim_sensor))
        luu = np.zeros((ts, du, du))
        lxx = np.zeros((ts, dx, dx))
        lux = np.zeros((ts, du, dx))
        return l, lx, lu, lxx, luu, lux

    def __repr__(self):
        if self._weights is None:
            return "c_state_linear({})".format(self._sensor)
        return "c_state_linear({}, {})".format(self._weights, self._sensor)
