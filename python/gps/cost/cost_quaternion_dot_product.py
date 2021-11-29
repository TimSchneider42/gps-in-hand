from typing import Any, Tuple

import numpy as np

from gps.cost.cost import Cost
from gps.sample import Sample


class CostQuaternionDotProduct(Cost):
    """
    Computes q_state * q
    """

    def __init__(self, sensor: Any, quaternion: np.ndarray):
        """

        :param sensor:              Sensor which measures the quaternion.
        :param quaternion:          Quaternion to compute dot product with.
        """
        self._sensor = sensor
        self._quaternion = quaternion

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

        # Extract sensor state
        q = sample.full_state[self._sensor]
        assert q.shape[1] == 4, f"Sensor has wrong dimensions (expected 4, got {q.shape[1]})"

        l = q.dot(self._quaternion)
        lx = np.zeros((ts, dx))
        if self._sensor in sample.state_packer.labels:
            lx[sample.state_packer.label_slices[self._sensor]] = self._quaternion
        lu = np.zeros((ts, du))
        lxx = np.zeros((ts, dx, dx))
        luu = np.zeros((ts, du, du))
        lux = np.zeros((ts, du, dx))

        return l, lx, lu, lxx, luu, lux

    def __repr__(self) -> str:
        return "c_state({0})".format(self._sensor)
