from typing import Any, Tuple

import numpy as np

from gps.cost.cost import Cost
from gps.sample import Sample


class CostTargetQuaternion(Cost):
    """
    Computes 1 - (q_state * q_target)^2
    """

    def __init__(self, sensor: Any, target_quaternion: np.ndarray):
        """

        :param sensor:              Sensor which measures the quaternion.
        :param target_quaternion:   Quaternion to compute distance to.
        """
        self._sensor = sensor
        self._target_quaternion = target_quaternion

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

        dot_prod = q.dot(self._target_quaternion)

        l = 1 - dot_prod ** 2
        lx = np.zeros((ts, dx))
        lu = np.zeros((ts, du))
        lxx = np.zeros((ts, dx, dx))
        luu = np.zeros((ts, du, du))
        lux = np.zeros((ts, du, dx))

        ls = -2 * dot_prod.reshape((100, 1)) @ self._target_quaternion.reshape((1, 4))
        lss = np.tile(-2 * np.outer(self._target_quaternion, self._target_quaternion), (ts, 1, 1))

        if self._sensor in sample.state_packer.labels:
            state_slices = sample.state_packer.label_slices
            lx[:, state_slices[self._sensor]] = ls
            lxx[:, state_slices[self._sensor], state_slices[self._sensor]] = lss
        return l, lx, lu, lxx, luu, lux

    def __repr__(self) -> str:
        return "c_state({0})".format(self._sensor)
