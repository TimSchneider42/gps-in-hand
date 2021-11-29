""" This file defines the state target cost. """
from typing import Any, Tuple, Optional, Union

import numpy as np

from gps.cost.cost import Cost
from gps.cost.cost_utils import evall1l2term, get_ramp_multiplier, RampOptions
from gps.sample import Sample


class CostTargetState(Cost):
    """
    Computes l1/l2 distance to a fixed target state for a single sensor.
    costs(t) =  r(t) * ((0.5 * l2 * d(t)^T * diag(w) * d(t)) + (l1 * sqrt(alpha + d(t)^T * diag(w) * d(t))))
    where r(t) is the ramp multiplier w.r.t t and w are the sensor weights.
    """

    def __init__(self, sensor: Any, target_state: Union[np.ndarray, Any], sensor_weights: Optional[np.ndarray] = None,
                 l1: float = 0.0, l2: float = 1.0, alpha: float = 1e-2, wp_final_multiplier: float = 1.0,
                 ramp_option: RampOptions = RampOptions.RAMP_CONSTANT):
        """

        :param sensor:              Sensor for which to compute l1/l2 distance.
        :param target_state:        Target state for this specific sensor (can be another sensor).
        :param sensor_weights:      Weights for each component of this sensor (Default: 1 for each component).
        :param l1:                  Weight of the l1 norm.
        :param l2:                  Weight of the l2 norm.
        :param alpha:               Constant summand inside the square root.
        :param wp_final_multiplier: Weight multiplier on final time step.
        :param ramp_option:         How target costs ramps over time.
        """
        self._sensor = sensor
        self._target_state = target_state
        self._sensor_weights = sensor_weights
        self._l1 = l1
        self._l2 = l2
        self._alpha = alpha
        self._wp_final_multiplier = wp_final_multiplier
        self._ramp_option = ramp_option

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

        final_l = np.zeros(ts)
        final_lx = np.zeros((ts, dx))
        final_lu = np.zeros((ts, du))
        final_lxx = np.zeros((ts, dx, dx))
        final_luu = np.zeros((ts, du, du))
        final_lux = np.zeros((ts, du, dx))

        if self._target_state is not self._sensor:
            if isinstance(self._target_state, np.ndarray):
                target_state = self._target_state
            else:
                target_state = sample.full_state[self._target_state]
            state_slices = sample.state_packer.label_slices

            # Extract sensor state
            state = sample.full_state[self._sensor]
            dim_sensor = state.shape[1]

            wpm = get_ramp_multiplier(
                self._ramp_option, ts,
                wp_final_multiplier=self._wp_final_multiplier
            )
            weights = self._sensor_weights if self._sensor_weights is not None else np.ones(dim_sensor)
            wp = weights * np.expand_dims(wpm, axis=-1)

            # Compute state penalty.
            dist = state - target_state

            # Compute scaled quantities.
            sqrtwp = np.sqrt(wp)
            dsclsq = dist * sqrtwp
            dscl = dist * wp
            dscls = dist * (wp ** 2)

            # Compute total cost.
            final_l = 0.5 * np.sum(dsclsq ** 2, axis=1) * self._l2 + \
                np.sqrt(self._alpha + np.sum(dscl ** 2, axis=1)) * self._l1

            if self._sensor in sample.state_packer.labels:
                # First order derivative terms.
                d1 = dscl * self._l2 + (
                        dscls / np.sqrt(self._alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * self._l1
                )
                lx = np.sum(np.eye(dim_sensor) * np.expand_dims(d1, axis=2), axis=1)

                # Second order terms.
                psq = np.expand_dims(
                    np.sqrt(self._alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1
                )
                d2 = self._l1 * (
                        (np.expand_dims(np.eye(wp.shape[1]), axis=0) *
                         (np.expand_dims(wp ** 2, axis=1) / psq)) -
                        ((np.expand_dims(dscls, axis=1) *
                          np.expand_dims(dscls, axis=2)) / psq ** 3)
                )
                d2 += self._l2 * (
                        np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [ts, 1, 1])
                )

                Jd_expand_1 = np.expand_dims(np.expand_dims(np.eye(dim_sensor), axis=2), axis=4)
                Jd_expand_2 = np.expand_dims(np.expand_dims(np.eye(dim_sensor), axis=1), axis=3)
                d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
                lxx = np.sum(np.sum(Jd_expand_1 * Jd_expand_2 * d2_expand, axis=1), axis=1)

                final_lx[:, state_slices[self._sensor]] = lx
                final_lxx[:, state_slices[self._sensor], state_slices[self._sensor]] = lxx

                if not isinstance(self._target_state, np.ndarray) and self._target_state in state_slices:
                    # Target is actually part of the state
                    final_lx[:, state_slices[self._target_state]] = -lx
                    final_lxx[:, state_slices[self._target_state], state_slices[self._target_state]] = lxx
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux

    def __repr__(self) -> str:
        return "c_state({0})".format(self._sensor)
