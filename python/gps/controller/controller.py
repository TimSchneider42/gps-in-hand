""" This file defines the base class for the controller. """
from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
import scipy.linalg as la

from gps.utility.general_utils import check_shape


class Controller(ABC):
    """ Computes actions from states/observations. """

    def __init__(self, covariance: np.ndarray, cholesky_covariance: Optional[np.ndarray] = None,
                 inv_covariance: Optional[np.ndarray] = None):
        """

        :param covariance:          Action covariance of this controller.
        :param cholesky_covariance: Cholesky decomposition of the action covariance of this controller (will be computed if
                                    not provided).
        :param inv_covariance:      Inverse of the action covariance of this controller (will be computed if not provided).
        """
        self.__covariance = covariance
        if cholesky_covariance is None:
            self.__cholesky_covariance = np.linalg.cholesky(covariance)
        else:
            self.__cholesky_covariance = cholesky_covariance
        if inv_covariance is None:
            chol_covar = self.__cholesky_covariance
            self.__inv_covariance = np.stack([la.solve_triangular(
                cc, la.solve_triangular(cc.T, np.eye(covariance.shape[1]), lower=True)) for cc in chol_covar])
        else:
            self.__inv_covariance = inv_covariance

        check_shape(self.__covariance, (self.time_steps, self.action_dimensions, self.action_dimensions))
        check_shape(self.__cholesky_covariance, (self.time_steps, self.action_dimensions, self.action_dimensions))
        check_shape(self.__inv_covariance, (self.time_steps, self.action_dimensions, self.action_dimensions))

        self.__entropy = 2 * np.sum(np.log(np.diagonal(self.cholesky_covariance, axis1=1, axis2=2)))

    def act(self, t: int, state: Optional[np.ndarray] = None, obs: Optional[np.ndarray] = None,
            noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the next action given the current state, observation and time step.
        :param t:       The current time step.
        :param state:   The current state vector (dX).
        :param obs:     The current observation vector (dO).
        :param noise:   Noise to be scaled and applied to the action (dU).
        :return: The next action vector (dU).
        """
        action = self._act_mean(t, state, obs)
        if noise is not None:
            if len(self.__cholesky_covariance.shape) == 3:
                action += self.__cholesky_covariance[t].T.dot(noise)
            else:
                action += self.__cholesky_covariance.T.dot(noise)
        return action

    def probe(self, states: Optional[np.ndarray] = None, observations: Optional[np.ndarray] = None,
              noise: Optional[np.ndarray] = None):
        observations = [None] * self.time_steps if observations is None else observations
        states = [None] * self.time_steps if states is None else states
        noise = [None] * self.time_steps if noise is None else noise
        son = zip(states, observations, noise)
        self.prepare_sampling()
        output = np.array([[self.act(t, s, o, n) for t in range(self.time_steps)] for s, o, n in son])
        self.sampling_done()
        return output

    @abstractmethod
    def _act_mean(self, t: int, state: Optional[np.ndarray], obs: Optional[np.ndarray]) -> np.ndarray:
        """
        Returns the next mean action given the current state, observation and time step.
        :param t:       The current time step.
        :param state:   The current state vector (dX).
        :param obs:     The current observation vector (dO).
        :return: The next mean action vector (dU).
        """
        pass

    def prepare_sampling(self):
        """
        Function to be called before sampling starts.
        Override this function to do initializations.
        :return:
        """
        pass

    def sampling_done(self):
        """
        Function to be called when sampling is done.
        Override this function to do deinitializations.
        :return:
        """
        pass

    @property
    def time_steps(self) -> int:
        """
        Number of time steps of a sample.
        :return:
        """
        return self.__covariance.shape[0]

    @property
    def action_dimensions(self) -> int:
        """
        Dimensions of the action vector.
        :return:
        """
        return self.__covariance.shape[1]

    @property
    def covariance(self) -> np.ndarray:
        """
        Action covariance of this controller.
        :return:
        """
        return self.__covariance

    @property
    def inv_covariance(self) -> np.ndarray:
        """
        Inverse of the action covariance of this controller.
        :return:
        """
        return self.__inv_covariance

    @property
    def cholesky_covariance(self) -> np.ndarray:
        """
        Cholesky decomposition of the action covariance of this controller.
        :return:
        """
        return self.__cholesky_covariance

    @property
    def entropy(self):
        """
        Entropy of this distribution.
        :return:
        """
        return self.__entropy
