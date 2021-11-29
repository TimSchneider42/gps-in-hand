""" This file defines a interface for dynamic priors. """
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from gps.sample import SampleList

LOGGER = logging.getLogger(__name__)


class DynamicsPrior(ABC):
    @property
    @abstractmethod
    def initial_state(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Dynamics prior for initial time step.
        :return:
            mu0:    Mean of initial state
            phi:    Covariance of initial state
            m:      TODO: figure out what that is
            n0:     TODO
        """
        pass

    @abstractmethod
    def update(self, sample_list: SampleList) -> "DynamicsPrior":
        """
        Create updated prior with additional data.
        :param sample_list: Samples to fit prior to
        :return: The updated prior
        """
        pass

    @abstractmethod
    def eval(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Evaluate prior at given points.
        :param points: A N x Dx+Du+Dx matrix specifying the points to evaluate prior at.
        :return:
            mu0:    Mean of initial state
            phi:    Covariance of initial state
            m:      TODO: figure out what that is
            n0:     TODO
        """
        pass
