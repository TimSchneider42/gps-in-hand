""" This file defines the base policy optimization class. """
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import numpy as np

from gps.policy import Policy

T = TypeVar("T", bound=Policy)


class PolicyOpt(ABC, Generic[T]):
    """ Policy optimization superclass. """

    @abstractmethod
    def update(self, policy: T, observation: np.ndarray, target_action_mean: np.ndarray,
               target_action_precision: np.ndarray, target_weight: np.ndarray,
               entropy_regularization: float = 0.0) -> T:
        """
        Returns an updated version of the given policy.
        :param policy:                  Policy to update
        :param observation:             Observation vectors (NUM_SAMPLES x NUM_TIMESTEPS x OBS_DIMS)
        :param target_action_mean:      Target mean controller outputs (NUM_SAMPLES x NUM_TIMESTEPS x ACT_DIMS)
        :param target_action_precision: Target precision matrices (NUM_SAMPLES x NUM_TIMESTEPS x ACT_DIMS x ACT_DIMS)
        :param target_weight:           Target weights (NUM_SAMPLES, NUM_TIMESTEPS)
        :param entropy_regularization:  Entropy regularization constant.
        :return: The updated policy
        """
        pass
