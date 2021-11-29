from abc import abstractmethod, ABC
from typing import Any

import numpy as np


class NoiseGenerator(ABC):
    """
    This is the abstract base class for noise generators. A noise generator generates a matrix of noise, given the
    number of time steps and the number of entries of the state/action vector.
    """

    @abstractmethod
    def generate_noise(self, time_steps: int, vector_dimensions: int) -> np.ndarray:
        """
        Generates a matrix of noise.
        :param time_steps:          The number of time steps.
        :param vector_dimensions:   Number of entries in the vector.
        :return: A time_steps x vector_dimensions matrix of noise
        """
        pass

    @property
    @abstractmethod
    def initial_random_state(self) -> Any:
        """
        Returns the initial random state
        :return:
        """
        pass

    @property
    @abstractmethod
    def random_state(self) -> Any:
        """
        State of the random number generator.
        :return:
        """
        pass

    @random_state.setter
    @abstractmethod
    def random_state(self, value: Any):
        pass
