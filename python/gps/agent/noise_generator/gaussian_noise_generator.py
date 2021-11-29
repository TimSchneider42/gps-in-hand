from typing import Optional, Any

import numpy as np

from .noise_generator import NoiseGenerator


class GaussianNoiseGenerator(NoiseGenerator):
    """
    Noise generator for gaussian distributed noise.
    """

    def __init__(self, sigma: float = 1.0, mu: float = 0.0, seed: Optional[int] = None):
        """

        :param sigma:   Standard deviation parameter of the gaussian distribution.
        :param mu:      Mean of the gaussian distribution.
        :param seed:                Seed for the random number generator.
        """
        self.__sigma = sigma
        self.__mu = mu
        self.__rng = np.random.RandomState(seed)
        self.__seed = seed

    def generate_noise(self, time_steps: int, vector_dimensions: int) -> np.ndarray:
        """
        Generates a gaussian distributed matrix of noise.
        :param time_steps:          The number of time steps.
        :param vector_dimensions:   Number of entries in the vector.
        :return: A time_steps x vector_dimensions matrix of noise
        """
        return self.__sigma * np.random.randn(time_steps, vector_dimensions) + self.__mu

    @property
    def initial_random_state(self) -> Any:
        return np.random.RandomState(self.__seed).get_state()

    @property
    def random_state(self) -> Any:
        return self.__rng.get_state()

    @random_state.setter
    def random_state(self, value: Any):
        self.__rng.set_state(value)

