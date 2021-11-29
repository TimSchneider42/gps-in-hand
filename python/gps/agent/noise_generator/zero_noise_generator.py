from typing import Any

import numpy as np

from .noise_generator import NoiseGenerator


class ZeroNoiseGenerator(NoiseGenerator):
    """
    Noise generator for zero noise.
    """
    def generate_noise(self, time_steps: int, vector_dimensions: int) -> np.ndarray:
        return np.zeros((time_steps, vector_dimensions))

    @property
    def initial_random_state(self) -> Any:
        return None

    @property
    def random_state(self) -> Any:
        return None

    @random_state.setter
    def random_state(self, value: Any):
        pass

