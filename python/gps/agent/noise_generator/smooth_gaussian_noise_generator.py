from typing import Optional

import numpy as np
import scipy.ndimage as sp_ndimage

from .gaussian_noise_generator import GaussianNoiseGenerator


class SmoothGaussianNoiseGenerator(GaussianNoiseGenerator):
    """
    Noise generator for smoothed gaussian distributed noise. This will apply a gaussian filter with a defined
    variance to a gaussian distributed noise vector.
    """

    def __init__(self, smooth_noise_var: float = 2.0, sigma: float = 1.0, mu: float = 0.0, seed: Optional[int] = None,
                 renormalize: bool = True):
        """

        :param sigma:               Standard deviation parameter of the gaussian distribution.
        :param mu:                  Mean of the gaussian distribution.
        :param seed:                Seed for the random number generator.
        :param smooth_noise_var:    Applies a Gaussian filter with this variance.
        :param renormalize:         Renormalize data to match the specified variance after smoothing again.
        """
        self._smooth_noise_var = smooth_noise_var
        self._renormalize = renormalize
        super(SmoothGaussianNoiseGenerator, self).__init__(sigma, mu, seed)

    def generate_noise(self, time_steps: int, vector_dimensions: int) -> np.ndarray:
        """
        Generates a matrix of noise.
        :param time_steps:          The number of time steps.
        :param vector_dimensions:   Number of entries in the vector.
        :return: A time_steps x vector_dimensions matrix of noise
        """
        noise = super(SmoothGaussianNoiseGenerator, self).generate_noise(time_steps, vector_dimensions)

        # Smooth noise. This violates the controller assumption, but might produce smoother motions.
        for i in range(vector_dimensions):
            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], self._smooth_noise_var)
        if self._renormalize:
            variance = np.var(noise, axis=0)
            noise = noise / np.sqrt(variance)
        return noise
