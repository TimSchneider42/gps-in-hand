from typing import Optional

import numpy as np

from gps.controller import Controller


class Policy(Controller):
    def __init__(self, observation_dimensions: int, covariance: np.ndarray,
                 cholesky_covariance: Optional[np.ndarray] = None, inv_covariance: Optional[np.ndarray] = None):
        super(Policy, self).__init__(covariance, cholesky_covariance, inv_covariance)
        self.__observation_dimensions = observation_dimensions

    @property
    def observation_dimensions(self) -> int:
        return self.__observation_dimensions
