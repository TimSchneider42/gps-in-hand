""" This file defines the linear Gaussian controller class. """
from typing import Optional

import numpy as np

from gps.controller import Controller
from gps.utility.general_utils import check_shape


class LinearGaussianController(Controller):
    """
    Time-varying linear Gaussian controller.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
    """

    def __init__(self, K: np.ndarray, k: np.ndarray, covariance: np.ndarray,
                 cholesky_covariance: Optional[np.ndarray] = None, inv_covariance: Optional[np.ndarray] = None):
        Controller.__init__(self, covariance, cholesky_covariance, inv_covariance)

        self.__K = K
        self.__k = k

        check_shape(k, (self.time_steps, self.action_dimensions))
        check_shape(K, (self.time_steps, self.action_dimensions, self.state_dimensions))

    def _act_mean(self, t: int, state: Optional[np.ndarray], obs: Optional[np.ndarray]) -> np.ndarray:
        return self.K[t].dot(state) + self.k[t]

    @property
    def state_dimensions(self) -> int:
        return self.__K.shape[2]

    @property
    def k(self) -> np.ndarray:
        return self.__k

    @property
    def K(self) -> np.ndarray:
        return self.__K
