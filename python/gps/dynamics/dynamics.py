""" This file defines the base class for dynamics estimation. """
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from gps.dynamics.dynamics_prior import DynamicsPrior
from gps.sample import SampleList


class Dynamics(ABC):
    """ Dynamics superclass. """

    def __init__(self, Fm: Optional[np.ndarray] = None, fv: Optional[np.ndarray] = None,
                 covariance: Optional[np.ndarray] = None, prior: Optional[DynamicsPrior] = None):
        # Fitted dynamics: x_t+1 = Fm * [x_t;u_t] + fv.
        self.__Fm = Fm
        self.__fv = fv
        self.__covariance = covariance
        self.__prior = prior

        variables_check = [v is not None for v in [Fm, fv, covariance]]
        if any(variables_check):
            assert all(variables_check), "Either all of Fm, fv and covariance or neither need to be set."

    @abstractmethod
    def fit(self, sample_list: SampleList):
        """ Fit dynamics. """
        pass

    @property
    def Fm(self) -> np.ndarray:
        return self.__Fm

    @property
    def fv(self) -> np.ndarray:
        return self.__fv

    @property
    def covariance(self) -> np.ndarray:
        return self.__covariance

    @property
    def prior(self) -> Optional[DynamicsPrior]:
        return self.__prior
