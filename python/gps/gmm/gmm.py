""" This file defines a a base class for Gaussian mixture models. """
from abc import abstractmethod, ABC
from typing import Tuple, Optional

import numpy as np

from gps.gmm.gmm_utils import compute_log_obs, logsum


class GMM(ABC):
    """ Gaussian Mixture Model. """

    def __init__(self, mu: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None, precisions: Optional[np.ndarray] = None):
        optional_variables = [mu, sigma, weights]

        self.__sigma = sigma
        self.__mu = mu
        self.__precisions = precisions
        self.__weights = weights

        if any(v is not None for v in optional_variables):
            assert mu is not None and weights is not None and (sigma is not None or precisions is not None)
            self.__log_weights = np.log(self.__weights)
            if self.__sigma is None:
                self.__sigma = np.linalg.inv(self.__precisions)
            if self.__precisions is None:
                self.__precisions = np.linalg.inv(self.__sigma)
        else:
            self.__log_weights = None

    def inference(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Evaluate dynamics prior and return an Inverse-Wishart prior.
        Args:
            points: A N x D array of points.
        """
        data_point_count = self.__mu.shape[1]

        # Compute posterior cluster weights.
        log_weights = self.__clusterweights(points)

        # Compute posterior mean and covariance.
        mu0, Phi = self.__moments(log_weights)

        # TODO: Figure out what is going on with that m
        # Set hyperparameters.
        m = data_point_count
        n0 = m - 2 - mu0.shape[0]

        # Normalize.
        m = float(m) / data_point_count
        n0 = float(n0) / data_point_count
        return mu0, Phi, m, n0

    def __moments(self, log_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the moments of the cluster mixture with logwts.
        Args:
            log_weights: A K x 1 array of log cluster probabilities.
        Returns:
            mu: A (D,) mean vector.
            sigma: A D x D covariance matrix.
        """
        # Exponentiate.
        wts = np.exp(log_weights)

        # Compute overall mean.
        mu = np.sum(self.__mu * wts, axis=0)

        # Compute overall covariance.
        diff = self.__mu - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(self.__mu, axis=1) * \
                      np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((self.__sigma + diff_expand) * wts_expand, axis=0)
        return mu, sigma

    def __clusterweights(self, data: np.ndarray) -> np.ndarray:
        """
        Compute cluster weights for specified points under GMM.
        Args:
            data: An N x D array of points
        Returns:
            A K x 1 array of average cluster log probabilities.
        """
        # Compute probability of each point under each cluster.
        logobs = compute_log_obs(data, self.__mu, self.__sigma, self.__log_weights)

        # Renormalize to get cluster weights.
        logwts = logobs - logsum(logobs, axis=1)

        # Average the cluster probabilities.
        logwts = logsum(logwts, axis=0) - np.log(data.shape[0])
        return logwts.T

    @abstractmethod
    def update(self, data: np.ndarray, cluster_count: int) -> "GMM":
        """
        Fit this GMM to the data and return the resulting GMM. Note that this GMM remains unchanged.
        :param data:            Data points to fit.
        :param cluster_count:   Number of clusters.
        :return:
        """
        pass

    @property
    def weights(self) -> np.ndarray:
        """
        Log weights for each cluster.
        :return:
        """
        return self.__weights

    @property
    def log_weights(self) -> np.ndarray:
        return self.__log_weights

    @property
    def covariances(self) -> np.ndarray:
        """
        Covariances for each cluster.
        :return:
        """
        return self.__sigma

    @property
    def precisions(self) -> np.ndarray:
        return self.__precisions

    @property
    def means(self):
        """
        Means for each cluster.
        :return:
        """
        return self.__mu
