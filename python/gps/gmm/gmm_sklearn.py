""" This file defines a Gaussian mixture model class. """
import logging
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from gps.gmm import GMM

LOGGER = logging.getLogger(__name__)


class GMMSklearn(GMM):
    """ Gaussian Mixture Model. """

    def __init__(self, means: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None,
                 precisions: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None,
                 regularization: float = 1e-6, max_iterations: int = 100):
        self.__max_iterations = max_iterations
        self.__regularization = regularization
        super(GMMSklearn, self).__init__(mu=means, sigma=sigma, precisions=precisions, weights=weights)

    def update(self, data: np.ndarray, cluster_count: int) -> "GMM":
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            cluster_count: Number of clusters to use.
        """
        LOGGER.debug(f"Fitting GMM with {cluster_count} clusters on {data.shape[0]} points using sklearn.")
        if self.covariances is not None and self.covariances.shape[0] == cluster_count:
            gmm = GaussianMixture(n_components=cluster_count, max_iter=self.__max_iterations,
                                  reg_covar=self.__regularization, weights_init=self.weights, means_init=self.means,
                                  precisions_init=self.precisions)
        else:
            gmm = GaussianMixture(n_components=cluster_count, max_iter=self.__max_iterations)
        gmm.fit(data)
        if gmm.converged_:
            LOGGER.debug(f"GMM converged. Log likelihood: {gmm.lower_bound_}")
        else:
            LOGGER.debug(f"GMM failed to converge. Log likelihood: {gmm.lower_bound_}")
        return GMMSklearn(gmm.means_, gmm.covariances_, gmm.precisions_, gmm.weights_, self.__regularization,
                          self.__max_iterations)
