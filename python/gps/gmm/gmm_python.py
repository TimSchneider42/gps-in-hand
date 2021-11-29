""" This file defines a Gaussian mixture model class. """
import logging
from typing import Optional

import numpy as np

from gps.gmm.gmm import GMM, compute_log_obs, logsum

LOGGER = logging.getLogger(__name__)


class GMMPython(GMM):
    """ Gaussian Mixture Model. """

    def __init__(self, init_sequential: bool = False, eigreg: bool = False,
                 mu: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None, regularization: float = 1e-6, max_iterations: int = 100):
        self.__init_sequential = init_sequential
        self.__eigreg = eigreg
        self.__regularization = regularization
        self.__max_iterations = max_iterations
        super(GMMPython, self).__init__(mu, sigma, weights)

    def update(self, data: np.ndarray, cluster_count: int) -> "GMM":
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            cluster_count: Number of clusters to use.
        """
        # Constants.
        point_count = data.shape[0]
        data_point_count = data.shape[1]

        LOGGER.debug(f"Fitting GMM with {cluster_count} clusters on {point_count} points using own implementation.")

        if self.covariances is None or cluster_count != self.covariances.shape[0]:
            # Initialization.
            LOGGER.debug("Initializing GMM.")
            new_sigma = np.zeros((cluster_count, data_point_count, data_point_count))
            new_mu = np.zeros((cluster_count, data_point_count))
            new_logmass = np.log((1.0 / cluster_count) * np.ones((cluster_count, 1)))

            # Set initial cluster indices.
            if not self.__init_sequential:
                cidx = np.random.randint(0, cluster_count, size=(1, point_count))
            else:
                raise NotImplementedError()

            # Initialize.
            for i in range(cluster_count):
                cluster_idx = (cidx == i)[0]
                local_mu = np.mean(data[cluster_idx, :], axis=0)
                diff = (data[cluster_idx, :] - local_mu).T
                local_sigma = (1.0 / cluster_count) * (diff.dot(diff.T))
                new_mu[i, :] = local_mu
                new_sigma[i, :, :] = local_sigma + np.eye(data_point_count) * 2e-6
        else:
            new_sigma = self.covariances
            new_mu = self.means
            new_logmass = self.log_weights

        prevll = -float("inf")
        max_iterations = self.__max_iterations
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logobs = compute_log_obs(data, new_mu, new_sigma, new_logmass)

            # Compute log-likelihood.
            ll = np.sum(logsum(logobs, axis=1))
            LOGGER.debug(f"GMM itr {itr}/{max_iterations}. Log likelihood: {ll}")
            if ll < prevll:
                # TODO: Why does log-likelihood decrease sometimes?
                LOGGER.debug(f"Log-likelihood decreased! Ending on itr={itr}/{max_iterations}")
                break
            if np.abs(ll - prevll) < 1e-5 * prevll:
                LOGGER.debug(f"GMM converged on itr={itr}/{max_iterations}")
                break
            prevll = ll

            # Renormalize to get cluster weights.
            logw = logobs - logsum(logobs, axis=1)
            assert logw.shape == (point_count, cluster_count)

            # Renormalize again to get weights for refitting clusters.
            logwn = logw - logsum(logw, axis=0)
            assert logwn.shape == (point_count, cluster_count)
            w = np.exp(logwn)

            # M-step: update clusters.
            # Fit cluster mass.
            new_logmass = logsum(logw, axis=0).T
            new_logmass = new_logmass - logsum(new_logmass, axis=0)
            assert new_logmass.shape == (cluster_count, 1)
            new_mass = np.exp(new_logmass)
            # Reboot small clusters.
            w[:, (new_mass < (1.0 / cluster_count) * 1e-4)[:, 0]] = 1.0 / point_count
            # Fit cluster means.
            w_expand = np.expand_dims(w, axis=2)
            data_expand = np.expand_dims(data, axis=1)
            new_mu = np.sum(w_expand * data_expand, axis=0)
            # Fit covariances.
            wdata = data_expand * np.sqrt(w_expand)
            assert wdata.shape == (point_count, cluster_count, data_point_count)
            for i in range(cluster_count):
                # Compute weighted outer product.
                XX = wdata[:, i, :].T.dot(wdata[:, i, :])
                new_sigma[i, :, :] = XX - np.outer(new_mu[i], new_mu[i])

                if self.__eigreg:  # Use eigenvalue regularization.
                    raise NotImplementedError()
                else:  # Use quick and dirty regularization. TODO
                    new_sigma[i, :, :] = 0.5 * (new_sigma[i, :, :] + new_sigma[i, :, :].T) + \
                                         self.__regularization * np.eye(data_point_count)
                    assert np.all(np.linalg.eigvals(new_sigma) > 0), "Covariance is not positive definite anymore!"
        return GMMPython(self.__init_sequential, self.__eigreg, new_mu, new_sigma, np.exp(new_logmass),
                         self.__regularization, self.__max_iterations)
