""" This file defines the GMM prior for dynamics estimation. """
import logging
from typing import Tuple, Optional

import numpy as np

from gps.dynamics import DynamicsPrior
from gps.sample import SampleList
from gps.gmm import GMM, GMMSklearn

LOGGER = logging.getLogger(__name__)


class DynamicsPriorGMM(DynamicsPrior):
    """
    A dynamics prior encoded as a GMM over [x_t, u_t, x_t+1] points.
    See:
        S. Levine*, C. Finn*, T. Darrell, P. Abbeel, "End-to-end
        training of Deep Visuomotor Policies", arXiv:1504.00702,
        Appendix A.3.
    """

    def __init__(self, min_samples_per_cluster: int = 20, max_clusters: int = 50, max_samples: int = 20,
                 strength: float = 1.0, state_dataset: Optional[np.ndarray] = None,
                 action_dataset: Optional[np.ndarray] = None, gmm: Optional[GMM] = None):
        """

        :param min_samples_per_cluster: Minimum number of samples per cluster
        :param max_clusters:            Maximum number of clusters to fit
        :param max_samples:             Maximum number of trajectories to use for fitting the GMM at any given time.
        :param strength:                Adjusts the strength of the prior.
        """
        self.__gmm = GMMSklearn() if gmm is None else gmm
        self.__min_samples_per_cluster = min_samples_per_cluster
        self.__max_samples = max_samples
        self.__max_clusters = max_clusters
        self.__strength = strength
        self.__state_dataset = state_dataset
        self.__action_dataset = action_dataset

    def initial_state(self):
        """ Return dynamics prior for initial time step. """
        # Compute mean and covariance.
        mu0 = np.mean(self.__state_dataset[:, 0, :], axis=0)
        Phi = np.diag(np.var(self.__state_dataset[:, 0, :], axis=0))

        # Factor in multiplier.
        n0 = self.__state_dataset.shape[2] * self.__strength
        m = self.__state_dataset.shape[2] * self.__strength

        # Multiply Phi by m (since it was normalized before).
        Phi = Phi * m
        return mu0, Phi, m, n0

    def update(self, sample_list: SampleList) -> "DynamicsPriorGMM":
        # Constants.
        time_steps = sample_list.actions.shape[1]
        states = sample_list.states
        actions = sample_list.actions

        # Append data to dataset.
        if self.__state_dataset is None:
            state_dataset = states
        else:
            state_dataset = np.concatenate([self.__state_dataset, states], axis=0)

        if self.__action_dataset is None:
            action_dataset = actions
        else:
            action_dataset = np.concatenate([self.__action_dataset, actions], axis=0)

        # Remove excess samples from dataset.
        start = max(0, action_dataset.shape[0] - self.__max_samples)
        state_dataset = state_dataset[start:, :]
        action_dataset = action_dataset[start:, :]

        # Compute cluster dimensionality.
        Do = states.shape[2] + actions.shape[2] + states.shape[2]

        # Create dataset.
        N = state_dataset.shape[0]
        xux = np.reshape(np.c_[state_dataset[:, :time_steps, :],
                               action_dataset[:, :time_steps, :],
                               state_dataset[:, 1:(time_steps + 1), :]],
                         [time_steps * N, Do])

        # Choose number of clusters.
        K = int(max(2, min(self.__max_clusters,
                           np.floor(float(N * time_steps) / self.__min_samples_per_cluster))))
        LOGGER.debug('Generating %d clusters for dynamics GMM.', K)

        # Update GMM.
        new_gmm = self.__gmm.update(xux, K)

        return DynamicsPriorGMM(self.__min_samples_per_cluster, self.__max_clusters, self.__max_samples,
                                self.__strength, state_dataset, action_dataset, new_gmm)

    def eval(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        # Perform query and fix mean.
        mu0, Phi, m, n0 = self.__gmm.inference(points)

        # Factor in multiplier.
        n0 = n0 * self.__strength
        m = m * self.__strength

        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0
