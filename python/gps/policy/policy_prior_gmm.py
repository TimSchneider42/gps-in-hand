""" This file defines a GMM prior for controller linearization. """
import logging
from typing import Optional, Tuple

import numpy as np

from gps.policy import Policy, PolicyPrior
from gps.sample import SampleList
from gps.gmm import GMM, GMMSklearn

LOGGER = logging.getLogger(__name__)


class PolicyPriorGMM(PolicyPrior):
    """
    A controller prior encoded as a GMM over [x_t, u_t] points, where u_t is
    the output of the controller for the given state x_t. This prior is used
    when computing the linearization of the controller.

    See the method AlgorithmBADMM._update_policy_fit, in
    python/gps/algorithm.algorithm_badmm.py.

    Also see the GMM dynamics prior, in
    python/gps/algorithm/dynamics/dynamics_prior_gmm.py. This is a
    similar GMM prior that is used for the dynamics estimate.
    """

    def __init__(self, min_samples_per_cluster: int = 20, max_clusters: int = 50,
                 max_samples: int = 20, strength: float = 1.0, state_buffer: Optional[np.ndarray] = None,
                 observation_buffer: Optional[np.ndarray] = None, gmm: Optional[GMM] = None):
        """

        :param min_samples_per_cluster: Minimum number of samples per cluster.
        :param max_clusters:            Maximum number of clusters to fit.
        :param max_samples:             Maximum number of trajectories to use for fitting the GMM at any given time.
        :param strength:                The strength of the prior.
        """
        super(PolicyPriorGMM, self).__init__(strength)

        self.__state_buffer = state_buffer
        self.__observation_buffer = observation_buffer
        self.__gmm = GMMSklearn() if gmm is None else gmm
        self.__min_samples_per_cluster = min_samples_per_cluster
        self.__max_samples = max_samples
        self.__max_clusters = max_clusters

        opt_vars = [state_buffer, observation_buffer]
        if any(v is not None for v in opt_vars):
            assert all(v is not None for v in opt_vars)
            assert state_buffer.shape[:2] == observation_buffer.shape[:2]

    def update(self, samples: SampleList, policy: Policy, replace_samples: bool = False) -> "PolicyPriorGMM":

        if self.__state_buffer is None or replace_samples:
            state_buffer = samples.states
            observation_buffer = samples.observations
        else:
            state_buffer = np.concatenate([self.__state_buffer, samples.states], axis=0)
            observation_buffer = np.concatenate([self.__observation_buffer, samples.observations], axis=0)
            # Trim extra samples
            buffer_size = state_buffer.shape[0]
            if buffer_size > self.__max_samples:
                start = buffer_size - self.__max_samples
                state_buffer = state_buffer[start:, :, :]
                observation_buffer = observation_buffer[start:, :, :]

        buffer_size, ts = state_buffer.shape[:2]

        # Evaluate controller at samples to get mean controller action.
        actions = np.array([policy.probe(observations=s) for s in observation_buffer])

        # Create the dataset
        dxu = state_buffer.shape[2] + actions.shape[2]
        xu = np.reshape(np.concatenate([state_buffer, actions], axis=2), [ts * buffer_size, dxu])
        # Choose number of clusters.
        k = int(max(2, min(self.__max_clusters,
                           np.floor(float(buffer_size * ts) / self.__min_samples_per_cluster))))

        LOGGER.debug(f"Generating {k} clusters for controller prior GMM.")
        gmm = self.__gmm.update(xu, k)
        return PolicyPriorGMM(self.__min_samples_per_cluster, self.__max_clusters, self.__max_samples, self.strength,
                              state_buffer, observation_buffer, gmm)

    def eval(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """ Evaluate prior. """
        # Construct query data point.
        pts = np.concatenate((states, actions), axis=1)
        # Perform query.
        mu0, Phi, m, n0 = self.__gmm.inference(pts)
        # Factor in multiplier.
        n0 *= self.strength
        m *= self.strength
        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0
