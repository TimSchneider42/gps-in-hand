from typing import Tuple

import numpy as np

from gps.policy import Policy, PolicyPrior
from gps.sample.sample_list import SampleList


class PolicyPriorConstant(PolicyPrior):
    """ Constant controller prior. """

    def update(self, samples: SampleList, policy: Policy, replace_samples: bool = False) -> "PolicyPrior":
        """
        Updates the controller prior.
        :param samples:         Samples to update prior with.
        :param policy:          Policy to compute prior for.
        :param replace_samples: True, if previously recorded samples
        :return:
        """
        # Nothing to update for constant controller prior.
        return self

    def eval(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Evaluate the controller prior for the given states and actions.
        :param states:  N x STATE_DIMS array of states
        :param actions: N x ACTION_DIMS array of actions
        :return:
        """
        # Ignore states and actions as this is a constant prior.
        dx, du = states.shape[-1], actions.shape[-1]
        prior_fd = np.zeros((du, dx))
        prior_cond = 1e-5 * np.eye(du)
        sig = np.eye(dx)
        Phi = self.strength * np.vstack([
            np.hstack([sig, sig.dot(prior_fd.T)]),
            np.hstack([prior_fd.dot(sig),
                       prior_fd.dot(sig).dot(prior_fd.T) + prior_cond])
        ])
        return np.zeros(dx + du), Phi, 0.0, self.strength
