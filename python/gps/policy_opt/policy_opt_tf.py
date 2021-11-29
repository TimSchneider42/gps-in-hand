""" This file defines controller optimization for a tensorflow controller. """
import logging
from typing import Optional

import numpy as np

import tensorflow as tf

from gps.policy import PolicyTf
from gps.policy_opt import PolicyOpt

LOGGER = logging.getLogger(__name__)


class PolicyOptTf(PolicyOpt[PolicyTf]):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """

    def __init__(self, training_iterations: int = 5000, batch_size: int = 25, tf_device_string: Optional[str] = None):

        self.__batch_size = batch_size
        self.__tf_device_string = tf_device_string
        self.__training_iterations = training_iterations

    def update(self, policy: PolicyTf, observation: np.ndarray, target_action_mean: np.ndarray,
               target_action_precision: np.ndarray, target_weight: np.ndarray,
               entropy_regularization: float = 0.0) -> PolicyTf:
        num_samples, ts, obs_dims = observation.shape
        action_dims = target_action_mean.shape[2]

        random_seed = policy.random_seed
        np_rng = np.random.RandomState(random_seed)

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(target_action_precision, [num_samples * ts, action_dims, action_dims])

        # Renormalize weights.
        target_weight *= (float(num_samples * ts) / np.sum(target_weight))
        # Allow weights to be at most twice the robust median.
        mn = np.median(target_weight[(target_weight > 1e-2).nonzero()])
        for n in range(num_samples):
            for t in range(ts):
                target_weight[n, t] = min(target_weight[n, t], 2 * mn)
        # Robust median should be around one.
        target_weight /= mn

        # Reshape inputs.
        obs = np.reshape(observation, (num_samples * ts, obs_dims))
        tgt_mu = np.reshape(target_action_mean, (num_samples * ts, action_dims))
        tgt_prc = np.reshape(target_action_precision, (num_samples * ts, action_dims, action_dims))
        target_weight = np.reshape(target_weight, (num_samples * ts, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = target_weight * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalization at the beginning.
        if policy.scale is None:
            # 1e-3 to avoid infs if some state dimensions don't change in the first batch of samples
            scale = np.diag(1.0 / np.maximum(np.std(obs, axis=0), 1e-3))
            bias = - np.mean(obs.dot(scale), axis=0)
            scale = np.diag(scale)
        else:
            scale = policy.scale
            bias = policy.bias
        obs_scaled = obs * scale + bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(num_samples * ts / self.__batch_size)
        idx = list(range(num_samples * ts))
        np_rng.shuffle(idx)

        nn = policy.neural_network

        # actual training.
        training_iterations = self.__training_iterations
        average_loss = 0
        with tf.Session(graph=policy.neural_network.graph) as sess:
            with tf.device(self.__tf_device_string):
                sess.run([var.assign(val) for var, val in policy.nn_variable_values.items()])
                for i in range(training_iterations):
                    # Load in data for this batch.
                    start_idx = int(i * self.__batch_size % (batches_per_epoch * self.__batch_size))
                    idx_i = idx[start_idx:start_idx + self.__batch_size]
                    feed_dict = {nn.observation_in: obs_scaled[idx_i],
                                 nn.action_in: tgt_mu[idx_i],
                                 nn.action_precision_in: tgt_prc[idx_i]}
                    _, train_loss = sess.run([nn.training_op, nn.loss], feed_dict=feed_dict)

                    average_loss += train_loss
                    if (i + 1) % 50 == 0:
                        LOGGER.info(f"tensorflow iteration {i + 1}, average loss {average_loss / 50}")
                        average_loss = 0
                variables = list(policy.nn_variable_values.keys())
                new_variable_values_list = sess.run(variables)
                new_variable_values = dict(zip(variables, new_variable_values_list))

                feed_dict = {nn.observation_in: obs_scaled,
                             nn.action_in: tgt_mu,
                             nn.action_precision_in: tgt_prc}
                total_loss = sess.run(nn.loss, feed_dict=feed_dict)
                LOGGER.info(f"Total loss after {training_iterations} iterations: {total_loss}")

        # Optimize variance.
        precision = np.sum(tgt_prc, 0) + 2 * num_samples * ts * entropy_regularization * np.ones(
            (action_dims, action_dims))
        precision = precision / np.sum(target_weight)

        # TODO - Use dense covariance?
        prec_chol = np.linalg.cholesky(precision)
        covar = np.linalg.inv(precision)
        covar = 0.5 * (covar.T + covar) + np.eye(action_dims) * 1e-6
        chol_pol_covar = np.linalg.inv(prec_chol)

        def e(m: np.ndarray):
            return np.tile(np.eye(policy.action_dimensions) * m, (policy.time_steps, 1, 1))

        return PolicyTf(neural_network=policy.neural_network, covariance=e(covar),
                        cholesky_covariance=e(chol_pol_covar),
                        inv_covariance=e(precision),
                        nn_variable_values=new_variable_values, scale=scale, bias=bias,
                        random_seed=random_seed + 1, device_string=policy.device_string)
