""" This file defines utilities for trajectory optimization. """
import abc
import logging

import numpy as np
import scipy as sp


LOGGER = logging.getLogger(__name__)

# Constants used in TrajOptLQR.
DGD_MAX_ITER = 50
DGD_MAX_LS_ITER = 20
DGD_MAX_GD_ITER = 200

ALPHA, BETA1, BETA2, EPS = 0.005, 0.9, 0.999, 1e-8  # Adam parameters


def traj_distr_kl(new_mu, new_sigma, new_traj_distr, prev_traj_distr, tot=True):
    """
    Compute KL divergence between new and previous trajectory
    distributions.
    Args:
        new_mu: T x dX, mean of new trajectory distribution.
        new_sigma: T x dX x dX, variance of new trajectory distribution.
        new_traj_distr: A linear Gaussian controller object, new
            distribution.
        prev_traj_distr: A linear Gaussian controller object, previous
            distribution.
        tot: Whether or not to sum KL across all time steps.
    Returns:
        kl_div: The KL divergence between the new and previous
            trajectories.
    """
    # Constants.
    T = new_mu.shape[0]
    dU = new_traj_distr.action_dimensions

    # Initialize vector of divergences for each time step.
    kl_div = np.zeros(T)

    # Step through trajectory.
    for t in range(T):
        # Fetch matrices and vectors from trajectory distributions.
        mu_t = new_mu[t, :]
        sigma_t = new_sigma[t, :, :]
        K_prev = prev_traj_distr.K[t, :, :]
        K_new = new_traj_distr.K[t, :, :]
        k_prev = prev_traj_distr.k[t, :]
        k_new = new_traj_distr.k[t, :]
        chol_prev = prev_traj_distr.cholesky_covariance[t, :, :]
        chol_new = new_traj_distr.cholesky_covariance[t, :, :]

        # Compute log determinants and precision matrices.
        logdet_prev = 2 * sum(np.log(np.diag(chol_prev)))
        logdet_new = 2 * sum(np.log(np.diag(chol_new)))
        prc_prev = sp.linalg.solve_triangular(
            chol_prev, sp.linalg.solve_triangular(chol_prev.T, np.eye(dU),
                                                  lower=True)
        )
        prc_new = sp.linalg.solve_triangular(
            chol_new, sp.linalg.solve_triangular(chol_new.T, np.eye(dU),
                                                 lower=True)
        )

        # Construct matrix, vector, and constants.
        M_prev = np.r_[
            np.c_[K_prev.T.dot(prc_prev).dot(K_prev), -K_prev.T.dot(prc_prev)],
            np.c_[-prc_prev.dot(K_prev), prc_prev]
        ]
        M_new = np.r_[
            np.c_[K_new.T.dot(prc_new).dot(K_new), -K_new.T.dot(prc_new)],
            np.c_[-prc_new.dot(K_new), prc_new]
        ]
        v_prev = np.r_[K_prev.T.dot(prc_prev).dot(k_prev),
                       -prc_prev.dot(k_prev)]
        v_new = np.r_[K_new.T.dot(prc_new).dot(k_new), -prc_new.dot(k_new)]
        c_prev = 0.5 * k_prev.T.dot(prc_prev).dot(k_prev)
        c_new = 0.5 * k_new.T.dot(prc_new).dot(k_new)

        # Compute KL divergence at timestep t.
        kl_div[t] = max(
            0,
            -0.5 * mu_t.T.dot(M_new - M_prev).dot(mu_t) -
            mu_t.T.dot(v_new - v_prev) - c_new + c_prev -
            0.5 * np.sum(sigma_t * (M_new-M_prev)) - 0.5 * logdet_new +
            0.5 * logdet_prev
        )

    # Add up divergences across time to get total divergence.
    return np.sum(kl_div) if tot else kl_div


def traj_distr_kl_alt(new_mu, new_sigma, new_traj_distr, prev_traj_distr, tot=True):
    """
    This function computes the same quantity as the function above.
    However, it is easier to modify and understand this function, i.e.,
    passing in a different mu and sigma to this function will behave properly.
    """
    T, dX, dU = new_mu.shape[0], new_traj_distr.state_dimensions, new_traj_distr.action_dimensions
    kl_div = np.zeros(T)

    for t in range(T):
        K_prev = prev_traj_distr.K[t, :, :]
        K_new = new_traj_distr.K[t, :, :]

        k_prev = prev_traj_distr.k[t, :]
        k_new = new_traj_distr.k[t, :]

        sig_prev = prev_traj_distr.covariance[t, :, :]
        sig_new = new_traj_distr.covariance[t, :, :]

        chol_prev = prev_traj_distr.cholesky_covariance[t, :, :]
        chol_new = new_traj_distr.cholesky_covariance[t, :, :]

        inv_prev = prev_traj_distr.inv_covariance[t, :, :]
        inv_new = new_traj_distr.inv_covariance[t, :, :]

        logdet_prev = 2 * sum(np.log(np.diag(chol_prev)))
        logdet_new = 2 * sum(np.log(np.diag(chol_new)))

        K_diff, k_diff = K_prev - K_new, k_prev - k_new
        mu, sigma = new_mu[t, :dX], new_sigma[t, :dX, :dX]

        kl_div[t] = max(
                0,
                0.5 * (logdet_prev - logdet_new - new_traj_distr.action_dimensions +
                       np.sum(np.diag(inv_prev.dot(sig_new))) +
                       k_diff.T.dot(inv_prev).dot(k_diff) +
                       mu.T.dot(K_diff.T).dot(inv_prev).dot(K_diff).dot(mu) +
                       np.sum(np.diag(K_diff.T.dot(inv_prev).dot(K_diff).dot(sigma))) +
                       2 * k_diff.T.dot(inv_prev).dot(K_diff).dot(mu))
        )

    return np.sum(kl_div) if tot else kl_div
