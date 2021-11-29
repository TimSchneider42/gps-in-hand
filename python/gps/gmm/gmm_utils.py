from typing import Optional

import numpy as np
import scipy


def logsum(vec, axis=0, keepdims=True):
    # TODO: Add a docstring.
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec - maxv), axis=axis, keepdims=keepdims)) + maxv


def compute_log_obs(data: np.ndarray, means: Optional[np.ndarray] = None, covariances: Optional[np.ndarray] = None,
                    log_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute log observation probabilities for the given data points for the given GMM parameters.
    :param data:        Data points to compute log observation probabilities for.
    :param means:          Means of each cluster.
    :param covariances:       Covariance of each cluster.
    :param log_weights: Log weights of each cluster.
    :return: A (NUM_POINTS x NUM_CLUSTERS) array of log log probabilities (for each point on each cluster).
    """
    # Constants.
    point_count, dimensions = data.shape
    cluster_count = covariances.shape[0]

    log_obs = -0.5 * np.ones((point_count, cluster_count)) * dimensions * np.log(2 * np.pi)
    for i in range(cluster_count):
        L = scipy.linalg.cholesky(covariances[i], lower=True)
        log_obs[:, i] -= np.sum(np.log(np.diag(L)))

        diff = (data - means[i]).T
        soln = scipy.linalg.solve_triangular(L, diff, lower=True)
        log_obs[:, i] -= 0.5 * np.sum(soln ** 2, axis=0)

    log_obs += log_weights.T
    return log_obs


def gauss_fit_joint_prior_old(pts, mu0, Phi, m, n0, dwts, dXU, dX, sig_reg):
    """ Perform Gaussian fit to data with a prior. """
    # Build weights matrix.
    D = np.diag(dwts)
    # Compute empirical mean and covariance.
    mun = np.sum((pts.T * dwts).T, axis=0)
    diff = pts - mun
    empsig = diff.T.dot(D).dot(diff)
    empsig = 0.5 * (empsig + empsig.T)
    # MAP estimate of joint distribution.
    N = dwts.shape[0]
    # mu = mun
    mu = (m * mu0 + n0 * mun) / (m + n0)
    sigma = (N * empsig + Phi + (N * m) / (N + m) *
             np.outer(mun - mu0, mun - mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T)
    # Add sigma regularization.
    sigma += sig_reg
    # Conditioning to get dynamics.
    fd = np.linalg.solve(sigma[:dXU, :dXU], sigma[:dXU, dXU:dXU + dX]).T
    fc = mu[dXU:dXU + dX] - fd.dot(mu[:dXU])
    dynsig = sigma[dXU:dXU + dX, dXU:dXU + dX] - fd.dot(sigma[:dXU, :dXU]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)
    return fd, fc, dynsig


def gauss_fit_joint_prior(pts, mu0, Phi, m, n0, dwts, dXU, dX, sig_reg):
    """ Perform Gaussian fit to data with a prior. """
    # Build weights matrix.
    D = np.diag(dwts)
    # Compute empirical mean and covariance.
    mun = np.sum((pts.T * dwts).T, axis=0)
    diff = pts - mun
    empsig = diff.T.dot(D).dot(diff)
    empsig = 0.5 * (empsig + empsig.T)

    # MAP estimate of joint distribution.
    N = dwts.shape[0]
    # mu = mun
    mu = (m * mu0 + n0 * mun) / (m + n0)
    # mu = mun
    sigma = (N * empsig + Phi + (N * m) / (N + m) *
             np.outer(mun - mu0, mun - mu0)) / (N + n0)
    # sigma = empsig

    # sig_diag = np.diag(sigma)
    # avg_sig = np.average(sig_diag)
    # normal_variance = (sig_diag >= 0.001 * avg_sig)[:dXU]
    normal_variance_indices = np.nonzero(np.diag(empsig)[:dXU])[0]
    normal_sigma_xu = sigma[normal_variance_indices, :][:, normal_variance_indices]
    normal_sigma_xux = sigma[normal_variance_indices, :][:, dXU:dXU + dX]
    sigma = 0.5 * (sigma + sigma.T)
    # Add sigma regularization.
    sigma += sig_reg
    # Conditioning to get dynamics.
    normal_fd = np.linalg.solve(normal_sigma_xu, normal_sigma_xux).T
    fd = np.zeros((dX, dXU))
    fd[:, normal_variance_indices] = normal_fd
    fc = mu[dXU:dXU + dX] - fd.dot(mu[:dXU])
    dynsig = sigma[dXU:dXU + dX, dXU:dXU + dX] - fd.dot(sigma[:dXU, :dXU]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)
    return fd, fc, dynsig
