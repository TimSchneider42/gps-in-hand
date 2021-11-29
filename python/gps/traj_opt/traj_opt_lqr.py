""" This file defines code for iLQG-based trajectory optimization. """
import logging
from typing import Optional, Callable, Tuple, Union

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

from gps.dynamics import Dynamics
from gps.controller import LinearGaussianController
from gps.traj_opt.traj_opt_utils import \
    DGD_MAX_ITER, DGD_MAX_LS_ITER, DGD_MAX_GD_ITER, \
    ALPHA, BETA1, BETA2, EPS, \
    traj_distr_kl, traj_distr_kl_alt

LOGGER = logging.getLogger(__name__)


class TrajOptLQR:
    """ LQR trajectory optimization, Python implementation. """

    def __init__(self, min_eta: float = 1e-8, max_eta: float = 1e16, del0: float = 1e-4, cons_per_step: bool = False,
                 usr_prev_distr: bool = False, update_in_bwd_pass: bool = True):
        """

        :param min_eta:
        :param max_eta:
        :param del0:                Dual variable updates for non-positive-definite Q-function (initial value).
        :param cons_per_step:       Whether or not to enforce separate KL constraints at each time step.
                                    (different eta for each time step)
        :param usr_prev_distr:      Whether or not to measure expected KL under the previous traj distr.
        :param update_in_bwd_pass:  Whether or not to update the TVLG controller during the bwd pass.
        """
        self._cons_per_step = cons_per_step
        self._use_prev_distr = usr_prev_distr
        self._update_in_bwd_pass = update_in_bwd_pass
        self._min_eta = min_eta
        self._max_eta = max_eta
        self._del0 = del0

    # TODO - Add arg and return spec on this function.
    def update(self, eta: Union[float, np.ndarray], kl_step: float, traj_distr: LinearGaussianController,
               x0mu: np.ndarray, x0sigma: np.ndarray, dynamics: Dynamics,
               cost_function: Callable[[float, bool], Tuple[np.ndarray, np.ndarray]],
               pol_wt: Optional[np.ndarray] = None) -> Tuple[LinearGaussianController, float]:
        """ Run dual gradient decent to optimize trajectories. """
        T = traj_distr.time_steps
        if self._cons_per_step and type(eta) in (int, float):
            eta = np.ones(T) * eta

        # For BADMM/trajopt, constrain to previous LG controller
        prev_traj_distr = traj_distr

        # Set KL-divergence step size (epsilon).
        if not self._cons_per_step:
            kl_step *= T

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self._cons_per_step:
            min_eta = self._min_eta
            max_eta = self._max_eta
            LOGGER.debug(f"Running DGD, eta: {eta}")
        else:
            min_eta = np.ones(T) * self._min_eta
            max_eta = np.ones(T) * self._max_eta
            LOGGER.debug(f"Running DGD, avg eta: {np.mean(eta[:-1])}")

        max_itr = (DGD_MAX_LS_ITER if self._cons_per_step else DGD_MAX_ITER)
        con = np.inf
        kl_div = None
        for itr in range(max_itr):
            if not self._cons_per_step:
                LOGGER.debug(f"Iteration {itr}, bracket: ({min_eta:.2e} , {eta:.2e} , {max_eta:.2e})")

            # Run fwd/bwd pass, note that eta may be updated.
            # Compute KL divergence constraint violation.
            traj_distr, eta = self.backward(prev_traj_distr, eta, dynamics, cost_function, pol_wt)

            if not self._use_prev_distr:
                new_mu, new_sigma = TrajOptLQR.forward(traj_distr, dynamics, x0mu, x0sigma)
                kl_div = traj_distr_kl(
                    new_mu, new_sigma, traj_distr, prev_traj_distr,
                    tot=(not self._cons_per_step)
                )
            else:
                prev_mu, prev_sigma = TrajOptLQR.forward(prev_traj_distr, dynamics, x0mu, x0sigma)
                kl_div = traj_distr_kl_alt(
                    prev_mu, prev_sigma, traj_distr, prev_traj_distr,
                    tot=(not self._cons_per_step)
                )

            con = kl_div - kl_step

            # Convergence check - constraint satisfaction.
            if self._conv_check(con, kl_step):
                if not self._cons_per_step:
                    LOGGER.debug(f"KL: {kl_div} / {kl_step}, converged iteration {itr}")
                else:
                    LOGGER.debug(f"KL: {np.mean(kl_div[:-1])} / {np.mean(kl_step[:-1])}, converged iteration {itr}")
                break

            if not self._cons_per_step:
                # Choose new eta (bisect bracket or multiply by constant)
                if con < 0:  # Eta was too big.
                    max_eta = eta
                    geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
                    new_eta = max(geom, 0.1 * max_eta)
                    LOGGER.debug(f"KL: {kl_div} / {kl_step}, eta too big, new eta: {new_eta}")
                else:  # Eta was too small.
                    min_eta = eta
                    geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
                    new_eta = min(geom, 10.0 * min_eta)
                    LOGGER.debug(f"KL: {kl_div} / {kl_step}, eta too small, new eta: {new_eta}")

                # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
                eta = new_eta
            else:
                for t in range(T):
                    if con[t] < 0:
                        max_eta[t] = eta[t]
                        geom = np.sqrt(min_eta[t] * max_eta[t])
                        eta[t] = max(geom, 0.1 * max_eta[t])
                    else:
                        min_eta[t] = eta[t]
                        geom = np.sqrt(min_eta[t] * max_eta[t])
                        eta[t] = min(geom, 10.0 * min_eta[t])
                if itr % 10 == 0:
                    LOGGER.debug(
                        f"avg KL: {np.mean(kl_div[:-1])} / {np.mean(kl_step[:-1])}, avg new eta: {np.mean(eta[:-1])}")

        if self._cons_per_step and not self._conv_check(con, kl_step):
            m_b, v_b = np.zeros(T - 1), np.zeros(T - 1)

            for itr in range(DGD_MAX_GD_ITER):
                traj_distr, eta = self.backward(prev_traj_distr, eta, dynamics, cost_function, pol_wt)

                if not self._use_prev_distr:
                    new_mu, new_sigma = TrajOptLQR.forward(traj_distr, dynamics, x0mu, x0sigma)
                    kl_div = traj_distr_kl(
                        new_mu, new_sigma, traj_distr, prev_traj_distr,
                        tot=False
                    )
                else:
                    prev_mu, prev_sigma = TrajOptLQR.forward(prev_traj_distr, dynamics, x0mu, x0sigma)
                    kl_div = traj_distr_kl_alt(
                        prev_mu, prev_sigma, traj_distr, prev_traj_distr,
                        tot=False
                    )

                con = kl_div - kl_step
                if self._conv_check(con, kl_step):
                    LOGGER.debug(f"KL: {np.mean(kl_div[:-1])} / {np.mean(kl_step[:-1])}, converged iteration {itr}")
                    break

                m_b = (BETA1 * m_b + (1 - BETA1) * con[:-1])
                m_u = m_b / (1 - BETA1 ** (itr + 1))
                v_b = (BETA2 * v_b + (1 - BETA2) * np.square(con[:-1]))
                v_u = v_b / (1 - BETA2 ** (itr + 1))
                eta[:-1] = np.minimum(
                    np.maximum(eta[:-1] + ALPHA * m_u / (np.sqrt(v_u) + EPS), self._min_eta),
                    self._max_eta
                )

                if itr % 10 == 0:
                    LOGGER.debug(
                        f"avg KL: {np.mean(kl_div[:-1])} / {np.mean(kl_step[:-1])}, avg new eta: {np.mean(eta[:-1])}")

        if (np.mean(kl_div) > np.mean(kl_step) and
                not self._conv_check(con, kl_step)):
            LOGGER.warning("Final KL divergence after DGD convergence is too high.")
        return traj_distr, eta

    @staticmethod
    def forward(traj_distr: LinearGaussianController, dynamics: Dynamics, x0mu: np.ndarray, x0sigma: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Perform LQR forward pass. Computes state-action marginals from
        dynamics and controller.
        :param traj_distr:  Policy to be used.
        :param dynamics:    Dynamics to be used.
        :param x0mu:        Initial state mean.
        :param x0sigma:     Initial state sigma.
        :return:
            mu:             (T x (dX + du)) mean state/action vector
            sigma:          (T x (dX + dU) x (dX + dU)) covariance matrix
        """
        # Compute state-action marginals from specified conditional
        # parameters and current traj_info.
        T = traj_distr.time_steps
        dU = traj_distr.action_dimensions
        dX = traj_distr.state_dimensions

        # Constants.
        idx_x = slice(dX)

        # Allocate space.
        sigma = np.zeros((T, dX + dU, dX + dU))
        mu = np.zeros((T, dX + dU))

        # Pull out dynamics.
        Fm = dynamics.Fm
        fv = dynamics.fv
        dyn_covar = dynamics.covariance

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = x0sigma
        mu[0, idx_x] = x0mu

        for t in range(T):
            sigma[t, :, :] = np.vstack([
                np.hstack([
                    sigma[t, idx_x, idx_x],
                    sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)
                ]),
                np.hstack([
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(
                        traj_distr.K[t, :, :].T
                    ) + traj_distr.covariance[t, :, :]
                ])
            ])
            mu[t, :] = np.hstack([
                mu[t, idx_x],
                traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]
            ])
            if t < T - 1:
                sigma[t + 1, idx_x, idx_x] = \
                    Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \
                    dyn_covar[t, :, :]
                mu[t + 1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
        return mu, sigma

    def backward(self, prev_traj_distr: LinearGaussianController, eta: Union[float, np.ndarray], dynamics: Dynamics,
                 cost_function: Callable[[float, bool], Tuple[np.ndarray, np.ndarray]],
                 pol_wt: Optional[np.ndarray] = None) -> Tuple[LinearGaussianController, Union[float, np.ndarray]]:
        """
        Perform LQR backward pass. This computes a new linear Gaussian
        controller object.
        :param prev_traj_distr: A linear Gaussian controller object from previous iteration.
        :param eta:             Dual variable.
        :param dynamics:        Computed dynamics for prev_traj_distr.
        :param cost_function:   Function returning the tailor expansion of the cost function around the trajectory mean
                                fCm (T x (dX + dU) x (dx + dU)) and fCv (T x (dX + dU)) (the constant term is omitted),
                                given eta and boolean value indicating whether the cost function shall be augmented
                                with a term to penalize KL divergence or not.
        :param pol_wt:
        :return:
            traj_distr:         A new linear Gaussian controller.
            new_eta:            The updated dual variable. Updates happen if the Q-function is not PD.
        """
        # Constants.
        T = prev_traj_distr.time_steps
        dU = prev_traj_distr.action_dimensions
        dX = prev_traj_distr.state_dimensions

        # Variables
        if not self._update_in_bwd_pass:
            K = prev_traj_distr.K
            k = prev_traj_distr.k
            pol_covar = prev_traj_distr.covariance
            chol_pol_covar = prev_traj_distr.cholesky_covariance
            inv_pol_covar = prev_traj_distr.inv_covariance
        else:
            K = np.zeros((T, dU, dX))
            k = np.zeros((T, dU))
            pol_covar = np.zeros((T, dU, dU))
            chol_pol_covar = np.zeros((T, dU, dU))
            inv_pol_covar = np.zeros((T, dU, dU))

        # Pull out dynamics.
        Fm = dynamics.Fm
        fv = dynamics.fv

        # Non-SPD correction terms.
        del_ = self._del0
        if self._cons_per_step:
            del_ = np.ones(T) * del_
        eta0 = eta

        # Allocate arrays here to save time
        Vxx = np.empty((T, dX, dX))
        Vx = np.empty((T, dX))
        Qtt = np.empty((T, dX + dU, dX + dU))
        Qt = np.empty((T, dX + dU))

        idx_x = slice(dX)
        idx_u = slice(dX, dX + dU)

        Qxx = Qtt[:, idx_x, idx_x]
        Qux = Qtt[:, idx_u, idx_x]
        Qxu = Qtt[:, idx_x, idx_u]
        Quu = Qtt[:, idx_u, idx_u]

        Qx = Qt[:, idx_x]
        Qu = Qt[:, idx_u]

        # Run dynamic programming.
        fail = True
        while fail:
            fail = False  # Flip to true on non-symmetric PD.

            fCm, fcv = cost_function(eta, not self._cons_per_step)

            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                # Add in the cost.
                Qtt[t] = fCm[t, :, :]  # (X+U) x (X+U)
                Qt[t] = fcv[t, :]  # (X+U) x 1

                # Add in the value function from the next time step.
                if t < T - 1:
                    if pol_wt is not None:
                        multiplier = (pol_wt[t + 1] + eta) / (pol_wt[t] + eta)
                    else:
                        multiplier = 1.0
                    Qtt[t] += multiplier * Fm[t, :, :].T.dot(Vxx[t + 1, :, :]).dot(Fm[t, :, :])
                    Qt[t] += multiplier * Fm[t, :, :].T.dot(Vx[t + 1, :] + Vxx[t + 1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component.
                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                if not self._cons_per_step:
                    inv_term = Qtt[t, idx_u, idx_u]
                    k_term = Qu[t]
                    K_term = Qux[t]
                else:
                    inv_term = (1.0 / eta[t]) * Quu[t] + prev_traj_distr.inv_covariance[t]
                    k_term = (1.0 / eta[t]) * Qu[t] - prev_traj_distr.inv_covariance[t].dot(prev_traj_distr.k[t])
                    K_term = (1.0 / eta[t]) * Qux[t] - prev_traj_distr.inv_covariance[t].dot(prev_traj_distr.K[t])
                # Compute Cholesky decomposition of Q function action component.
                try:
                    inv_chol = sp.linalg.cholesky(inv_term)
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not
                    # symmetric positive definite.
                    LOGGER.debug(f"LinAlgError: {e}")
                    fail = t if self._cons_per_step else True
                    break

                # Store conditional covariance, inverse, and Cholesky.
                inv_pol_covar[t, :, :] = inv_term
                pol_covar[t, :, :] = sp.linalg.solve_triangular(
                    inv_chol, sp.linalg.solve_triangular(inv_chol.T, np.eye(dU), lower=True))
                chol_pol_covar[t, :, :] = sp.linalg.cholesky(pol_covar[t, :, :])

                # Compute mean terms.
                new_k = -sp.linalg.solve_triangular(
                    inv_chol, sp.linalg.solve_triangular(inv_chol.T, k_term, lower=True))
                new_K = -sp.linalg.solve_triangular(
                    inv_chol, sp.linalg.solve_triangular(inv_chol.T, K_term, lower=True))

                if self._update_in_bwd_pass:
                    k[t, :] = new_k
                    K[t, :, :] = new_K

                # Compute value function.
                if self._cons_per_step or not self._update_in_bwd_pass:
                    Vxx[t, :, :] = Qxx[t] + K[t].T.dot(Quu[t]).dot(K[t]) + (2 * Qxu[t]).dot(K[t])
                    Vx[t, :] = Qx[t].T + Qu[t].T.dot(K[t]) + k[t].T.dot(Quu[t]).dot(K[t]) + Qxu[t].dot(k[t])
                else:
                    Vxx[t, :, :] = Qxx[t] + Qxu[t].dot(K[t, :, :])
                    Vx[t, :] = Qx[t] + Qxu[t].dot(k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

                if not self._update_in_bwd_pass:
                    k[t, :] = new_k
                    K[t, :, :] = new_K

            # Increment eta on non-SPD Q-function.
            if fail:
                if not self._cons_per_step:
                    old_eta = eta
                    eta = eta0 + del_
                    LOGGER.debug(f"Increasing eta: {old_eta} -> {eta}")
                    del_ *= 2  # Increase del_ exponentially on failure.
                else:
                    old_eta = eta[fail]
                    eta[fail] = eta0[fail] + del_[fail]
                    LOGGER.debug(f"Increasing eta {fail}: {old_eta} -> {eta[fail]}")
                    del_[fail] *= 2  # Increase del_ exponentially on failure.
                if self._cons_per_step:
                    fail_check = (eta[fail] >= 1e16)
                else:
                    fail_check = (eta >= 1e16)
                if fail_check:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError("NaNs encountered in dynamics!")
                    raise ValueError("Failed to find PD solution even for very large eta (check that dynamics and cost "
                                     "are reasonably well conditioned)!")
        return LinearGaussianController(K, k, pol_covar, chol_pol_covar, inv_pol_covar), eta

    def _conv_check(self, con, kl_step):
        """Function that checks whether dual gradient descent has converged."""
        if self._cons_per_step:
            return all([abs(con[t]) < (0.1 * kl_step[t]) for t in range(con.size)])
        return abs(con) < 0.1 * kl_step
