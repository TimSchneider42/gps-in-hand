""" This file defines linear regression with an arbitrary prior. """
from typing import Optional

import numpy as np

from gps.dynamics import Dynamics, DynamicsPrior
from gps.gmm import gauss_fit_joint_prior
from gps.sample import SampleList


class DynamicsLR(Dynamics):
    """ Dynamics with linear regression, with arbitrary prior. """

    def __init__(self, Fm: Optional[np.ndarray] = None, fv: Optional[np.ndarray] = None,
                 covariance: Optional[np.ndarray] = None, prior: Optional[DynamicsPrior] = None,
                 regularization: float = 1e-6):
        super(DynamicsLR, self).__init__(Fm, fv, covariance, prior)
        self.__regularization = regularization

    def fit(self, sample_list: SampleList, update_prior: bool = True):
        # Update prior
        prior = self.prior
        if update_prior and prior is not None:
            prior = prior.update(sample_list)

        """ Fit dynamics. """
        X = sample_list.states
        U = sample_list.actions
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        Fm = np.zeros([T, dX, dX + dU])
        fv = np.zeros([T, dX])
        covariance = np.zeros([T, dX, dX])

        it = slice(dX + dU)
        ip = slice(dX + dU, dX + dU + dX)

        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T - 1):
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t + 1, :]]
            if prior is not None:
                # Obtain Normal-inverse-Wishart prior.
                mu0, Phi, mm, n0 = prior.eval(xux)
                sig_reg = np.zeros((dX + dU + dX, dX + dU + dX))

                sig_reg[it, it] = self.__regularization
                new_Fm, new_fv, new_covar = gauss_fit_joint_prior(xux, mu0, Phi, mm, n0, dwts, dX + dU, dX, sig_reg)
            else:
                xux_mean = np.mean(xux, axis=0)
                empsig = (xux - xux_mean).T.dot(xux - xux_mean) / N
                sigma = 0.5 * (empsig + empsig.T)
                sigma[it, it] += self.__regularization

                new_Fm = np.linalg.solve(sigma[it, it], sigma[it, ip]).T
                new_fv = xux_mean[ip] - new_Fm.dot(xux_mean[it])

                new_covar = sigma[ip, ip] - new_Fm.dot(sigma[it, it]).dot(new_Fm.T)
                new_covar = 0.5 * (new_covar + new_covar.T)

            Fm[t, :, :] = new_Fm
            fv[t, :] = new_fv
            covariance[t, :, :] = new_covar
        return DynamicsLR(Fm, fv, covariance, prior, self.__regularization)
