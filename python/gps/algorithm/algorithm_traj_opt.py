""" This file defines the iLQG-based trajectory optimization method. """
import logging
from typing import Dict, Optional, List, Tuple, Callable, Any

import numpy as np

from gps.agent import Agent
from gps.algorithm.iteration_data import AlgorithmConditionData, AlgorithmData, Trajectory, IterationData
from gps.algorithm.algorithm import ConditionDescription, Algorithm
from gps.controller import LinearGaussianController
from gps.dynamics import Dynamics
from gps.traj_opt import TrajOptLQR
from gps.utility.human_readable_formatter import human_readable_formatter as hr_formatter

LOGGER = logging.getLogger(__name__)


class TrajOptCondData(AlgorithmConditionData):
    def __init__(self, traj_distr: LinearGaussianController, eta: float, step_mult: float, kl_step: float):
        super(TrajOptCondData, self).__init__(traj_distr)
        self.eta = eta
        self.step_mult = step_mult
        self.kl_step = kl_step

    @property
    def display_data(self):
        return {"eta": self.eta, "kl_step": self.kl_step, "entropy": self.traj_distr.entropy}


class TrajOptData(AlgorithmData):
    def __init__(self):
        pass


# TODO documentation
class AlgorithmTrajOpt(Algorithm[TrajOptData, TrajOptCondData]):
    """ Sample-based trajectory optimization. """

    def __init__(self, agent: Agent, conditions: Dict[int, ConditionDescription], trajectory_optimizer: TrajOptLQR,
                 training_samples_per_condition: int,
                 training_conditions: Optional[List[int]] = None,
                 test_samples_per_condition: Optional[int] = 0,
                 test_conditions: Optional[List[int]] = None,
                 inner_iterations: int = 1, base_kl_step: float = 0.2, min_step_mult: float = 0.01,
                 max_step_mult: float = 10.0, max_entropy_weight: float = 0.0, initial_state_var: float = 1e-6):
        """

        :param agent:
        :param conditions:              Dictionary containing the condition descriptions for each condition.
        :param trajectory_optimizer:    Trajectory optimizer to use
        :param inner_iterations:        Number of trajectory optimization iterations to do in each iteration
        :param base_kl_step:
        :param min_step_mult:           Minimum of step multiplier of KL step
        :param max_step_mult:           Maximum of step multiplier of KL step
        :param max_entropy_weight:      Weight of maximum entropy term in trajectory optimization.
        :param training_conditions:     List of conditions to use for training
        """
        super(AlgorithmTrajOpt, self).__init__(
            agent=agent,
            conditions=conditions,
            training_samples_per_condition=training_samples_per_condition,
            training_conditions=training_conditions,
            test_samples_per_condition=test_samples_per_condition,
            test_conditions=test_conditions,
            policy_samples_per_condition=0,
            initial_state_var=initial_state_var,
            display_name="AlgorithmTrajOpt")
        self.__base_kl_step = base_kl_step
        self.__inner_iterations = inner_iterations
        self.__traj_opt = trajectory_optimizer
        self.__max_entropy_weight = max_entropy_weight
        self.__min_step_mult = min_step_mult
        self.__max_step_mult = max_step_mult

    def _initialize_new(self) -> Tuple[TrajOptData, Dict[int, TrajOptCondData]]:
        cond_data = {
            c: TrajOptCondData(self.conditions[c].initial_traj_distr, eta=1.0, step_mult=1.0,
                               kl_step=self.__base_kl_step)
            for c in self.training_conditions
        }
        return TrajOptData(), cond_data

    def _iteration(self, trajectories: Dict[int, Trajectory],
                   previous_iteration: IterationData[TrajOptData, TrajOptCondData]) \
            -> Tuple[TrajOptData, Dict[int, TrajOptCondData]]:
        """
        Run iteration of LQR.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """

        # Prepare variables for inner loop
        cond_data = previous_iteration.cond_data
        itr_data = {}
        for c in self.training_conditions:
            controller = trajectories[c].controller
            eta = cond_data[c].algorithm_data.eta
            if previous_iteration.iteration_no > 0:
                step_mult = self.__compute_new_step_mult(cond_data[c].algorithm_data.step_mult,
                                                         cond_data[c].training_trajectory,
                                                         trajectories[c])
            else:
                step_mult = cond_data[c].algorithm_data.step_mult
            itr_data[c] = (controller, eta, step_mult)

        # Run inner loop to compute new policies.
        for _ in range(self.__inner_iterations):
            for cond in self.training_conditions:
                traj_distr, eta, step_mult = itr_data[cond]
                traj = trajectories[cond]

                def cost_function(eta, augment):
                    """ Compute cost estimates used in the LQR backward pass. """
                    if not augment:  # Whether to augment cost with term to penalize KL
                        return traj.Cm, traj.cv

                    multiplier = self.__max_entropy_weight
                    fCm, fcv = traj.Cm / (eta + multiplier), traj.cv / (eta + multiplier)
                    K, ipc, k = traj_distr.K, traj_distr.inv_covariance, traj_distr.k

                    # Add in the trajectory divergence term.
                    for t in range(self.agent.time_steps - 1, -1, -1):
                        fCm[t, :, :] += eta / (eta + multiplier) * np.vstack([
                            np.hstack([
                                K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                                -K[t, :, :].T.dot(ipc[t, :, :])
                            ]),
                            np.hstack([
                                -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                            ])
                        ])
                        fcv[t, :] += eta / (eta + multiplier) * np.hstack([
                            K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),
                            -ipc[t, :, :].dot(k[t, :])
                        ])

                    return fCm, fcv

                # Run trajectory optimizer
                kl_step = self.__base_kl_step * step_mult
                traj_distr, eta = self.__traj_opt.update(eta, kl_step, traj.controller, traj.x0mu,
                                                         traj.x0sigma, traj.dynamics, cost_function)
                itr_data[cond] = traj_distr, eta, step_mult

        return TrajOptData(), {c: TrajOptCondData(*d, self.__base_kl_step * d[-1]) for c, d in itr_data.items()}

    def __compute_new_step_mult(
            self, previous_step_mult: float, previous_trajectory: Trajectory, current_trajectory: Trajectory) -> float:
        """ Evaluate costs on samples, and adjust the step size relative to previous iteration. """
        prev = previous_trajectory
        curr = current_trajectory
        prev_traj_distr: LinearGaussianController = prev.controller
        curr_traj_distr: LinearGaussianController = curr.controller

        # Compute values under Laplace approximation. This is the controller
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        previous_laplace_obj = AlgorithmTrajOpt.__estimate_cost(
            prev_traj_distr, prev.dynamics, prev.x0mu, prev.x0sigma, prev.cc, prev.cv, prev.Cm)

        # This is the controller that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_obj = AlgorithmTrajOpt.__estimate_cost(
            curr_traj_distr, prev.dynamics, prev.x0mu, prev.x0sigma, prev.cc, prev.cv, prev.Cm)

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        new_actual_laplace_obj = AlgorithmTrajOpt.__estimate_cost(
            curr_traj_distr, curr.dynamics, curr.x0mu, curr.x0sigma, curr.cc, curr.cv, curr.Cm)

        LOGGER.debug("Trajectory step: ent: {} cost: {} -> {}".format(
            curr_traj_distr.entropy, prev.mean_cost, curr.mean_cost))

        # Compute predicted and actual improvement.
        predicted_impr: float = np.sum(previous_laplace_obj) - \
                                np.sum(new_predicted_laplace_obj)
        actual_impr: float = np.sum(previous_laplace_obj) - \
                             np.sum(new_actual_laplace_obj)

        # Print improvement details.
        LOGGER.debug("Previous cost: Laplace: {} MC: {}".format(
            np.sum(previous_laplace_obj), prev.mean_cost))
        LOGGER.debug("Predicted new cost: Laplace: {} MC: {}".format(
            np.sum(new_predicted_laplace_obj), curr.mean_cost))
        LOGGER.debug("Actual new cost: Laplace: {} MC: {}".format(
            np.sum(new_actual_laplace_obj), curr.mean_cost))
        LOGGER.debug("Predicted/actual improvement: {} / {}".format(
            predicted_impr, actual_impr))

        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * previous_step_mult, self.__max_step_mult), self.__min_step_mult)

        if new_mult > 1:
            LOGGER.debug("Increasing step size multiplier to {}".format(new_step))
        else:
            LOGGER.debug("Decreasing step size multiplier to {}".format(new_step))

        return new_step

    @staticmethod
    def __estimate_cost(traj_distr: LinearGaussianController, dynamics: Dynamics, x0mu: np.ndarray, x0sigma: np.ndarray,
                        cc: np.ndarray, cv: np.ndarray, Cm: np.ndarray) -> np.ndarray:
        """
        Compute Laplace approximation to expected cost.
        :param traj_distr:  Trajectory distribution to be used for estimation
        :param dynamics:    Dynamics to be used for estimation
        :param x0mu:        Mean of initial state distribution
        :param x0sigma:     Covariance of initial state distribution
        :param cc:          Taylor approximation of cost function - constant term
        :param cv:          Taylor approximation of cost function - linear term
        :param Cm:          Taylor approximation of cost function - quadratic term
        :return:
        """
        # Perform forward pass
        mu, sigma = TrajOptLQR.forward(traj_distr, dynamics, x0mu, x0sigma)

        # Compute cost.
        predicted_cost = np.zeros(traj_distr.time_steps)
        for t in range(traj_distr.time_steps):
            predicted_cost[t] = cc[t] + \
                                mu[t, :].T.dot(cv[t, :]) + \
                                0.5 * np.sum(sigma[t, :, :] * Cm[t, :, :]) + \
                                0.5 * mu[t, :].T.dot(Cm[t, :, :]).dot(mu[t, :])
        return predicted_cost

    @property
    def _display_data_description(self) -> List[Tuple[bool, str, str, Callable[[Any], str], int]]:
        h = hr_formatter
        return [(True, "kl_step", "kl_step", h(2), 10),
                (True, "entropy", "entropy", h(2), 10),
                (True, "eta", "eta", h(2), 10)]
