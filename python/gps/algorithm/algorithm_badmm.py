""" This file defines the BADMM-based GPS algorithm. """
import logging
from enum import Enum
from typing import List, Dict, Optional, Tuple, NamedTuple, Union, TypeVar, Any, Callable

import numpy as np

from gps.agent import Agent
from gps.algorithm.iteration_data import AlgorithmData, AlgorithmConditionData, Trajectory, IterationData
from gps.algorithm.algorithm import Algorithm, ConditionDescription
from gps.controller import LinearGaussianController, Controller
from gps.dynamics import Dynamics
from gps.policy import Policy, PolicyPrior
from gps.policy_opt import PolicyOpt
from gps.sample import SampleList
from gps.traj_opt import TrajOptLQR
from gps.utility.human_readable_formatter import human_readable_formatter as hr_formatter

LOGGER = logging.getLogger(__name__)

POLICY_TYPE = TypeVar("PT", bound=Policy)


class LGStepMode(Enum):
    FIXED = 0
    CONSTRAINT_SAT = 1
    DIFF_AVERAGE = 2
    STANDARD = 3
    TARGET_KL = 4


PolicyInfo = NamedTuple("PolicyInfo",
                        (("estimated_kl", np.ndarray),  # Monte-Carlo estimate of KL divergence
                         ("lin_K", np.ndarray),  # Linearized controller attributes
                         ("lin_k", np.ndarray),
                         ("lin_covariance", np.ndarray),
                         ("lin_cholesky_covariance", np.ndarray),
                         ("action_mean", np.ndarray),  # Mean and covariance of the controller
                         ("action_covariance", np.ndarray),
                         ("action_covariance_inv", np.ndarray),
                         ("action_covariance_log_det", np.ndarray)
                         ))


class BADMMData(AlgorithmData):
    def __init__(self, policy: POLICY_TYPE):
        self.__policy = policy

    @property
    def policy(self) -> POLICY_TYPE:
        return self.__policy


class BADMMCondData(AlgorithmConditionData):
    def __init__(self, traj_distr: LinearGaussianController, policy_prior: PolicyPrior,
                 step_mult: float, kl_step: float, policy_weight: np.ndarray, eta: float, lambda_K: np.ndarray,
                 lambda_k: np.ndarray, mean_lambda: float, policy_info: Optional[PolicyInfo] = None):
        AlgorithmConditionData.__init__(self, traj_distr)
        self.policy_prior = policy_prior
        self.policy_info = policy_info
        self.step_mult = step_mult
        self.policy_weight = policy_weight
        self.eta = eta
        self.lambda_K = lambda_K
        self.lambda_k = lambda_k

        # Just for display
        self.kl_step = kl_step
        self.mean_lambda = mean_lambda

    @property
    def display_data(self):
        return {
            "eta": self.eta,
            "mean_policy_weight": np.mean(self.policy_weight),
            "mean_lambda": self.mean_lambda,
            "entropy": self.traj_distr.entropy,
            "kl_step": self.kl_step,
            "kl_div_pol": np.nan if self.policy_info is None else np.mean(self.policy_info.estimated_kl)
        }


class AlgorithmBADMM(Algorithm):
    """
    Sample-based joint controller learning and trajectory optimization with
    BADMM-based guided controller search algorithm.
    """

    def __init__(self, agent: Agent, conditions: Dict[int, ConditionDescription], initial_policy: POLICY_TYPE,
                 policy_optimizer: PolicyOpt[POLICY_TYPE],
                 initial_policy_prior: PolicyPrior,
                 trajectory_optimizer: TrajOptLQR,
                 training_samples_per_condition: int,
                 training_conditions: Optional[List[int]] = None,
                 test_samples_per_condition: int = 1,
                 test_conditions: Optional[List[int]] = None,
                 policy_samples_per_condition: int = 1,
                 policy_conditions: Optional[List[int]] = None,
                 initial_state_var: float = 1e-6, inner_iterations: int = 4, base_kl_step: float = 0.2,
                 min_kl_step_mult: float = 0.01, max_kl_step_mult: float = 10.0, max_entropy_weight: float = 0.0,
                 policy_dual_rate: float = 0.1, policy_dual_rate_covar: float = 0.0, init_policy_weight: float = 0.01,
                 lg_step_mode: LGStepMode = LGStepMode.STANDARD, lg_step_schedule: Union[np.ndarray, float] = 10.0,
                 entropy_regularization_schedule: Union[np.ndarray, float] = 0.0,
                 exp_step_increase: float = 2.0, exp_step_decrease: float = 0.5, exp_step_upper: float = 0.5,
                 exp_step_lower: float = 1.0, target_kl: float = 0.1,
                 prior_clear_on_itr_start: bool = False):
        """

        :param agent:
        :param conditions:
        :param initial_policy_prior:
        :param trajectory_optimizer:
        :param policy_optimizer:
        :param training_samples_per_condition:
        :param training_conditions:
        :param test_samples_per_condition:
        :param test_conditions:
        :param initial_state_var:
        :param inner_iterations:
        :param base_kl_step:
        :param min_kl_step_mult:
        :param max_kl_step_mult:
        :param max_entropy_weight:              Weight of maximum entropy term in trajectory optimization.
        :param prior_clear_on_itr_start:
        """
        super(AlgorithmBADMM, self).__init__(agent=agent, conditions=conditions,
                                             training_samples_per_condition=training_samples_per_condition,
                                             training_conditions=training_conditions,
                                             test_samples_per_condition=test_samples_per_condition,
                                             test_conditions=test_conditions,
                                             policy_samples_per_condition=policy_samples_per_condition,
                                             policy_conditions=policy_conditions,
                                             initial_state_var=initial_state_var,
                                             display_name="AlgorithmBADMM")

        self.__policy_optimizer = policy_optimizer
        self.__initial_policy = initial_policy
        self.__inner_iterations = inner_iterations
        self.__min_kl_step_mult = min_kl_step_mult
        self.__max_kl_step_mult = max_kl_step_mult
        self.__base_kl_step = base_kl_step
        self.__initial_policy_prior = initial_policy_prior
        self.__prior_clear_on_itr_start = prior_clear_on_itr_start
        self.__traj_opt = trajectory_optimizer
        self.__max_entropy_weight = max_entropy_weight
        self.__lg_step_mode = lg_step_mode
        self.__exp_step_increase = exp_step_increase
        self.__exp_step_decrase = exp_step_decrease
        self.__exp_step_upper = exp_step_upper
        self.__exp_step_lower = exp_step_lower
        self.__policy_dual_rate = policy_dual_rate
        self.__policy_dual_rate_covar = policy_dual_rate_covar
        self.__initial_policy_weight = init_policy_weight
        self.__target_kl = target_kl

        if not isinstance(lg_step_schedule, np.ndarray):
            lg_step_schedule = np.array([lg_step_schedule])
        self.__lg_step_schedule = lg_step_schedule
        if not isinstance(entropy_regularization_schedule, np.ndarray):
            entropy_regularization_schedule = np.array([entropy_regularization_schedule])
        self.__entropy_regularization_schedule = entropy_regularization_schedule

    def _initialize_new(self) -> \
            Tuple[BADMMData, Dict[int, BADMMCondData]]:
        ts = self.agent.time_steps
        du = self.agent.action_dimensions
        dx = self.agent.state_dimensions
        cond_data = {
            c: BADMMCondData(traj_distr=self.conditions[c].initial_traj_distr,
                             policy_prior=self.__initial_policy_prior,
                             step_mult=1.0,
                             kl_step=self.__base_kl_step,
                             policy_weight=self.__initial_policy_weight * np.ones(ts),
                             eta=1.0,
                             lambda_K=np.zeros((ts, du, dx)),
                             lambda_k=np.zeros((ts, du)),
                             mean_lambda=0)
            for c in self.training_conditions
        }

        data = BADMMData(self.__initial_policy)

        return data, cond_data

    def _iteration(self, trajectories: Dict[int, Trajectory],
                   previous_iteration: IterationData[BADMMData, BADMMCondData]) -> \
            Tuple[BADMMData, Dict[int, BADMMCondData]]:

        # Prepare iteration
        ent_schedule_index = max(previous_iteration.iteration_no, len(self.__entropy_regularization_schedule))
        entropy_regularization = self.__entropy_regularization_schedule[
            min(ent_schedule_index, len(self.__entropy_regularization_schedule) - 1)]

        lg_schedule_index = max(previous_iteration.iteration_no, len(self.__lg_step_schedule))
        lg_step = self.__lg_step_schedule[min(lg_schedule_index, len(self.__lg_step_schedule) - 1)]

        cond_data = {}
        previous_policy = previous_iteration.algorithm_data.policy
        for c in self.training_conditions:
            cd = previous_iteration.cond_data[c]
            ad = cd.algorithm_data

            # Update controller prior
            replace_samples = self.__prior_clear_on_itr_start
            policy_prior = ad.policy_prior.update(trajectories[c].samples, previous_policy, replace_samples)
            policy_info = self.__compute_policy_info(trajectories[c], previous_policy, policy_prior, ad.lambda_K,
                                                     ad.lambda_k)

            # Adjust step size relative to the previous iteration.
            if previous_iteration.iteration_no > 0:
                step_mult = self.__compute_new_step_mult(
                    cd.algorithm_data.step_mult, trajectories[c], policy_info, cd.training_trajectory,
                    ad.policy_info, ad.policy_weight)
            else:
                step_mult = cd.algorithm_data.step_mult
            cond_data[c] = BADMMCondData(
                traj_distr=ad.traj_distr,
                policy_prior=policy_prior,
                step_mult=step_mult,
                kl_step=step_mult * self.__base_kl_step,
                policy_weight=ad.policy_weight,
                eta=ad.eta,
                lambda_K=ad.lambda_K,
                lambda_k=ad.lambda_k,
                mean_lambda=self.__compute_mean_lambda(trajectories[c].samples, ad.lambda_K, ad.lambda_k),
                policy_info=policy_info)

        # Run inner loop to compute new policies.
        current_policy = previous_policy
        for inner_itr in range(self.__inner_iterations):
            first_iteration = previous_iteration.iteration_no == 0 and inner_itr == 0

            # No need to update the policy before the trajectory has been updated
            if not first_iteration:
                # Update the controller.
                current_policy = self.__update_policy(current_policy, cond_data, trajectories, entropy_regularization)

            for c, d in cond_data.items():
                if not first_iteration:
                    # Update controller prior
                    new_policy_prior = d.policy_prior.update(trajectories[c].samples, current_policy)
                    new_policy_info = self.__compute_policy_info(trajectories[c], current_policy, new_policy_prior,
                                                                 d.lambda_K, d.lambda_k)
                else:
                    # No need to update the policy prior and info if the policy has not been updated yet
                    new_policy_prior = d.policy_prior
                    new_policy_info = d.policy_info

                new_lambda_K = d.lambda_K
                new_lambda_k = d.lambda_k
                new_policy_weight = d.policy_weight

                if not first_iteration:
                    # Update dual variables.
                    new_lambda_K, new_lambda_k = self.__update_dual_variables(
                        trajectories[c], new_policy_info, new_policy_weight, d.traj_distr, d.lambda_K, d.lambda_k)
                    if inner_itr == self.__inner_iterations - 1:
                        new_policy_weight = self.__update_policy_weight(new_policy_info, d.policy_weight, lg_step,
                                                                        d.policy_info.estimated_kl)

                # Use LQR to update trajectory distributions
                t = trajectories[c]
                cost_function = self.__create_cost_function(
                    d.traj_distr, t, new_policy_info, new_policy_weight, new_lambda_K, new_lambda_k)
                traj_distr, eta = self.__traj_opt.update(
                    d.eta, d.step_mult * self.__base_kl_step, trajectories[c].controller, t.x0mu, t.x0sigma, t.dynamics,
                    cost_function, new_policy_weight)
                cond_data[c] = BADMMCondData(
                    traj_distr=traj_distr,
                    policy_prior=new_policy_prior,
                    step_mult=d.step_mult,
                    kl_step=d.step_mult * self.__base_kl_step,
                    policy_weight=new_policy_weight,
                    eta=eta,
                    lambda_K=new_lambda_K,
                    lambda_k=new_lambda_k,
                    mean_lambda=self.__compute_mean_lambda(trajectories[c].samples, new_lambda_K, new_lambda_k),
                    policy_info=new_policy_info)

        return BADMMData(current_policy), cond_data

    def __compute_mean_lambda(self, samples: SampleList, lambda_K: np.ndarray, lambda_k: np.ndarray) -> float:
        return float(
            np.mean([np.abs(lK.dot(x) + lk) for s in samples for lk, lK, x in zip(lambda_k, lambda_K, s.states)]))

    def __update_policy(self, policy: Policy, cond_data: Dict[int, BADMMCondData],
                        trajectories: Dict[int, Trajectory], entropy_regularization: float) -> Policy:
        """ Compute the new controller. """
        ts = self.agent.time_steps
        du = self.agent.action_dimensions
        do = self.agent.observation_dimensions
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, ts, do)), np.zeros((0, ts, du))
        tgt_prc, tgt_wt = np.zeros((0, ts, du, du)), np.zeros((0, ts))
        for c in self.training_conditions:
            cd = cond_data[c]
            traj_distr = cd.traj_distr
            samples = trajectories[c].samples

            states = samples.states
            n = len(samples)
            mu = np.zeros((n, ts, du))
            prc = np.zeros((n, ts, du, du))
            wt = np.zeros((n, ts))
            # Get time-indexed actions.
            for t in range(ts):
                # Compute actions along this trajectory.
                prc[:, t, :, :] = np.tile(traj_distr.inv_covariance[t, :, :], [n, 1, 1])
                for i in range(n):
                    lambda_sum = cd.lambda_K[t, :, :].dot(states[i, t, :]) + cd.lambda_k[t, :]
                    mu[i, t, :] = \
                        (traj_distr.K[t, :, :].dot(states[i, t, :]) + traj_distr.k[t, :]) - \
                        traj_distr.covariance[t, :, :] @ lambda_sum / cd.policy_weight[t]
                wt[:, t].fill(cd.policy_weight[t])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.observations))
        return self.__policy_optimizer.update(policy, obs_data, tgt_mu, tgt_prc, tgt_wt, entropy_regularization)

    def __compute_policy_info(self, trajectory: Trajectory, policy: Controller, policy_prior: PolicyPrior,
                              lambda_K: np.ndarray, lambda_k: np.ndarray) -> PolicyInfo:
        """
        Re-estimate the local controller values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
            init: Whether this is the initial fitting of the controller.
        """
        # Choose samples to use.
        samples = trajectory.samples
        obs = samples.observations
        action_mean = np.array([policy.probe(observations=s) for s in obs])
        action_covar = policy.covariance
        action_covar_inv = policy.inv_covariance
        sign, action_cov_log_det = np.linalg.slogdet(policy.covariance)

        # Fit linearization and store in pol_info.
        lin_K, lin_k, lin_covar = policy_prior.fit(samples.states, action_mean, action_covar)
        lin_chol_covar = np.array([np.linalg.cholesky(s) for s in lin_covar])

        estimated_kl = self.__policy_kl(trajectory, action_mean, action_covar_inv, action_cov_log_det, lambda_K,
                                        lambda_k)[0]

        return PolicyInfo(estimated_kl, lin_K, lin_k, lin_covar, lin_chol_covar, action_mean, action_covar,
                          action_covar_inv, action_cov_log_det)

    def __update_dual_variables(self, trajectory: Trajectory, new_policy_info: PolicyInfo, policy_weight: np.ndarray,
                                traj_distr: LinearGaussianController, lambda_K: np.ndarray, lambda_k: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the dual variables.
        Args:
            m: Condition
            step: Whether or not to update pol_wt.
        """
        action_dim = self.agent.action_dimensions
        ts = self.agent.time_steps
        num_samples = len(trajectory.samples)
        states = trajectory.samples.states

        new_lambda_K = lambda_K.copy()
        new_lambda_k = lambda_k.copy()

        # Compute trajectory action at each sampled state.
        traj_mu = np.zeros((num_samples, ts, action_dim))
        for i in range(num_samples):
            for t in range(ts):
                traj_mu[i, t, :] = traj_distr.K[t, :, :].dot(states[i, t, :]) + traj_distr.k[t, :]

        # Compute the difference and increment based on pol_wt.
        for t in range(ts):
            tU, pU = traj_mu[:, t, :], new_policy_info.action_mean[:, t, :]
            # Increment mean term.
            new_lambda_k[t, :] -= self.__policy_dual_rate * \
                                  policy_weight[t] * \
                                  traj_distr.inv_covariance[t, :, :].dot(np.mean(tU - pU, axis=0))
            # Increment covariance term.
            t_covar, p_covar = traj_distr.K[t, :, :], new_policy_info.lin_K[t, :, :]
            new_lambda_K[t, :, :] -= \
                self.__policy_dual_rate_covar * \
                policy_weight[t] * \
                traj_distr.inv_covariance[t, :, :].dot(t_covar - p_covar)
        return new_lambda_K, new_lambda_k

    def __update_policy_weight(self, new_policy_info: PolicyInfo, policy_weight: np.ndarray,
                               lg_step: Optional[np.ndarray] = None,
                               previous_kl: Optional[np.ndarray] = None) -> np.ndarray:
        # Get KL divergence.
        kl = new_policy_info.estimated_kl

        # Increment pol_wt based on change in KL divergence.
        if self.__lg_step_mode == LGStepMode.FIXED:
            # Take fixed size step.
            new_policy_weight = np.maximum(policy_weight + lg_step, 0)
        elif self.__lg_step_mode == LGStepMode.CONSTRAINT_SAT:
            # (In/De)crease based on change in constraint
            # satisfaction.
            new_policy_weight = policy_weight.copy()
            if previous_kl is not None:
                kl_change = kl / previous_kl
                new_policy_weight[kl_change < 0.8] *= 0.5
                new_policy_weight[kl_change >= 0.95] *= 2.0
        elif self.__lg_step_mode in [LGStepMode.DIFF_AVERAGE, LGStepMode.TARGET_KL]:
            # (In/De)crease based on difference from average.
            new_policy_weight = policy_weight.copy()
            if self.__lg_step_mode == LGStepMode.DIFF_AVERAGE:
                lower = np.mean(kl) - self.__exp_step_lower * np.std(kl)
                upper = np.mean(kl) + self.__exp_step_upper * np.std(kl)
            else:
                lower = upper = self.__target_kl
            new_policy_weight[kl < lower] *= self.__exp_step_decrase
            new_policy_weight[kl >= upper] *= self.__exp_step_increase
        else:
            # Standard DGD step.
            new_policy_weight = np.maximum(policy_weight + lg_step * kl, 0)
        return new_policy_weight

    def __compute_new_step_mult(self, previous_step_mult: float,
                                current_trajectory: Trajectory, current_policy_info: PolicyInfo,
                                previous_trajectory: Trajectory, previous_policy_info: PolicyInfo,
                                policy_weight: np.ndarray):
        """
        Evaluate costs on samples, and adjust the step size relative to previous iteration.
        """

        prev = previous_trajectory
        curr = current_trajectory
        prev_traj_distr: LinearGaussianController = prev.controller
        curr_traj_distr: LinearGaussianController = curr.controller

        # Compute values under Laplace approximation. This is the controller
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        prev_laplace_obj, prev_laplace_kl = AlgorithmBADMM.__estimate_cost(
            prev_traj_distr, prev.dynamics, prev.x0mu, prev.x0sigma, prev.cc, prev.cv, prev.Cm, previous_policy_info)

        # This is the controller that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_pred_laplace_obj, new_pred_laplace_kl = AlgorithmBADMM.__estimate_cost(
            curr_traj_distr, prev.dynamics, prev.x0mu, prev.x0sigma, prev.cc, prev.cv, prev.Cm, previous_policy_info)

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        new_actual_laplace_obj, new_actual_laplace_kl = AlgorithmBADMM.__estimate_cost(
            curr_traj_distr, curr.dynamics, curr.x0mu, curr.x0sigma, curr.cc, curr.cv, curr.Cm, current_policy_info)

        # Get the entropy of the current trajectory (for printout).
        ent = curr_traj_distr.entropy

        # Get sample-based estimate of KL divergence between controller and trajectories.
        new_mc_kl = current_policy_info.estimated_kl
        prev_mc_kl = previous_policy_info.estimated_kl

        new_mc_kl_sum = np.sum(new_mc_kl)
        prev_mc_kl_sum = np.sum(prev_mc_kl)

        # Compute full controller KL divergence objective terms by applying
        # the Lagrange multipliers.
        prev_laplace_kl_sum = np.sum(prev_laplace_kl * policy_weight)
        new_pred_laplace_kl_sum = np.sum(new_pred_laplace_kl * policy_weight)
        new_actual_laplace_kl_sum = np.sum(new_actual_laplace_kl * policy_weight)
        prev_mc_kl_w_sum = np.sum(prev_mc_kl * policy_weight)
        new_mc_kl_w_sum = np.sum(new_mc_kl * policy_weight)

        LOGGER.debug(
            f"Trajectory step: ent: {ent} cost: {prev.mean_cost} -> {curr.mean_cost} "
            f"KL: {prev_mc_kl_w_sum} -> {new_mc_kl_w_sum}")

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(prev_laplace_obj) + prev_laplace_kl_sum - \
                         np.sum(new_pred_laplace_obj) - new_pred_laplace_kl_sum
        actual_impr = np.sum(prev_laplace_obj) + prev_laplace_kl_sum - \
                      np.sum(new_actual_laplace_obj) - new_actual_laplace_kl_sum

        # Print improvement details.
        LOGGER.debug(f"Previous cost: Laplace: {np.sum(prev_laplace_obj)} MC: {prev.mean_cost}")
        LOGGER.debug(f"Predicted new cost: Laplace: {np.sum(new_pred_laplace_obj)} MC: {curr.mean_cost}")
        LOGGER.debug(f"Actual new cost: Laplace: {np.sum(new_actual_laplace_obj)} MC: {curr.mean_cost}")
        LOGGER.debug(f"Previous KL: Laplace: {np.sum(prev_laplace_kl)} MC: {np.sum(previous_policy_info.estimated_kl)}")
        LOGGER.debug(f"Predicted new KL: Laplace: {np.sum(new_pred_laplace_kl)} MC: {new_mc_kl_sum}")
        LOGGER.debug(f"Actual new KL: Laplace: {np.sum(new_actual_laplace_kl)} MC: {prev_mc_kl_sum}")
        LOGGER.debug(f"Previous w KL: Laplace: {prev_laplace_kl_sum} MC: {prev_mc_kl_w_sum}")
        LOGGER.debug(f"Predicted w new KL: Laplace: {new_pred_laplace_kl_sum} MC: {new_mc_kl_w_sum}")
        LOGGER.debug(f"Actual w new KL: Laplace {new_actual_laplace_kl_sum} MC: {new_mc_kl_w_sum}")
        LOGGER.debug(f"Predicted/actual improvement: {predicted_impr} / {actual_impr}")

        # Compute actual (estimated) KL step multiplier taken at last iteration.
        actual_step_mult = prev_mc_kl_sum / (self.__base_kl_step * prev_traj_distr.time_steps)
        if actual_step_mult < previous_step_mult:
            previous_step_mult = max(actual_step_mult, self.__min_kl_step_mult)

        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL" = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * previous_step_mult, self.__max_kl_step_mult), self.__min_kl_step_mult)

        LOGGER.debug(
            f"{'Increasing' if new_mult > previous_step_mult else 'Decreasing'} step size multiplier to {new_step}")

        return new_step

    def __policy_kl(self, trajectory: Trajectory, action_mean: np.ndarray, action_covariance_inv: np.ndarray,
                    action_covariance_log_det: np.ndarray, lambda_K: np.ndarray, lambda_k: np.ndarray):
        """
        Monte-Carlo estimate of KL divergence between controller and trajectory.
        """
        ts = self.agent.time_steps
        act_dim = self.agent.action_dimensions
        num_samples = len(trajectory.samples)
        states = trajectory.samples.states
        traj_distr: LinearGaussianController = trajectory.controller

        kl, kl_m = np.zeros((num_samples, ts)), np.zeros(ts)
        kl_l, kl_lm = np.zeros((num_samples, ts)), np.zeros(ts)

        # Compute controller mean and covariance at each sample.
        pol_mu = action_mean

        pol_precision = np.tile(action_covariance_inv, (num_samples, ts, 1, 1))
        pol_cov_log_det = np.tile(action_covariance_log_det, (num_samples, ts))

        # Compute KL divergence.
        for t in range(ts):
            # Compute trajectory action at sample.
            traj_mu = np.zeros((num_samples, act_dim))
            for i in range(num_samples):
                traj_mu[i, :] = traj_distr.K[t, :, :].dot(states[i, t, :]) + traj_distr.k[t, :]
            diff = pol_mu[:, t, :] - traj_mu
            tr_pp_ct = pol_precision[:, t, :, :] * traj_distr.covariance[t, :, :]
            k_ln_det_ct = 0.5 * act_dim + np.sum(
                np.log(np.diag(traj_distr.cholesky_covariance[t, :, :]))
            )
            ln_det_cp = pol_cov_log_det[:, t]
            # IMPORTANT: Note that this assumes that pol_prec does not
            #            depend on state!!!!
            #            (Only the last term makes this assumption.)
            d_pp_d = np.sum(diff * (diff.dot(pol_precision[1, t, :, :])), axis=1)
            kl[:, t] = 0.5 * np.sum(np.sum(tr_pp_ct, axis=1), axis=1) - \
                       k_ln_det_ct + 0.5 * ln_det_cp + 0.5 * d_pp_d
            tr_pp_ct_m = np.mean(tr_pp_ct, axis=0)
            kl_m[t] = 0.5 * np.sum(np.sum(tr_pp_ct_m, axis=0), axis=0) - \
                      k_ln_det_ct + 0.5 * np.mean(ln_det_cp) + \
                      0.5 * np.mean(d_pp_d)
            # Compute trajectory action at sample with Lagrange multiplier.
            traj_mu = np.zeros((num_samples, act_dim))
            for i in range(num_samples):
                traj_mu[i, :] = (traj_distr.K[t, :, :] - lambda_K[t, :, :]).dot(states[i, t, :]) \
                                + (traj_distr.k[t, :] - lambda_k[t, :])
            # Compute KL divergence with Lagrange multiplier.
            diff_l = pol_mu[:, t, :] - traj_mu
            d_pp_d_l = np.sum(diff_l * (diff_l.dot(pol_precision[1, t, :, :])), axis=1)
            kl_l[:, t] = 0.5 * np.sum(np.sum(tr_pp_ct, axis=1), axis=1) \
                         - k_ln_det_ct + 0.5 * ln_det_cp + 0.5 * d_pp_d_l
            kl_lm[t] = 0.5 * np.sum(np.sum(tr_pp_ct_m, axis=0), axis=0) \
                       - k_ln_det_ct + 0.5 * np.mean(ln_det_cp) \
                       + 0.5 * np.mean(d_pp_d_l)
        return kl_m, kl, kl_lm, kl_l

    @staticmethod
    def __estimate_cost(traj_distr: LinearGaussianController, dynamics: Dynamics, x0mu: np.ndarray, x0sigma: np.ndarray,
                        cc: np.ndarray, cv: np.ndarray, cm: np.ndarray, policy_info: PolicyInfo) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Laplace approximation to expected cost.
        :param traj_distr:  Trajectory distribution to be used for estimation
        :param dynamics:    Dynamics to be used for estimation
        :param x0mu:        Mean of initial state distribution
        :param x0sigma:     Covariance of initial state distribution
        :param cc:          Taylor approximation of cost function - constant term
        :param cv:          Taylor approximation of cost function - linear term
        :param cm:          Taylor approximation of cost function - quadratic term
        :param policy_info: PolicyInfo object for the controller to measure cost for
        :return:
        """
        # Perform forward pass
        mu, sigma = TrajOptLQR.forward(traj_distr, dynamics, x0mu, x0sigma)

        # Compute cost.
        predicted_cost = np.zeros(traj_distr.time_steps)
        for t in range(traj_distr.time_steps):
            predicted_cost[t] = cc[t] + \
                                mu[t, :].T.dot(cv[t, :]) + \
                                0.5 * np.sum(sigma[t, :, :] * cm[t, :, :]) + \
                                0.5 * mu[t, :].T.dot(cm[t, :, :]).dot(mu[t, :])

        # Compute KL divergence.
        dx = traj_distr.state_dimensions
        du = traj_distr.action_dimensions
        predicted_kl = np.zeros(traj_distr.time_steps)
        for t in range(traj_distr.time_steps):
            inv_pol_covar = np.linalg.solve(
                policy_info.lin_cholesky_covariance[t, :, :],
                np.linalg.solve(policy_info.lin_cholesky_covariance[t, :, :].T, np.eye(du)))
            action_means = policy_info.lin_K[t, :, :].dot(mu[t, :dx].T) + policy_info.lin_k[t, :]
            diff = mu[t, dx:] - action_means
            kbar = traj_distr.K[t, :, :] - policy_info.lin_K[t, :, :]
            predicted_kl[t] = 0.5 * (diff).dot(inv_pol_covar).dot(diff) + \
                              0.5 * np.sum(traj_distr.covariance[t, :, :] * inv_pol_covar) + \
                              0.5 * np.sum(sigma[t, :dx, :dx] * kbar.T.dot(inv_pol_covar).dot(kbar)) + \
                              np.sum(np.log(np.diag(policy_info.lin_cholesky_covariance[t, :, :]))) - \
                              np.sum(np.log(np.diag(traj_distr.cholesky_covariance[t, :, :]))) + \
                              0.5 * du

        return predicted_cost, predicted_kl

    def __create_cost_function(self, traj_distr: LinearGaussianController, trajectory: Trajectory,
                               policy_info: PolicyInfo,
                               policy_weight: np.ndarray, lambda_K: np.ndarray, lambda_k: np.ndarray):
        """
        Generates a cost function used in LQR backward pass.
        :param traj_distr:      Current trajectory distribution.
        :param trajectory:      Previously drawn trajectory.
        :param policy_info:     Policy info of current controller.
        :param policy_weight:
        :return:
        """

        def cost_function(eta: float, augment: bool):
            """ Compute cost estimates used in the LQR backward pass. """
            if not augment:  # Whether to augment cost with term to penalize KL
                return trajectory.Cm, trajectory.cv

            multiplier = self.__max_entropy_weight
            ts, du, dx = traj_distr.time_steps, traj_distr.action_dimensions, traj_distr.state_dimensions
            Cm, cv = np.copy(trajectory.Cm), np.copy(trajectory.cv)

            # Modify controller action via Lagrange multiplier.
            cv[:, dx:] -= lambda_k
            Cm[:, dx:, :dx] -= lambda_K
            Cm[:, :dx, dx:] -= np.transpose(lambda_K, [0, 2, 1])

            # Pre-process the costs with KL-divergence terms.
            TKLm = np.zeros((ts, dx + du, dx + du))
            TKLv = np.zeros((ts, dx + du))
            PKLm = np.zeros((ts, dx + du, dx + du))
            PKLv = np.zeros((ts, dx + du))
            fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)

            # TODO: remove
            l_K = np.zeros((ts, dx + du, dx + du))
            l_K[:, dx:, :dx] = -lambda_K
            l_K[:, :dx, dx:] = -np.transpose(lambda_K, [0, 2, 1])
            l_k = np.zeros((ts, dx + du))
            l_k[:, dx:] = -lambda_k

            pcm = np.zeros(ts - 1)
            plk = np.zeros(ts - 1)
            ptkl = np.zeros(ts - 1)
            ppkl = np.zeros(ts - 1)
            lK = np.zeros(ts - 1)
            pcmv = np.zeros(ts - 1)
            plkv = np.zeros(ts - 1)
            ptklv = np.zeros(ts - 1)
            ppklv = np.zeros(ts - 1)
            lk = np.zeros(ts - 1)

            for t in range(ts):
                K, k = traj_distr.K[t, :, :], traj_distr.k[t, :]
                inv_traj_covar = traj_distr.inv_covariance[t, :, :]

                # Trajectory KL-divergence terms.
                TKLm[t, :, :] = np.vstack([
                    np.hstack([
                        K.T.dot(inv_traj_covar).dot(K),
                        -K.T.dot(inv_traj_covar)]),
                    np.hstack([-inv_traj_covar.dot(K), inv_traj_covar])
                ])
                TKLv[t, :] = np.concatenate([
                    K.T.dot(inv_traj_covar).dot(k), -inv_traj_covar.dot(k)
                ])

                # Policy KL-divergence terms.
                inv_pol_covar = policy_info.action_covariance_inv[t, :, :]
                KB, kB = policy_info.lin_K[t, :, :], policy_info.lin_k[t, :]
                PKLm[t, :, :] = np.vstack([
                    np.hstack([KB.T.dot(inv_pol_covar).dot(KB), -KB.T.dot(inv_pol_covar)]),
                    np.hstack([-inv_pol_covar.dot(KB), inv_pol_covar])
                ])
                PKLv[t, :] = np.concatenate([
                    KB.T.dot(inv_pol_covar).dot(kB), -inv_pol_covar.dot(kB)
                ])
                wt = policy_weight[t]
                fCm[t, :, :] = (Cm[t, :, :] + TKLm[t, :, :] * eta +
                                PKLm[t, :, :] * wt) / (eta + wt + multiplier)
                fcv[t, :] = (cv[t, :] + TKLv[t, :] * eta +
                             PKLv[t, :] * wt) / (eta + wt + multiplier)

                if t < ts - 1:
                    pcm[t], plk[t], ptkl[t], ppkl[t], lK[t] = [
                        np.mean([np.abs(np.concatenate((s.states[t], s.actions[t])).dot(
                            o.dot(np.concatenate((s.states[t], s.actions[t]))))) for s in trajectory.samples])
                        for o in [trajectory.Cm[t], Cm[t], TKLm[t, :, :] * eta, PKLm[t, :, :] * wt, l_K[t]]]
                    pcmv[t], plkv[t], ptklv[t], ppklv[t], lk[t] = [
                        np.abs(
                            np.mean([np.concatenate((s.states[t], s.actions[t])).dot(o) for s in trajectory.samples]))
                        for o in [trajectory.cv[t], cv[t], TKLv[t, :] * eta, PKLv[t, :] * wt, l_k[t]]]
            LOGGER.info(f"""
            =====================================
            cv: {np.mean(pcmv)}
            cm-l: {np.mean(plkv)}
            trajkl: {np.mean(ptklv)}
            polkl: {np.mean(ppklv)}
            lambda: {np.mean(lk)}
            -------------------------------------
            Cm: {np.mean(pcm)}
            cm-l: {np.mean(plk)}
            trajkl: {np.mean(ptkl)}
            polkl: {np.mean(ppkl)}
            lambda: {np.mean(lK)}
            #####################################
            """)
            return fCm, fcv

        return cost_function

    @property
    def _display_data_description(self) -> List[Tuple[bool, str, str, Callable[[Any], str], int]]:
        h = hr_formatter(3, min_exp_tol=3, max_exp_tol=4)
        return [(True, "kl_step", "kl_step", h, 10),
                (True, "entropy", "entropy", h, 10),
                (True, "eta", "eta", h, 10),
                (True, "mean_policy_weight", "pol wt", h, 10),
                (True, "mean_lambda", "lambda", h, 10),
                (True, "kl_div_pol", "pol kl", h, 10)]
