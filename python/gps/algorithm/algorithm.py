""" This file defines the base algorithm class. """

import logging
from threading import Lock

import numpy as np

from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Tuple, TypeVar, Generic, NamedTuple, Iterable, Callable, Any

from allegro_pybullet.util import ReadOnlyDict
from gps.agent import Agent, Sampler
from gps.agent.noise_generator import NoiseGenerator
from gps.algorithm.iteration_data import AlgorithmData, AlgorithmConditionData, IterationData, ConditionData, \
    Trajectory, TrajectoryCost
from gps.controller import LinearGaussianController, Controller
from gps.cost import Cost
from gps.dynamics import Dynamics
from gps.sample import Sample
from gps.traj_opt import TrajOptLQR
from gps.utility.human_readable_formatter import human_readable_formatter as hr_formatter
from gps.utility.abortable_worker import AbortableWorker

LOGGER = logging.getLogger(__name__)

AD = TypeVar("AD", bound=AlgorithmData)
ACD = TypeVar("ACD", bound=AlgorithmConditionData)

ConditionDescription = NamedTuple("ConditionDescription",
                                  (("cost_function", Cost),
                                   # Initial trajectory distribution for this condition
                                   ("initial_traj_distr", LinearGaussianController),
                                   # Initial dynamics for this condition
                                   ("initial_dynamics", Dynamics),
                                   # Noise generator used to generate control noise for training trajectories
                                   ("control_noise_generator", Optional[NoiseGenerator])))


class Algorithm(ABC, Generic[AD, ACD], AbortableWorker[None]):
    """ Algorithm superclass. """

    def __init__(self, agent: Agent, conditions: Dict[int, ConditionDescription],
                 training_samples_per_condition: int,
                 training_conditions: Optional[Iterable[int]] = None,
                 test_samples_per_condition: int = 1,
                 test_conditions: Optional[Iterable[int]] = None,
                 policy_samples_per_condition: int = 0,
                 policy_conditions: Optional[Iterable[int]] = None,
                 initial_state_var: float = 1e-6, display_name: Optional[str] = None):
        """

        :param agent:           Agent to run controller on.
        :param conditions:      Dictionary containing the condition descriptions for each condition.
        :param display_name:    Display name of this algorithm for the gui.
        """
        AbortableWorker.__init__(self)
        self.__display_name = display_name
        self.__training_conditions = tuple(
            range(agent.condition_count) if training_conditions is None else training_conditions)

        if test_samples_per_condition > 0:
            self.__test_conditions = tuple(self.__training_conditions if test_conditions is None else test_conditions)
        else:
            self.__test_conditions: Tuple[int] = ()

        if policy_samples_per_condition > 0:
            self.__policy_conditions = tuple(conditions.keys() if policy_conditions is None else policy_conditions)
        else:
            self.__policy_conditions: Tuple[int] = ()

        invalid_conditions = [c for c in self.__training_conditions + self.__policy_conditions if c not in conditions]
        assert len(invalid_conditions) == 0, f"Invalid conditions: {invalid_conditions}."

        invalid_conditions = [c for c in self.__test_conditions if c not in self.__training_conditions]
        assert len(invalid_conditions) == 0, \
            f"Cannot test controller on conditions which have not been trained (conditions: {invalid_conditions})."

        self.__training_samples_per_condition = training_samples_per_condition
        self.__test_samples_per_condition = test_samples_per_condition
        self.__policy_samples_per_condition = policy_samples_per_condition

        self.__agent = agent
        self.__current_iteration: IterationData[AD, ACD] = None
        self.__conditions = ReadOnlyDict(conditions.copy())
        self.__training_samplers: ReadOnlyDict[int, Sampler] = ReadOnlyDict({
            c: Sampler(self.agent, c, self.__conditions[c].control_noise_generator)
            for c in self.__training_conditions
        })
        self.__test_samplers = ReadOnlyDict({c: Sampler(self.agent, c) for c in self.__test_conditions})
        self.__policy_samplers = ReadOnlyDict({c: Sampler(self.agent, c) for c in self.__policy_conditions})

        self.__initial_state_var = initial_state_var
        self.__abort_lock = Lock()

    def initialize_new(self):
        """
        Initializes a new run of the algorithm.
        :return:
        """
        # Initialize iteration 0 just containing the initial distribution and dynamics
        algorithm_data, algorithm_condition_data = self._initialize_new()
        condition_data = {
            c: ConditionData(algorithm_condition_data[c],
                             self.__training_samplers[c].control_noise_generator.initial_random_state,
                             Trajectory(self.__conditions[c].initial_dynamics))
            for c in self.__training_conditions
        }
        self.set_iteration(IterationData(0, algorithm_data, condition_data))

    def set_iteration(self, iteration_data: IterationData):
        self.__current_iteration = iteration_data

    @abstractmethod
    def _initialize_new(self) -> Tuple[AD, Dict[int, ACD]]:
        pass

    def _run(self):
        """
        Run one iteration of the algorithm
        :return:
        """
        # Collect trajectories for each condition
        training_trajectories = {}
        for cond in self.training_conditions:
            cd = self.__current_iteration.cond_data[cond]
            self.__training_samplers[cond].control_noise_generator.random_state = cd.control_noise_rng_state
            policy = cd.algorithm_data.traj_distr
            sampler = self.__training_samplers[cond]
            sample_count = self.__training_samples_per_condition
            training_trajectories[cond] = self.__take_samples(sampler, cond, policy, sample_count, True)
            if self.aborting:
                return
        algorithm_data, algo_cond_data = self._iteration(training_trajectories, self.current_iteration)

        # Run controller tests after controller has been updated
        test_trajectories = {}
        for cond in self.__test_conditions:
            sampler = self.__test_samplers[cond]
            sample_count = self.__test_samples_per_condition
            test_trajectories[cond] = self.__take_samples(sampler, cond, algo_cond_data[cond].traj_distr, sample_count,
                                                          False)
            if self.aborting:
                return

        policy_trajectories = {}

        # Run policy tests
        for cond in self.__policy_conditions:
            assert algorithm_data.policy is not None
            sampler = self.__policy_samplers[cond]
            sample_count = self.__policy_samples_per_condition
            policy_trajectories[cond] = self.__take_samples(sampler, cond, algorithm_data.policy, sample_count,
                                                            False)
            if self.aborting:
                return

        if self.aborting:
            return
        condition_data = {
            c: ConditionData(algo_cond_data[c], self.__training_samplers[c].control_noise_generator.random_state,
                             training_trajectories.get(c), test_trajectories.get(c), policy_trajectories.get(c))
            for c in self.training_conditions}
        condition_data.update(
            {c: ConditionData(None, None, test_trajectory=test_trajectories.get(c),
                              policy_trajectory=policy_trajectories.get(c))
             for c in set(self.test_conditions + self.policy_conditions) if c not in condition_data})
        self.__current_iteration = IterationData(self.current_iteration.iteration_no + 1, algorithm_data,
                                                 condition_data)

    def __take_samples(self, sampler: Sampler, condition: int, policy: Controller, sample_count: int,
                       compute_dynamics: Optional[bool] = False) -> Optional[Trajectory]:
        # Sampler has to be started like this to avoid race conditions with the abort function
        with self.__abort_lock:
            if self.aborting:
                return None
            sampler.setup(policy, sample_count)
        sample_list = sampler.run()
        if self.aborting:
            return None
        cost = [self.__eval_cost(s, self.__conditions[condition].cost_function) for s in sample_list]

        cc = np.mean([c.cc for c in cost], axis=0)
        cv = np.mean([c.cv for c in cost], axis=0)
        Cm = np.mean([c.Cm for c in cost], axis=0)
        mean_cost = float(np.mean(np.sum([c.l for c in cost], axis=1)))

        x0 = sample_list.states[:, 0, :]
        x0mu = np.mean(x0, axis=0)
        x0sigma = np.diag(np.maximum(np.var(x0, axis=0), self.__initial_state_var))

        expected_mu = None
        expected_sigma = None

        if compute_dynamics:
            # Update prior and fit dynamics.
            new_dynamics = self.current_iteration.cond_data[condition].training_trajectory.dynamics.fit(sample_list)
            if new_dynamics.prior is not None:
                mu0, phi, priorm, n0 = new_dynamics.prior.initial_state()
                x0sigma += phi + (len(sample_list) * priorm) / (len(sample_list) + priorm) * \
                           np.outer(x0mu - mu0, x0mu - mu0) / (len(sample_list) + n0)
            if isinstance(policy, LinearGaussianController):
                # Compute expected mu and sigma under current dynamics
                expected_mu, expected_sigma = TrajOptLQR.forward(policy, new_dynamics, x0mu, x0sigma)
        else:
            new_dynamics = None

        return Trajectory(new_dynamics, policy, cost, mean_cost, sample_list, x0mu, x0sigma, expected_mu,
                          expected_sigma, cc, cv, Cm)

    def __eval_cost(self, sample: Sample, cost_function: Cost) -> TrajectoryCost:
        """
        Generates trajectory cost objects for a sample list.
        :param sample: Sample to compute cost for.
        :param cost_function: Cost function to use
        :return:
        """
        # Get costs.
        l, lx, lu, lxx, luu, lux = cost_function.eval(sample)

        # Compute cost.
        cc = l.copy()

        # Assemble matrix and vector.
        cv = np.c_[lx, lu]
        Cm = np.concatenate((np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]), axis=1)

        # Adjust for expanding cost around a sample.
        X = sample.states
        U = np.zeros((sample.time_steps, sample.action_dimensions))
        U[0:-1] = sample.actions
        yhat = np.c_[X, U]
        rdiff = -yhat
        rdiff_expand = np.expand_dims(rdiff, axis=2)
        cv_update = np.sum(Cm * rdiff_expand, axis=1)
        cc += np.sum(rdiff * cv, axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
        cv += cv_update

        return TrajectoryCost(l, lx, lu, lxx, luu, lux, cc, cv, Cm)

    def _on_abort(self):
        with self.__abort_lock:
            for s in self.__training_samplers.values():
                s.abort()
            for s in self.__test_samplers.values():
                s.abort()

    @abstractmethod
    def _iteration(self, trajectories: Dict[int, Trajectory],
                   previous_iteration: IterationData[AD, ACD]) -> Tuple[AD, Dict[int, ACD]]:
        pass

    @property
    def display_name(self) -> str:
        return self.__display_name if self.__display_name is not None else type(self).__name__

    @property
    def current_iteration(self) -> IterationData[AD, ACD]:
        return self.__current_iteration

    @property
    def test_samples_per_condition(self) -> int:
        return self.__test_samples_per_condition

    @property
    def training_samples_per_condition(self) -> int:
        return self.__training_samples_per_condition

    @property
    def policy_samples_per_condition(self) -> int:
        return self.__policy_samples_per_condition

    @property
    def conditions(self) -> ReadOnlyDict[int, ConditionDescription]:
        return self.__conditions

    @property
    def training_conditions(self) -> Tuple[int]:
        return self.__training_conditions

    @property
    def test_conditions(self) -> Tuple[int]:
        return self.__test_conditions

    @property
    def policy_conditions(self) -> Tuple[int]:
        return self.__policy_conditions

    @property
    def training_samplers(self) -> ReadOnlyDict[int, Sampler]:
        return self.__training_samplers

    @property
    def test_samplers(self) -> ReadOnlyDict[int, Sampler]:
        return self.__test_samplers

    @property
    def policy_samplers(self) -> ReadOnlyDict[int, Sampler]:
        return self.__policy_samplers

    @property
    def display_data_description(self) -> List[Tuple[bool, str, str, Callable[[float], int], int]]:
        """
        Description of the data to display by the GUI.
        Each entry is specified by a tuple (IS_CONDITION_RELATED, IDENTIFIER, DISPLAY_NAME, FORMAT, REQUIRED_CHARS).
        :return:
        """
        h = hr_formatter(3, max_exp_tol=4, min_exp_tol=3)
        desc = [(False, "itr", "itr", "{:03d}".format, 3),
                (False, "mean_training_cost", "tr_cost", h, 10),
                (True, "training_cost", "tr_cost", h, 10)]
        if self.test_samples_per_condition > 0 and len(self.training_conditions) > 0:
            desc += ((False, "mean_test_cost", "tst_cost", h, 10), (True, "test_cost", "tst_cost", h, 10))
        if self.policy_samples_per_condition > 0 and len(self.policy_conditions) > 0:
            desc += ((False, "mean_policy_cost", "pol_cost", h, 10), (True, "policy_cost", "pol_cost", h, 10))
        return desc + self._display_data_description

    @property
    def _display_data_description(self) -> List[Tuple[bool, str, str, Callable[[Any], str], int]]:
        """
        This can be implemented by subclasses to add own descriptions.
        :return:
        """
        return []

    @property
    def agent(self):
        return self.__agent
