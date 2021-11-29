from typing import NamedTuple, Dict, List, Optional, TypeVar, Generic, Any

import math
import numpy as np

from gps.dynamics import Dynamics
from gps.controller import LinearGaussianController, Controller
from gps.sample import SampleList

TrajectoryCost = NamedTuple("TrajectoryCost",
                            # Cost and derivatives
                            (("l", np.ndarray),  # Actual cost (T)
                             ("lx", np.ndarray),  # Cost by x (T x dX)
                             ("lu", np.ndarray),  # Cost by u (T x dU)
                             ("lxx", np.ndarray),  # Cost by x by x (T x dX x dX)
                             ("luu", np.ndarray),  # Cost by u by u (T x dU x dU)
                             ("lux", np.ndarray),  # Cost by x by x (T x dU x dX)
                             # Taylor approximation
                             ("cc", np.ndarray),  # Constant term (T)
                             ("cv", np.ndarray),  # Linear term (T x (dX + dU))
                             ("Cm", np.ndarray)))  # Quadratic term (T x (dX + dU) x (dX + dU))

args = (("dynamics", Dynamics),)  # Dynamics fitted to this trajectory
opt_args = (("controller", Optional[Controller]),  # Controller used for this trajectory
            ("cost", Optional[List[TrajectoryCost]]),  # Costs for each sample
            ("mean_cost", Optional[float]),  # Mean cost of all samples
            ("samples", Optional[SampleList]),  # Samples taken for this trajectory
            ("x0mu", Optional[np.ndarray]),  # Mean for the initial state, used by the dynamics.
            ("x0sigma", Optional[np.ndarray]),  # Covariance for the initial state distribution.
            ("expected_mu", Optional[np.ndarray]),  # Expected mean of this trajectory under current dynamics
            ("expected_sigma", Optional[np.ndarray]),  # Expected sigma of this trajectory under current dynamics
            # Taylor approximation of mean cost
            ("cc", Optional[np.ndarray]),  # Constant term (T)
            ("cv", Optional[np.ndarray]),  # Linear term (T x (dX + dU))
            ("Cm", Optional[np.ndarray]))  # Quadratic term (T x (dX + dU) x (dX + dU))

Trajectory = NamedTuple("Trajectory", args + opt_args)

Trajectory.__new__.__defaults__ = (None,) * len(opt_args)


class AlgorithmConditionData:
    """
    Class for condition specific data generated by the algorithm
    """

    def __init__(self, trajectory_distribution: LinearGaussianController):
        self.__trajectory_distribution = trajectory_distribution

    @property
    def traj_distr(self) -> LinearGaussianController:
        """
        The optimized trajectory distribution for this condition.
        :return:
        """
        return self.__trajectory_distribution

    @property
    def display_data(self) -> Dict[str, Any]:
        """
        Data to display by the gui as defined by Algorithm.display_data_description.
        :return: A mapping (IDENTIFIER -> DATA)
        """
        return {}


class AlgorithmData:
    @property
    def policy(self) -> Optional[Controller]:
        """
        Policy computed in this iteration
        :return:
        """
        return None

    @property
    def display_data(self) -> Dict[str, Any]:
        """
        Data to display by the gui as defined by Algorithm.display_data_description.
        :return: A mapping (IDENTIFIER -> DATA)
        """
        return {}


AD = TypeVar("AD", bound=AlgorithmData)
ACD = TypeVar("ACD", bound=AlgorithmConditionData)


class ConditionData(Generic[ACD]):
    def __init__(self, algorithm_data: Optional[ACD], control_noise_rng_state: Any,
                 training_trajectory: Optional[Trajectory] = None, test_trajectory: Optional[Trajectory] = None,
                 policy_trajectory: Optional[Trajectory] = None):
        self.__training_trajectory = training_trajectory
        self.__test_trajectory = test_trajectory
        self.__policy_trajectory = policy_trajectory
        self.__algorithm_data = algorithm_data
        self.__control_noise_rng_state = control_noise_rng_state

    @property
    def training_trajectory(self) -> Optional[Trajectory]:
        return self.__training_trajectory

    @property
    def test_trajectory(self) -> Optional[Trajectory]:
        return self.__test_trajectory

    @property
    def policy_trajectory(self) -> Optional[Trajectory]:
        return self.__policy_trajectory

    @property
    def algorithm_data(self) -> Optional[ACD]:
        return self.__algorithm_data

    @property
    def control_noise_rng_state(self) -> Any:
        return self.__control_noise_rng_state

    @property
    def display_data(self) -> Dict[str, Any]:
        disp = {} if self.__algorithm_data is None else self.__algorithm_data.display_data.copy()
        disp["training_cost"] = np.nan if self.training_trajectory is None else self.training_trajectory.mean_cost
        disp["test_cost"] = np.nan if self.test_trajectory is None else self.test_trajectory.mean_cost
        disp["policy_cost"] = np.nan if self.policy_trajectory is None else self.policy_trajectory.mean_cost
        return disp


class IterationData(Generic[AD, ACD]):
    def __init__(self, iteration_no: int, algorithm_data: AD, condition_data: Dict[int, ConditionData[ACD]]):
        self.__iteration_no = iteration_no
        self.__condition_data = condition_data
        self.__algorithm_data = algorithm_data

    @property
    def iteration_no(self) -> int:
        return self.__iteration_no

    @property
    def cond_data(self) -> Dict[int, ConditionData[ACD]]:
        return self.__condition_data

    @property
    def algorithm_data(self) -> AD:
        return self.__algorithm_data

    @property
    def display_data(self) -> Dict[str, Any]:
        algorithm_display_data = self.__algorithm_data.display_data.copy()
        training_cost = [c.training_trajectory.mean_cost for c in self.cond_data.values() if
                         c.training_trajectory is not None and not math.isnan(c.training_trajectory.mean_cost)]
        test_cost = [c.test_trajectory.mean_cost for c in self.cond_data.values() if
                     c.test_trajectory is not None and not math.isnan(c.test_trajectory.mean_cost)]
        policy_cost = [c.policy_trajectory.mean_cost for c in self.cond_data.values() if
                       c.policy_trajectory is not None and not math.isnan(c.policy_trajectory.mean_cost)]
        mean_training_cost = sum(training_cost) / len(training_cost) if len(training_cost) > 0 else np.nan
        mean_test_cost = sum(test_cost) / len(test_cost) if len(test_cost) > 0 else np.nan
        mean_policy_cost = sum(policy_cost) / len(policy_cost) if len(policy_cost) > 0 else np.nan
        algorithm_display_data.update(
            {"mean_training_cost": mean_training_cost,
             "mean_test_cost": mean_test_cost,
             "mean_policy_cost": mean_policy_cost,
             "itr": self.iteration_no})
        return algorithm_display_data
