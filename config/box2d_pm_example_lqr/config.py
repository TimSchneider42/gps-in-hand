""" Configuration for Box2d Point Mass setup """
from typing import List

import numpy as np

import scipy as sp

from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.arm_world import ArmWorld
from gps.agent.box2d.point_mass_world import PointMassWorld
from gps.agent.noise_generator import SmoothGaussianNoiseGenerator
from gps.algorithm import ConditionDescription
from gps.algorithm import AlgorithmTrajOpt
from gps.config import DebugCostFunction
from gps.controller import LinearGaussianController
from gps.cost import Cost, CostTargetState, CostAction
from gps.dynamics import DynamicsLR, DynamicsPriorGMM
from gps.traj_opt import TrajOptLQR
from gps import Config
from gps.gmm import GMMSklearn, GMMPython


def create_config(args: List[str]) -> Config:
    initial_state = {
        "pos": [5, 35]
    }

    initial_states = [initial_state]

    agent = AgentBox2D(time_steps=100,
                       initial_states=initial_states,
                       framework=PointMassWorld([0, 20], False),
                       state_data_types=["pos"],
                       observation_data_types=["pos"],
                       tracking_data_types=["pos"])

    t = agent.time_steps
    du = agent.action_dimensions
    K = np.zeros((t, du, agent.state_packer.dimensions))  # Controller gains matrix.
    k = np.zeros((t, du))  # Controller bias term.
    covar_single = np.diag([0.1] * du)  # Covariance matrix
    chol = np.tile(sp.linalg.cholesky(covar_single), (t, 1, 1))  # Cholesky decomposition.
    inv_covar = np.tile(np.linalg.inv(covar_single), (t, 1, 1))  # Inverse of covariance.
    covar = np.tile(covar_single, (t, 1, 1))

    init_traj_distr = LinearGaussianController(K, k, covar, chol, inv_covar)

    action_cost = 1e-5 * CostAction()

    state_cost = CostTargetState(sensor="pos", target_state=np.array([0, 20, 0]))

    cost: Cost = action_cost + state_cost

    dynamics_prior = DynamicsPriorGMM(min_samples_per_cluster=40, max_clusters=20, max_samples=20, gmm=GMMPython())
    initial_dynamics = DynamicsLR(regularization=1e-6, prior=dynamics_prior)

    conditions = {
        c: ConditionDescription(cost, init_traj_distr, initial_dynamics, SmoothGaussianNoiseGenerator(seed=c))
        for c in range(len(initial_states))
    }

    algorithm = AlgorithmTrajOpt(agent, conditions, TrajOptLQR(),
                                 training_samples_per_condition=30, test_samples_per_condition=1)
    debug_cost_functions = [DebugCostFunction(action_cost, "action_cost"),
                            DebugCostFunction(state_cost, "state_cost")]
    return Config(algorithm, 10, {c: debug_cost_functions for c in conditions})
