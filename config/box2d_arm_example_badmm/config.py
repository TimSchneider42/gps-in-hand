""" Configuration for Box2d Point Mass setup """
from typing import List

import numpy as np

import scipy as sp

from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.arm_world import ArmWorld
from gps.agent.noise_generator import GaussianNoiseGenerator
from gps.algorithm import ConditionDescription, AlgorithmBADMM, LGStepMode
from gps.config import DebugCostFunction
from gps.controller import LinearGaussianController
from gps.cost import Cost, CostTargetState, CostAction
from gps.dynamics import DynamicsLR, DynamicsPriorGMM
from gps.neural_network import NeuralNetworkFullyConnected
from gps.policy import PolicyTf, PolicyPriorGMM
from gps.policy_opt import PolicyOptTf
from gps.traj_opt import TrajOptLQR
from gps import Config
from gps.gmm import GMMSklearn


def create_config(args: List[str]) -> Config:
    sensor_dims = {
        "angles": 2,
        "vel": 2,
        "tracking": 3
    }

    initial_states = [
        {
            "angles": np.array([0.75 * np.pi, 0.5 * np.pi]),
            "vel": np.zeros(sensor_dims["vel"]),
            "tracking": np.zeros(sensor_dims["tracking"])
        },
        {
            "angles": np.array([0.25 * np.pi, 0.5 * np.pi]),
            "vel": np.zeros(sensor_dims["vel"]),
            "tracking": np.zeros(sensor_dims["tracking"])
        },
        {
            "angles": np.array([0.5 * np.pi, 0.25 * np.pi]),
            "vel": np.zeros(sensor_dims["vel"]),
            "tracking": np.zeros(sensor_dims["tracking"])
        },
        {
            "angles": np.array([0.25 * np.pi, 0.75 * np.pi]),
            "vel": np.zeros(sensor_dims["vel"]),
            "tracking": np.zeros(sensor_dims["tracking"])
        },
        {
            "angles": np.array([0.25 * np.pi, 0.25 * np.pi]),
            "vel": np.zeros(sensor_dims["vel"]),
            "tracking": np.zeros(sensor_dims["tracking"])
        }
    ]

    agent = AgentBox2D(time_steps=100,
                       initial_states=initial_states,
                       framework=ArmWorld(np.array([0, 0]), True),
                       state_data_types=["angles", "vel", "tracking"],
                       observation_data_types=["angles", "vel", "tracking"],
                       tracking_data_types=["tracking"])

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

    state_cost = CostTargetState(sensor="angles", target_state=np.array([0, 0]))

    cost: Cost = action_cost + state_cost

    dynamics_prior = DynamicsPriorGMM(min_samples_per_cluster=40, max_clusters=20, max_samples=20, gmm=GMMSklearn())
    initial_dynamics = DynamicsLR(regularization=1e-6, prior=dynamics_prior)

    conditions = {
        c: ConditionDescription(cost, init_traj_distr, initial_dynamics, GaussianNoiseGenerator(seed=c))
        for c in range(len(initial_states))
    }

    observation_dimensions = sum(sensor_dims.values())

    neural_network = NeuralNetworkFullyConnected(action_dimensions=du,
                                                 observation_dimensions=observation_dimensions,
                                                 layer_units=[40, 40])

    initial_policy = PolicyTf(neural_network=neural_network,
                              covariance=np.tile(np.eye(du) * 0.1, (agent.time_steps, 1, 1)))

    algorithm = AlgorithmBADMM(agent=agent,
                               conditions=conditions,
                               initial_policy=initial_policy,
                               policy_optimizer=PolicyOptTf(),
                               initial_policy_prior=PolicyPriorGMM(),
                               trajectory_optimizer=TrajOptLQR(),
                               training_samples_per_condition=5,
                               test_samples_per_condition=1,
                               lg_step_mode=LGStepMode.CONSTRAINT_SAT,
                               training_conditions=[0, 1, 3, 4],
                               test_conditions=[0, 1, 2, 3, 4])
    debug_cost_functions = [DebugCostFunction(action_cost, "action_cost"),
                            DebugCostFunction(state_cost, "state_cost")]
    return Config(algorithm, 10, {c: debug_cost_functions for c in conditions})
