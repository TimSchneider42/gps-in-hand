""" Configuration for Darias hand target angle setup """
from typing import List

import numpy as np

import scipy as sp

from gps.agent.darias_vrep.agent_darias_vrep import AgentDariasVrep, Properties
from gps.agent.noise_generator import GaussianNoiseGenerator
from gps.algorithm.algorithm import ConditionDescription
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.controller.linear_gaussian_controller import LinearGaussianController
from gps.cost import Cost, CostTargetState, CostAction
from gps.dynamics import DynamicsPriorGMM, DynamicsLR
from gps.traj_opt.traj_opt_lqr import TrajOptLQR

from gps.config import Config
from vrep_interface import SimulationClient
from vrep_interface.darias import DariasHand


def create_config(args: List[str]) -> Config:
    hand = DariasHand("right_hand")

    # spread, proximal, distal
    initial_angles = [
        {
            hand.thumb: np.array([0.0, 0.2, 0.2])
        },
        {
            hand.thumb: np.array([0.0, 0.1, 0.1])
        },
        {
            hand.thumb: np.array([0.0, 0.2, 0.2])
        }
    ]
    num_conditions = 1

    initial_states = initial_angles[0:num_conditions]

    for f in hand.fingers:
        f.torques = 0.1, -0.1, -0.1
        f.initial_torques = f.torques

    agent = AgentDariasVrep(time_steps=40, control_cycle_time_ms=50, action_delay_ms=10, hand=hand,
                            used_fingers=[hand.thumb],
                            initial_states=initial_states, simulation_client=SimulationClient())

    action_cost = CostAction()
    state_cost = CostTargetState(sensor=(Properties.JOINT_ANGLES, hand.thumb.name_prefix),
                                 target_state=np.array([0.0, 0.2, 0.2]))
    cost: Cost = 1e-5 * action_cost + state_cost

    t = agent.time_steps
    du = agent.action_dimensions
    K = np.zeros((t, du, agent.state_packer.dimensions))  # Controller gains matrix.
    k = np.zeros((t, du))  # Controller bias term.
    covar_single = np.diag([0.1] * du)  # Covariance matrix
    chol = np.tile(sp.linalg.cholesky(covar_single), (t, 1, 1))  # Cholesky decomposition.
    inv_covar = np.tile(np.linalg.inv(covar_single), (t, 1, 1))  # Inverse of covariance.
    covar = np.tile(covar_single, (t, 1, 1))

    init_traj_distr = LinearGaussianController(K, k, covar, chol, inv_covar)

    dynamics_prior = DynamicsPriorGMM(min_samples_per_cluster=40, max_clusters=20, max_samples=20)
    initial_dynamics = DynamicsLR(regularization=1e-6, prior=dynamics_prior)

    conditions = {
        c: ConditionDescription(cost, init_traj_distr, initial_dynamics, GaussianNoiseGenerator(seed=c))
        for c in range(len(initial_states))
    }

    algorithm = AlgorithmTrajOpt(agent, conditions, TrajOptLQR(), training_samples_per_condition=5)
    return Config(algorithm, 100)
