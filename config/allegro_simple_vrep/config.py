""" Configuration for Allegro target angle setup """
import math
from typing import List

import numpy as np

import scipy as sp

from gps.agent.allegro_vrep import AgentAllegroVrep, Properties
from gps.agent.noise_generator import GaussianNoiseGenerator
from gps.algorithm.algorithm import ConditionDescription
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.controller.linear_gaussian_controller import LinearGaussianController
from gps.cost import CostSum, Cost, CostTargetState, CostAction
from gps.dynamics import DynamicsPriorGMM, DynamicsLR
from gps.traj_opt.traj_opt_lqr import TrajOptLQR

from gps.config import Config
from vrep_interface import SimulationClient
from vrep_interface.allegro import AllegroHand, AllegroFingerType


def create_config(args: List[str]) -> Config:
    hand = AllegroHand("right")

    num_conditions = 1
    use_finger = {
        AllegroFingerType.THUMB: True,
        AllegroFingerType.INDEX: True,
        AllegroFingerType.MIDDLE: True,
        AllegroFingerType.SMALL: True
    }

    used_fingers = [t for t, b in use_finger.items() if b]

    np.random.seed(0)
    # twist, proximal, middle, distal
    initial_angles = [{t: np.random.uniform(0.05, 0.2, 4) for t in used_fingers} for _ in range(num_conditions)]

    full_target_state = {
        AllegroFingerType.THUMB: np.array([0.2, 0.5, 0.0, math.pi / 2]),
        AllegroFingerType.INDEX: np.array([math.pi / 4, -0.2, math.pi / 2, math.pi / 2]),
        AllegroFingerType.MIDDLE: np.array([0.05, 0.05, 0.05, math.pi / 2]),
        AllegroFingerType.SMALL: np.array([-math.pi / 4, -0.1, math.pi / 2, math.pi / 2])
    }

    target_states = [{t: full_target_state[t] for t in used_fingers}] * 3

    control_cycle_time_ms = 10

    initial_states = initial_angles[0:num_conditions]

    for f in hand.fingers.values():
        f.torques = 0.0, 0.0, 0.0, 0.0
        f.initial_torques = f.torques

    target_indicator_hand = AllegroHand("right_ti")

    agent = AgentAllegroVrep(time_steps=100, control_cycle_time_ms=control_cycle_time_ms, action_delay_ms=4, hand=hand,
                             used_fingers=used_fingers, initial_states=initial_states,
                             simulation_client=SimulationClient(), target_indicator_hand=target_indicator_hand,
                             target_states=target_states)
    cost = {}
    for cond in range(num_conditions):
        action_cost = CostAction()
        state_cost = [CostTargetState(sensor=(Properties.JOINT_ANGLES, t),
                                      target_state=np.array(target_states[cond][t])) for t in used_fingers]
        c: Cost = 1e-5 * action_cost + CostSum(state_cost)
        cost[cond] = c

    t = agent.time_steps
    du = agent.action_dimensions
    K = np.zeros((t, du, agent.state_packer.dimensions))  # Controller gains matrix.
    k = np.zeros((t, du))  # Controller bias term.
    covar_single = np.diag([0.0001] * du)  # Covariance matrix
    chol = np.tile(sp.linalg.cholesky(covar_single), (t, 1, 1))  # Cholesky decomposition.
    inv_covar = np.tile(np.linalg.inv(covar_single), (t, 1, 1))  # Inverse of covariance.
    covar = np.tile(covar_single, (t, 1, 1))

    init_traj_distr = LinearGaussianController(K, k, covar, chol, inv_covar)

    dynamics_prior = DynamicsPriorGMM(min_samples_per_cluster=40, max_clusters=20, max_samples=20)
    initial_dynamics = DynamicsLR(regularization=1e-6, prior=dynamics_prior)

    conditions = {
        c: ConditionDescription(cost[c], init_traj_distr, initial_dynamics, GaussianNoiseGenerator(seed=c))
        for c in range(len(initial_states))
    }

    algorithm = AlgorithmTrajOpt(agent=agent,
                                 conditions=conditions,
                                 trajectory_optimizer=TrajOptLQR(),
                                 training_samples_per_condition=30,
                                 test_samples_per_condition=1,
                                 base_kl_step=0.2,
                                 max_step_mult=10.0,
                                 initial_state_var=1e-4)
    return Config(algorithm, 100)
