"""
Configuration for Allegro hand cylinder setup

args:
1: used_fingers - any combination of "t" (thumb), "i" (index), "m" (middle), "s" (small), i.e. "tim"
2: rewarded cylinder direction - "cw" for clock-wise, "ccw" for counter-clock-wise

"""
from typing import List

import numpy as np
from numpy.random.mtrand import multivariate_normal

import scipy as sp

import argparse

from gps.agent.allegro_vrep import AgentAllegroVrep, Properties
from gps.agent.noise_generator import GaussianNoiseGenerator
from gps.algorithm.algorithm import ConditionDescription
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.controller.linear_gaussian_controller import LinearGaussianController
from gps.cost import CostAction, CostStateLinear

from gps.dynamics import DynamicsLR, DynamicsPriorGMM
from gps.traj_opt.traj_opt_lqr import TrajOptLQR

from gps.config import Config
from vrep_interface import SimulationClient
from vrep_interface.allegro import AllegroHand, AllegroFingerType


def create_config(args: List[str]) -> Config:
    parser = argparse.ArgumentParser(description="Allegro cylinder setup", prog="--args")
    for t in AllegroFingerType:
        name = t.name.lower()
        parser.add_argument("-{}".format(name[0]), "--use-{}".format(name), action="append_const", dest="used_fingers",
                            const=t, help="Set this to use finger {}.".format(name))

    parser.add_argument("cylinder_direction", type=str, choices=["cw", "ccw"],
                        help="Direction of the cylinder. Use \"cw\" for clockwise and \"ccw\" for counter-clockwise "
                             "reward.")

    parser.add_argument("--use-tactile-sensors", action="store_true", help="Set this to use the tactile sensors in LQR")

    parsed_args = parser.parse_args(args)

    hand = AllegroHand("right")

    num_conditions = 1

    used_fingers = sorted(parsed_args.used_fingers, key=lambda t: t.value)
    cylinder_direction = parsed_args.cylinder_direction
    state_properties = None
    if not parsed_args.use_tactile_sensors:
        state_properties = [p for p in Properties if p != Properties.TACTILE_FORCES]

    np.random.seed(0)
    # twist, proximal, middle, distal
    initial_angles = {
        AllegroFingerType.THUMB: np.array([0.0, 1.0, -0.3, 0.9]),
        AllegroFingerType.INDEX: np.array([0.0, 0.4, 1.0, 0.7]),
        AllegroFingerType.MIDDLE: np.array([0.0, 0.4, 0.8, 0.7]),
        AllegroFingerType.SMALL: np.array([0.0, 0.4, 0.8, 0.7])
    }

    control_cylcle_time_ms = 10

    sigma = {
        AllegroFingerType.THUMB: np.array([0.01, 0.01, 0.01, 0.01]),
        AllegroFingerType.INDEX: np.array([0.01, 0.01, 0.01, 0.01]),
        AllegroFingerType.MIDDLE: np.array([0.01, 0.01, 0.01, 0.01]),
        AllegroFingerType.SMALL: np.array([0.01, 0.01, 0.01, 0.01])
    }
    initial_states = [{t: multivariate_normal(initial_angles[t], np.diag(sigma[t])) for t in used_fingers} for _ in
                      range(num_conditions)]

    for f in hand.fingers.values():
        f.torques = 0.0, 0.0, 0.0, 0.0
        f.initial_torques = f.torques
        f.initial_angles = 0.0, 0.0, 0.0, 0.0

    target_indicator_hand = AllegroHand("right_ti")

    agent = AgentAllegroVrep(time_steps=200,
                             control_cycle_time_ms=control_cylcle_time_ms,
                             action_delay_ms=4,
                             hand=hand,
                             used_fingers=used_fingers,
                             initial_states=initial_states,
                             simulation_client=SimulationClient(connection_port=19998),
                             target_indicator_hand=target_indicator_hand,
                             target_states=initial_states,  # This is just used for the target indicator
                             state_properties=state_properties)
    action_cost = CostAction()
    cost_clockwise = -CostStateLinear(sensor=Properties.CYLINDER_ANGLE)
    cost_counter_clockwise = CostStateLinear(sensor=Properties.CYLINDER_ANGLE)
    cost = 1e-5 * action_cost + (cost_clockwise if cylinder_direction == "cw" else cost_counter_clockwise)

    t = agent.time_steps
    du = agent.action_dimensions
    K = np.zeros((t, du, agent.state_packer.dimensions))  # Controller gains matrix.
    k = np.zeros((t, du))  # Controller bias term.
    covar_single = np.diag([0.0001] * du)  # Covariance matrix
    chol = np.tile(sp.linalg.cholesky(covar_single), (t, 1, 1))  # Cholesky decomposition.
    inv_covar = np.tile(np.linalg.inv(covar_single), (t, 1, 1))  # Inverse of covariance.
    covar = np.tile(covar_single, (t, 1, 1))

    init_traj_distr = LinearGaussianController(K, k, covar, chol, inv_covar)

    dynamics_prior = DynamicsPriorGMM(min_samples_per_cluster=400, max_clusters=40, max_samples=100)
    initial_dynamics = DynamicsLR(regularization=1e-6, prior=dynamics_prior)

    conditions = {
        c: ConditionDescription(cost, init_traj_distr, initial_dynamics, GaussianNoiseGenerator(seed=c))
        for c in range(num_conditions)
    }

    algorithm = AlgorithmTrajOpt(agent=agent,
                                 conditions=conditions,
                                 trajectory_optimizer=TrajOptLQR(del0=1e-6),
                                 training_samples_per_condition=30,
                                 test_samples_per_condition=1,
                                 base_kl_step=0.2,
                                 max_step_mult=10.0)

    return Config(algorithm, 100)
