"""
Configuration for Allegro hand cylinder setup using pybullet.
"""
import json
import math
from random import Random
from types import SimpleNamespace
from typing import List

import numpy as np
import os

import pybullet
import scipy as sp

import argparse

from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerType, AllegroFingerJointType
from gps.agent.allegro_pybullet.agent_allegro_pybullet import AgentAllegroPybullet, Property
from gps.agent.allegro_pybullet.cuboid_plugin import CuboidPlugin, CuboidProperty
from gps.agent.allegro_pybullet.cylinder_plugin import CylinderPlugin, CylinderProperty
from gps.agent.allegro_pybullet.initialization_sequence_target_angles import InitializationSequenceTargetAngles
from gps.agent.allegro_pybullet.initialization_sequence_set_pos import InitializationSequenceSetPos
from gps.agent.noise_generator import GaussianNoiseGenerator
from gps.algorithm.algorithm import ConditionDescription
from gps.algorithm.algorithm_badmm import AlgorithmBADMM, LGStepMode
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.cost.cost_sqrt import CostSqrt
from gps.policy import PolicyPriorGMM
from gps.policy import PolicyTf
from gps.cost import CostAction, CostStateLinear, CostScalar, CostTargetState, CostLog, CostSum, CostAbs, CostMin, \
    CostSigmoid, CostArccos, CostQuaternionDotProduct

from gps.dynamics import DynamicsLR, DynamicsPriorGMM
from gps.controller.linear_gaussian_controller import LinearGaussianController
from gps.neural_network import NeuralNetworkFullyConnected
from gps.policy_opt import PolicyOptTf
from gps.traj_opt.traj_opt_lqr import TrajOptLQR

from gps.config import Config, DebugCostFunction

import logging

from gps.gmm import GMMPython
from gps.gmm.gmm_sklearn import GMMSklearn

LOGGER = logging.getLogger(__file__)


def create_config(args: List[str]) -> Config:
    parser = argparse.ArgumentParser(description="Allegro cylinder setup", prog="--args")
    for t in AllegroFingerType:
        name = t.name.lower()
        parser.add_argument("-{}".format(name[0]), "--use-{}".format(name), action="append_const", dest="used_fingers",
                            const=t, help="Set this to use finger {}.".format(name))

    parser.add_argument("--use-tactile-sensors", action="store_true", help="Set this to use the tactile sensors in LQR")
    parser.add_argument("-g", "--gui", action="store_true", help="Show the pybullet debug GUI.")
    parser.add_argument("-r", "--real-time-factor", type=float,
                        help="By setting the real-time factor, real time mode will be enabled and the simulator will "
                             "try to run in real-time multiplied by the specified factor.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument("--gui-options", type=str, default="", help="Options passed to the pybullet GUI.")
    parser.add_argument("--rand-hp", action="store_true", help="Randomize the hyperparameters.")
    parser.add_argument("--hp-seed", type=int, default=0, help="Random seed for hyperparameters (default: 0).")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of GPS iterations to run (default: 50).")
    parser.add_argument("-l", "--log-mp4", type=str, default=None,
                        help="Directory for mp4 logs (Default: do not create mp4 logs).")
    parser.add_argument("-n", "--num-conditions", type=int, default=4, help="Number of conditions to generate.")
    parser.add_argument("-a", "--algorithm", type=str, choices=["lqr", "badmm"], default="lqr",
                        help="Algorithm to use.")
    parser.add_argument("-d", "--device-string", type=str, default=None, help="Tensorflow device string")
    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument("--policy-only-conditions", type=int, nargs="+", default=None,
                        help="Conditions on which not to train but to test policy only.")
    mutex.add_argument("--training-conditions", type=int, nargs="+", default=None,
                       help="Conditions on which to train.")
    parser.add_argument("--tip-pos", action="store_true", help="Include tip positions in state.")
    parser.add_argument("--no-sklearn", action="store_true", help="Use own GMM implementation instead of sklearn.")

    subparsers = parser.add_subparsers(title="environment", dest="environment")
    parser_cylinder = subparsers.add_parser("cylinder", help="Use cylinder environment")
    parser_cylinder.add_argument("cylinder_direction", type=str, choices=["cw", "ccw", "tar", "arb"],
                                 help="Direction of the cylinder. Use \"cw\" for clockwise, \"ccw\" for "
                                      "counter-clockwise, \"arb\" for arbitrary reward and \"tar\" to set "
                                      "a 180 deg rotation as target state.")

    parser_cuboid = subparsers.add_parser("cuboid", help="Use cuboid environment.")
    parser_fpos = subparsers.add_parser("fpos", help="Use finger positioning environment.")

    parsed_args = parser.parse_args(args)

    hp = SimpleNamespace()

    # Set values for parameters
    hp.min_samples_per_cluster = 1000  # 20
    hp.max_clusters = 50  # 50
    hp.max_samples = 150  # 20
    hp.init_var = 1e-4
    hp.base_kl_step = 0.2
    hp.min_step_mult = 0.01
    hp.max_step_mult = 10.0
    hp.max_entropy_weight = 0.0
    hp.regularization = 1e-6
    hp.num_samples = 30
    hp.max_joint_torque = 0.1  # robot max: 0.7
    hp.policy_min_samples_per_cluster = 500
    hp.policy_max_clusters = 50
    hp.policy_max_samples = 150
    hp.policy_strength = 1.0
    hp.lg_step = 0.0002
    hp.lg_step = 0.0
    hp.init_policy_weight = 0.01
    hp.strength = 100.0
    hp.badmm_dual_rate = 0.01
    hp.badmm_dual_rate = 0.0
    hp.inner_iterations = 1
    if parsed_args.rand_hp:
        # Randomize hyperparameters
        hp_rng = Random(parsed_args.hp_seed)
        u = hp_rng.uniform
        g = hp_rng.gauss

        def eu(a: float, b: float):
            return 10 ** u(a, b)

        def eg(mu: float, sigma: float):
            return 10 ** g(mu, sigma)

        # f = u(0.5, 6)
        # hp.max_clusters = int(50 * f)  # int(eg(1.5, 0.3))
        # hp.max_samples = int(20 * f)  # int(eg(1.75, 0.3))
        hp.min_step_mult = eu(-1, -4)

    LOGGER.info("Hyperparameters:")
    for n, v in vars(hp).items():
        LOGGER.info(f"{n}: {v}")

    if os.path.split(os.path.dirname(__file__))[-1] == "config":
        # Save hyperparameters in JSON file
        with open(os.path.join(os.path.dirname(__file__), "hyperparameters.json"), "w") as f:
            json.dump(hp.__dict__, f, indent=2)

    used_fingers_arg = [] if parsed_args.used_fingers is None else parsed_args.used_fingers

    if len(used_fingers_arg) == 0:
        raise ValueError("At least one finger needs to be specified")

    num_conditions = parsed_args.num_conditions
    if parsed_args.training_conditions is not None:
        training_conditions = parsed_args.training_conditions
    elif parsed_args.policy_only_conditions is not None:
        training_conditions = [c for c in range(num_conditions) if c not in parsed_args.policy_only_conditions]
    else:
        training_conditions = list(range(num_conditions))
    policy_only_conditions = [c for c in range(num_conditions) if c not in training_conditions]

    used_fingers = sorted(used_fingers_arg, key=lambda t: t.value)
    state_properties = [Property.JOINT_ANGLES, Property.JOINT_VELOCITIES]
    if parsed_args.tip_pos:
        state_properties += [Property.TIP_POSITION]
    observation_properties = state_properties.copy()
    if parsed_args.use_tactile_sensors:
        # state_properties.append(Property.SUMMED_TACTILE_FORCES)
        observation_properties.append(Property.TACTILE_FORCES)

    plugins = []
    debug_cost_functions = []
    time_steps = 100
    ground_plane_collision = False

    hand_pose = (np.array([0.0, -3.0, 0.1]), np.array([math.pi / 2, 0, math.pi]))

    np_rng = np.random.RandomState(0)

    if parsed_args.environment == "cylinder":
        # twist, proximal, middle, distal
        initial_angles_mean = {
            AllegroFingerType.THUMB: np.array([0.0, 1.3, -0.3, 0.9]),
            AllegroFingerType.INDEX: np.array([0.0, 0.3, 0.7, 0.6]),
            AllegroFingerType.MIDDLE: np.array([0.0, 0.4, 0.8, 0.6]),
            AllegroFingerType.SMALL: np.array([0.0, 0.4, 0.8, 0.6])
        }

        initial_angles_sigma = {
            AllegroFingerType.THUMB: np.array([0.01, 0.01, 0.01, 0.01]),
            AllegroFingerType.INDEX: np.array([0.01, 0.01, 0.01, 0.01]),
            AllegroFingerType.MIDDLE: np.array([0.01, 0.01, 0.01, 0.01]),
            AllegroFingerType.SMALL: np.array([0.01, 0.01, 0.01, 0.01])
        }

        initial_target_angles = {
            AllegroFingerType.THUMB: np.array([0.0, 1.3, 0.35, 1.0]),
            AllegroFingerType.INDEX: np.array([0.0, 0.7, 0.9, 0.8]),
            AllegroFingerType.MIDDLE: np.array([0.0, 0.7, 0.9, 0.8]),
            AllegroFingerType.SMALL: np.array([0.0, 0.7, 0.9, 0.8])
        }

        initial_positions = [
            {
                t: np_rng.multivariate_normal(initial_angles_mean[t], np.diag(initial_angles_sigma[t]))
                if t in used_fingers else np.zeros(4)
                for t in AllegroFingerType
            } for _ in range(num_conditions)]

        initial_target_positions = [{
            t: a if t in used_fingers else np.zeros(4)
            for t, a in initial_target_angles.items()
        }] * num_conditions

        initialization_sequences = [
            InitializationSequenceTargetAngles(initial_positions[c], initial_target_positions[c], 15)
            for c in range(num_conditions)
        ]

        np_rng = np.random.RandomState(1)
        used_joints = [AllegroFingerJointType.PROXIMAL, AllegroFingerJointType.MIDDLE, AllegroFingerJointType.DISTAL]
        cylinder_pos_range = np.array([[0.095, 0.105], [-0.06, 0.02], [-0.1, -0.1]])
        cylinder_pos_range_pol = np.array([[0.097, 0.103], [-0.05, 0.00], [-0.1, -0.1]])
        cylinder_positions = [
            np_rng.uniform(cylinder_pos_range[:, 0], cylinder_pos_range[:, 1])
            if c not in policy_only_conditions else
            np_rng.uniform(cylinder_pos_range_pol[:, 0], cylinder_pos_range_pol[:, 1])
            for c in range(num_conditions)
        ]
        """
        cylinder_positions = [
            np.array([0.105, -0.05, -0.1]),
            np.array([0.105, 0.0, -0.1]),
            np.array([0.1, 0.01, -0.1]),
            np.array([0.11, 0.01, -0.1]),
            np.array([0.1, -0.02, -0.1])
        ]
        """

        for i, d in enumerate(initial_positions):
            LOGGER.info(f" Condition {i}:")
            LOGGER.info("Initial finger angles:")
            for t, v in d.items():
                LOGGER.info(f"  {t.name}: {v}")
            LOGGER.info(f"Cylinder position: {cylinder_positions[i]}")

        state_probs = [CylinderProperty.VELOCITY]

        # Create cost function
        target_tactel_force = 0.01
        penalty_at_target_force = 0.02
        x_offset = target_tactel_force / 2.0
        steepness = -np.log(1.0 / penalty_at_target_force - 1.0) / (target_tactel_force - x_offset)
        contact_cost = 0.5 / len(used_fingers) * CostSum([
            CostSigmoid(CostStateLinear((Property.SUMMED_TACTILE_FORCES, t)), x_offset, steepness) for t in
            used_fingers])
        action_cost = CostAction()
        d = parsed_args.cylinder_direction
        if d == "cw":
            state_cost = -CostStateLinear(sensor=CylinderProperty.VELOCITY)
        elif d == "ccw":
            state_cost = CostStateLinear(sensor=CylinderProperty.VELOCITY)
        elif d == "tar":
            cts1 = CostTargetState(sensor=CylinderProperty.ANGLE, target_state=np.array([math.pi]), l1=1.0, l2=0.0)
            cts2 = CostTargetState(sensor=CylinderProperty.ANGLE, target_state=np.array([-math.pi]), l1=1.0, l2=0.0)
            state_cost = CostMin([cts1, cts2])
            state_probs.append(CylinderProperty.ANGLE)
            pos_err = CostMin([CostAbs(CostStateLinear(sensor=CylinderProperty.ANGLE) - math.pi),
                               CostAbs(CostStateLinear(sensor=CylinderProperty.ANGLE) + math.pi)])
            debug_cost_functions.append((pos_err, "pos_err"))
            debug_cost_functions.append((CostScalar(np.array([0] * (time_steps - 1) + [1])) * pos_err, "pos_err_final"))
            debug_cost_functions.append((CostStateLinear(sensor=CylinderProperty.ANGLE), "pos"))
        else:
            state_cost_cw = -CostStateLinear(sensor=CylinderProperty.VELOCITY)
            state_cost_ccw = CostStateLinear(sensor=CylinderProperty.VELOCITY)
            state_cost = CostMin([state_cost_cw, state_cost_ccw])
        cost = action_cost + state_cost

        debug_cost_functions.append((action_cost, "action_cost"))
        debug_cost_functions.append((state_cost, "state_cost"))
        debug_cost_functions.append((contact_cost, "contact_cost"))

        obs_props = [CylinderProperty.POSITION]

        plugins.append(
            CylinderPlugin(cylinder_positions, state_properties=state_probs, observation_properties=obs_props))
        enable_gravity = False
    elif parsed_args.environment == "cuboid":
        hand_pose[0][2] = 0.5
        ground_plane_collision = True
        """
        finger_tip_pos_cuboid_frame = {
            AllegroFingerType.THUMB: np.array([0, 0.03, 0]),
            AllegroFingerType.INDEX: np.array([0, -0.03, 0.05]),
            AllegroFingerType.MIDDLE: np.array([0, -0.03, 0]),
            AllegroFingerType.SMALL: np.array([0, -0.03, -0.05])
        }

        finger_tip_pos_world_frame = [{
            t: np.array(pybullet.multiplyTransforms(p, np.array([0, 0, 0, 1]), cp, co)[0]) for t, p in
            finger_tip_pos_cuboid_frame.items()
        } for cp, co in zip(cuboid_positions, cuboid_orientations)]

        initialization_sequences = [InitializationSequenceIK(p) for p in finger_tip_pos_world_frame]
        """

        # twist, proximal, middle, distal
        initial_angles_mean = {
            AllegroFingerType.THUMB: np.array([0.4, 1.7, -0.3, 0.9]),
            AllegroFingerType.INDEX: np.array([0.0, 0.4, 0.8, 0.7]),
            AllegroFingerType.MIDDLE: np.array([0.0, 0.4, 0.8, 0.7]),
            AllegroFingerType.SMALL: np.array([0.0, 0.4, 0.8, 0.7])
        }

        initial_angles_sigma = {
            AllegroFingerType.THUMB: np.array([0.01, 0.01, 0.01, 0.01]),
            AllegroFingerType.INDEX: np.array([0.01, 0.01, 0.01, 0.01]),
            AllegroFingerType.MIDDLE: np.array([0.01, 0.01, 0.01, 0.01]),
            AllegroFingerType.SMALL: np.array([0.01, 0.01, 0.01, 0.01])
        }

        initial_target_angles = {
            AllegroFingerType.THUMB: np.array([0.4, 1.7, 0.35, 1.115]),
            AllegroFingerType.INDEX: np.array([0.0, 0.75, 0.95, 1.05]),
            AllegroFingerType.MIDDLE: np.array([0.0, 0.75, 0.95, 1.05]),
            AllegroFingerType.SMALL: np.array([0.0, 0.75, 0.95, 1.05])
        }

        initial_positions = [
            {
                t: np_rng.multivariate_normal(initial_angles_mean[t], np.diag(initial_angles_sigma[t]))
                if t in used_fingers else np.zeros(4)
                for t in AllegroFingerType
            } for _ in range(num_conditions)]

        initial_target_positions = [{
            t: a if t in used_fingers else np.zeros(4)
            for t, a in initial_target_angles.items()
        }] * num_conditions

        initialization_sequences = [
            InitializationSequenceTargetAngles(initial_positions[c], initial_target_positions[c], 20)
            for c in range(num_conditions)
        ]

        np_rng = np.random.RandomState(1)
        used_joints = list(AllegroFingerJointType)
        # cuboid_pos_range = np.array([[0.095, 0.105], [-0.06, 0.01], [0.0, 0.0]])
        cuboid_pos_range = np.array([[0.095, 0.105], [-0.01, 0.01], [0.0, 0.0]])
        cuboid_rot_range = np.array([[-0.2, 0.2], [-0.03, 0.03], [-math.pi / 16 - 0.03, -math.pi / 16 + 0.03]])
        cuboid_positions = [np_rng.uniform(cuboid_pos_range[:, 0], cuboid_pos_range[:, 1]) for _ in
                            range(num_conditions)]
        cuboid_orientations = [
            pybullet.getQuaternionFromEuler(np_rng.uniform(cuboid_rot_range[:, 0], cuboid_rot_range[:, 1])) for _ in
            range(num_conditions)]

        for i, d in enumerate(initial_positions):
            LOGGER.info(f" Condition {i}:")
            LOGGER.info("Initial finger angles:")
            for t, v in d.items():
                LOGGER.info(f"  {t.name}: {v}")
            LOGGER.info(f"Cuboid position: {cuboid_positions[i]}")

        target_position = np.array([0.1, -0.0, 0.0])
        target_orientation = np.array(pybullet.getQuaternionFromEuler([math.pi / 4, 0, -math.pi / 16]))

        cuboid_base_frame = pybullet.multiplyTransforms(
            hand_pose[0], pybullet.getQuaternionFromEuler(hand_pose[1]),
            np.zeros(3), pybullet.getQuaternionFromEuler([-math.pi / 2, 0, 0]))
        tpos_wf, trot_wf = pybullet.multiplyTransforms(*cuboid_base_frame, target_position, target_orientation)
        tpos_wf = np.array(tpos_wf)
        trot_wf = np.array(trot_wf)

        used_tracking_points = ["top_ne", "top_sw", "bottom_nw", "bottom_se"]

        # Create cost function
        ts = time_steps + 1
        ramp = np.array([ts / (ts - i) for i in range(ts)])
        ramp *= ts / np.sum(ramp)
        ramp = CostScalar(np.arange(1, 100, 1) / 50)
        ramp = 1

        target_tactel_force = 0.01
        penalty_at_target_force = 0.02
        x_offset = target_tactel_force / 2.0
        steepness = -np.log(1.0 / penalty_at_target_force - 1.0) / (target_tactel_force - x_offset)
        contact_cost = 0.05 / len(used_fingers) * CostSum([
            CostSigmoid(CostStateLinear((Property.SUMMED_TACTILE_FORCES, t)), x_offset, steepness) for t in
            used_fingers])
        tip_distance_cost = 1.0 / len(used_fingers) * CostStateLinear(sensor=CuboidProperty.FINGER_TIP_DISTANCES)

        action_cost = 16 / (len(used_fingers) * len(used_joints)) * CostAction()
        center_dist = CostTargetState(sensor=CuboidProperty.POSITION, target_state=tpos_wf)
        target_orientation_conj = trot_wf * np.array([1, 1, 1, -1])
        error_angle = 2 * CostMin([
            CostArccos(CostQuaternionDotProduct(CuboidProperty.QUATERNION, target_orientation_conj)),
            CostArccos(CostQuaternionDotProduct(CuboidProperty.QUATERNION, trot_wf))
        ])
        # state_cost_quat = CostTargetQuaternion(sensor=CuboidProperty.QUATERNION, target_quaternion=target_orientation)
        # state_cost = state_cost_pos + 0.1 * state_cost_quat
        state_diff = CostTargetState(sensor=CuboidProperty.TRACKING_POINT_POSITIONS,
                                     target_state=CuboidProperty.TRACKING_POINT_TARGET_POSITIONS)
        state_diff_scaled = ramp * state_diff
        state_diff_log_scaled = 1e-3 * ramp * CostLog(state_diff + 1e-6)
        vel_cost_scaled = 1e-1 * ramp * CostTargetState(
            sensor=CuboidProperty.TRACKING_POINT_VELOCITIES,
            target_state=np.zeros(len(used_tracking_points) * 3))
        # TODO: vel cost?
        cost = state_diff_scaled + state_diff_log_scaled + action_cost
        avg_tp_dist = CostSum([
            CostSqrt(
                CostTargetState(
                    sensor=CuboidProperty.TRACKING_POINT_POSITIONS,
                    target_state=CuboidProperty.TRACKING_POINT_TARGET_POSITIONS,
                    sensor_weights=np.array([0] * (i * 3) + [1] * 3 + [0] * (3 * (len(used_tracking_points) - i - 1)))))
            for i in range(len(used_tracking_points))
        ]) * CostScalar(1.0 / len(used_tracking_points))

        debug_cost_functions.append((state_diff_scaled, "state_diff"))
        debug_cost_functions.append((state_diff_log_scaled, "state_diff_log"))
        # debug_cost_functions.append((vel_cost_scaled, "vel_cost"))
        debug_cost_functions.append((action_cost, "action_cost"))
        # debug_cost_functions.append((contact_cost, "contact_cost"))
        # debug_cost_functions.append((tip_distance_cost, "tip_distance_cost"))
        debug_cost_functions.append((avg_tp_dist, "avg_tp_dist", False))
        debug_cost_functions.append((center_dist, "center_dist", False))
        debug_cost_functions.append((error_angle, "error_angle", False))

        plugins.append(
            CuboidPlugin(zip(cuboid_positions, cuboid_orientations),
                         state_properties=[CuboidProperty.TRACKING_POINT_POSITIONS,
                                           CuboidProperty.TRACKING_POINT_VELOCITIES,
                                           CuboidProperty.FINGER_TIP_DISTANCES],
                         observation_properties=[],
                         target_poses=[(target_position, target_orientation)] * num_conditions,
                         used_fingers=used_fingers,
                         used_tracking_points=used_tracking_points))
        enable_gravity = True
    elif parsed_args.environment == "fpos":
        # twist, proximal, middle, distal
        initial_angles_mean = {t: np.array([0.0, 0.25, 0.25, 0.25]) for t in AllegroFingerType}

        initial_angles_sigma = {t: np.ones(4) * 0.05 for t in AllegroFingerType}

        pos_rng = np.random.RandomState(1)
        initial_positions = [
            {
                t: pos_rng.multivariate_normal(
                    initial_angles_mean[t],
                    np.diag(initial_angles_sigma[t] / (1.0 if c in policy_only_conditions else 1.0)))
                if t in used_fingers else np.zeros(4)
                for t in AllegroFingerType
            } for c in range(num_conditions)]

        target_angles = {t: a for t, a in initial_angles_mean.items() if t in used_fingers}

        cost_action = CostAction()
        # ramp = CostScalar(np.array([100.0 / (time_steps - i) for i in range(time_steps)]))
        cost = cost_action + CostSum(
            [CostTargetState(sensor=(Property.JOINT_ANGLES, t), target_state=s, wp_final_multiplier=1.0) for t, s in
             target_angles.items()])
        # init_angles = {t: np.array([0.0, 0.25, 0.25, 0.25]) for t in used_fingers}
        initialization_sequences = [InitializationSequenceSetPos(initial_positions[c]) for c in range(num_conditions)]
        used_joints = list(AllegroFingerJointType)
        enable_gravity = True
        debug_cost_functions.append((cost_action, "action"))


        summed_err = CostSum(
            [CostTargetState(sensor=(Property.JOINT_ANGLES, t), target_state=target_angles[t], l1=1.0, l2=0.0,
                             alpha=0.0) for t in used_fingers])
        avg_err = summed_err * (1.0 / (len(used_fingers) * len(used_joints)))

        debug_cost_functions.append((avg_err, "avg_err"))
        for t in used_fingers:
            for i, jt in enumerate(used_joints):
                debug_cost_functions.append(
                    (CostTargetState(sensor=(Property.JOINT_ANGLES, t), target_state=target_angles[t], l1=1.0, l2=0.0,
                                     alpha=0.0,
                                     sensor_weights=np.array([int(i == j) for j in range(len(used_joints))])),
                     f"diff {t.name}.{jt.name}"))
    else:
        raise ValueError(f"Unknown environment: {parsed_args.environment}")

    agent = AgentAllegroPybullet(time_steps=time_steps,
                                 time_step_seconds=0.01,
                                 used_fingers=used_fingers,
                                 initialization_sequences=initialization_sequences,
                                 state_properties=state_properties,
                                 observation_properties=observation_properties,
                                 show_gui=parsed_args.gui,
                                 real_time_factor=parsed_args.real_time_factor,
                                 pre_start_pause=0.0 if parsed_args.real_time_factor is None else 1.0,
                                 sub_steps=10,
                                 gui_options=parsed_args.gui_options,
                                 used_joints=used_joints,
                                 recording_directory=parsed_args.log_mp4,
                                 max_joint_torque=hp.max_joint_torque,
                                 environment_plugins=plugins,
                                 enable_gravity=enable_gravity,
                                 hand_pose=(hand_pose[0], pybullet.getQuaternionFromEuler(hand_pose[1])),
                                 ground_plane_collision=ground_plane_collision)

    # Create initial controller
    t = agent.time_steps
    du = agent.action_dimensions
    K = np.zeros((t, du, agent.state_packer.dimensions))  # Controller gains matrix.
    k = np.zeros((t, du))  # Controller bias term.
    covar_single = np.diag([hp.init_var] * du)  # Covariance matrix
    chol = np.tile(sp.linalg.cholesky(covar_single), (t, 1, 1))  # Cholesky decomposition.
    inv_covar = np.tile(np.linalg.inv(covar_single), (t, 1, 1))  # Inverse of covariance.
    covar = np.tile(covar_single, (t, 1, 1))
    init_traj_distr = LinearGaussianController(K, k, covar, chol, inv_covar)

    if parsed_args.no_sklearn:
        gmm = GMMPython(regularization=hp.regularization)
    else:
        gmm = GMMSklearn(regularization=hp.regularization)

    # Create initial dynamics
    dynamics_prior = DynamicsPriorGMM(min_samples_per_cluster=hp.min_samples_per_cluster, max_clusters=hp.max_clusters,
                                      max_samples=hp.max_samples, gmm=gmm, strength=hp.strength)
    initial_dynamics = DynamicsLR(regularization=hp.regularization, prior=dynamics_prior)

    conditions = {
        c: ConditionDescription(cost, init_traj_distr, initial_dynamics,
                                GaussianNoiseGenerator(seed=(c + parsed_args.seed) % 2 ** 32))
        for c in range(num_conditions)
    }

    if parsed_args.algorithm == "lqr":
        algorithm = AlgorithmTrajOpt(agent=agent,
                                     conditions=conditions,
                                     trajectory_optimizer=TrajOptLQR(),
                                     training_samples_per_condition=hp.num_samples,
                                     # training_conditions=[3],  # None,
                                     test_samples_per_condition=1,
                                     base_kl_step=hp.base_kl_step,
                                     min_step_mult=hp.min_step_mult,
                                     max_step_mult=hp.max_step_mult,
                                     max_entropy_weight=hp.max_entropy_weight,
                                     inner_iterations=hp.inner_iterations)
    elif parsed_args.algorithm == "badmm":
        neural_network = NeuralNetworkFullyConnected(action_dimensions=du,
                                                     observation_dimensions=agent.observation_dimensions,
                                                     layer_units=6 * [150])

        initial_policy = PolicyTf(neural_network=neural_network,
                                  covariance=np.tile(np.eye(du) * 0.1, (agent.time_steps, 1, 1)),
                                  device_string=parsed_args.device_string)

        algorithm = AlgorithmBADMM(agent=agent,
                                   conditions=conditions,
                                   initial_policy=initial_policy,
                                   initial_policy_prior=PolicyPriorGMM(
                                       max_clusters=hp.policy_max_clusters,
                                       max_samples=hp.policy_max_samples,
                                       min_samples_per_cluster=hp.policy_min_samples_per_cluster,
                                       strength=hp.policy_strength,
                                       gmm=gmm),
                                   policy_optimizer=PolicyOptTf(batch_size=100, training_iterations=10000),
                                   trajectory_optimizer=TrajOptLQR(),
                                   training_samples_per_condition=hp.num_samples,
                                   training_conditions=training_conditions,
                                   test_samples_per_condition=1,
                                   policy_samples_per_condition=1,
                                   base_kl_step=hp.base_kl_step,
                                   max_entropy_weight=hp.max_entropy_weight,
                                   lg_step_schedule=hp.lg_step,
                                   init_policy_weight=hp.init_policy_weight,
                                   lg_step_mode=LGStepMode.TARGET_KL,
                                   target_kl=0.2,
                                   exp_step_increase=1.05,
                                   exp_step_decrease=0.75,
                                   policy_dual_rate=hp.badmm_dual_rate,
                                   policy_dual_rate_covar=hp.badmm_dual_rate,
                                   inner_iterations=hp.inner_iterations)
    else:
        raise ValueError(f"Unknown algorithm: {parsed_args.algorithm}")
    return Config(algorithm, parsed_args.iterations,
                  {c: [DebugCostFunction(*d) for d in debug_cost_functions] for c in range(num_conditions)})
