import math
from enum import Enum
from pybullet import multiplyTransforms
from typing import Dict, Any, Iterable, Tuple, Optional, List

import numpy as np
import os

import pybullet

from allegro_pybullet import PhysicsClient
from allegro_pybullet.simulation_body import URDFBody
from allegro_pybullet.simulation_body.allegro_hand import AllegroRightHand, AllegroFingerLinkType, AllegroFingerType
from allegro_pybullet.simulation_object import Link
from gps.agent.allegro_pybullet.environment_plugin import EnvironmentPlugin


class CuboidProperty(Enum):
    POSITION = 0
    QUATERNION = 1
    EULER_ANGLES = 2
    LINEAR_VELOCITY = 3
    ANGULAR_VELOCITY = 4
    TRACKING_POINT_POSITIONS = 5
    TRACKING_POINT_VELOCITIES = 6
    TRACKING_POINT_TARGET_POSITIONS = 7
    FINGER_TIP_DISTANCES = 8


class CuboidPlugin(EnvironmentPlugin):
    TRACKING_POINTS = {
        "top_ne": np.array([0.025, 0.025, 0.1]),
        "top_nw": np.array([-0.025, 0.025, 0.1]),
        "top_se": np.array([0.025, -0.025, 0.1]),
        "top_sw": np.array([-0.025, -0.025, 0.1]),
        "bottom_ne": np.array([0.025, 0.025, -0.1]),
        "bottom_nw": np.array([-0.025, 0.025, -0.1]),
        "bottom_se": np.array([0.025, -0.025, -0.1]),
        "bottom_sw": np.array([-0.025, -0.025, -0.1])
    }

    def __init__(self, base_poses_hand_frame: Iterable[Tuple[np.ndarray, np.ndarray]],
                 used_fingers: List[AllegroFingerType],
                 state_properties: Optional[Iterable[CuboidProperty]] = None,
                 observation_properties: Optional[Iterable[CuboidProperty]] = None,
                 target_poses: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                 used_tracking_points: Optional[Iterable[str]] = None):
        if used_tracking_points is None:
            used_tracking_points = self.TRACKING_POINTS.keys()
        assert all(tp in self.TRACKING_POINTS for tp in used_tracking_points)
        self.__used_tracking_points = used_tracking_points
        labels = [(CuboidProperty.POSITION, 3), (CuboidProperty.QUATERNION, 4), (CuboidProperty.EULER_ANGLES, 3),
                  (CuboidProperty.TRACKING_POINT_POSITIONS, 3 * len(used_tracking_points)),
                  (CuboidProperty.TRACKING_POINT_VELOCITIES, 3 * len(used_tracking_points)),
                  (CuboidProperty.FINGER_TIP_DISTANCES, len(used_fingers))]
        base_poses_hand_frame = list(base_poses_hand_frame)
        super(CuboidPlugin, self).__init__(
            [l for l in labels if state_properties is None or l[0] in state_properties],
            [l for l in labels if observation_properties is None or l[0] in observation_properties])

        self.__body = URDFBody(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "allegro_pybullet_data/cuboid_description.urdf")),
            use_fixed_base=False, base_position=base_poses_hand_frame[0][0],
            base_orientation=base_poses_hand_frame[0][1])
        if target_poses is not None:
            self.__static_body = URDFBody(
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__),
                                 "allegro_pybullet_data/cuboid_no_collision_description.urdf")),
                use_fixed_base=True, base_position=target_poses[0][0], base_orientation=target_poses[0][1])
        else:
            self.__static_body = None
        self.__target_poses = target_poses
        self.__base_poses_hand_frame = base_poses_hand_frame
        self.__tracking_links: Optional[List[Link]] = None
        self.__target_tracking_links: Optional[List[Link]] = None
        self.__hand: Optional[AllegroRightHand] = None
        self.__physics_client: Optional[PhysicsClient] = None
        self.__used_fingers = used_fingers
        self.__current_constraint = None

    def on_initialize(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        physics_client.add_body(self.__body)
        self.__physics_client = physics_client

        if self.__target_poses is not None:
            physics_client.add_body(self.__static_body)
            self.__target_tracking_links: List[Link] \
                = [self.__static_body.simulation_objects_by_name[f"{n}_link"] for n in self.__used_tracking_points]

        # Set lateral friction of cuboid
        self.__body.base_link.lateral_friction = 1.5

        self.__tracking_links: List[Link] \
            = [self.__body.simulation_objects_by_name[f"{n}_link"] for n in self.__used_tracking_points]
        self.__hand = hand

    def on_terminate(self):
        pass

    def reset(self, condition: int):
        # Set cuboid base position
        hand_pose = pybullet.multiplyTransforms(*self.__hand.base_link.initial_pose, np.zeros(3),
                                                pybullet.getQuaternionFromEuler([-math.pi / 2, 0, 0]))
        pose_hand_frame = self.__base_poses_hand_frame[condition]
        pos, rot = pybullet.multiplyTransforms(*hand_pose, *pose_hand_frame)

        self.__body.base_link.initial_pose = pos, rot
        self.__current_constraint = self.__body.call(
            pybullet.createConstraint, self.__body.base_link.link_index, -1, -1, pybullet.JOINT_FIXED,
            np.array([1, 0, 0]), np.array([0, 0, 0]), pos, np.array([0, 0, 0, 1]), rot)

        if self.__target_poses is not None:
            pos, rot = pybullet.multiplyTransforms(*hand_pose, *self.__target_poses[condition])
            self.__static_body.base_link.initial_position = pos
            self.__static_body.base_link.initial_orientation = rot

    def on_start(self):
        self.__physics_client.call(pybullet.removeConstraint, self.__current_constraint)

    def on_sample_complete(self):
        pass

    def get_state(self) -> Dict[Any, np.ndarray]:
        max_dist = 0.1
        tip_distances = np.zeros(len(self.__used_fingers))
        for i, t in enumerate(self.__used_fingers):
            closest_points = self.__physics_client.call(
                pybullet.getClosestPoints, self.__body.body_unique_id, self.__hand.body_unique_id, max_dist,
                self.__body.base_link.link_index, self.__hand.fingers[t].links[AllegroFingerLinkType.TIP].link_index)
            if len(closest_points) > 0:
                tip_distances[i] = max(min([c[8] for c in closest_points]), 0.0)
            else:
                tip_distances[i] = max_dist
        state = {
            CuboidProperty.POSITION: self.__body.base_link.observed_position,
            CuboidProperty.QUATERNION: self.__body.base_link.observed_quaternion,
            CuboidProperty.LINEAR_VELOCITY: self.__body.base_link.observed_linear_velocity,
            CuboidProperty.ANGULAR_VELOCITY: self.__body.base_link.observed_angular_velocity,
            CuboidProperty.EULER_ANGLES: self.__body.base_link.observed_euler_angles,
            CuboidProperty.TRACKING_POINT_POSITIONS: np.array(
                [t.observed_position for t in self.__tracking_links]).reshape((-1,)),
            CuboidProperty.TRACKING_POINT_VELOCITIES: np.array(
                [t.observed_linear_velocity for t in self.__tracking_links]).reshape((-1,)),
            CuboidProperty.FINGER_TIP_DISTANCES: tip_distances
        }
        if self.__target_tracking_links is not None:
            state[CuboidProperty.TRACKING_POINT_TARGET_POSITIONS] = np.array(
                [t.observed_position for t in self.__target_tracking_links]).reshape((-1,))
        return state

    def compute_tracking_points(self, base_pose: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return np.array([
            np.array(multiplyTransforms(self.TRACKING_POINTS[n], np.array([0, 0, 0, 1]), *base_pose)[0])
            for n in self.__used_tracking_points
        ])

    @property
    def tracking_point_labels(self):
        return []
