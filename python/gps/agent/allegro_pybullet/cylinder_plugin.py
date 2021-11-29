import math
from enum import Enum
from typing import Dict, Any, Iterable, Optional, List

import numpy as np
import os

import pybullet

from allegro_pybullet import PhysicsClient
from allegro_pybullet.simulation_body import URDFBody
from allegro_pybullet.simulation_body.allegro_hand import AllegroRightHand
from allegro_pybullet.simulation_object import Joint, JointControlMode
from gps.agent.allegro_pybullet.environment_plugin import EnvironmentPlugin


class CylinderProperty(Enum):
    ANGLE = 0
    VELOCITY = 1
    POSITION = 2


class CylinderPlugin(EnvironmentPlugin):
    def __init__(self, cylinder_base_positions_hand_frame: List[np.ndarray],
                 state_properties: Optional[Iterable[CylinderProperty]] = None,
                 observation_properties: Optional[Iterable[CylinderProperty]] = None):
        labels = [(CylinderProperty.ANGLE, 1), (CylinderProperty.VELOCITY, 1)]
        super(CylinderPlugin, self).__init__(
            [l for l in labels if state_properties is None or l[0] in state_properties],
            [l for l in labels if observation_properties is None or l[0] in observation_properties])

        self.__cylinder_body = URDFBody(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "allegro_pybullet_data/cylinder_description.urdf")),
            use_fixed_base=True, base_position=cylinder_base_positions_hand_frame[0])
        self.__cylinder_joint: Optional[Joint] = None
        self.__cylinder_base_positions_hand_frame = cylinder_base_positions_hand_frame
        self.__hand: Optional[AllegroRightHand] = None

    def on_initialize(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        physics_client.add_body(self.__cylinder_body)

        # Add simulated joint friction
        cylinder_joint = self.__cylinder_body.joints["cylinder_joint"]
        cylinder_joint.set_control_mode(JointControlMode.VELOCITY_CONTROL)

        # Set lateral friction of cylinder
        self.__cylinder_body.links["cylinder_link"].lateral_friction = 1.5

        self.__cylinder_joint = cylinder_joint
        self.__hand = hand

    def on_terminate(self):
        pass

    def reset(self, condition: int):
        # Set cylinder base position
        hand_pose = pybullet.multiplyTransforms(*self.__hand.base_link.initial_pose, np.zeros(3),
                                                pybullet.getQuaternionFromEuler([-math.pi / 2, 0, 0]))
        pos_hand_frame = self.__cylinder_base_positions_hand_frame[condition]
        pos, rot = pybullet.multiplyTransforms(*hand_pose, pos_hand_frame, np.array([0, 0, 0, 1]))
        self.__cylinder_body.base_link.initial_pose = pos, rot

    def on_sample_complete(self):
        pass

    def on_start(self):
        self.__cylinder_joint.initial_position = 0.0
        self.__cylinder_body._set_initial_state()
        self.__cylinder_body._observe()

    def get_state(self) -> Dict[Any, np.ndarray]:
        return {
            CylinderProperty.ANGLE: np.array([self.__cylinder_joint.observed_position]),
            CylinderProperty.VELOCITY: np.array([self.__cylinder_joint.observed_velocity]),
            CylinderProperty.POSITION: self.__cylinder_body.base_link.observed_position
        }
