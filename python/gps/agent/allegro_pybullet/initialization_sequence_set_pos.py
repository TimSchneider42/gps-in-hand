from typing import Dict, Optional

import numpy as np

from allegro_pybullet import PhysicsClient
from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerJointType, AllegroRightHand, AllegroFingerType
from allegro_pybullet.simulation_object import JointControlMode
from gps.agent.allegro_pybullet.initialization_sequence import InitializationSequence


class InitializationSequenceSetPos(InitializationSequence):
    def __init__(self, initial_angles: Dict[AllegroFingerType, np.ndarray]):
        self.__initial_angles = initial_angles

    def on_initialize(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        # Fill up initial positions with values from the URDF
        for t in AllegroFingerType:
            if t not in self.__initial_angles:
                self.__initial_angles[t] = np.array(
                    [hand.fingers[t].joints[jt].initial_position for jt in AllegroFingerJointType])

    def run(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        # Set initial positions and target positions
        for t in AllegroFingerType:
            for i, jt in enumerate(AllegroFingerJointType):
                joint = hand.fingers[t].joints[jt]
                joint.initial_position = self.__initial_angles[t][i]
                joint.set_control_mode(JointControlMode.POSITION_CONTROL)
                joint.target_velocity = 0.5
                joint.torque_force = 0.05

        # Make initialization steps so that the joints have time to reach their target positions
        physics_client.reset_to_initial_state()
