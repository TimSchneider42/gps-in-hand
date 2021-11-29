from typing import Dict

import numpy as np
import pybullet

from allegro_pybullet import PhysicsClient
from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerJointType, AllegroRightHand, AllegroFingerType, \
    AllegroFingerLinkType
from allegro_pybullet.simulation_object import JointControlMode, JointTypes
from gps.agent.allegro_pybullet.initialization_sequence import InitializationSequence


class InitializationSequenceIK(InitializationSequence):
    def __init__(self, target_tip_positions: Dict[AllegroFingerType, np.ndarray]):
        self.__target_tip_positions = target_tip_positions
        self.__initial_angles = None

    def on_initialize(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        self.__initial_angles = {}
        joint_indices = {}
        current_index = 0
        for j in sorted(hand.joints.values(), key=lambda j: j.joint_index):
            if j.joint_type != JointTypes.FIXED:
                joint_indices[j] = current_index
                current_index += 1

        # Compute inverse kinematics
        for t, p in self.__target_tip_positions.items():
            index = hand.fingers[t].links[AllegroFingerLinkType.TIP].link_index
            angles = hand.call(pybullet.calculateInverseKinematics, index, p, maxNumIterations=1000)
            self.__initial_angles[t] = np.array(
                [angles[joint_indices[hand.fingers[t].joints[jt]]] for jt in AllegroFingerJointType])

    def run(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        # Set initial positions and target positions
        for t in AllegroFingerType:
            for i, jt in enumerate(AllegroFingerJointType):
                joint = hand.fingers[t].joints[jt]
                joint.initial_position = self.__initial_angles[t][i]
                joint.target_position = joint.initial_position
                joint.set_control_mode(JointControlMode.POSITION_CONTROL)

        # Make initialization steps so that the joints have time to reach their target positions
        physics_client.reset_to_initial_state()

        x = 3