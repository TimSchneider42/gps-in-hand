from typing import Dict, Optional

import numpy as np

from allegro_pybullet import PhysicsClient
from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerJointType, AllegroRightHand, AllegroFingerType
from allegro_pybullet.simulation_object import JointControlMode
from gps.agent.allegro_pybullet.initialization_sequence import InitializationSequence


class InitializationSequenceTargetAngles(InitializationSequence):
    def __init__(self, initial_angles: Dict[AllegroFingerType, np.ndarray],
                 initial_target_angles: Optional[Dict[AllegroFingerType, np.ndarray]] = None,
                 initialization_iterations: int = 10, stop_init_on_contact: bool = True):
        self.__initial_angles = initial_angles
        self.__initial_target_angles = initial_target_angles
        self.__initialization_iterations = initialization_iterations
        self.__stop_init_on_contact = stop_init_on_contact

    def on_initialize(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        # Fill up initial positions with values from the URDF
        for t in AllegroFingerType:
            if t not in self.__initial_angles:
                self.__initial_angles[t] = np.array(
                    [hand.fingers[t].joints[jt].initial_position for jt in AllegroFingerJointType])

        if self.__initial_target_angles is None:
            self.__initial_target_angles = {}
        for t in AllegroFingerType:
            if t not in self.__initial_target_angles:
                # If the initial target position of a finger is not set, we use its initial position
                self.__initial_target_angles[t] = self.__initial_angles[t]

    def run(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        # Set initial positions and target positions
        for t in AllegroFingerType:
            for i, jt in enumerate(AllegroFingerJointType):
                joint = hand.fingers[t].joints[jt]
                joint.initial_position = self.__initial_angles[t][i]
                joint.target_position = self.__initial_target_angles[t][i]
                joint.set_control_mode(JointControlMode.POSITION_CONTROL)
                joint.target_velocity = 0.5
                joint.torque_force = 0.05

        # Make initialization steps so that the joints have time to reach their target positions
        physics_client.reset_to_initial_state()

        disabled_fingers = set()

        for i in range(self.__initialization_iterations):
            physics_client.step_simulation()
            if self.__stop_init_on_contact:
                # Stop finger when it is in contact
                for t in AllegroFingerType:
                    if t not in disabled_fingers and np.any(hand.fingers[t].tactile_sensor.tactel_forces != 0):
                        for jt in AllegroFingerJointType:
                            joint = hand.fingers[t].joints[jt]
                            joint.set_control_mode(JointControlMode.POSITION_CONTROL)
                            joint.target_position = joint.observed_position
                    disabled_fingers.add(t)
