from enum import Enum
from typing import Optional, List, TYPE_CHECKING

import pybullet

import numpy as np

from .simulation_object import SimulationObject

if TYPE_CHECKING:
    from allegro_pybullet.simulation_body import SimulationBody
    from allegro_pybullet.physics_client import PhysicsClient


class JointControlMode(Enum):
    POSITION_CONTROL = 0
    VELOCITY_CONTROL = 1
    TORQUE_CONTROL = 2
    NONE = 3


class RevoluteJoint(SimulationObject):
    """
    Represents a revolute joint in the simulation.
    """

    def __init__(self, joint_index: int, name: str, initial_angle: float,
                 initial_velocity: float, control_mode: JointControlMode, torque: float = 0.0,
                 target_velocity: float = 0.0):
        """
        :param joint_index:         Index of this joint in the simulation.
        :param name:                Name of this joint in the simulation.
        :param initial_angle:       Initial angle of this joint.
        :param initial_velocity:    Initial velocity of this joint.
        :param control_mode:        Control mode for this joint.
        """
        self.initial_angle = initial_angle
        self.initial_velocity = initial_velocity

        self.torque = torque
        self.target_velocity = target_velocity
        self.target_angle = 0.0
        self.control_mode = control_mode

        self.__observed_angle = None  # type: Optional[float]
        self.__observed_velocity = None  # type: Optional[float]

        self.__joint_index = joint_index
        super(RevoluteJoint, self).__init__(name)

    @property
    def joint_index(self) -> int:
        """
        Index of this joint in the simulation.
        :return:
        """
        return self.__joint_index

    @property
    def observed_angle(self) -> float:
        """
        The currently observed angle of this joint in rad.
        :return:
        """
        return self.__observed_angle

    @property
    def observed_velocity(self) -> float:
        """
        The currently observed angular velocity of this joint in rad/s.
        :return:
        """
        return self.__observed_velocity

    @classmethod
    def _set_initial_state(cls, body: "SimulationBody", simulation_objects: List["RevoluteJoint"]):
        assert all(isinstance(so, RevoluteJoint) for so in simulation_objects)
        for joint in simulation_objects:
            body.call(pybullet.resetJointState, joint.joint_index, joint.initial_angle, joint.initial_velocity)

    @classmethod
    def _observe(cls, body: "SimulationBody", simulation_objects: List["RevoluteJoint"]):
        assert all(isinstance(so, RevoluteJoint) for so in simulation_objects)
        states = body.call(pybullet.getJointStates, [so.joint_index for so in simulation_objects])
        if states is not None:
            for state, so in zip(states, simulation_objects):
                so.__observed_angle = state[0]
                so.__observed_velocity = state[1]

    @classmethod
    def _act(cls, body: "SimulationBody", simulation_objects: List["RevoluteJoint"]):
        assert all(isinstance(so, RevoluteJoint) for so in simulation_objects)

        # Handle torque controlled joints
        torque_control_joints = [rj for rj in simulation_objects if rj.control_mode == JointControlMode.TORQUE_CONTROL]
        if len(torque_control_joints) > 0:
            joint_indices, target_velocities, forces = zip(
                *[(j.joint_index, 1e10 * np.sign(j.torque), abs(j.torque)) for j in torque_control_joints])
            # This is a hack to ensure that the joint will hold the torque
            body.call(pybullet.setJointMotorControlArray, jointIndices=joint_indices,
                      controlMode=pybullet.VELOCITY_CONTROL, targetVelocities=target_velocities, forces=forces)

        # Handle velocity controlled joints
        velocity_control_joints = [rj for rj in simulation_objects if
                                   rj.control_mode == JointControlMode.VELOCITY_CONTROL]
        if len(velocity_control_joints) > 0:
            joint_indices, target_velocities, forces = zip(
                *[(j.joint_index, j.target_velocity, j.torque) for j in velocity_control_joints])
            body.call(pybullet.setJointMotorControlArray, jointIndices=joint_indices,
                      controlMode=pybullet.VELOCITY_CONTROL, targetVelocities=target_velocities, forces=forces)

        # Handle position controlled joints
        position_control_joints = [rj for rj in simulation_objects if
                                   rj.control_mode == JointControlMode.POSITION_CONTROL]
        if len(position_control_joints) > 0:
            joint_indices, target_positions, target_velocities, forces = zip(
                *[(j.joint_index, j.target_angle, j.target_velocity, j.torque) for j in position_control_joints])
            body.call(pybullet.setJointMotorControlArray, jointIndices=joint_indices,
                      controlMode=pybullet.POSITION_CONTROL, targetPositions=target_positions,
                      targetVelocities=target_velocities, forces=forces)

    @classmethod
    def retrieve_all_objects(
            cls, physics_client: "PhysicsClient", body_unique_id: int,
            control_mode: JointControlMode = JointControlMode.POSITION_CONTROL) -> List["RevoluteJoint"]:
        joints = []
        num_joints = physics_client.call(pybullet.getNumJoints, body_unique_id)
        states = physics_client.call(pybullet.getJointStates, body_unique_id, range(num_joints))
        if states is not None:
            for i, state in enumerate(states):
                info = physics_client.call(pybullet.getJointInfo, body_unique_id, i)
                if info[2] == pybullet.JOINT_REVOLUTE:
                    torque = 0.0
                    target_velocity = 0.0
                    if control_mode in [JointControlMode.POSITION_CONTROL, JointControlMode.VELOCITY_CONTROL]:
                        torque = info[10]
                        if control_mode == JointControlMode.POSITION_CONTROL:
                            target_velocity = info[11]
                    joints.append(RevoluteJoint(i, info[1].decode("utf-8"), state[0], state[1], control_mode, torque,
                                                target_velocity))
        return joints
