from enum import Enum
from typing import List, Dict, Mapping

from allegro_pybullet.simulation_object import RevoluteJoint, Link
from allegro_pybullet.simulation_object import TactileSensor
from allegro_pybullet.util import ReadOnlyOrderedDict


class AllegroFingerTypes(Enum):
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    SMALL = 3


class AllegroFingerJointTypes(Enum):
    TWIST = 0
    PROXIMAL = 1
    MIDDLE = 2
    DISTAL = 3


class AllegroFingerLinkTypes(Enum):
    METACARPAL = 0
    PROXIMAL = 1
    MIDDLE = 2
    DISTAL = 3
    TIP = 4


class AllegroFinger:
    """
    Represents a finger of the allegro hand with all its joints.
    """

    def __init__(self, finger_type: AllegroFingerTypes, joints: Dict[AllegroFingerJointTypes, RevoluteJoint],
                 links: Dict[AllegroFingerLinkTypes, Link], tactile_sensor: TactileSensor):
        assert all(t in joints for t in AllegroFingerJointTypes)
        assert all(l in links for l in AllegroFingerLinkTypes)
        self.__joints = ReadOnlyOrderedDict(((t, joints[t]) for t in AllegroFingerJointTypes))
        self.__links = ReadOnlyOrderedDict(((t, links[t]) for t in AllegroFingerLinkTypes))
        self.__tactile_sensor = tactile_sensor
        self.__finger_type = finger_type

    @property
    def joints(self) -> Mapping[AllegroFingerJointTypes, RevoluteJoint]:
        """
        All joints of this finger.
        :return:
        """
        return self.__joints

    @property
    def links(self) -> Mapping[AllegroFingerLinkTypes, RevoluteJoint]:
        """
        All links of this finger.
        :return:
        """
        return self.__links

    @property
    def tactile_sensor(self) -> TactileSensor:
        return self.__tactile_sensor

    @property
    def finger_type(self) -> AllegroFingerTypes:
        return self.__finger_type

    @property
    def angles(self) -> List[float]:
        return [j.observed_angle for j in self.joints.values()]

    @property
    def angular_velocities(self) -> List[float]:
        return [j.observed_velocity for j in self.joints.values()]

    @property
    def initial_angles(self) -> List[float]:
        return [j.initial_angle for j in self.joints.values()]

    @initial_angles.setter
    def initial_angles(self, value: List[float]):
        for j, v in zip(self.joints.values(), value):
            j.initial_angle = v

    @property
    def torques(self) -> List[float]:
        return [j.torque for j in self.joints.values()]

    @torques.setter
    def torques(self, value: List[float]):
        for j, v in zip(self.joints.values(), value):
            j.torque = v

    def __repr__(self) -> str:
        return "Allegro {} finger" if self.__finger_type != AllegroFingerTypes.THUMB else "Allegro thumb"
