from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pybullet

from allegro_pybullet.simulation_object import SimulationObject

if TYPE_CHECKING:
    from allegro_pybullet.physics_client import PhysicsClient
    from allegro_pybullet.simulation_body import SimulationBody


class Link(SimulationObject):
    """
    Represents a link in the simulation.
    """

    def __init__(self, link_index: int, name: str):
        """

        :param link_index:          Index of this link in the simulation.
        :param name:                Name of this link in the simulation.
        """
        self.__observed_position = None  # type: Optional[np.ndarray]
        self.__observed_quaternion = None  # type: Optional[np.ndarray]
        self.__link_index = link_index
        super(Link, self).__init__(name)

    @property
    def observed_euler_angles(self) -> np.ndarray:
        """
        The currently observed orientation in euler angles (rad).
        :return:
        """
        return pybullet.getEulerFromQuaternion(self.__observed_quaternion)

    @property
    def observed_quaternion(self) -> np.ndarray:
        """
        The currently observed orientation in quaternions.
        :return:
        """
        return self.__observed_quaternion

    @property
    def observed_position(self) -> np.ndarray:
        """
        The currently observed position in meters.
        :return:
        """
        return self.__observed_position

    @property
    def link_index(self) -> int:
        """
        Index of this link in the simulation.
        :return:
        """
        return self.__link_index

    @classmethod
    def _observe(cls, body: "SimulationBody", simulation_objects: List["Link"]):
        assert all(isinstance(so, Link) for so in simulation_objects)
        for so in simulation_objects:
            state = body.call(pybullet.getLinkState, so.link_index)
            so.__observed_position = np.array(state[0])
            so.__observed_quaternion = np.array(state[1])

    @classmethod
    def retrieve_all_objects(cls, physics_client: "PhysicsClient", body_unique_id: int) -> List["Link"]:
        # This is a ugly hack as it is not possible to get all links directly. The only way of accessing link names is
        # by calling getJointInfo, which will return the parent link index and the child link name, however for some
        # reason it won't return parent link name and child link index. So this is all about guessing.
        from .base_link import BaseLink
        links: List[Link] = BaseLink.retrieve_all_objects(physics_client, body_unique_id)  # Just the base link
        num_joints = physics_client.call(pybullet.getNumJoints, body_unique_id)
        for i in range(num_joints):
            info = physics_client.call(pybullet.getJointInfo, body_unique_id, i)
            # Just assume that the index is always the next free index
            links.append(Link(links[-1].link_index + 1, info[12].decode("utf-8")))
        return links
