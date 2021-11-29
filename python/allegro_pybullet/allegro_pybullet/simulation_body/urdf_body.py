from typing import Tuple, List, Optional, Dict, TYPE_CHECKING

import pybullet
import numpy as np

from allegro_pybullet.simulation_body import SimulationBody
from allegro_pybullet.util import ReadOnlyDict
from allegro_pybullet.simulation_object import RevoluteJoint, Link, BaseLink

if TYPE_CHECKING:
    from allegro_pybullet.simulation_object import SimulationObject


class URDFBody(SimulationBody):
    def __init__(self, urdf_filename: str, base_position: Optional[np.ndarray] = None,
                 base_orientation: Optional[np.ndarray] = None, use_maximal_coordinates: bool = False,
                 use_fixed_base: bool = False, flags: int = 0, global_scaling: float = 1.0):
        self.__urdf_filename = urdf_filename
        self.__base_position = np.zeros(3) if base_position is None else base_position
        self.__base_orientation = np.array([0, 0, 0, 1]) if base_orientation is None else base_orientation
        self.__use_maximal_coordinates = use_maximal_coordinates
        self.__use_fixed_base = use_fixed_base
        self.__flags = flags
        self.__global_scaling = global_scaling
        self.__revolute_joints = None  # type: Optional[Dict[str, RevoluteJoint]]
        self.__links = None  # type: Optional[Dict[str, Link]]
        super(URDFBody, self).__init__()

    def _setup(self) -> Tuple[int, List["SimulationObject"]]:
        bid = self.physics_client.call(pybullet.loadURDF, self.__urdf_filename, self.__base_position,
                                       self.__base_orientation, self.__use_maximal_coordinates,
                                       self.__use_fixed_base, self.__flags, self.__global_scaling)
        revolute_joints = RevoluteJoint.retrieve_all_objects(self.physics_client, bid)
        links = Link.retrieve_all_objects(self.physics_client, bid)
        self.__revolute_joints = ReadOnlyDict({rj.name: rj for rj in revolute_joints})
        self.__links = ReadOnlyDict({l.name: l for l in links})
        return bid, revolute_joints

    @property
    def revolute_joints(self) -> Optional[Dict[str, RevoluteJoint]]:
        return self.__revolute_joints

    @property
    def links(self) -> Optional[Dict[str, Link]]:
        return self.__links

    @property
    def base_link(self) -> Optional[BaseLink]:
        return self.__links["base_link"] if self.__links is not None else None
