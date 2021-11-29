from typing import Dict, Mapping, Iterable, Tuple
from typing import Optional

import numpy as np
import pybullet

import os

from allegro_pybullet.simulation_object import TactileSensor
from allegro_pybullet.simulation_object.revolute_joint import JointControlMode
from .allegro_finger import AllegroFinger, AllegroFingerJointTypes, AllegroFingerLinkTypes, AllegroFingerTypes
from allegro_pybullet.simulation_body import URDFBody


class AllegroHand(URDFBody):
    def __init__(self, urdf_filename: str, base_position: Optional[np.ndarray] = None,
                 base_orientation: Optional[np.ndarray] = None, global_scaling: float = 1.0,
                 joint_control_mode: JointControlMode = JointControlMode.POSITION_CONTROL):
        self.__urdf_filename = urdf_filename
        self.__fingers = None  # type: Optional[Dict[AllegroFingerTypes, AllegroFinger]]
        self.__name_prefix = os.path.splitext(urdf_filename)[0].split("_")[-1]  # type: str
        self.__joint_control_mode = joint_control_mode

        flags = [pybullet.URDF_USE_INERTIA_FROM_FILE, pybullet.URDF_USE_SELF_COLLISION,
                 pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS]
        super(AllegroHand, self).__init__(self.__urdf_filename, base_position, base_orientation,
                                          use_maximal_coordinates=False, use_fixed_base=True, flags=sum(flags),
                                          global_scaling=global_scaling)

    def _setup(self):
        bid, sim_objs = super(AllegroHand, self)._setup()
        for rj in self.revolute_joints.values():
            rj.control_mode = self.__joint_control_mode
        self.__fingers = {
            t: AllegroFinger(
                t,
                {
                    jt: self.revolute_joints["_".join([self.__name_prefix, t.name.lower(), jt.name.lower(), "joint"])]
                    for jt in AllegroFingerJointTypes
                },
                {
                    lt: self.links["_".join([self.__name_prefix, t.name.lower(), lt.name.lower(), "link"])]
                    for lt in AllegroFingerLinkTypes
                },
                TactileSensor(self.links["_".join([self.__name_prefix, t.name.lower(), "tip_link"])],
                              tactel_density=50.0, force_loss=1.0,
                              tactel_regions=[(np.array([0.3, 0, 0.5]), 0.5),
                                              (np.array([0.7, 0, 0.2]), 1.4),
                                              (np.array([0.5, 0, -0.4]), 0.6)])
            )
            for t in AllegroFingerTypes
        }
        return bid, sim_objs + [f.tactile_sensor for f in self.__fingers.values()]

    @property
    def fingers(self) -> Optional[Mapping[AllegroFingerTypes, AllegroFinger]]:
        return self.__fingers

    @property
    def name_prefix(self) -> str:
        return self.__name_prefix

    def _reset(self):
        self.__fingers = None

    def __repr__(self):
        return "Allegro {} hand".format(self.name_prefix)
