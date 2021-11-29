from typing import Optional

import numpy as np
import os

from allegro_pybullet.simulation_object.revolute_joint import JointControlMode
from .allegro_hand import AllegroHand


class AllegroRightHand(AllegroHand):
    def __init__(self, base_position: Optional[np.ndarray] = None,
                 base_orientation: Optional[np.ndarray] = None, global_scaling: float = 1.0,
                 joint_control_mode: JointControlMode = JointControlMode.POSITION_CONTROL):
        urdf_filename = os.path.join(
            os.path.dirname(__file__), "../../allegro_hand_description/allegro_hand_description_right.urdf")
        super(AllegroRightHand, self).__init__(urdf_filename, base_position, base_orientation, global_scaling,
                                               joint_control_mode)
