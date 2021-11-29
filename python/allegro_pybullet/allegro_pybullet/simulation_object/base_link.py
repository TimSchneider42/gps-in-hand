from typing import List, TYPE_CHECKING

import numpy as np
import pybullet

from allegro_pybullet.simulation_object.link import Link

if TYPE_CHECKING:
    from allegro_pybullet.simulation_body import SimulationBody
    from allegro_pybullet import PhysicsClient


class BaseLink(Link):
    """
    Represents the base link of a body in the simulation.
    """

    def __init__(self, initial_position: np.ndarray, initial_orientation: np.ndarray):
        """

        :param initial_position:    Initial position of this link.
        :param initial_orientation: Initial orientation of this link in quaternions.
        """
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        super(BaseLink, self).__init__(-1, "base_link")

    @classmethod
    def _set_initial_state(cls, body: "SimulationBody", simulation_objects: List["BaseLink"]):
        assert len(simulation_objects) == 1
        bl = simulation_objects[0]
        assert isinstance(bl, BaseLink)
        body.call(pybullet.resetBasePositionAndOrientation, bl.initial_position, bl.initial_orientation)

    @classmethod
    def retrieve_all_objects(cls, physics_client: "PhysicsClient", body_unique_id: int) -> List["BaseLink"]:
        position, orientation = (np.array(v) for v in
                                 physics_client.call(pybullet.getBasePositionAndOrientation, body_unique_id))
        return [BaseLink(position, orientation)]
