from typing import Optional, List, TYPE_CHECKING, Tuple, Iterable

import math
import numpy as np
import pybullet

from allegro_pybullet.simulation_object import Link

if TYPE_CHECKING:
    from allegro_pybullet.simulation_body import SimulationBody


class TactileSensor(Link):
    def __init__(self, link: "Link", position_offset: Optional[np.ndarray] = None,
                 orientation_offset: Optional[np.ndarray] = None, tactel_density: float = 50.0,
                 force_loss: float = 1.0, tactel_regions: Optional[Iterable[Tuple[np.ndarray, float]]] = None,
                 display_tactels: bool = False):
        self.__position_offset = np.zeros(3) if position_offset is None else position_offset
        self.__orientation_offset = np.array([0, 0, 0, 1]) if orientation_offset is None else orientation_offset
        if tactel_regions is None:
            tactel_regions = [(np.array([0, 0, 1]), math.pi / 2)]
        self.__tactel_vectors = _filter_tactel_vectors(_distribute_uniformly(1.0, tactel_density), tactel_regions)
        self.__display_tactels = display_tactels
        self.__line_ids = []
        self.__force_loss = force_loss
        self.__tactel_forces = np.zeros(len(self.__tactel_vectors))
        super(TactileSensor, self).__init__(link.link_index, "_".join([link.name, "tactilesensor"]))

    def _update_tactels(self, body: "SimulationBody"):
        pc = body.physics_client

        # Link to world frame
        lf_to_wf = self.observed_position, self.observed_quaternion

        # Tactile sensor center to link frame
        sf_to_lf = self.__position_offset, self.__orientation_offset

        # Tactile sensor center to world frame
        sf_to_wf = pybullet.multiplyTransforms(*lf_to_wf, *sf_to_lf)

        # World frame to tactile sensor
        wf_to_sf = pybullet.invertTransform(*sf_to_wf)

        # Clear previous visuals
        for lid in self.__line_ids:
            pc.call(pybullet.removeUserDebugItem, lid)
        self.__line_ids.clear()

        tactel_forces = np.zeros(len(self.__tactel_vectors))

        contact_points_wf = [(cp[5], cp[9]) for cp in body.call(pybullet.getContactPoints, linkIndexA=self.link_index)]

        for cp_wf, force in contact_points_wf:
            contact_point_sf, _ = pybullet.multiplyTransforms(*wf_to_sf, cp_wf, np.array([0, 0, 0, 1]))
            for i, tv in enumerate(self.__tactel_vectors):
                tactel_forces[i] += (self.__compute_tactel_force(tv, contact_point_sf, force))

        for i, tv in enumerate(self.__tactel_vectors):
            if self.__display_tactels:
                line_end_pos_wf, _ = pybullet.multiplyTransforms(*sf_to_wf, tv * 0.02, np.array([0, 0, 0, 1]))
                color = [1.0, 0, 0] if tactel_forces[i] == 0.0 else [0.0, 1.0, 0.0]
                self.__line_ids.append(
                    pc.call(pybullet.addUserDebugLine, sf_to_wf[0], line_end_pos_wf, color, 2.0))

        self.__tactel_forces = tactel_forces

    def __compute_tactel_force(self, tactel_vector: np.ndarray, contact_point_sf: np.ndarray, force: float) -> float:
        return max(0.0, force * (1 - self.__force_loss * _angle(tactel_vector, contact_point_sf)))

    @property
    def tactel_vectors(self) -> Tuple[np.ndarray, ...]:
        return tuple(self.__tactel_vectors)

    @property
    def tactel_forces(self) -> np.ndarray:
        return self.__tactel_forces

    @property
    def display_tactels(self) -> bool:
        return self.__display_tactels

    @display_tactels.setter
    def display_tactels(self, value: bool):
        self.__display_tactels = value

    @classmethod
    def _observe(cls, body: "SimulationBody", simulation_objects: List["TactileSensor"]):
        Link._observe(body, simulation_objects)
        for ts in simulation_objects:
            ts._update_tactels(body)


def _normalize(vector: np.ndarray) -> np.ndarray:
    """ Normalizes the given vector to length 1.  """
    return vector / np.linalg.norm(vector)


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Returns the angle in radians between the given vectors
    :param v1:
    :param v2:
    :return:
    """
    v1_u = _normalize(v1)
    v2_u = _normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def _distribute_uniformly(radius: float, target_tactel_count: float) -> List[np.ndarray]:
    tactel_directions = []

    a = 4 * math.pi * radius ** 2 / target_tactel_count
    d = math.sqrt(a)
    m_theta = int(math.ceil(math.pi / d))
    d_theta = math.pi / m_theta
    d_phi = a / d_theta

    for m in range(m_theta):
        theta = math.pi * (m + 0.5) / m_theta
        m_phi = int(math.ceil(2 * math.pi * math.sin(theta) / d_phi))
        for n in range(m_phi):
            phi = 2 * math.pi * n / m_phi
            direction = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
            tactel_directions.append(direction / np.linalg.norm(direction))

    return tactel_directions


def _filter_tactel_vectors(tactel_vectors: Iterable[np.ndarray], allowed_regions: Iterable[Tuple[np.ndarray, float]]):
    return [v for v in tactel_vectors if any([_angle(v, d) <= a for d, a in allowed_regions])]
