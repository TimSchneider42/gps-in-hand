from typing import Dict, Any, Iterable, Tuple

import numpy as np

from allegro_pybullet import PhysicsClient
from allegro_pybullet.simulation_body.allegro_hand import AllegroRightHand


class EnvironmentPlugin:
    def __init__(self, state_labels: Iterable[Tuple[Any, int]],
                 observation_labels: Iterable[Tuple[Any, int]]):
        self.__state_labels = state_labels
        self.__observation_labels = observation_labels

    def on_initialize(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        pass

    def on_terminate(self):
        pass

    def reset(self, condition: int):
        pass

    def on_start(self):
        pass

    def on_sample_complete(self):
        pass

    def get_state(self) -> Dict[Any, np.ndarray]:
        pass

    @property
    def state_labels(self) -> Iterable[Tuple[Any, int]]:
        return self.__state_labels

    @property
    def observation_labels(self) -> Iterable[Tuple[Any, int]]:
        return self.__observation_labels

    @property
    def tracking_point_labels(self):
        return []
